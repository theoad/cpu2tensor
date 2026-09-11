// SPDX-License-Identifier: AGPL-3.0-only
#include <cpu2tensor/trace.hpp>
#include <cpu2tensor/register_selection.hpp>
#include <cpu2tensor/frame_ring.hpp>
#include <cpu2tensor/transition_window.hpp>
#include <dlfcn.h>
#include <qemu-plugin.h>

#include <cerrno>
#include <atomic>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <pthread.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using namespace cpu2tensor;

static_assert(sizeof(uintptr_t) == sizeof(uint64_t), "Capture needs a 64-bit host");
static_assert(max_frame_bytes <= PIPE_BUF, "Frames must fit in one atomic pipe write");
constexpr int capture_exit_code = 125;
constexpr uint64_t clone_vm = 0x100;
constexpr uint64_t clone_thread = 0x10000;
constexpr size_t register_scratch_bytes = 65536;
constexpr size_t register_prefix_bytes = 16;
constexpr size_t memory_prefix_bytes = 24;
constexpr uint16_t initial_register = 1u << 8;

enum class RegisterProfile : uint8_t { none, general, all, selected };
enum class Checkpoint : uint16_t { block = 1, syscall = 2, exit = 3 };

struct Register final {
    qemu_plugin_register* handle = nullptr;
    char name[64]{};
    uint8_t* previous = nullptr;
    uint32_t width = 0;
    bool sampled = false;
    uint8_t raw_field = 255;
};

enum class State : uint8_t { unused, active, ended };

// QEMU calls execution and exit callbacks on the owning vCPU. The final callback
// runs after instrumentation has stopped, so it can drain the remaining sources.
// A source never shares its event buffer or sequence counter with another vCPU.
struct alignas(frame_ring_cache_line_bytes) Source final {
    // Only lifecycle/idle/drain callbacks take this lock. Execution callbacks
    // own their source while running; external drains require a QMP world stop.
    pthread_mutex_t cold_mutex = PTHREAD_MUTEX_INITIALIZER;
    uint8_t bytes[max_frame_bytes]{};
    uint64_t next = 0;
    uint32_t count = 0;
    uint32_t used = 0;
    Kind kind = Kind::blocks;
    Kind run_kind = Kind::blocks;
    uint32_t run_offset = 0;
    FrameRing<> ring;
    std::atomic<uint64_t> frames_queued{0};
    std::atomic<uint64_t> frames_sent{0};
    uint64_t full_waits = 0;
    Register* registers = nullptr;
    GByteArray* scratch = nullptr;
    uint32_t register_count = 0;
    Register controls[4]{};
    uint64_t raw_state[28]{};
    uint64_t context[6]{};
    uint64_t context_sequence = 0;
    bool context_sampled = false;
    // The descriptor at index zero can have a null but valid QEMU handle.
    int32_t pc_register = -1;
    State state = State::unused;
    bool recording = false;
    uint64_t generation = 0;
    std::atomic<uint64_t> published{0};
    std::atomic<uint64_t> drained{0};
};

Source sources[max_sources];
int output_fd = -1;
Architecture architecture{};
RegisterProfile register_profile = RegisterProfile::general;
char selected_registers[1024]{};
using ReadX86State = bool (*)(uint64_t*, size_t);
ReadX86State read_x86_state = nullptr;
bool capture_context = false;
bool requested_context = false;
bool capture_memory = true;
bool capture_values = false;
bool capture_stdio = false;
bool capture_system = false;
bool capture_kernel = false;
bool capture_layout = false;
bool capture_blocks = true;
pthread_once_t layout_once = PTHREAD_ONCE_INIT;
unsigned int system_cpus = 0;
bool has_start_pc = false;
uint64_t start_pc = 0;
bool has_stop_pc = false;
uint64_t stop_pc = 0;
bool has_window_start_pc = false;
bool has_window_end_pc = false;
bool has_window_abort_pc = false;
uint64_t window_start_pc = 0;
uint64_t window_end_pc = 0;
uint64_t window_abort_pc = 0;
bool reduce_transitions = false;
uint32_t transition_capacity = 4096;
TransitionWindow transition_window;
bool action_window_expected = false;
bool mixed_batches = false;
bool ring_publication = false;
pthread_t collector_thread{};
std::atomic<bool> collector_done{false};
enum class CaptureState : uint8_t { waiting, running, stopped };
std::atomic<CaptureState> capture_state{CaptureState::running};
int control_fd = -1;
pthread_t control_thread{};
std::atomic<bool> closing{false};

class ColdLock final {
public:
    explicit ColdLock(Source& source) : _mutex(&source.cold_mutex) {
        if (pthread_mutex_lock(_mutex) != 0) _exit(capture_exit_code);
    }
    ~ColdLock() { pthread_mutex_unlock(_mutex); }
    ColdLock(const ColdLock&) = delete;
    ColdLock& operator=(const ColdLock&) = delete;
private:
    pthread_mutex_t* _mutex;
};

class PublishChanges final {
public:
    explicit PublishChanges(Source& source) : _source(source) {
        if (capture_kernel) (void)_source.drained.load(std::memory_order_acquire);
    }
    ~PublishChanges() {
        if (capture_kernel) _source.published.store(++_source.generation, std::memory_order_release);
    }
    PublishChanges(const PublishChanges&) = delete;
    PublishChanges& operator=(const PublishChanges&) = delete;
private:
    Source& _source;
};

Result<Done> publish(const uint8_t* bytes, size_t size)
{
    // Block SIGPIPE only in this host thread. Changing its process-wide signal
    // disposition could change how QEMU delivers signals to the guest.
    sigset_t blocked;
    sigset_t previous;
    sigemptyset(&blocked);
    sigaddset(&blocked, SIGPIPE);
    if (pthread_sigmask(SIG_BLOCK, &blocked, &previous) != 0) {
        return Result<Done>::failure("Cannot block SIGPIPE for trace publication");
    }
    ssize_t written;
    do {
        written = write(output_fd, bytes, size);
    } while (written < 0 && errno == EINTR);
    if (written != static_cast<ssize_t>(size)) {
        // The caller terminates immediately. Keep SIGPIPE blocked so a pending
        // broken-pipe signal cannot replace our explicit capture failure.
        return Result<Done>::failure("Cannot publish a complete trace frame");
    }
    if (pthread_sigmask(SIG_SETMASK, &previous, nullptr) != 0) {
        return Result<Done>::failure("Cannot restore the host signal mask");
    }
    return Result<Done>::success({});
}

[[noreturn]] void fail(Failure reason, const char* message)
{
    uint8_t bytes[header_bytes];
    encode_header(bytes, {Kind::error, 0, 0, 0, static_cast<uint64_t>(reason)});
    (void)publish(bytes, sizeof(bytes));
    // QEMU's plugin log is disabled unless the operator requests it. Capture
    // failures must remain visible in the worker's ordinary error output.
    std::fputs(message, stderr);
    _exit(capture_exit_code);
}

void send_header(const Header& header)
{
    uint8_t bytes[header_bytes];
    encode_header(bytes, header);
    const auto sent = publish(bytes, sizeof(bytes));
    if (!sent.ok()) {
        // Retrying on a broken transport cannot restore lost data.
        _exit(capture_exit_code);
    }
}

Source& active_source(unsigned int index)
{
    if (index >= max_sources || sources[index].state != State::active) {
        fail(Failure::unsupported_target, "cpu2tensor: unexpected vCPU index or lifetime\n");
    }
    return sources[index];
}

void wait_for_frames(Source& source)
{
    const auto expected = source.frames_queued.load(std::memory_order_acquire);
    while (source.frames_sent.load(std::memory_order_acquire) != expected) {
        if (collector_done.load(std::memory_order_acquire))
            fail(Failure::transport, "cpu2tensor: collector stopped before draining the source\n");
        const timespec delay{0, 50000};
        nanosleep(&delay, nullptr);
    }
}

void* collect_frames(void*)
{
    unsigned first = 0;
    for (;;) {
        bool found = false;
        for (unsigned step = 0; step < max_sources; ++step) {
            auto& source = sources[(first + step) % max_sources];
            const auto frame = source.ring.front();
            if (frame.data == nullptr) continue;
            found = true;
            if (!publish(frame.data, frame.size).ok()) _exit(capture_exit_code);
            source.ring.pop();
            source.frames_sent.fetch_add(1, std::memory_order_release);
        }
        first = (first + 1) % max_sources;
        // Finish joins only after every producer has stopped and drained.
        if (collector_done.load(std::memory_order_acquire)) return nullptr;
        if (!found) {
            const timespec delay{0, 50000};
            nanosleep(&delay, nullptr);
        }
    }
}

void flush(unsigned int index, Source& source)
{
    if (source.count == 0) {
        return;
    }
    const bool schema = source.kind == Kind::register_schema;
    encode_header(source.bytes, {source.kind, index, source.count,
                                  schema ? 0 : source.next - source.count,
                                  source.kind == Kind::blocks ? 0 : source.used});
    if (ring_publication) {
        for (;;) {
            const auto queued = source.ring.try_push(source.bytes, header_bytes + source.used);
            if (!queued.ok()) fail(Failure::capture, queued.error());
            if (queued.value() == FramePush::stored) break;
            ++source.full_waits;
            if (collector_done.load(std::memory_order_acquire))
                fail(Failure::transport, "cpu2tensor: collector stopped with a full ring\n");
            const timespec delay{0, 50000};
            nanosleep(&delay, nullptr);
        }
        source.frames_queued.fetch_add(1, std::memory_order_release);
    } else {
        if (!publish(source.bytes, header_bytes + source.used).ok()) _exit(capture_exit_code);
    }
    source.count = 0;
    source.used = 0;
}

uint8_t* append(unsigned int index, Source& source, Kind kind, size_t size)
{
    if (mixed_batches && kind != Kind::register_schema) {
        if (source.kind != Kind::mixed || source.count == max_addresses ||
            source.used + size + ((source.count == 0 || source.run_kind != kind) ? 8 : 0) > max_payload_bytes)
            flush(index, source);
        source.kind = Kind::mixed;
        if (source.count == 0 || source.run_kind != kind) {
            source.run_kind = kind;
            source.run_offset = source.used;
            uint8_t* run = source.bytes + header_bytes + source.used;
            store_u16(run, static_cast<uint16_t>(kind));
            store_u16(run + 2, 0);
            store_u32(run + 4, 0);
            source.used += 8;
        }
        uint8_t* run = source.bytes + header_bytes + source.run_offset;
        store_u16(run + 2, load_u16(run + 2) + 1);
        store_u32(run + 4, load_u32(run + 4) + static_cast<uint32_t>(size));
        if (source.next == UINT64_MAX) fail(Failure::capture, "cpu2tensor: source sequence overflow\n");
        ++source.next;
        ++source.count;
        uint8_t* row = source.bytes + header_bytes + source.used;
        source.used += static_cast<uint32_t>(size);
        return row;
    }
    if (source.kind != kind || source.used + size > max_payload_bytes ||
        (kind == Kind::blocks && source.count == max_addresses)) {
        flush(index, source);
    }
    source.kind = kind;
    if (kind != Kind::register_schema) {
        if (source.next == UINT64_MAX) {
            fail(Failure::capture, "cpu2tensor: source event sequence overflow\n");
        }
        ++source.next;
    }
    uint8_t* row = source.bytes + header_bytes + source.used;
    source.used += static_cast<uint32_t>(size);
    ++source.count;
    return row;
}

bool general_register(const char* name)
{
    constexpr const char* arm_names[] = {
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        "sp", "pc", "cpsr"
    };
    constexpr const char* x86_names[] = {
        "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "r8", "r9",
        "r10", "r11", "r12", "r13", "r14", "r15", "rip", "eflags",
        "cs", "ss", "ds", "es", "fs", "gs", "fs_base", "gs_base"
    };
    const bool arm = architecture == Architecture::aarch64;
    const char* const* names = arm ? arm_names : x86_names;
    const size_t count = arm ? sizeof(arm_names) / sizeof(*arm_names) :
                               sizeof(x86_names) / sizeof(*x86_names);
    for (size_t index = 0; index < count; ++index) {
        if (std::strcmp(name, names[index]) == 0) {
            return true;
        }
    }
    return false;
}

void read_register(Source& source, const Register& reg)
{
    if (reg.raw_field != 255) {
        g_byte_array_set_size(source.scratch, 8);
        store_u64(source.scratch->data, source.raw_state[reg.raw_field]);
        return;
    }
    // Public QEMU register readers append. Reserve the largest supported target
    // reader at init so these repeated reads keep the same allocation.
    g_byte_array_set_size(source.scratch, 0);
    if (!qemu_plugin_read_register(reg.handle, source.scratch)) {
        fail(Failure::capture, "cpu2tensor: QEMU could not read a selected register\n");
    }
    if (source.scratch->len == 0 || source.scratch->len > max_register_bytes ||
        (reg.width != 0 && source.scratch->len != reg.width)) {
        char message[192];
        std::snprintf(message, sizeof(message),
            "cpu2tensor: register %s has width %u; expected %u or an initial width of 1..%u bytes\n",
            reg.name, source.scratch->len, reg.width, max_register_bytes);
        fail(Failure::unsupported_target, message);
    }
}

void register_setup(unsigned int index, Source& source)
{
    // A context-only hook read needs no register schema, descriptors or baselines.
    if (register_profile == RegisterProfile::none && (!capture_context || read_x86_state != nullptr)) return;
    source.registers = new (std::nothrow) Register[max_registers];
    source.scratch = g_byte_array_sized_new(register_scratch_bytes);
    if (source.registers == nullptr) {
        fail(Failure::capture, "cpu2tensor: cannot allocate register state\n");
    }
    GArray* descriptors = qemu_plugin_get_registers();
    if (descriptors == nullptr) {
        fail(Failure::unsupported_target, "cpu2tensor: QEMU exposes no register descriptors\n");
    }
    for (guint position = 0; position < descriptors->len; ++position) {
        const auto& descriptor = g_array_index(descriptors, qemu_plugin_reg_descriptor, position);
        if (descriptor.name == nullptr) {
            fail(Failure::unsupported_target, "cpu2tensor: QEMU exposes an unnamed register\n");
        }
        constexpr const char* controls[] = {"cr0", "cr3", "cr4", "efer"};
        for (unsigned field = 0; capture_context && field < 4; ++field) {
            if (std::strcmp(descriptor.name, controls[field]) == 0) {
                source.controls[field].handle = descriptor.handle;
                source.controls[field].width = 8;
                std::strcpy(source.controls[field].name, descriptor.name);
            }
        }
        const bool pc = std::strcmp(descriptor.name,
            architecture == Architecture::aarch64 ? "pc" : "rip") == 0;
        const bool selected = register_profile == RegisterProfile::all ||
            (register_profile == RegisterProfile::general && general_register(descriptor.name)) ||
            (register_profile == RegisterProfile::selected && (pc || listed_register(selected_registers, descriptor.name)));
        if (!selected) continue;
        if (architecture == Architecture::x86_64 && unavailable_x86_register(descriptor.name, capture_system)) {
            if (register_profile == RegisterProfile::selected && listed_register(selected_registers, descriptor.name))
                fail(Failure::unsupported_target, "cpu2tensor: requested register has no trustworthy backend value\n");
            if (index == 0) std::fprintf(stderr, "cpu2tensor: omitting unavailable register %s\n", descriptor.name);
            continue;
        }
        const size_t length = strnlen(descriptor.name, sizeof(Register::name));
        if (length == 0 || length >= sizeof(Register::name) ||
            source.register_count == max_registers) {
            fail(Failure::unsupported_target, "cpu2tensor: unsupported register name or count\n");
        }
        for (size_t character = 0; character < length; ++character) {
            const auto value = static_cast<unsigned char>(descriptor.name[character]);
            if (value > 127) {
                fail(Failure::unsupported_target, "cpu2tensor: register names must be ASCII\n");
            }
        }
        for (uint32_t previous = 0; previous < source.register_count; ++previous) {
            if (std::strcmp(source.registers[previous].name, descriptor.name) == 0) {
                fail(Failure::unsupported_target, "cpu2tensor: QEMU exposes duplicate register names\n");
            }
        }
        auto& reg = source.registers[source.register_count];
        reg.handle = descriptor.handle;
        if (capture_context && read_x86_state != nullptr) {
            constexpr const char* raw_names[] = {
                "cr0", "cr2", "cr3", "cr4", "", "efer", "", "",
                "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
                "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
                "rip", "fs_base", "gs_base", "k_gs_base"};
            for (unsigned field = 0; field < 28; ++field)
                if (std::strcmp(descriptor.name, raw_names[field]) == 0) reg.raw_field = field;
        }
        std::memcpy(reg.name, descriptor.name, length);
        read_register(source, reg);
        reg.width = source.scratch->len;
        reg.previous = static_cast<uint8_t*>(std::malloc(reg.width));
        if (reg.previous == nullptr) {
            fail(Failure::capture, "cpu2tensor: cannot allocate previous register value\n");
        }
        if (std::strcmp(reg.name, architecture == Architecture::aarch64 ? "pc" : "rip") == 0) {
            if (reg.width != sizeof(uint64_t)) {
                fail(Failure::unsupported_target, "cpu2tensor: unsupported program counter width\n");
            }
            source.pc_register = static_cast<int32_t>(source.register_count);
        }
        uint8_t* row = append(index, source, Kind::register_schema, register_schema_bytes);
        store_u32(row, source.register_count);
        store_u32(row + 4, reg.width);
        std::memcpy(row + 8, reg.name, sizeof(reg.name));
        ++source.register_count;
    }
    g_array_free(descriptors, true);
    for (unsigned field = 0; capture_context && field < 4; ++field)
        if (source.controls[field].name[0] == 0)
            fail(Failure::unsupported_target, "cpu2tensor: x86 paging controls unavailable\n");
    if (register_profile == RegisterProfile::selected) {
        char names[sizeof(selected_registers)];
        std::strcpy(names, selected_registers);
        for (char* name = names; name != nullptr;) {
            char* next = std::strchr(name, ':');
            if (next != nullptr) *next++ = 0;
            bool found = false;
            for (uint32_t field = 0; field < source.register_count; ++field)
                if (std::strcmp(name, source.registers[field].name) == 0) found = true;
            if (!found) fail(Failure::unsupported_target, "cpu2tensor: requested register unavailable\n");
            name = next;
        }
    }
    if (register_profile != RegisterProfile::none && (source.register_count == 0 || source.pc_register < 0)) {
        fail(Failure::unsupported_target, "cpu2tensor: selected registers or program counter unavailable\n");
    }
    flush(index, source);
}

void read_context(Source& source, bool full) {
    if (!capture_context) return;
    if (read_x86_state != nullptr) {
        if (!read_x86_state(source.raw_state, full ? 28 : 8))
            fail(Failure::capture, "cpu2tensor: x86 state read outside a supported callback\n");
    } else {
        constexpr unsigned fields[] = {0, 2, 3, 5};
        for (unsigned field = 0; field < 4; ++field) {
            read_register(source, source.controls[field]);
            source.raw_state[fields[field]] = load_u64(source.scratch->data);
        }
    }
}

void emit_context(unsigned int index, Source& source, uint64_t pc) {
    if (!capture_context) return;
    constexpr unsigned fields[] = {0, 2, 3, 5, 6, 7};
    bool changed = !source.context_sampled;
    for (unsigned field = 0; field < 6; ++field)
        changed |= source.context[field] != source.raw_state[fields[field]];
    if (!changed) return;
    uint8_t* row = append(index, source, Kind::address_context, context_bytes);
    source.context_sequence = source.next - 1;
    store_u64(row, pc);
    for (unsigned field = 0; field < 6; ++field) {
        source.context[field] = source.raw_state[fields[field]];
        store_u64(row + 8 + field * 8, source.context[field]);
    }
    store_u64(row + 56, read_x86_state == nullptr ? 15 : 63);
    source.context_sampled = true;
}

uint64_t current_pc(Source& source)
{
    const auto& reg = source.registers[source.pc_register];
    read_register(source, reg);
    // Both supported QEMU target register interfaces use little-endian bytes.
    return load_u64(source.scratch->data);
}

void sample_registers(unsigned int index, Source& source, uint64_t pc, Checkpoint checkpoint)
{
    for (uint32_t id = 0; id < source.register_count; ++id) {
        auto& reg = source.registers[id];
        read_register(source, reg);
        if (reg.sampled && std::memcmp(reg.previous, source.scratch->data, reg.width) == 0) {
            continue;
        }
        uint8_t* row = append(index, source, Kind::registers, register_prefix_bytes + reg.width);
        store_u64(row, pc);
        store_u32(row + 8, id);
        store_u16(row + 12, static_cast<uint16_t>(reg.width));
        store_u16(row + 14, static_cast<uint16_t>(checkpoint) |
                            (reg.sampled ? 0 : initial_register));
        std::memcpy(row + register_prefix_bytes, source.scratch->data, reg.width);
        std::memcpy(reg.previous, source.scratch->data, reg.width);
        reg.sampled = true;
    }
}

void release_registers(Source& source)
{
    for (uint32_t id = 0; id < source.register_count; ++id) {
        std::free(source.registers[id].previous);
    }
    delete[] source.registers;
    source.registers = nullptr;
    if (source.scratch != nullptr) {
        g_byte_array_free(source.scratch, true);
        source.scratch = nullptr;
    }
}

void source_start(qemu_plugin_id_t, unsigned int index)
{
    if ((capture_stdio && index != 0) || (capture_kernel && index >= system_cpus) ||
        index >= max_sources || sources[index].state != State::unused) {
        fail(Failure::unsupported_target, "cpu2tensor: too many vCPUs or a reused vCPU index\n");
    }
    const ColdLock lock(sources[index]);
    sources[index].state = State::active;
    register_setup(index, sources[index]);
}

void end_source(unsigned int index, Source& source)
{
    flush(index, source);
    if (ring_publication) wait_for_frames(source);
    send_header({Kind::source_end, index, 0, source.next, 0});
    release_registers(source);
    source.state = State::ended;
}

void source_end(qemu_plugin_id_t, unsigned int index)
{
    auto& source = active_source(index);
    const ColdLock lock(source);
    if (source.register_count != 0 && (!has_start_pc || source.next != 0) &&
        (!has_stop_pc || capture_state.load(std::memory_order_acquire) == CaptureState::running) &&
        (!has_window_start_pc || source.recording)) {
        read_context(source, true);
        const auto offset = current_pc(source);
        // Register RIP is raw storage; event PCs use the linear code address.
        const auto pc = offset + ((capture_context && source.raw_state[7] != 64) ? source.raw_state[6] : 0);
        emit_context(index, source, pc);
        sample_registers(index, source, pc, Checkpoint::exit);
    }
    end_source(index, source);
}

void block_entry(unsigned int index, void* address)
{
    auto& source = active_source(index);
    const auto pc = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address));
    if (has_start_pc && pc == start_pc) {
        auto expected = CaptureState::waiting;
        capture_state.compare_exchange_strong(expected, CaptureState::running, std::memory_order_acq_rel);
    }
    if (has_window_start_pc && pc == window_start_pc) {
        const auto opened = transition_window.begin();
        if (!opened.ok()) fail(Failure::capture, opened.error());
        source.recording = false;
        return;
    }
    if (has_window_end_pc && pc == window_end_pc) {
        const auto ended = transition_window.end();
        if (!ended.ok()) fail(Failure::capture, ended.error());
        source.recording = false;
        return;
    }
    if (has_window_abort_pc && pc == window_abort_pc) {
        const auto aborted = transition_window.abort();
        if (!aborted.ok()) fail(Failure::capture, aborted.error());
        source.recording = false;
        return;
    }
    if (has_stop_pc && pc == stop_pc) {
        auto expected = CaptureState::running;
        capture_state.compare_exchange_strong(expected, CaptureState::stopped, std::memory_order_acq_rel);
        // Even an early stop marker racing another CPU's start is excluded.
        source.recording = false;
        return;
    }
    const BlockAdmission admission = has_window_start_pc ? transition_window.admit() :
        BlockAdmission{0, true};
    source.recording = capture_state.load(std::memory_order_acquire) == CaptureState::running &&
        admission.included;
    if (!source.recording) return;
    if (reduce_transitions) {
        const auto observed = transition_window.observe(index, pc, admission);
        if (!observed.ok()) fail(Failure::capture, observed.error());
    }
    const PublishChanges publication(source);
    read_context(source, source.register_count != 0);
    emit_context(index, source, pc);
    // These changes precede this block; they are not effects of its execution.
    sample_registers(index, source, pc, Checkpoint::block);
    if (capture_blocks) {
        store_u64(append(index, source, Kind::blocks, sizeof(uint64_t)), pc);
        if (source.count == max_addresses) {
            flush(index, source);
        }
    }
}

void memory_access(unsigned int index, qemu_plugin_meminfo_t info, uint64_t address, void* pc)
{
    auto& source = active_source(index);
    if (!source.recording ||
        (has_stop_pc && capture_state.load(std::memory_order_acquire) != CaptureState::running)) return;
    const PublishChanges publication(source);
    const unsigned int shift = qemu_plugin_mem_size_shift(info);
    if (shift >= 32 || (capture_values && shift > 4)) {
        fail(Failure::unsupported_target, "cpu2tensor: memory access width is unsupported\n");
    }
    const uint32_t size = uint32_t{1} << shift;
    read_context(source, false);
    emit_context(index, source, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pc)));
    const size_t prefix = capture_context ? 48 : memory_prefix_bytes;
    uint8_t* row = append(index, source, Kind::memory, prefix + (capture_values ? 16 : 0));
    store_u64(row, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pc)));
    store_u64(row + 8, address);
    store_u32(row + 16, size);
    store_u32(row + 20, (qemu_plugin_mem_is_store(info) ? 1u : 0u) |
                        (qemu_plugin_mem_is_big_endian(info) ? 2u : 0u));
    if (capture_context) {
        // The API describes the callback address. Do not query another page or
        // claim the rest of a cross-page transaction is physically contiguous.
        const auto* mapping = qemu_plugin_get_hwaddr(info, address);
        uint64_t physical = 0;
        uint32_t mapped = 0;
        uint32_t flags = 0;
        if (mapping != nullptr) {
            physical = qemu_plugin_hwaddr_phys_addr(mapping);
            const uint32_t remaining = 4096 - (address & 4095);
            mapped = size < remaining ? size : remaining;
            flags = 1; // Physical prefix known; QEMU's dispatch path may be unknown.
            if (read_x86_state != nullptr)
                flags |= 2 | (qemu_plugin_hwaddr_is_io(mapping) ? 4 : 0);
        }
        store_u64(row + 24, physical);
        store_u32(row + 32, mapped);
        store_u32(row + 36, flags);
        store_u64(row + 40, source.context_sequence);
    }
    if (!capture_values) return;
    // This returns the actual transaction value, not a subsequent memory read.
    const auto value = qemu_plugin_mem_get_value(info);
    if (static_cast<unsigned int>(value.type) != shift) {
        fail(Failure::capture, "cpu2tensor: QEMU memory value width does not match the access\n");
    }
    uint8_t* bytes = row + prefix;
    std::memset(bytes, 0, 16);
    switch (value.type) {
    case QEMU_PLUGIN_MEM_VALUE_U8: bytes[0] = value.data.u8; break;
    case QEMU_PLUGIN_MEM_VALUE_U16: store_u16(bytes, value.data.u16); break;
    case QEMU_PLUGIN_MEM_VALUE_U32: store_u32(bytes, value.data.u32); break;
    case QEMU_PLUGIN_MEM_VALUE_U64: store_u64(bytes, value.data.u64); break;
    case QEMU_PLUGIN_MEM_VALUE_U128:
        store_u64(bytes, value.data.u128.low);
        store_u64(bytes + 8, value.data.u128.high);
        break;
    }
}

void emit_layout()
{
    // The loader has finished by first translation. These APIs are too early
    // during plugin installation. This record is outside every vCPU sequence.
    uint8_t bytes[header_bytes + layout_bytes];
    encode_header(bytes, {Kind::executable_layout, 0, 1, 0, layout_bytes});
    store_u64(bytes + header_bytes, qemu_plugin_start_code());
    store_u64(bytes + header_bytes + 8, qemu_plugin_end_code());
    store_u64(bytes + header_bytes + 16, qemu_plugin_entry_code());
    if (load_u64(bytes + header_bytes) >= load_u64(bytes + header_bytes + 8))
        fail(Failure::capture, "cpu2tensor: QEMU did not provide a nonempty executable code span\n");
    if (!publish(bytes, sizeof(bytes)).ok()) _exit(capture_exit_code);
}

void translate(qemu_plugin_id_t, qemu_plugin_tb* block)
{
    // Translation is cold compared with execution. Once returns only after the
    // initial record is published; no execution or memory callback pays this cost.
    if (capture_layout && pthread_once(&layout_once, emit_layout) != 0)
        fail(Failure::capture, "cpu2tensor: cannot publish executable layout\n");
    // The userdata is an address bit pattern, never a pointer to dereference.
    // QEMU retains it with the translated block; no per-block allocation exists.
    const auto address = static_cast<uintptr_t>(qemu_plugin_tb_vaddr(block));
    const auto flags = register_profile == RegisterProfile::none && !capture_context ?
        QEMU_PLUGIN_CB_NO_REGS : QEMU_PLUGIN_CB_R_REGS;
    qemu_plugin_register_vcpu_tb_exec_cb(block, block_entry, flags,
                                        reinterpret_cast<void*>(address));
    if (capture_memory) {
        for (size_t index = 0; index < qemu_plugin_tb_n_insns(block); ++index) {
            auto* instruction = qemu_plugin_tb_get_insn(block, index);
            const auto pc = static_cast<uintptr_t>(qemu_plugin_insn_vaddr(instruction));
            qemu_plugin_register_vcpu_mem_cb(instruction, memory_access,
                capture_context ? QEMU_PLUGIN_CB_R_REGS : QEMU_PLUGIN_CB_NO_REGS,
                QEMU_PLUGIN_MEM_RW, reinterpret_cast<void*>(pc));
        }
    }
}

void syscall_entry(qemu_plugin_id_t, unsigned int index, int64_t number, uint64_t first,
                   uint64_t second, uint64_t requested, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
    auto& source = active_source(index);
    if (source.register_count != 0 && (!has_start_pc || source.recording) &&
        (!has_stop_pc || capture_state.load(std::memory_order_acquire) == CaptureState::running) &&
        (!has_window_start_pc || source.recording)) {
        sample_registers(index, source, current_pc(source), Checkpoint::syscall);
    }
    const bool arm = architecture == Architecture::aarch64;
    if (capture_stdio) {
        const bool closes_stdin = (number == (arm ? 57 : 3) && first == STDIN_FILENO) ||
                                  (number == 436 && first == STDIN_FILENO);
        const bool replaces_stdin = ((number == (arm ? 24 : 292) || (!arm && number == 33)) && second == STDIN_FILENO);
        if (number == (arm ? 134 : 13) && first == SIGCONT && second != 0) {
            fail(Failure::unsupported_target, "cpu2tensor: SIGCONT handlers need a later pause adapter\n");
        }
        if (closes_stdin || replaces_stdin) {
            fail(Failure::unsupported_target, "cpu2tensor: stdin remapping needs a later adapter\n");
        }
    }
    if (capture_stdio && number == (arm ? 63 : 0) && first == STDIN_FILENO && requested != 0) {
        flush(index, source);
        if (ring_publication) wait_for_frames(source);
        const auto limit = requested < max_action_bytes ? requested : max_action_bytes;
        send_header({Kind::input_request, index, 0, source.next, limit});
        // The worker waits for WIFSTOPPED before exposing this boundary. Only
        // one vCPU is allowed, so its published tail contains all observations.
        if (kill(getpid(), SIGSTOP) != 0) {
            fail(Failure::capture, "cpu2tensor: cannot stop at stdin request\n");
        }
    }
    const int64_t clone_number = arm ? 220 : 56;
    const bool exec = number == (arm ? 221 : 59) || number == (arm ? 281 : 322);
    const bool fork = !arm && (number == 57 || number == 58);
    const bool new_process = number == clone_number &&
        (first & (clone_vm | clone_thread)) != (clone_vm | clone_thread);
    if (exec || fork || new_process) {
        fail(Failure::unsupported_target, "cpu2tensor: fork and exec need a later process contract\n");
    }
}

void syscall_return(qemu_plugin_id_t, unsigned int, int64_t number, int64_t result)
{
    // This QEMU returns ENOSYS for clone3, which libc may try before clone.
    // Preserve that normal fallback, but never report a successful clone3 run
    // as complete until its address-space and lifetime semantics are supported.
    constexpr int64_t clone3_number = 435;
    if (number == clone3_number && result >= 0) {
        fail(Failure::unsupported_target, "cpu2tensor: successful clone3 is not supported\n");
    }
}

void source_idle(qemu_plugin_id_t, unsigned int index)
{
    if (index >= max_sources) return;
    auto& source = sources[index];
    const ColdLock lock(source);
    if (source.state == State::active) flush(index, source);
}

bool publish_transition_window()
{
    const auto closed = transition_window.close();
    if (!closed.ok()) fail(Failure::capture, closed.error());
    const auto summary = closed.value();
    if (summary.status == WindowStatus::none) return false;
    constexpr uint32_t rows_per_frame =
        static_cast<uint32_t>(max_payload_bytes / transition_count_bytes);
    uint8_t frame[max_frame_bytes];
    for (uint32_t source = 0; source < summary.sources; ++source) {
        const uint32_t rows = transition_window.row_count(source, summary.id);
        for (uint32_t first = 0; first < rows; first += rows_per_frame) {
            const uint32_t remaining = rows - first;
            const uint32_t count = remaining < rows_per_frame ? remaining : rows_per_frame;
            encode_header(frame, {Kind::block_transitions, source, count, summary.id,
                                  count * transition_count_bytes});
            for (uint32_t index = 0; index < count; ++index) {
                const auto row = transition_window.row(source, summary.id, first + index);
                uint8_t* output = frame + header_bytes + index * transition_count_bytes;
                store_u64(output, row.from_address);
                store_u64(output + 8, row.destination);
                store_u64(output + 16, row.count);
            }
            if (!publish(frame, header_bytes + count * transition_count_bytes).ok())
                fail(Failure::transport, "cpu2tensor: cannot publish transition rows\n");
        }
    }
    encode_header(frame, {Kind::transition_window, 0, 1, summary.id,
                          transition_window_bytes});
    uint8_t* output = frame + header_bytes;
    std::memset(output, 0, transition_window_bytes);
    store_u32(output, static_cast<uint32_t>(summary.status));
    store_u32(output + 4, summary.sources);
    store_u32(output + 8, summary.capacity_per_source);
    store_u64(output + 16, summary.distinct);
    store_u64(output + 24, summary.observed);
    store_u64(output + 32, summary.overflow);
    if (!publish(frame, header_bytes + transition_window_bytes).ok())
        fail(Failure::transport, "cpu2tensor: cannot publish transition window\n");
    return true;
}

void* control(void*)
{
    while (!closing.load(std::memory_order_acquire)) {
        pollfd descriptor{control_fd, POLLIN, 0};
        const int ready = poll(&descriptor, 1, 100);
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0) fail(Failure::transport, "cpu2tensor: cannot poll kernel control pipe\n");
        if (ready == 0) continue;
        char command = 0;
        if (read(control_fd, &command, 1) != 1) return nullptr;
        if (command != 'D') fail(Failure::transport, "cpu2tensor: unknown kernel control command\n");
        // The sole worker sends D only after QMP confirms every vCPU stopped.
        // An idle callback may still be flushing: its cold lock gives this
        // thread exclusive buffer ownership without locking execution callbacks.
        for (unsigned int index = 0; index < system_cpus; ++index) {
            auto& source = sources[index];
            const ColdLock lock(source);
            if (source.state != State::active)
                fail(Failure::unsupported_target, "cpu2tensor: kernel boundary needs every configured vCPU to be active\n");
            // QMP established quiescence. Pair with the owning callback's
            // final release so its captured bytes are visible to this reader.
            const auto generation = source.published.load(std::memory_order_acquire);
            if (source.state == State::active) flush(index, source);
            if (ring_publication) wait_for_frames(source);
            // The next execution callback acquires this before reusing the
            // buffer/count fields changed by the stopped-world drain.
            source.drained.store(generation, std::memory_order_release);
        }
        if (transition_window.open())
            fail(Failure::capture,
                 "cpu2tensor: guest requested an action before ending or aborting its window\n");
        const bool published_window = publish_transition_window();
        if (has_window_start_pc && published_window != action_window_expected)
            fail(Failure::capture, action_window_expected ?
                 "cpu2tensor: guest requested its next action without a completed window\n" :
                 "cpu2tensor: guest completed an action window before receiving an action\n");
        action_window_expected = false;
        if (!closing.load(std::memory_order_acquire)) {
            send_header({Kind::kernel_request, 0, 0, 0, 127});
            if (has_window_start_pc) action_window_expected = true;
        }
    }
    return nullptr;
}

void finish(qemu_plugin_id_t, void*)
{
    if (has_start_pc && capture_state.load(std::memory_order_acquire) == CaptureState::waiting)
        fail(Failure::capture, "cpu2tensor: target exited before the requested capture start block\n");
    if (has_stop_pc && capture_state.load(std::memory_order_acquire) != CaptureState::stopped)
        fail(Failure::capture, "cpu2tensor: target exited before the requested capture stop block\n");
    if (has_stop_pc) std::fprintf(stderr, "cpu2tensor: capture stop reached at 0x%llx\n",
                                 static_cast<unsigned long long>(stop_pc));
    if (capture_kernel) {
        closing.store(true, std::memory_order_release);
        pthread_join(control_thread, nullptr);
    }
    if (has_window_start_pc) {
        // Atexit runs after execution callbacks stop. Drain raw tails first so
        // reducer metadata cannot claim a frontier that raw publication lacks.
        for (unsigned int index = 0; index < system_cpus; ++index) {
            const ColdLock lock(sources[index]);
            if (sources[index].state == State::active) {
                flush(index, sources[index]);
                if (ring_publication) wait_for_frames(sources[index]);
            }
        }
        const bool published_window = publish_transition_window();
        if (published_window != action_window_expected)
            fail(Failure::capture, action_window_expected ?
                 "cpu2tensor: guest exited without a completed action window\n" :
                 "cpu2tensor: guest exited with an unexpected action window\n");
        action_window_expected = false;
    }
    for (unsigned int index = 0; index < max_sources; ++index) {
        const ColdLock lock(sources[index]);
        if (sources[index].state == State::active) {
            // Atexit cannot read registers. Drain only the events already seen.
            end_source(index, sources[index]);
        }
    }
    // This seals callbacks only. The worker supplies the actual child exit code.
    if (ring_publication) {
        collector_done.store(true, std::memory_order_release);
        pthread_join(collector_thread, nullptr);
        uint64_t frames = 0, waits = 0;
        for (const auto& source : sources) {
            frames += source.frames_sent.load(std::memory_order_acquire);
            waits += source.full_waits;
        }
        std::fprintf(stderr, "cpu2tensor: ring frames=%llu full_waits=%llu\n",
                     static_cast<unsigned long long>(frames), static_cast<unsigned long long>(waits));
    }
    send_header({Kind::complete});
}

} // namespace

extern "C" {
QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t* info,
                                          int argc, char** argv)
{
    capture_system = info->system_emulation;
    if (std::strcmp(info->target_name, "aarch64") == 0) {
        architecture = Architecture::aarch64;
    } else if (std::strcmp(info->target_name, "x86_64") == 0) {
        architecture = Architecture::x86_64;
    } else {
        std::fputs("cpu2tensor: unsupported guest architecture\n", stderr);
        return 1;
    }
    bool registers_seen = false;
    bool memory_seen = false;
    bool values_seen = false;
    bool context_seen = false;
    bool stdio_seen = false;
    bool kernel_seen = false;
    bool layout_seen = false;
    bool batching_seen = false;
    bool publication_seen = false;
    bool blocks_seen = false;
    bool reducer_seen = false;
    bool capacity_seen = false;
    for (int index = 0; index < argc; ++index) {
        const char* argument = argv[index];
        if (std::strncmp(argument, "fd=", 3) == 0 && output_fd < 0) {
            char* end = nullptr;
            errno = 0;
            const long parsed = std::strtol(argument + 3, &end, 10);
            if (argument[3] == '\0' || errno != 0 || *end != '\0' ||
                parsed < 3 || parsed > INT_MAX) {
                std::fputs("cpu2tensor: trace fd must be an open pipe descriptor above stderr\n", stderr);
                return 1;
            }
            output_fd = static_cast<int>(parsed);
        } else if (std::strncmp(argument, "control=", 8) == 0 && control_fd < 0) {
            char* end = nullptr;
            errno = 0;
            const long parsed = std::strtol(argument + 8, &end, 10);
            if (end == argument + 8 || errno != 0 || *end != '\0' || parsed < 3 || parsed > INT_MAX)
                return 1;
            control_fd = static_cast<int>(parsed);
        } else if (std::strncmp(argument, "publication=", 12) == 0 && !publication_seen) {
            publication_seen = true;
            if (std::strcmp(argument + 12, "ring") == 0) ring_publication = true;
            else if (std::strcmp(argument + 12, "pipe") != 0) return 1;
        } else if (std::strncmp(argument, "stop=", 5) == 0 && !has_stop_pc) {
            char* end = nullptr;
            errno = 0;
            stop_pc = std::strtoull(argument + 5, &end, 0);
            if (end == argument + 5 || errno != 0 || *end != '\0' || argument[5] == '-') return 1;
            has_stop_pc = true;
        } else if (std::strncmp(argument, "window_start=", 13) == 0 && !has_window_start_pc) {
            char* end = nullptr;
            errno = 0;
            window_start_pc = std::strtoull(argument + 13, &end, 0);
            if (end == argument + 13 || errno != 0 || *end != '\0' || argument[13] == '-') return 1;
            has_window_start_pc = true;
        } else if (std::strncmp(argument, "window_end=", 11) == 0 && !has_window_end_pc) {
            char* end = nullptr;
            errno = 0;
            window_end_pc = std::strtoull(argument + 11, &end, 0);
            if (end == argument + 11 || errno != 0 || *end != '\0' || argument[11] == '-') return 1;
            has_window_end_pc = true;
        } else if (std::strncmp(argument, "window_abort=", 13) == 0 && !has_window_abort_pc) {
            char* end = nullptr;
            errno = 0;
            window_abort_pc = std::strtoull(argument + 13, &end, 0);
            if (end == argument + 13 || errno != 0 || *end != '\0' || argument[13] == '-') return 1;
            has_window_abort_pc = true;
        } else if (std::strncmp(argument, "blocks=", 7) == 0 && !blocks_seen) {
            blocks_seen = true;
            if (std::strcmp(argument + 7, "on") == 0) capture_blocks = true;
            else if (std::strcmp(argument + 7, "off") == 0) capture_blocks = false;
            else return 1;
        } else if (std::strncmp(argument, "reducer=", 8) == 0 && !reducer_seen) {
            reducer_seen = true;
            if (std::strcmp(argument + 8, "block-transitions") == 0) reduce_transitions = true;
            else if (std::strcmp(argument + 8, "none") != 0) return 1;
        } else if (std::strncmp(argument, "transition_capacity=", 20) == 0 && !capacity_seen) {
            capacity_seen = true;
            char* end = nullptr;
            errno = 0;
            const unsigned long parsed = std::strtoul(argument + 20, &end, 10);
            if (end == argument + 20 || errno != 0 || *end != '\0' || parsed < 2 ||
                parsed > 65536 || (parsed & (parsed - 1)) != 0) return 1;
            transition_capacity = static_cast<uint32_t>(parsed);
        } else if (std::strncmp(argument, "batching=", 9) == 0 && !batching_seen) {
            batching_seen = true;
            if (std::strcmp(argument + 9, "mixed") == 0) mixed_batches = true;
            else if (std::strcmp(argument + 9, "legacy") != 0) return 1;
        } else if (std::strncmp(argument, "start=", 6) == 0 && !has_start_pc) {
            char* end = nullptr;
            errno = 0;
            start_pc = std::strtoull(argument + 6, &end, 0);
            if (end == argument + 6 || errno != 0 || *end != '\0' || argument[6] == '-') {
                std::fputs("cpu2tensor: start needs a guest block address\n", stderr);
                return 1;
            }
            has_start_pc = true;
            capture_state.store(CaptureState::waiting, std::memory_order_relaxed);
        } else if (std::strncmp(argument, "layout=", 7) == 0 && !layout_seen) {
            layout_seen = true;
            if (std::strcmp(argument + 7, "on") == 0) capture_layout = true;
            else if (std::strcmp(argument + 7, "off") == 0) capture_layout = false;
            else return 1;
        } else if (std::strncmp(argument, "kernel=", 7) == 0 && !kernel_seen) {
            kernel_seen = true;
            if (std::strcmp(argument + 7, "on") == 0) capture_kernel = true;
            else if (std::strcmp(argument + 7, "off") == 0) capture_kernel = false;
            else return 1;
        } else if (std::strncmp(argument, "stdio=", 6) == 0 && !stdio_seen) {
            stdio_seen = true;
            if (std::strcmp(argument + 6, "on") == 0) capture_stdio = true;
            else if (std::strcmp(argument + 6, "off") == 0) capture_stdio = false;
            else { std::fputs("cpu2tensor: stdio must be on or off\n", stderr); return 1; }
        } else if (std::strncmp(argument, "registers=", 10) == 0 && !registers_seen) {
            registers_seen = true;
            const char* profile = argument + 10;
            if (std::strcmp(profile, "none") == 0) {
                register_profile = RegisterProfile::none;
            } else if (std::strcmp(profile, "general") == 0) {
                register_profile = RegisterProfile::general;
            } else if (std::strcmp(profile, "all") == 0) {
                register_profile = RegisterProfile::all;
            } else if (valid_register_selection(profile)) {
                register_profile = RegisterProfile::selected;
                std::strcpy(selected_registers, profile);
            } else {
                std::fputs("cpu2tensor: invalid register name list\n", stderr);
                return 1;
            }
        } else if (std::strncmp(argument, "context=", 8) == 0 && !context_seen) {
            context_seen = true;
            if (std::strcmp(argument + 8, "on") == 0) requested_context = true;
            else if (std::strcmp(argument + 8, "auto") != 0) {
                std::fputs("cpu2tensor: context must be auto or on\n", stderr);
                return 1;
            }
        } else if (std::strncmp(argument, "memory=", 7) == 0 && !memory_seen) {
            memory_seen = true;
            if (std::strcmp(argument + 7, "on") == 0) {
                capture_memory = true;
            } else if (std::strcmp(argument + 7, "off") == 0) {
                capture_memory = false;
            } else {
                std::fputs("cpu2tensor: memory must be on or off\n", stderr);
                return 1;
            }
        } else if (std::strncmp(argument, "values=", 7) == 0 && !values_seen) {
            values_seen = true;
            if (std::strcmp(argument + 7, "on") == 0) {
                capture_values = true;
            } else if (std::strcmp(argument + 7, "off") == 0) {
                capture_values = false;
            } else {
                std::fputs("cpu2tensor: values must be on or off\n", stderr);
                return 1;
            }
        } else {
            std::fputs("cpu2tensor: unknown or repeated plugin argument\n", stderr);
            return 1;
        }
    }
    if (output_fd < 0 || (capture_values && !capture_memory)) {
        std::fputs("cpu2tensor: fd is required, and memory values require memory capture\n", stderr);
        return 1;
    }
    if (capture_layout && capture_system) {
        std::fputs("cpu2tensor: executable layout is only available for user-mode targets\n", stderr);
        return 1;
    }
    if (has_stop_pc && (capture_stdio || capture_kernel || (has_start_pc && start_pc == stop_pc))) {
        std::fputs("cpu2tensor: stop requires observation-only capture and distinct start/stop addresses\n", stderr);
        return 1;
    }
    if ((capture_system && capture_stdio) || (capture_kernel && !capture_system) ||
        (capture_kernel != (control_fd >= 0))) {
        std::fputs("cpu2tensor: kernel interaction requires system emulation and its control pipe\n", stderr);
        return 1;
    }
    const bool any_window = has_window_start_pc || has_window_end_pc || has_window_abort_pc;
    if (any_window && (!has_window_start_pc || !has_window_end_pc || !has_window_abort_pc ||
        !capture_kernel || window_start_pc == window_end_pc || window_start_pc == window_abort_pc ||
        window_end_pc == window_abort_pc)) {
        std::fputs("cpu2tensor: action windows need three distinct markers and kernel interaction\n", stderr);
        return 1;
    }
    if (reduce_transitions != any_window || (!capture_blocks && !reduce_transitions)) {
        std::fputs("cpu2tensor: action windows and transition reduction must be enabled together\n", stderr);
        return 1;
    }
    if (capture_kernel) {
        if (info->system.smp_vcpus < 1 || info->system.smp_vcpus > static_cast<int>(max_sources) ||
            info->system.smp_vcpus != info->system.max_vcpus) {
            std::fputs("cpu2tensor: kernel actions need a fixed vCPU count without hotplug slots\n", stderr);
            return 1;
        }
        system_cpus = static_cast<unsigned int>(info->system.smp_vcpus);
        if (any_window) {
            const auto configured = transition_window.configure(system_cpus, transition_capacity);
            if (!configured.ok()) {
                std::fputs(configured.error(), stderr);
                std::fputc('\n', stderr);
                return 1;
            }
        }
    }
    if (requested_context && (!capture_system || architecture != Architecture::x86_64)) {
        std::fputs("cpu2tensor: context=on requires x86 system emulation\n", stderr);
        return 1;
    }
    capture_context = capture_system && architecture == Architecture::x86_64 &&
        (requested_context || capture_memory || register_profile != RegisterProfile::none);
    if (capture_context) {
        read_x86_state = reinterpret_cast<ReadX86State>(dlsym(RTLD_DEFAULT, "qemu_plugin_cpu2tensor_x86_state_v1"));
        if (read_x86_state == nullptr && register_profile != RegisterProfile::none) {
            std::fputs("cpu2tensor: exact x86 system registers require the optional x86-state-v1 QEMU hook; "
                "see docs/qemu-state-hook.md, or select registers=none\n", stderr);
            return 1;
        }
        if (read_x86_state == nullptr)
            std::fputs("cpu2tensor: execution mode, CS base and RAM/MMIO classification unavailable in this backend\n", stderr);
    }
    if (control_fd >= 0) {
        struct stat control_status{};
        const int control_flags = fcntl(control_fd, F_GETFL);
        if (fstat(control_fd, &control_status) != 0 || !S_ISFIFO(control_status.st_mode) ||
            control_flags < 0 || (control_flags & O_ACCMODE) != O_RDONLY) return 1;
    }
    struct stat status{};
    const int flags = fcntl(output_fd, F_GETFL);
    if (fstat(output_fd, &status) != 0 || !S_ISFIFO(status.st_mode) || flags < 0 ||
        (flags & O_NONBLOCK) != 0 || (flags & O_ACCMODE) != O_WRONLY) {
        std::fputs("cpu2tensor: trace fd must be a blocking pipe write end\n", stderr);
        return 1;
    }
    const uint64_t features = (capture_memory ? feature_memory : 0) |
        (register_profile != RegisterProfile::none ? feature_registers : 0) |
        (capture_values ? feature_memory_values : 0) | (capture_stdio ? feature_stdio : 0) |
        (capture_system ? feature_system : 0) | (capture_kernel ? feature_kernel : 0) |
        (has_start_pc ? feature_window : 0) |
        (capture_context ? feature_address_context : 0) |
        (capture_context && capture_memory ? feature_system_memory : 0) |
        (capture_layout ? feature_executable_layout : 0) | (mixed_batches ? feature_mixed : 0) |
        (has_stop_pc ? feature_stop : 0) | (any_window ? feature_transition_windows : 0);
    send_header({Kind::hello, 0, 0, 0, static_cast<uint64_t>(architecture) | features});
    if (ring_publication && pthread_create(&collector_thread, nullptr, collect_frames, nullptr) != 0) {
        std::fputs("cpu2tensor: cannot start trace collector\n", stderr);
        return 1;
    }
    qemu_plugin_register_vcpu_init_cb(id, source_start);
    qemu_plugin_register_vcpu_exit_cb(id, source_end);
    qemu_plugin_register_vcpu_tb_trans_cb(id, translate);
    if (capture_system) {
        qemu_plugin_register_vcpu_idle_cb(id, source_idle);
    } else {
        qemu_plugin_register_vcpu_syscall_cb(id, syscall_entry);
        qemu_plugin_register_vcpu_syscall_ret_cb(id, syscall_return);
    }
    if (capture_kernel && pthread_create(&control_thread, nullptr, control, nullptr) != 0) {
        std::fputs("cpu2tensor: cannot start kernel control thread\n", stderr);
        return 1;
    }
    qemu_plugin_register_atexit_cb(id, finish, nullptr);
    return 0;
}
}
