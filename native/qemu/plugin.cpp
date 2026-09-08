// SPDX-License-Identifier: AGPL-3.0-only
#include <cpu2tensor/trace.hpp>
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

enum class RegisterProfile : uint8_t { none, general, all };
enum class Checkpoint : uint16_t { block = 1, syscall = 2, exit = 3 };

struct Register final {
    qemu_plugin_register* handle = nullptr;
    char name[64]{};
    uint8_t* previous = nullptr;
    uint32_t width = 0;
    bool sampled = false;
};

enum class State : uint8_t { unused, active, ended };

// QEMU calls execution and exit callbacks on the owning vCPU. The final callback
// runs after instrumentation has stopped, so it can drain the remaining sources.
// A source never shares its event buffer or sequence counter with another vCPU.
struct alignas(64) Source final {
    // Only lifecycle/idle/drain callbacks take this lock. Execution callbacks
    // own their source while running; external drains require a QMP world stop.
    pthread_mutex_t cold_mutex = PTHREAD_MUTEX_INITIALIZER;
    uint8_t bytes[max_frame_bytes]{};
    uint64_t next = 0;
    uint32_t count = 0;
    uint32_t used = 0;
    Kind kind = Kind::blocks;
    Register* registers = nullptr;
    GByteArray* scratch = nullptr;
    uint32_t register_count = 0;
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
bool capture_memory = true;
bool capture_values = false;
bool capture_stdio = false;
bool capture_system = false;
bool capture_kernel = false;
unsigned int system_cpus = 0;
bool has_start_pc = false;
uint64_t start_pc = 0;
std::atomic<bool> recording{true};
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

void flush(unsigned int index, Source& source)
{
    if (source.count == 0) {
        return;
    }
    const bool schema = source.kind == Kind::register_schema;
    encode_header(source.bytes, {source.kind, index, source.count,
                                  schema ? 0 : source.next - source.count,
                                  source.kind == Kind::blocks ? 0 : source.used});
    const auto sent = publish(source.bytes, header_bytes + source.used);
    if (!sent.ok()) {
        _exit(capture_exit_code);
    }
    source.count = 0;
    source.used = 0;
}

uint8_t* append(unsigned int index, Source& source, Kind kind, size_t size)
{
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
    if (register_profile == RegisterProfile::none) {
        return;
    }
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
        // Upstream x86 system TCG keeps arithmetic flags in lazy internal state.
        // Its public GDB register reader returns stale eflags inside a block.
        // Omitting the field is preferable to emitting a false architectural value.
        if (capture_system && architecture == Architecture::x86_64 &&
            std::strcmp(descriptor.name, "eflags") == 0) continue;
        if (register_profile == RegisterProfile::general && !general_register(descriptor.name)) {
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
    if (source.register_count == 0 || source.pc_register < 0) {
        fail(Failure::unsupported_target, "cpu2tensor: selected registers or program counter unavailable\n");
    }
    flush(index, source);
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
    send_header({Kind::source_end, index, 0, source.next, 0});
    release_registers(source);
    source.state = State::ended;
}

void source_end(qemu_plugin_id_t, unsigned int index)
{
    auto& source = active_source(index);
    const ColdLock lock(source);
    if (source.register_count != 0 && (!has_start_pc || source.next != 0)) {
        sample_registers(index, source, current_pc(source), Checkpoint::exit);
    }
    end_source(index, source);
}

void block_entry(unsigned int index, void* address)
{
    auto& source = active_source(index);
    const auto pc = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address));
    if (has_start_pc && pc == start_pc) recording.store(true, std::memory_order_release);
    source.recording = !has_start_pc || recording.load(std::memory_order_acquire);
    if (!source.recording) return;
    const PublishChanges publication(source);
    // These changes precede this block; they are not effects of its execution.
    sample_registers(index, source, pc, Checkpoint::block);
    store_u64(append(index, source, Kind::blocks, sizeof(uint64_t)), pc);
    if (source.count == max_addresses) {
        flush(index, source);
    }
}

void memory_access(unsigned int index, qemu_plugin_meminfo_t info, uint64_t address, void* pc)
{
    auto& source = active_source(index);
    if (!source.recording) return;
    const PublishChanges publication(source);
    const unsigned int shift = qemu_plugin_mem_size_shift(info);
    if (shift >= 32 || (capture_values && shift > 4)) {
        fail(Failure::unsupported_target, "cpu2tensor: memory access width is unsupported\n");
    }
    const uint32_t size = uint32_t{1} << shift;
    uint8_t* row = append(index, source, Kind::memory,
                          memory_prefix_bytes + (capture_values ? 16 : 0));
    store_u64(row, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pc)));
    store_u64(row + 8, address);
    store_u32(row + 16, size);
    store_u32(row + 20, (qemu_plugin_mem_is_store(info) ? 1u : 0u) |
                        (qemu_plugin_mem_is_big_endian(info) ? 2u : 0u));
    if (!capture_values) {
        return;
    }
    // This returns the actual transaction value, not a subsequent memory read.
    const auto value = qemu_plugin_mem_get_value(info);
    if (static_cast<unsigned int>(value.type) != shift) {
        fail(Failure::capture, "cpu2tensor: QEMU memory value width does not match the access\n");
    }
    uint8_t* bytes = row + memory_prefix_bytes;
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

void translate(qemu_plugin_id_t, qemu_plugin_tb* block)
{
    // The userdata is an address bit pattern, never a pointer to dereference.
    // QEMU retains it with the translated block; no per-block allocation exists.
    const auto address = static_cast<uintptr_t>(qemu_plugin_tb_vaddr(block));
    const auto flags = register_profile == RegisterProfile::none ?
        QEMU_PLUGIN_CB_NO_REGS : QEMU_PLUGIN_CB_R_REGS;
    qemu_plugin_register_vcpu_tb_exec_cb(block, block_entry, flags,
                                        reinterpret_cast<void*>(address));
    if (capture_memory) {
        for (size_t index = 0; index < qemu_plugin_tb_n_insns(block); ++index) {
            auto* instruction = qemu_plugin_tb_get_insn(block, index);
            const auto pc = static_cast<uintptr_t>(qemu_plugin_insn_vaddr(instruction));
            qemu_plugin_register_vcpu_mem_cb(instruction, memory_access,
                QEMU_PLUGIN_CB_NO_REGS, QEMU_PLUGIN_MEM_RW, reinterpret_cast<void*>(pc));
        }
    }
}

void syscall_entry(qemu_plugin_id_t, unsigned int index, int64_t number, uint64_t first,
                   uint64_t second, uint64_t requested, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
    auto& source = active_source(index);
    if (source.register_count != 0 && (!has_start_pc || source.recording)) {
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
            // The next execution callback acquires this before reusing the
            // buffer/count fields changed by the stopped-world drain.
            source.drained.store(generation, std::memory_order_release);
        }
        if (!closing.load(std::memory_order_acquire))
            send_header({Kind::kernel_request, 0, 0, 0, 127});
    }
    return nullptr;
}

void finish(qemu_plugin_id_t, void*)
{
    if (has_start_pc && !recording.load(std::memory_order_acquire))
        fail(Failure::capture, "cpu2tensor: target exited before the requested capture start block\n");
    if (capture_kernel) {
        closing.store(true, std::memory_order_release);
        pthread_join(control_thread, nullptr);
    }
    for (unsigned int index = 0; index < max_sources; ++index) {
        const ColdLock lock(sources[index]);
        if (sources[index].state == State::active) {
            // Atexit cannot read registers. Drain only the events already seen.
            end_source(index, sources[index]);
        }
    }
    // This seals callbacks only. The worker supplies the actual child exit code.
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
    bool stdio_seen = false;
    bool kernel_seen = false;
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
        } else if (std::strncmp(argument, "start=", 6) == 0 && !has_start_pc) {
            char* end = nullptr;
            errno = 0;
            start_pc = std::strtoull(argument + 6, &end, 0);
            if (end == argument + 6 || errno != 0 || *end != '\0' || argument[6] == '-') {
                std::fputs("cpu2tensor: start needs a guest block address\n", stderr);
                return 1;
            }
            has_start_pc = true;
            recording.store(false, std::memory_order_relaxed);
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
            } else {
                std::fputs("cpu2tensor: registers must be none, general, or all\n", stderr);
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
    if ((capture_system && capture_stdio) || (capture_kernel && !capture_system) ||
        (capture_kernel != (control_fd >= 0))) {
        std::fputs("cpu2tensor: kernel interaction requires system emulation and its control pipe\n", stderr);
        return 1;
    }
    if (capture_kernel) {
        if (info->system.smp_vcpus < 1 || info->system.smp_vcpus > static_cast<int>(max_sources) ||
            info->system.smp_vcpus != info->system.max_vcpus) {
            std::fputs("cpu2tensor: kernel actions need a fixed vCPU count without hotplug slots\n", stderr);
            return 1;
        }
        system_cpus = static_cast<unsigned int>(info->system.smp_vcpus);
    }
    if (capture_system && architecture == Architecture::x86_64 && register_profile != RegisterProfile::none)
        std::fputs("cpu2tensor: omitting x86 eflags; system TCG exposes lazy flags at block callbacks\n", stderr);
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
        (has_start_pc ? feature_window : 0);
    send_header({Kind::hello, 0, 0, 0, static_cast<uint64_t>(architecture) | features});
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
