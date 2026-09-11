// SPDX-License-Identifier: AGPL-3.0-only
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/reboot.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

enum {
    max_memory_bytes = 65536,
    max_pipe_bytes = 256,
    max_command_bytes = 128,
    max_report_padding_bytes = 64,
    max_kernel_command_bytes = 4096,
    max_event_bytes = 1024,
    default_memory_bytes = 4096,
    default_seed = 17,
    default_parallel_timeout_seconds = 30
};

// Guest wall-clock budgets include delays caused by lossless trace backpressure.
static uint32_t parallel_timeout_seconds = default_parallel_timeout_seconds;
static bool event_channel_failed = false;
static int event_channel = STDOUT_FILENO;

struct options {
    bool interactive;
    bool context_filter;
    uint32_t seed;
    uint32_t bytes;
};

// The worker may use this stable guest PC for explicit postboot capture.
// It changes no guest state and introduces no instrumentation transport ABI.
__attribute__((noinline)) void cpu2tensor_capture_begin(void)
{
    __asm__ volatile("" ::: "memory");
}

// The context-filter acceptance workload latches at this user-space block. Its
// two stores also stay in user space, so the fixture does not claim that a
// KPTI-paired kernel page table belongs to the nominated process.
__attribute__((noinline)) void cpu2tensor_context_gate(void)
{
    __asm__ volatile("" ::: "memory");
}

__attribute__((noinline)) void cpu2tensor_context_first(volatile uint64_t *values)
{
    values[0] = UINT64_C(0x1122334455667788);
}

__attribute__((noinline)) void cpu2tensor_context_second(volatile uint64_t *values)
{
    values[1] = UINT64_C(0x8877665544332211);
}

__attribute__((noinline)) uint64_t cpu2tensor_context_background(
    volatile uint64_t *value, uint32_t iterations)
{
    for (uint32_t index = 0; index < iterations; ++index) {
        *value = *value * UINT64_C(6364136223846793005) + index + UINT64_C(1);
    }
    return *value;
}

// These function addresses are the guest-to-plugin action boundary contract.
// They perform no transport and deliberately remain separate basic blocks.
__attribute__((noinline)) void cpu2tensor_action_begin(void)
{
    __asm__ volatile("" ::: "memory");
}

__attribute__((noinline)) void cpu2tensor_action_end(void)
{
    __asm__ volatile("" ::: "memory");
}

__attribute__((noinline)) void cpu2tensor_action_abort(void)
{
    __asm__ volatile("" ::: "memory");
}

static bool emit_event(const char *format, ...)
{
    if (event_channel_failed) {
        return false;
    }
    char bytes[max_event_bytes];
    va_list arguments;
    va_start(arguments, format);
    const int formatted = vsnprintf(bytes, sizeof(bytes), format, arguments);
    va_end(arguments);
    if (formatted <= 0 || (size_t)formatted >= sizeof(bytes)) {
        event_channel_failed = true;
        return false;
    }
    const size_t size = (size_t)formatted;
    if (size < 5 || memcmp(bytes, "C2T ", 4) != 0 || bytes[size - 1] != '\n') {
        event_channel_failed = true;
        return false;
    }
    // POSIX reports EINTR only when write transferred no bytes. A short
    // positive result is fatal; never append to a truncated protocol stream.
    // The dedicated adapter UART, not this syscall boundary, excludes printk.
    ssize_t written;
    do {
        written = write(event_channel, bytes, size);
    } while (written < 0 && errno == EINTR);
    if (written != (ssize_t)size) {
        event_channel_failed = true;
        return false;
    }
    return true;
}

static bool open_event_channel(void)
{
    const int descriptor = open("/dev/ttyS1", O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
    if (descriptor < 0) {
        return false;
    }
    struct termios settings;
    const int flags = fcntl(descriptor, F_GETFL);
    if (tcgetattr(descriptor, &settings) != 0 || flags < 0) {
        close(descriptor);
        return false;
    }
    settings.c_cflag |= CLOCAL | CREAD;
    settings.c_iflag &= ~INLCR;
    settings.c_lflag |= ICANON;
    settings.c_lflag &= ~(ECHO | ECHONL);
    if (tcsetattr(descriptor, TCSANOW, &settings) != 0 ||
        fcntl(descriptor, F_SETFL, flags & ~O_NONBLOCK) != 0) {
        close(descriptor);
        return false;
    }
    event_channel = descriptor;
    return true;
}

enum command_read {
    command_read_ok,
    command_read_too_long,
    command_read_failed
};

static enum command_read read_command(char *line, size_t capacity)
{
    size_t used = 0;
    while (used + 1 < capacity) {
        const size_t offset = used;
        ssize_t size;
        do {
            size = read(event_channel, line + used, capacity - 1 - used);
        } while (size < 0 && errno == EINTR);
        if (size <= 0) {
            return command_read_failed;
        }
        used += (size_t)size;
        const char *newline = memchr(line + offset, '\n', (size_t)size);
        if (newline != NULL) {
            if (newline != line + used - 1) {
                return command_read_failed;
            }
            line[used] = 0;
            return command_read_ok;
        }
    }
    char discard[64];
    for (;;) {
        ssize_t size;
        do {
            size = read(event_channel, discard, sizeof(discard));
        } while (size < 0 && errno == EINTR);
        if (size <= 0) {
            return command_read_failed;
        }
        if (memchr(discard, '\n', (size_t)size) != NULL) {
            return command_read_too_long;
        }
    }
}

static bool parse_number(const char *text, uint32_t limit, uint32_t *value)
{
    if (text == NULL || text[0] < '0' || text[0] > '9') {
        return false;
    }
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed > limit) {
        return false;
    }
    *value = (uint32_t)parsed;
    return true;
}

static bool error_event(uint64_t step, const char *message)
{
    // Messages are fixed strings without JSON control characters.
    return emit_event("C2T {\"event\":\"error\",\"step\":%" PRIu64
                      ",\"message\":\"%s\"}\n", step, message);
}

static uint8_t next_byte(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (uint8_t)(*state >> 24);
}

// This bounded action is an oracle for the action-window contract. Equivalent
// numeric commands execute the same body even when their text and reports have
// different sizes. A different iteration count changes the enclosed trace.
__attribute__((noinline)) uint64_t cpu2tensor_compute(uint32_t iterations)
{
    volatile uint64_t value = UINT64_C(0x9e3779b97f4a7c15);
    for (uint32_t index = 0; index < iterations; ++index) {
        value ^= value << 7;
        value ^= value >> 9;
        value += index;
    }
    return value;
}

static bool compute_result(uint64_t step, uint32_t iterations, uint32_t padding,
                           uint64_t value)
{
    char report_padding[max_report_padding_bytes + 1];
    memset(report_padding, 'x', padding);
    report_padding[padding] = 0;
    return emit_event("C2T {\"event\":\"result\",\"step\":%" PRIu64
                      ",\"action\":\"compute\",\"iterations\":%" PRIu32
                      ",\"value\":%" PRIu64 ",\"padding\":\"%s\"}\n",
                      step, iterations, value, report_padding);
}

static void begin_action(bool windowed)
{
    if (windowed) {
        cpu2tensor_action_begin();
    }
}

static void end_action(bool windowed, bool ok)
{
    if (!windowed) {
        return;
    }
    if (ok) {
        cpu2tensor_action_end();
    } else {
        cpu2tensor_action_abort();
    }
}

static bool action_error(uint64_t step, bool windowed, const char *message)
{
    end_action(windowed, false);
    (void)error_event(step, message);
    return false;
}

static bool memory_action(uint64_t step, uint32_t seed, uint32_t count, bool windowed)
{
    begin_action(windowed);
    if (count == 0 || count > max_memory_bytes) {
        return action_error(step, windowed, "guest memory size is outside the allowed range");
    }
    void *mapping = mmap(NULL, count, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) {
        return action_error(step, windowed, "cannot allocate guest memory");
    }
    // Keep actual loads and stores visible to instrumentation at any build level.
    volatile uint8_t *bytes = mapping;
    uint32_t state = seed;
    for (uint32_t index = 0; index < count; ++index) {
        bytes[index] = next_byte(&state);
    }
    uint64_t checksum = 0;
    for (uint32_t index = 0; index < count; ++index) {
        checksum += bytes[index];
    }
    if (munmap(mapping, count) != 0) {
        return action_error(step, windowed, "cannot release guest memory");
    }
    end_action(windowed, true);
    return emit_event("C2T {\"event\":\"result\",\"step\":%" PRIu64
                      ",\"action\":\"memory\",\"seed\":%" PRIu32
                      ",\"bytes\":%" PRIu32 ",\"checksum\":%" PRIu64 "}\n",
                      step, seed, count, checksum);
}

static bool pipe_action(uint64_t step, uint32_t seed, uint32_t count, bool windowed)
{
    begin_action(windowed);
    if (count == 0 || count > max_pipe_bytes) {
        return action_error(step, windowed, "guest pipe size is outside the allowed range");
    }
    int descriptors[2];
    if (pipe(descriptors) != 0) {
        return action_error(step, windowed, "cannot create guest pipe");
    }
    uint8_t sent[max_pipe_bytes];
    uint8_t received[max_pipe_bytes];
    uint32_t state = seed;
    for (uint32_t index = 0; index < count; ++index) {
        sent[index] = next_byte(&state);
    }
    // The fixed bound fits an empty pipe without needing a second process.
    ssize_t written;
    do {
        written = write(descriptors[1], sent, count);
    } while (written < 0 && errno == EINTR);
    close(descriptors[1]);
    if (written != (ssize_t)count) {
        close(descriptors[0]);
        return action_error(step, windowed, "cannot write complete guest pipe message");
    }
    size_t total = 0;
    uint64_t checksum = 0;
    while (total < count) {
        const ssize_t size = read(descriptors[0], received, sizeof(received));
        if (size < 0 && errno == EINTR) {
            continue;
        }
        if (size <= 0 || (size_t)size > count - total) {
            close(descriptors[0]);
            return action_error(step, windowed, "cannot read complete guest pipe message");
        }
        if (memcmp(sent + total, received, (size_t)size) != 0) {
            close(descriptors[0]);
            return action_error(step, windowed, "guest pipe message changed");
        }
        for (size_t index = 0; index < (size_t)size; ++index) {
            checksum += received[index];
        }
        total += (size_t)size;
    }
    close(descriptors[0]);
    end_action(windowed, true);
    return emit_event("C2T {\"event\":\"result\",\"step\":%" PRIu64
                      ",\"action\":\"pipe\",\"seed\":%" PRIu32
                      ",\"bytes\":%" PRIu32 ",\"checksum\":%" PRIu64 "}\n",
                      step, seed, count, checksum);
}

struct start_gate {
    atomic_uint ready[2];
};

enum gate_state { gate_waiting, gate_ready, gate_failed };

struct parallel_result {
    uint64_t checksum;
    int cpu;
    bool ok;
};

static int remaining_parallel_ms(const struct timespec *started)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0;
    }
    const int64_t elapsed = (now.tv_sec - started->tv_sec) * INT64_C(1000) +
                            (now.tv_nsec - started->tv_nsec) / INT64_C(1000000);
    const int64_t remaining = parallel_timeout_seconds * INT64_C(1000) - elapsed;
    return remaining > 0 ? (int)remaining : 0;
}

static bool pin_cpu(int cpu)
{
    cpu_set_t selected;
    CPU_ZERO(&selected);
    CPU_SET(cpu, &selected);
    return sched_setaffinity(0, sizeof(selected), &selected) == 0 && sched_getcpu() == cpu;
}

static bool start_together(struct start_gate *gate, unsigned int worker,
                           const struct timespec *started)
{
    // Synchronize only the start. Each CPU owns its private memory during work;
    // there is no lock, shared counter, or barrier inside the memory loops.
    atomic_store_explicit(&gate->ready[worker], gate_ready, memory_order_release);
    unsigned int other;
    while ((other = atomic_load_explicit(&gate->ready[1 - worker], memory_order_acquire)) == gate_waiting) {
        if (remaining_parallel_ms(started) == 0 || sched_yield() != 0) {
            return false;
        }
    }
    return other == gate_ready;
}

// Require this entry PC on two trace sources to distinguish active work from
// merely booting a guest with two configured CPUs.
__attribute__((noinline)) uint64_t cpu2tensor_parallel_memory(
    volatile uint8_t *bytes, uint32_t seed, uint32_t count)
{
    uint32_t state = seed;
    for (uint32_t index = 0; index < count; ++index) {
        bytes[index] = next_byte(&state);
    }
    uint64_t checksum = 0;
    for (uint32_t index = 0; index < count; ++index) {
        checksum += bytes[index];
    }
    return checksum;
}

static bool read_parallel_result(int descriptor, struct parallel_result *result,
                                 const struct timespec *started)
{
    size_t received = 0;
    while (received < sizeof(*result)) {
        const int timeout = remaining_parallel_ms(started);
        if (timeout == 0) {
            return false;
        }
        struct pollfd watch = { descriptor, POLLIN, 0 };
        const int ready = poll(&watch, 1, timeout);
        if (ready < 0 && errno == EINTR) {
            continue;
        }
        if (ready <= 0) {
            return false;
        }
        const ssize_t count = read(descriptor, (uint8_t *)result + received,
                                    sizeof(*result) - received);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        received += (size_t)count;
    }
    return true;
}

static bool parallel_action(uint64_t step, uint32_t seed, uint32_t count, bool windowed)
{
    begin_action(windowed);
    cpu_set_t original;
    if (count == 0 || count > max_memory_bytes ||
        sched_getaffinity(0, sizeof(original), &original) != 0) {
        return action_error(step, windowed, "cannot configure bounded parallel work");
    }
    int cpus[2] = { -1, -1 };
    for (int cpu = 0, found = 0; cpu < CPU_SETSIZE && found < 2; ++cpu) {
        if (CPU_ISSET(cpu, &original)) {
            cpus[found++] = cpu;
        }
    }
    if (cpus[1] < 0) {
        return action_error(step, windowed, "parallel work needs two available CPUs");
    }
    struct start_gate *gate = mmap(NULL, sizeof(*gate), PROT_READ | PROT_WRITE,
                                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (gate == MAP_FAILED) {
        return action_error(step, windowed, "cannot create parallel start gate");
    }
    atomic_init(&gate->ready[0], gate_waiting);
    atomic_init(&gate->ready[1], gate_waiting);
    if (!atomic_is_lock_free(&gate->ready[0])) {
        munmap(gate, sizeof(*gate));
        return action_error(step, windowed, "parallel start flags must be lock free");
    }
    volatile uint8_t *bytes = mmap(NULL, count, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int descriptors[2] = { -1, -1 };
    pid_t child = -1;
    struct parallel_result parent_result = { 0, -1, false };
    struct parallel_result child_result = { 0, -1, false };
    struct timespec started;
    bool ok = false;
    const char *error = "cannot allocate parallel memory or result pipe";
    if (bytes == MAP_FAILED || pipe(descriptors) != 0 ||
        clock_gettime(CLOCK_MONOTONIC, &started) != 0) {
        goto cleanup;
    }
    child = fork();
    if (child < 0) {
        error = "cannot start parallel child";
        goto cleanup;
    }
    if (child == 0) {
        close(descriptors[0]);
        const bool pinned = pin_cpu(cpus[1]);
        if (!pinned) {
            atomic_store_explicit(&gate->ready[1], gate_failed, memory_order_release);
        }
        if (pinned && start_together(gate, 1, &started)) {
            child_result.checksum = cpu2tensor_parallel_memory(bytes, seed + UINT32_C(1), count);
            child_result.cpu = sched_getcpu();
            child_result.ok = child_result.cpu == cpus[1];
        }
        ssize_t written;
        do {
            written = write(descriptors[1], &child_result, sizeof(child_result));
        } while (written < 0 && errno == EINTR);
        close(descriptors[1]);
        _exit(written == (ssize_t)sizeof(child_result) && child_result.ok ? 0 : 1);
    }
    close(descriptors[1]);
    descriptors[1] = -1;
    if (!pin_cpu(cpus[0]) || !start_together(gate, 0, &started)) {
        error = "parallel workers did not reach their assigned CPUs";
        goto cleanup;
    }
    parent_result.checksum = cpu2tensor_parallel_memory(bytes, seed, count);
    parent_result.cpu = sched_getcpu();
    parent_result.ok = parent_result.cpu == cpus[0];
    if (!parent_result.ok || !read_parallel_result(descriptors[0], &child_result, &started) ||
        !child_result.ok || child_result.cpu != cpus[1]) {
        error = "parallel child result was incomplete or had an incorrect CPU";
        goto cleanup;
    }
    ok = true;

cleanup:
    if (child > 0) {
        if (!ok) {
            kill(child, SIGKILL);
        }
        int status = 0;
        pid_t waited;
        do {
            waited = waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            if (ok) {
                error = "parallel child did not finish successfully";
            }
            ok = false;
        }
    }
    for (unsigned int index = 0; index < 2; ++index) {
        if (descriptors[index] >= 0) {
            close(descriptors[index]);
        }
    }
    if (bytes != MAP_FAILED && munmap((void *)bytes, count) != 0) {
        ok = false;
        error = "cannot release parallel memory";
    }
    const bool gate_released = munmap(gate, sizeof(*gate)) == 0;
    const bool affinity_restored = sched_setaffinity(0, sizeof(original), &original) == 0;
    if (!gate_released || !affinity_restored) {
        ok = false;
        error = "cannot restore parallel workload resources";
    }
    if (!ok) {
        return action_error(step, windowed, error);
    }
    end_action(windowed, true);
    return emit_event("C2T {\"event\":\"result\",\"step\":%" PRIu64
                      ",\"action\":\"parallel\",\"seed\":%" PRIu32
                      ",\"bytes\":%" PRIu32 ",\"cpu0\":%d,\"cpu1\":%d"
                      ",\"checksum0\":%" PRIu64 ",\"checksum1\":%" PRIu64 "}\n",
                      step, seed, count, parent_result.cpu, child_result.cpu,
                      parent_result.checksum, child_result.checksum);
}

enum context_child_stage {
    context_child_waiting,
    context_child_ready,
    context_child_moved,
    context_child_failed
};

struct context_filter_gate {
    atomic_uint child_stage;
    atomic_uint parent_stage;
    uint64_t background;
};

static bool complete_event(uint64_t steps, bool ok);

static bool wait_for_context_stage(atomic_uint *stage, unsigned int expected,
                                   const struct timespec *started)
{
    unsigned int current;
    while ((current = atomic_load_explicit(stage, memory_order_acquire)) < expected) {
        if (current == context_child_failed || remaining_parallel_ms(started) == 0 ||
            sched_yield() != 0) {
            return false;
        }
    }
    return current == expected;
}

static bool context_filter_observe(void)
{
    cpu_set_t original;
    if (sched_getaffinity(0, sizeof(original), &original) != 0) {
        return false;
    }
    int cpus[2] = { -1, -1 };
    for (int cpu = 0, found = 0; cpu < CPU_SETSIZE && found < 2; ++cpu) {
        if (CPU_ISSET(cpu, &original)) {
            cpus[found++] = cpu;
        }
    }
    if (cpus[1] < 0) {
        return false;
    }
    struct context_filter_gate *gate = mmap(NULL, sizeof(*gate), PROT_READ | PROT_WRITE,
                                            MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    volatile uint64_t *values = mmap(NULL, 2 * sizeof(*values), PROT_READ | PROT_WRITE,
                                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (gate == MAP_FAILED || values == MAP_FAILED) {
        if (gate != MAP_FAILED) munmap(gate, sizeof(*gate));
        if (values != MAP_FAILED) munmap((void *)values, 2 * sizeof(*values));
        return false;
    }
    atomic_init(&gate->child_stage, context_child_waiting);
    atomic_init(&gate->parent_stage, 0);
    gate->background = UINT64_C(0xa5a5a5a5a5a5a5a5);
    if (!atomic_is_lock_free(&gate->child_stage)) {
        munmap((void *)values, 2 * sizeof(*values));
        munmap(gate, sizeof(*gate));
        return false;
    }
    struct timespec started;
    if (clock_gettime(CLOCK_MONOTONIC, &started) != 0) {
        munmap((void *)values, 2 * sizeof(*values));
        munmap(gate, sizeof(*gate));
        return false;
    }
    const pid_t child = fork();
    if (child == 0) {
        if (!pin_cpu(cpus[1])) {
            atomic_store_explicit(&gate->child_stage, context_child_failed,
                                  memory_order_release);
            _exit(1);
        }
        atomic_store_explicit(&gate->child_stage, context_child_ready, memory_order_release);
        while (atomic_load_explicit(&gate->parent_stage, memory_order_acquire) == 0 &&
               remaining_parallel_ms(&started) != 0) {
            cpu2tensor_context_background(&gate->background, 256);
        }
        cpu2tensor_context_background(&gate->background, max_memory_bytes);
        if (!pin_cpu(cpus[0])) {
            atomic_store_explicit(&gate->child_stage, context_child_failed,
                                  memory_order_release);
            _exit(1);
        }
        atomic_store_explicit(&gate->child_stage, context_child_moved, memory_order_release);
        while (atomic_load_explicit(&gate->parent_stage, memory_order_acquire) < 2 &&
               remaining_parallel_ms(&started) != 0) {
            cpu2tensor_context_background(&gate->background, 256);
        }
        _exit(atomic_load_explicit(&gate->parent_stage, memory_order_acquire) == 2 ? 0 : 1);
    }
    bool ok = child > 0 && pin_cpu(cpus[0]) &&
              wait_for_context_stage(&gate->child_stage, context_child_ready, &started);
    if (ok) {
        cpu2tensor_context_gate();
        cpu2tensor_context_first(values);
        atomic_store_explicit(&gate->parent_stage, 1, memory_order_release);
        ok = wait_for_context_stage(&gate->child_stage, context_child_moved, &started) &&
             pin_cpu(cpus[1]);
    }
    if (ok) {
        cpu2tensor_context_second(values);
        atomic_store_explicit(&gate->parent_stage, 2, memory_order_release);
    }
    int status = 0;
    if (child > 0) {
        if (!ok) kill(child, SIGKILL);
        pid_t waited;
        do {
            waited = waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        ok = ok && waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    const uint64_t first = values[0];
    const uint64_t second = values[1];
    ok = ok && first == UINT64_C(0x1122334455667788) &&
         second == UINT64_C(0x8877665544332211);
    const bool affinity_restored = sched_setaffinity(0, sizeof(original), &original) == 0;
    const bool values_released = munmap((void *)values, 2 * sizeof(*values)) == 0;
    const bool gate_released = munmap(gate, sizeof(*gate)) == 0;
    if (!ok || !affinity_restored || !values_released || !gate_released) {
        return false;
    }
    return emit_event("C2T {\"event\":\"result\",\"step\":0,\"action\":\"context-filter\""
                      ",\"cpu0\":%d,\"cpu1\":%d,\"first\":%" PRIu64
                      ",\"second\":%" PRIu64 "}\n",
                      cpus[0], cpus[1], first, second) && complete_event(1, true);
}

static bool getpid_action(uint64_t step, bool windowed)
{
    begin_action(windowed);
    const pid_t value = getpid();
    end_action(windowed, true);
    return emit_event("C2T {\"event\":\"result\",\"step\":%" PRIu64
                      ",\"action\":\"getpid\",\"value\":%jd}\n",
                      step, (intmax_t)value);
}

static bool complete_event(uint64_t steps, bool ok)
{
    return emit_event("C2T {\"event\":\"complete\",\"steps\":%" PRIu64
                      ",\"ok\":%s}\n", steps, ok ? "true" : "false");
}

static bool observe(const struct options *options)
{
    if (!getpid_action(0, false)) {
        return false;
    }
    if (!memory_action(1, options->seed, options->bytes, false)) {
        (void)complete_event(1, false);
        return false;
    }
    if (!pipe_action(2, options->seed, max_pipe_bytes, false)) {
        (void)complete_event(2, false);
        return false;
    }
    if (!parallel_action(3, options->seed, options->bytes, false)) {
        (void)complete_event(3, false);
        return false;
    }
    return complete_event(4, true);
}

static bool interact(void)
{
    uint64_t step = 0;
    char line[max_command_bytes];
    for (;;) {
        if (!emit_event("C2T {\"event\":\"ready\",\"step\":%" PRIu64 "}\n", step)) {
            return false;
        }
        const enum command_read received = read_command(line, sizeof(line));
        if (received == command_read_failed) {
            (void)error_event(step, "guest command input closed");
            (void)complete_event(step, false);
            return false;
        }
        if (received == command_read_too_long) {
            begin_action(true);
            end_action(true, false);
            if (!error_event(step, "command must fit one short line")) {
                return false;
            }
            continue;
        }
        char *position = NULL;
        char *action = strtok_r(line, " \t\r\n", &position);
        char *first = strtok_r(NULL, " \t\r\n", &position);
        char *second = strtok_r(NULL, " \t\r\n", &position);
        char *extra = strtok_r(NULL, " \t\r\n", &position);
        if (action != NULL && strcmp(action, "quit") == 0 && first == NULL) {
            begin_action(true);
            end_action(true, true);
            return complete_event(step, true);
        }
        if (action != NULL && strcmp(action, "getpid") == 0 && first == NULL) {
            if (!getpid_action(step, true)) {
                (void)complete_event(step, false);
                return false;
            }
            ++step;
            continue;
        }
        const bool memory = action != NULL && strcmp(action, "memory") == 0;
        const bool pipe = action != NULL && strcmp(action, "pipe") == 0;
        const bool parallel = action != NULL && strcmp(action, "parallel") == 0;
        const bool compute = action != NULL && strcmp(action, "compute") == 0;
        uint32_t seed = 0;
        uint32_t count = 0;
        const uint32_t first_limit = compute ? max_memory_bytes : UINT32_MAX;
        const uint32_t limit = memory || parallel ? max_memory_bytes :
                               compute ? max_report_padding_bytes : max_pipe_bytes;
        if ((!memory && !pipe && !parallel && !compute) || extra != NULL ||
            !parse_number(first, first_limit, &seed) ||
            !parse_number(second, limit, &count) || (!compute && count == 0) ||
            (compute && seed == 0)) {
            begin_action(true);
            end_action(true, false);
            if (!error_event(step, "expected getpid, memory/pipe/parallel SEED BYTES, compute ITERATIONS PADDING, or quit")) {
                return false;
            }
            continue;
        }
        bool ok;
        if (compute) {
            begin_action(true);
            const uint64_t value = cpu2tensor_compute(seed);
            end_action(true, true);
            ok = compute_result(step, seed, count, value);
        } else {
            ok = memory ? memory_action(step, seed, count, true) :
                 parallel ? parallel_action(step, seed, count, true) :
                 pipe_action(step, seed, count, true);
        }
        if (!ok) {
            (void)complete_event(step, false);
            return false;
        }
        ++step;
    }
}

static bool read_options(struct options *options)
{
    if (mkdir("/proc", 0555) != 0 && errno != EEXIST) {
        return false;
    }
    if (mount("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC, NULL) != 0 &&
        errno != EBUSY) {
        return false;
    }
    FILE *file = fopen("/proc/cmdline", "r");
    if (file == NULL) {
        return false;
    }
    char line[max_kernel_command_bytes];
    const bool read_ok = fgets(line, sizeof(line), file) != NULL &&
                         strchr(line, '\n') != NULL;
    fclose(file);
    if (!read_ok) {
        return false;
    }
    char *position = NULL;
    bool mode_seen = false;
    bool seed_seen = false;
    bool bytes_seen = false;
    bool timeout_seen = false;
    bool context_filter_seen = false;
    const char timeout_key[] = "cpu2tensor.parallel_timeout=";
    for (char *word = strtok_r(line, " \t\r\n", &position); word != NULL;
         word = strtok_r(NULL, " \t\r\n", &position)) {
        if (strncmp(word, "cpu2tensor.mode=", 16) == 0) {
            if (mode_seen) {
                return false;
            }
            mode_seen = true;
            if (strcmp(word + 16, "interactive") == 0) {
                options->interactive = true;
            } else if (strcmp(word + 16, "observe") != 0) {
                return false;
            }
        } else if (strncmp(word, "cpu2tensor.seed=", 16) == 0) {
            if (seed_seen || !parse_number(word + 16, UINT32_MAX, &options->seed)) {
                return false;
            }
            seed_seen = true;
        } else if (strncmp(word, timeout_key, sizeof(timeout_key) - 1) == 0) {
            if (timeout_seen || !parse_number(word + sizeof(timeout_key) - 1, 3600, &parallel_timeout_seconds) ||
                parallel_timeout_seconds == 0) return false;
            timeout_seen = true;
        } else if (strncmp(word, "cpu2tensor.bytes=", 17) == 0) {
            if (bytes_seen || !parse_number(word + 17, max_memory_bytes, &options->bytes) ||
                options->bytes == 0) {
                return false;
            }
            bytes_seen = true;
        } else if (strncmp(word, "cpu2tensor.context_filter=", 26) == 0) {
            if (context_filter_seen || strcmp(word + 26, "on") != 0) {
                return false;
            }
            context_filter_seen = true;
            options->context_filter = true;
        }
    }
    return !options->context_filter || !options->interactive;
}

static void poweroff(void)
{
    sync();
    if (reboot(RB_POWER_OFF) != 0) {
        error_event(0, "guest poweroff failed");
    }
    // PID 1 must stay alive if the machine does not support poweroff.
    for (;;) {
        pause();
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    struct options options = { false, false, default_seed, default_memory_bytes };
    if (argc == 2 && strcmp(argv[1], "--check") == 0) {
        // This safe host check never mounts filesystems or shuts down a machine.
        return observe(&options) ? 0 : 1;
    }
    // Linux can pass unrecognized boot words to init as arguments. Guest options
    // come from /proc/cmdline; extra argv words must not make PID 1 exit.
    if (getpid() != 1) {
        fputs("kernel_init: run as guest PID 1, or use --check for the fixed workload\n", stderr);
        return 1;
    }
    if (!open_event_channel()) {
        event_channel_failed = true;
        fputs("kernel_init: cannot open dedicated adapter channel\n", stderr);
        poweroff();
    }
    if (!read_options(&options)) {
        (void)error_event(0, "cannot read valid guest workload options");
        (void)complete_event(0, false);
        poweroff();
    }
    cpu2tensor_capture_begin();
    if (!emit_event("C2T {\"event\":\"start\",\"mode\":\"%s\",\"seed\":%" PRIu32
                    ",\"bytes\":%" PRIu32 "}\n",
                    options.interactive ? "interactive" : "observe",
                    options.seed, options.bytes)) {
        poweroff();
    }
    if (options.interactive) {
        interact();
    } else if (options.context_filter) {
        if (!context_filter_observe()) {
            (void)error_event(0, "context filter fixture failed");
            (void)complete_event(0, false);
        }
    } else {
        observe(&options);
    }
    poweroff();
    return 0;
}
