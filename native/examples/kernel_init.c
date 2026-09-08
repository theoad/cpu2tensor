// SPDX-License-Identifier: AGPL-3.0-only
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
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
#include <time.h>
#include <unistd.h>

enum {
    max_memory_bytes = 65536,
    max_pipe_bytes = 256,
    max_command_bytes = 128,
    max_kernel_command_bytes = 4096,
    default_memory_bytes = 4096,
    default_seed = 17,
    parallel_timeout_seconds = 30
};

struct options {
    bool interactive;
    uint32_t seed;
    uint32_t bytes;
};

// The worker may use this stable guest PC for explicit postboot capture.
// It changes no guest state and introduces no instrumentation transport ABI.
__attribute__((noinline)) void cpu2tensor_capture_begin(void)
{
    __asm__ volatile("" ::: "memory");
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

static void error_event(uint64_t step, const char *message)
{
    // Messages are fixed strings without JSON control characters.
    printf("C2T {\"event\":\"error\",\"step\":%" PRIu64
           ",\"message\":\"%s\"}\n", step, message);
}

static uint8_t next_byte(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (uint8_t)(*state >> 24);
}

static bool memory_action(uint64_t step, uint32_t seed, uint32_t count)
{
    if (count == 0 || count > max_memory_bytes) {
        error_event(step, "guest memory size is outside the allowed range");
        return false;
    }
    void *mapping = mmap(NULL, count, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) {
        error_event(step, "cannot allocate guest memory");
        return false;
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
        error_event(step, "cannot release guest memory");
        return false;
    }
    printf("C2T {\"event\":\"result\",\"step\":%" PRIu64
           ",\"action\":\"memory\",\"seed\":%" PRIu32
           ",\"bytes\":%" PRIu32 ",\"checksum\":%" PRIu64 "}\n",
           step, seed, count, checksum);
    return true;
}

static bool pipe_action(uint64_t step, uint32_t seed, uint32_t count)
{
    if (count == 0 || count > max_pipe_bytes) {
        error_event(step, "guest pipe size is outside the allowed range");
        return false;
    }
    int descriptors[2];
    if (pipe(descriptors) != 0) {
        error_event(step, "cannot create guest pipe");
        return false;
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
        error_event(step, "cannot write complete guest pipe message");
        return false;
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
            error_event(step, "cannot read complete guest pipe message");
            return false;
        }
        if (memcmp(sent + total, received, (size_t)size) != 0) {
            close(descriptors[0]);
            error_event(step, "guest pipe message changed");
            return false;
        }
        for (size_t index = 0; index < (size_t)size; ++index) {
            checksum += received[index];
        }
        total += (size_t)size;
    }
    close(descriptors[0]);
    printf("C2T {\"event\":\"result\",\"step\":%" PRIu64
           ",\"action\":\"pipe\",\"seed\":%" PRIu32
           ",\"bytes\":%" PRIu32 ",\"checksum\":%" PRIu64 "}\n",
           step, seed, count, checksum);
    return true;
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

static bool parallel_action(uint64_t step, uint32_t seed, uint32_t count)
{
    cpu_set_t original;
    if (count == 0 || count > max_memory_bytes ||
        sched_getaffinity(0, sizeof(original), &original) != 0) {
        error_event(step, "cannot configure bounded parallel work");
        return false;
    }
    int cpus[2] = { -1, -1 };
    for (int cpu = 0, found = 0; cpu < CPU_SETSIZE && found < 2; ++cpu) {
        if (CPU_ISSET(cpu, &original)) {
            cpus[found++] = cpu;
        }
    }
    if (cpus[1] < 0) {
        error_event(step, "parallel work needs two available CPUs");
        return false;
    }
    struct start_gate *gate = mmap(NULL, sizeof(*gate), PROT_READ | PROT_WRITE,
                                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (gate == MAP_FAILED) {
        error_event(step, "cannot create parallel start gate");
        return false;
    }
    atomic_init(&gate->ready[0], gate_waiting);
    atomic_init(&gate->ready[1], gate_waiting);
    if (!atomic_is_lock_free(&gate->ready[0])) {
        munmap(gate, sizeof(*gate));
        error_event(step, "parallel start flags must be lock free");
        return false;
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
        error_event(step, error);
        return false;
    }
    printf("C2T {\"event\":\"result\",\"step\":%" PRIu64
           ",\"action\":\"parallel\",\"seed\":%" PRIu32
           ",\"bytes\":%" PRIu32 ",\"cpu0\":%d,\"cpu1\":%d"
           ",\"checksum0\":%" PRIu64 ",\"checksum1\":%" PRIu64 "}\n",
           step, seed, count, parent_result.cpu, child_result.cpu,
           parent_result.checksum, child_result.checksum);
    return true;
}

static void getpid_action(uint64_t step)
{
    printf("C2T {\"event\":\"result\",\"step\":%" PRIu64
           ",\"action\":\"getpid\",\"value\":%jd}\n", step, (intmax_t)getpid());
}

static void complete_event(uint64_t steps, bool ok)
{
    printf("C2T {\"event\":\"complete\",\"steps\":%" PRIu64
           ",\"ok\":%s}\n", steps, ok ? "true" : "false");
}

static bool observe(const struct options *options)
{
    getpid_action(0);
    if (!memory_action(1, options->seed, options->bytes)) {
        complete_event(1, false);
        return false;
    }
    if (!pipe_action(2, options->seed, max_pipe_bytes)) {
        complete_event(2, false);
        return false;
    }
    if (!parallel_action(3, options->seed, options->bytes)) {
        complete_event(3, false);
        return false;
    }
    complete_event(4, true);
    return true;
}

static bool interact(void)
{
    uint64_t step = 0;
    char line[max_command_bytes];
    for (;;) {
        printf("C2T {\"event\":\"ready\",\"step\":%" PRIu64 "}\n", step);
        if (fgets(line, sizeof(line), stdin) == NULL) {
            error_event(step, "guest command input closed");
            complete_event(step, false);
            return false;
        }
        if (strchr(line, '\n') == NULL) {
            int byte;
            while ((byte = getchar()) != '\n' && byte != EOF) {
            }
            error_event(step, "command must fit one short line");
            continue;
        }
        char *position = NULL;
        char *action = strtok_r(line, " \t\r\n", &position);
        char *first = strtok_r(NULL, " \t\r\n", &position);
        char *second = strtok_r(NULL, " \t\r\n", &position);
        char *extra = strtok_r(NULL, " \t\r\n", &position);
        if (action != NULL && strcmp(action, "quit") == 0 && first == NULL) {
            complete_event(step, true);
            return true;
        }
        if (action != NULL && strcmp(action, "getpid") == 0 && first == NULL) {
            getpid_action(step++);
            continue;
        }
        const bool memory = action != NULL && strcmp(action, "memory") == 0;
        const bool pipe = action != NULL && strcmp(action, "pipe") == 0;
        const bool parallel = action != NULL && strcmp(action, "parallel") == 0;
        uint32_t seed = 0;
        uint32_t count = 0;
        const uint32_t limit = memory || parallel ? max_memory_bytes : max_pipe_bytes;
        if ((!memory && !pipe && !parallel) || extra != NULL ||
            !parse_number(first, UINT32_MAX, &seed) ||
            !parse_number(second, limit, &count) || count == 0) {
            error_event(step, "expected getpid, memory/pipe/parallel SEED BYTES, or quit");
            continue;
        }
        const bool ok = memory ? memory_action(step, seed, count) :
                        parallel ? parallel_action(step, seed, count) :
                        pipe_action(step, seed, count);
        if (!ok) {
            complete_event(step, false);
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
        } else if (strncmp(word, "cpu2tensor.bytes=", 17) == 0) {
            if (bytes_seen || !parse_number(word + 17, max_memory_bytes, &options->bytes) ||
                options->bytes == 0) {
                return false;
            }
            bytes_seen = true;
        }
    }
    return true;
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
    struct options options = { false, default_seed, default_memory_bytes };
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
    if (!read_options(&options)) {
        error_event(0, "cannot read valid guest workload options");
        complete_event(0, false);
        poweroff();
    }
    cpu2tensor_capture_begin();
    printf("C2T {\"event\":\"start\",\"mode\":\"%s\",\"seed\":%" PRIu32
           ",\"bytes\":%" PRIu32 "}\n",
           options.interactive ? "interactive" : "observe", options.seed, options.bytes);
    if (options.interactive) {
        interact();
    } else {
        observe(&options);
    }
    poweroff();
}
