// SPDX-License-Identifier: AGPL-3.0-only
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/reboot.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

enum { max_command_bytes = 96, max_repetitions = 64 };

// The operator passes these three addresses from this exact ELF to the worker.
// Different no-op bodies keep the three externally resolved PCs distinct. The
// plugin excludes the marker blocks from the compared workload.
__attribute__((noinline)) void cpu2tensor_action_begin(void)
{
    __asm__ volatile("nop" ::: "memory");
}

__attribute__((noinline)) void cpu2tensor_action_end(void)
{
    __asm__ volatile("nop\nnop" ::: "memory");
}

__attribute__((noinline)) void cpu2tensor_action_abort(void)
{
    __asm__ volatile("nop\nnop\nnop" ::: "memory");
}

static bool parse_repetitions(const char *text, uint32_t *value)
{
    if (text == NULL || text[0] < '0' || text[0] > '9') {
        return false;
    }
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed == 0 || parsed > max_repetitions) {
        return false;
    }
    *value = (uint32_t)parsed;
    return true;
}

static void error_event(uint64_t step, const char *message)
{
    printf("C2T {\"event\":\"error\",\"step\":%" PRIu64
           ",\"message\":\"%s\"}\n", step, message);
}

static void ready_event(uint64_t step)
{
    printf("C2T {\"event\":\"ready\",\"step\":%" PRIu64 "}\n", step);
}

static void complete_event(uint64_t steps, bool ok)
{
    printf("C2T {\"event\":\"complete\",\"steps\":%" PRIu64
           ",\"ok\":%s}\n", steps, ok ? "true" : "false");
}

static bool open_sequence(int flags, uint32_t repetitions)
{
    for (uint32_t index = 0; index < repetitions; ++index) {
        const int descriptor = (int)syscall(SYS_openat, AT_FDCWD, "/proc/version",
                                            flags, 0);
        if (descriptor < 0) {
            return false;
        }
        uint8_t byte = 0;
        ssize_t count;
        do {
            count = syscall(SYS_read, descriptor, &byte, sizeof(byte));
        } while (count < 0 && errno == EINTR);
        int closed;
        do {
            closed = (int)syscall(SYS_close, descriptor);
        } while (closed < 0 && errno == EINTR);
        if (count != (ssize_t)sizeof(byte) || closed != 0) {
            return false;
        }
    }
    return true;
}

static bool run_action(uint64_t step, const char *variant, uint32_t repetitions)
{
    const bool close_on_exec = strcmp(variant, "cloexec") == 0;
    if (!close_on_exec && strcmp(variant, "plain") != 0) {
        return false;
    }
    const int flags = O_RDONLY | (close_on_exec ? O_CLOEXEC : 0);
    cpu2tensor_action_begin();
    const bool ok = open_sequence(flags, repetitions);
    if (!ok) {
        cpu2tensor_action_abort();
        error_event(step, "open-read-close sequence failed");
        return false;
    }
    cpu2tensor_action_end();
    printf("C2T {\"event\":\"result\",\"step\":%" PRIu64
           ",\"action\":\"open_sequence\",\"variant\":\"%s\""
           ",\"repetitions\":%" PRIu32 ",\"syscalls\":%" PRIu32
           ",\"open_flags\":%d}\n",
           step, variant, repetitions, repetitions * UINT32_C(3), flags);
    return true;
}

static bool interact(void)
{
    uint64_t step = 0;
    char line[max_command_bytes];
    for (;;) {
        ready_event(step);
        if (fgets(line, sizeof(line), stdin) == NULL) {
            error_event(step, "guest command input closed");
            complete_event(step, false);
            return false;
        }
        if (strchr(line, '\n') == NULL) {
            int byte;
            while ((byte = getchar()) != '\n' && byte != EOF) {
            }
            cpu2tensor_action_begin();
            cpu2tensor_action_abort();
            error_event(step, "command must fit one short line");
            continue;
        }
        char *position = NULL;
        char *action = strtok_r(line, " \t\r\n", &position);
        char *variant = strtok_r(NULL, " \t\r\n", &position);
        char *count_text = strtok_r(NULL, " \t\r\n", &position);
        char *extra = strtok_r(NULL, " \t\r\n", &position);
        if (action != NULL && strcmp(action, "quit") == 0 && variant == NULL) {
            cpu2tensor_action_begin();
            cpu2tensor_action_end();
            complete_event(step, true);
            return true;
        }
        uint32_t repetitions = 0;
        if (action == NULL || strcmp(action, "open") != 0 || extra != NULL ||
            !parse_repetitions(count_text, &repetitions) ||
            (strcmp(variant, "plain") != 0 && strcmp(variant, "cloexec") != 0)) {
            cpu2tensor_action_begin();
            cpu2tensor_action_abort();
            error_event(step, "expected open plain|cloexec REPETITIONS, or quit");
            continue;
        }
        if (!run_action(step, variant, repetitions)) {
            complete_event(step, false);
            return false;
        }
        ++step;
    }
}

static bool mount_proc(void)
{
    return (mkdir("/proc", 0555) == 0 || errno == EEXIST) &&
           (mount("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC,
                  NULL) == 0 || errno == EBUSY);
}

static void poweroff(void)
{
    sync();
    reboot(RB_POWER_OFF);
    for (;;) {
        pause();
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    const bool host_check = argc == 2 && strcmp(argv[1], "--check") == 0;
    if (!host_check && (argc != 1 || getpid() != 1)) {
        fputs("kernel_custom_actions: run as guest PID 1, or use --check\n", stderr);
        return 1;
    }
    if (!host_check && !mount_proc()) {
        error_event(0, "cannot mount procfs");
        complete_event(0, false);
        poweroff();
    }
    printf("C2T {\"event\":\"start\",\"adapter\":\"open-sequence-v1\"}\n");
    const bool ok = interact();
    if (!host_check) {
        poweroff();
    }
    return ok ? 0 : 1;
}
