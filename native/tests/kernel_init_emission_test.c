// SPDX-License-Identifier: AGPL-3.0-only
#define main kernel_init_program_main
#define write simulated_write
#include "../examples/kernel_init.c"
#undef write
#undef main

#include <assert.h>

enum write_mode {
    write_complete,
    write_partial,
    write_interrupted_once
};

static enum write_mode mode;
static size_t write_calls;
static char captured[max_event_bytes];
static size_t captured_size;

ssize_t simulated_write(int descriptor, const void *bytes, size_t size)
{
    assert(descriptor == STDOUT_FILENO);
    ++write_calls;
    if (mode == write_interrupted_once && write_calls == 1) {
        errno = EINTR;
        return -1;
    }
    if (mode == write_partial) {
        return (ssize_t)(size - 1);
    }
    assert(size <= sizeof(captured));
    memcpy(captured, bytes, size);
    captured_size = size;
    return (ssize_t)size;
}

static void reset(enum write_mode selected)
{
    mode = selected;
    write_calls = 0;
    captured_size = 0;
    event_channel_failed = false;
    event_channel = STDOUT_FILENO;
}

static void check_complete_write(void)
{
    static const char expected[] = "C2T {\"event\":\"complete\",\"steps\":4,\"ok\":true}\n";
    reset(write_complete);
    assert(emit_event("C2T {\"event\":\"complete\",\"steps\":%d,\"ok\":true}\n", 4));
    assert(write_calls == 1);
    assert(captured_size == sizeof(expected) - 1);
    assert(memcmp(captured, expected, sizeof(expected) - 1) == 0);
}

static void check_eintr_retry(void)
{
    reset(write_interrupted_once);
    assert(emit_event("C2T {\"event\":\"ready\",\"step\":0}\n"));
    assert(write_calls == 2);
    assert(!event_channel_failed);
}

static void check_partial_write_latches_failure(void)
{
    reset(write_partial);
    assert(!emit_event("C2T {\"event\":\"result\",\"step\":0}\n"));
    assert(write_calls == 1);
    mode = write_complete;
    assert(!emit_event("C2T {\"event\":\"complete\",\"steps\":0,\"ok\":false}\n"));
    assert(write_calls == 1);
}

static void check_format_failures_do_not_write(void)
{
    char oversized[max_event_bytes];
    memset(oversized, 'x', sizeof(oversized) - 1);
    oversized[sizeof(oversized) - 1] = 0;
    reset(write_complete);
    assert(!emit_event("C2T %s\n", oversized));
    assert(write_calls == 0);
    assert(!emit_event("C2T {\"event\":\"complete\"}\n"));
    assert(write_calls == 0);

    reset(write_complete);
    assert(!emit_event("ordinary output\n"));
    assert(write_calls == 0);
}

static void check_compute_report_is_outside_the_action_body(void)
{
    reset(write_complete);
    const uint64_t first = cpu2tensor_compute(257);
    const uint64_t second = cpu2tensor_compute(257);
    const uint64_t different = cpu2tensor_compute(521);
    assert(first == second);
    assert(first != different);
    assert(compute_result(3, 257, 5, first));
    static const char padding[] = "\"padding\":\"xxxxx\"";
    assert(memmem(captured, captured_size, padding, sizeof(padding) - 1) != NULL);
}

int main(void)
{
    check_complete_write();
    check_eintr_retry();
    check_partial_write_latches_failure();
    check_format_failures_do_not_write();
    check_compute_report_is_outside_the_action_body();
}
