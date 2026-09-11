// SPDX-License-Identifier: AGPL-3.0-only
#define main kernel_custom_actions_program_main
#define write simulated_write
#include "../examples/kernel_custom_actions.c"
#undef write
#undef main

#include <assert.h>

enum write_mode {
    write_complete,
    write_partial,
};

static enum write_mode mode;
static size_t write_calls;

ssize_t simulated_write(int descriptor, const void *bytes, size_t size)
{
    assert(descriptor == STDOUT_FILENO);
    assert(bytes != NULL);
    ++write_calls;
    return mode == write_partial ? (ssize_t)(size - 1) : (ssize_t)size;
}

static void reset(enum write_mode selected)
{
    mode = selected;
    write_calls = 0;
    event_channel = STDOUT_FILENO;
    command_channel = STDIN_FILENO;
    event_channel_failed = false;
}

static void check_event_helpers_report_success(void)
{
    reset(write_complete);
    assert(ready_event(0));
    assert(error_event(0, "invalid command"));
    assert(complete_event(0, false));
    assert(write_calls == 3);
}

static void check_partial_write_latches_failure(void)
{
    reset(write_partial);
    assert(!interact());
    assert(event_channel_failed);
    assert(write_calls == 1);

    mode = write_complete;
    assert(!complete_event(0, false));
    assert(write_calls == 1);
}

int main(void)
{
    check_event_helpers_report_success();
    check_partial_write_latches_failure();
}
