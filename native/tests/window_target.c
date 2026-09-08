// SPDX-License-Identifier: AGPL-3.0-only
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { word_count = 8, workload_steps = 64, tail_steps = 512 };
volatile uint64_t window_words[word_count];
volatile unsigned window_starts;
volatile unsigned window_stops;

// Separate symbols give the trace test an oracle from this target's own ELF.
__attribute__((noinline)) void window_start_marker(void) { ++window_starts; }
__attribute__((noinline)) void window_stop_marker(void) { ++window_stops; }

__attribute__((noinline)) void window_workload(void)
{
    for (unsigned step = 0; step < workload_steps; ++step) {
        window_words[step % word_count] += 3 * step + 1;
    }
}

__attribute__((noinline)) void window_shutdown_tail(void)
{
    for (unsigned step = 0; step < tail_steps; ++step) {
        window_words[step % word_count] += 5 * step + 7;
    }
}

int main(int argc, char** argv)
{
    const char* mode = argc == 2 ? argv[1] : "normal";
    const int missing_start = strcmp(mode, "missing-start") == 0;
    const int missing_stop = strcmp(mode, "missing-stop") == 0;
    const int stop_before_start = strcmp(mode, "stop-before-start") == 0;
    const int repeat = strcmp(mode, "repeat") == 0;
    if (argc > 2 || (!missing_start && !missing_stop && !stop_before_start &&
                    !repeat && strcmp(mode, "normal") != 0)) {
        fputs("unknown window target mode\n", stderr);
        return 2;
    }
    if (stop_before_start) window_stop_marker();
    if (!missing_start) window_start_marker();
    window_workload();
    if (!missing_stop) window_stop_marker();
    if (repeat) {
        window_start_marker();
        window_workload();
        window_stop_marker();
    }
    window_shutdown_tail();
    uint64_t checksum = 0;
    for (unsigned index = 0; index < word_count; ++index) checksum += window_words[index];
    if (printf("window: checksum=%" PRIu64 "\n", checksum) < 0) return 1;
    return fflush(stdout) == 0 ? 0 : 1;
}
