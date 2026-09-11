// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cpu2tensor/result.hpp>

namespace cpu2tensor {
struct KernelOptions final {
    const char* qemu;
    const char* plugin;
    const char* registers;
    const char* memory;
    const char* values;
    const char* context;
    const char* start_pc;
    const char* stop_pc;
    const char* window_start_pc;
    const char* window_end_pc;
    const char* window_abort_pc;
    const char* reducer;
    const char* blocks;
    int transition_capacity;
    const char* batching;
    const char* publication;
    int timeout_ms;
    int maximum_ms;
    bool interactive;
    char** arguments;
};
// True means the consumer cancelled; all child resources have been released.
Result<bool> run_kernel(const KernelOptions& options, int listener);
}
