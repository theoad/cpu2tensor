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
    const char* start_pc;
    int timeout_ms;
    char** arguments;
};
// True means the consumer cancelled; all child resources have been released.
Result<bool> run_kernel(const KernelOptions& options, int listener);
}
