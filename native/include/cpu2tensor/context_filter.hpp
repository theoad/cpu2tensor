// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <atomic>
#include <cstdint>

namespace cpu2tensor {

enum class ContextFilterPolicy : uint32_t { drop = 1, keep = 2 };
enum class ContextRelation : uint32_t { unknown = 1, matching = 2, foreign = 3 };
enum class ContextLatch : uint32_t { waiting = 0, writing = 1, known = 2, unknown = 3 };

struct ContextFilterSummary final {
    ContextFilterPolicy policy = ContextFilterPolicy::drop;
    ContextLatch latch = ContextLatch::waiting;
    uint32_t source = 0;
    uint64_t gate_pc = 0;
    uint64_t cr3 = 0;
    uint64_t paging_root = 0;
    uint64_t kept = 0;
    uint64_t dropped = 0;
    uint64_t matching = 0;
    uint64_t foreign = 0;
    uint64_t unknown = 0;
};

struct ContextFilterCounts final {
    uint64_t kept = 0;
    uint64_t dropped = 0;
    uint64_t matching = 0;
    uint64_t foreign = 0;
    uint64_t unknown = 0;

    void add(const ContextFilterCounts& other);
};

// One filter belongs to one QEMU process. Latching is a one-time cross-vCPU
// operation; ordinary decisions use only atomics and never serialize sources.
class ContextFilter final {
public:
    void configure(ContextFilterPolicy policy);
    bool latch(uint32_t source, uint64_t gate_pc, uint64_t cr3, bool known);
    ContextRelation relation(uint64_t cr3, bool known) const;
    bool keep_registers(uint64_t cr3, bool known) const;
    bool keep_memory(ContextFilterCounts& counts, uint64_t cr3, bool known) const;
    ContextFilterSummary summary(const ContextFilterCounts& counts = {}) const;

private:
    static uint64_t paging_root(uint64_t cr3);
    bool keep(ContextRelation relation) const;

    ContextFilterPolicy _policy = ContextFilterPolicy::drop;
    std::atomic<ContextLatch> _latch{ContextLatch::waiting};
    std::atomic<uint32_t> _source{0};
    std::atomic<uint64_t> _gate_pc{0};
    std::atomic<uint64_t> _cr3{0};
    std::atomic<uint64_t> _paging_root{0};
};

} // namespace cpu2tensor
