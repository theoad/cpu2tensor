// SPDX-License-Identifier: AGPL-3.0-only
#include <cassert>
#include <cpu2tensor/context_filter.hpp>

using namespace cpu2tensor;

int main()
{
    constexpr uint64_t target = 0x12000;
    constexpr uint64_t background = 0x34000;
    ContextFilter filter;
    ContextFilterCounts source0;
    ContextFilterCounts source1;
    filter.configure(ContextFilterPolicy::drop);

    // Source zero reaches the target gate. The same context then migrates to
    // source one while background work continues on both sources.
    assert(!filter.keep_memory(source0, target, true));
    assert(filter.latch(0, 0xffffffff81001000, target, true));
    assert(!filter.latch(1, 0xffffffff81001000, background, true));
    assert(filter.keep_registers(target, true));
    assert(!filter.keep_registers(background, true));
    assert(filter.keep_memory(source0, target | 7, true));
    assert(!filter.keep_memory(source0, background, true));
    assert(filter.keep_memory(source1, target, true));
    assert(!filter.keep_memory(source1, background, true));

    source0.add(source1);
    const auto summary = filter.summary(source0);
    assert(summary.policy == ContextFilterPolicy::drop);
    assert(summary.latch == ContextLatch::known);
    assert(summary.source == 0);
    assert(summary.gate_pc == 0xffffffff81001000);
    assert(summary.cr3 == target);
    assert(summary.paging_root == target);
    assert(summary.kept == 2);
    assert(summary.dropped == 3);
    assert(summary.matching == 2);
    assert(summary.foreign == 2);
    assert(summary.unknown == 1);

    ContextFilter permissive;
    ContextFilterCounts permissive_counts;
    permissive.configure(ContextFilterPolicy::keep);
    assert(permissive.keep_memory(permissive_counts, background, true));
    assert(permissive.latch(1, 7, target, true));
    assert(permissive.keep_memory(permissive_counts, background, true));
    assert(permissive.keep_memory(permissive_counts, target, true));
    const auto kept = permissive.summary(permissive_counts);
    assert(kept.kept == 3 && kept.dropped == 0);
    assert(kept.matching == 1 && kept.foreign == 1 && kept.unknown == 1);
}
