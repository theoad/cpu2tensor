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

    // The preceding block is outside the observation window, so its memory
    // delta is discarded before the gate publishes its paging root. The same
    // context then migrates to source one while background work continues.
    assert(!filter.admits(filter.relation(target, true)));
    assert(filter.latch(0, 0xffffffff81001000, target, true));
    assert(!filter.latch(1, 0xffffffff81001000, background, true));
    assert(filter.admits(filter.relation(target, true)));
    assert(!filter.admits(filter.relation(background, true)));
    assert(filter.account(source0, filter.relation(target | 7, true), 1));
    assert(filter.account(source0, filter.relation(background, true), 1));
    assert(filter.account(source1, filter.relation(target, true), 1));
    assert(filter.account(source1, filter.relation(background, true), 1));

    assert(source0.add(source1));
    const auto summary = filter.summary(source0);
    assert(summary.policy == ContextFilterPolicy::drop);
    assert(summary.latch == ContextLatch::known);
    assert(summary.source == 0);
    assert(summary.gate_pc == 0xffffffff81001000);
    assert(summary.cr3 == target);
    assert(summary.paging_root == target);
    assert(summary.kept == 2);
    assert(summary.dropped == 2);
    assert(summary.matching == 2);
    assert(summary.foreign == 2);
    assert(summary.unknown == 0);

    ContextFilter permissive;
    ContextFilterCounts permissive_counts;
    permissive.configure(ContextFilterPolicy::keep);
    // Keep begins at the same gate and does not retain a pre-gate access.
    assert(permissive.latch(1, 7, target, true));
    assert(permissive.account(permissive_counts, permissive.relation(background, true), 1));
    assert(permissive.account(permissive_counts, permissive.relation(target, true), 1));
    const auto kept = permissive.summary(permissive_counts);
    assert(kept.kept == 2 && kept.dropped == 0);
    assert(kept.matching == 1 && kept.foreign == 1 && kept.unknown == 0);

    ContextFilterCounts grouped;
    assert(filter.account(grouped, ContextRelation::unknown, 11));
    assert(filter.account(grouped, ContextRelation::matching, 7));
    assert(filter.account(grouped, ContextRelation::foreign, 5));
    assert(grouped.kept == 7 && grouped.dropped == 16);
    assert(grouped.matching == 7 && grouped.foreign == 5 && grouped.unknown == 11);

    ContextFilterCounts overflow;
    overflow.foreign = UINT64_MAX;
    assert(!filter.account(overflow, ContextRelation::foreign, 1));
    assert(overflow.foreign == UINT64_MAX && overflow.dropped == 0);

    ContextFilterCounts add_overflow;
    ContextFilterCounts add_one;
    add_overflow.kept = UINT64_MAX;
    add_one.kept = 1;
    assert(!add_overflow.add(add_one));
    assert(add_overflow.kept == UINT64_MAX);
}
