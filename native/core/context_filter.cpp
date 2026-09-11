// SPDX-License-Identifier: AGPL-3.0-only
#include <cpu2tensor/context_filter.hpp>

#include <limits>

namespace cpu2tensor {

bool ContextFilterCounts::add(const ContextFilterCounts& other)
{
    constexpr auto maximum = std::numeric_limits<uint64_t>::max();
    if (kept > maximum - other.kept || dropped > maximum - other.dropped ||
        matching > maximum - other.matching || foreign > maximum - other.foreign ||
        unknown > maximum - other.unknown) return false;
    kept += other.kept;
    dropped += other.dropped;
    matching += other.matching;
    foreign += other.foreign;
    unknown += other.unknown;
    return true;
}

void ContextFilter::configure(ContextFilterPolicy policy) { _policy = policy; }

uint64_t ContextFilter::paging_root(uint64_t cr3) { return cr3 & ~UINT64_C(0xfff); }

bool ContextFilter::latch(uint32_t source, uint64_t gate_pc, uint64_t cr3, bool known)
{
    auto expected = ContextLatch::waiting;
    if (!_latch.compare_exchange_strong(expected, ContextLatch::writing,
                                        std::memory_order_acq_rel)) return false;
    _source.store(source, std::memory_order_relaxed);
    _gate_pc.store(gate_pc, std::memory_order_relaxed);
    _cr3.store(cr3, std::memory_order_relaxed);
    _paging_root.store(paging_root(cr3), std::memory_order_relaxed);
    _latch.store(known ? ContextLatch::known : ContextLatch::unknown,
                 std::memory_order_release);
    return true;
}

ContextRelation ContextFilter::relation(uint64_t cr3, bool known) const
{
    ContextLatch latch = _latch.load(std::memory_order_acquire);
    // A concurrent gate publication is unknown for this callback. Waiting here
    // would couple an unrelated vCPU's progress to the gate source.
    if (latch == ContextLatch::writing) return ContextRelation::unknown;
    if (!known || latch != ContextLatch::known) return ContextRelation::unknown;
    return paging_root(cr3) == _paging_root.load(std::memory_order_relaxed) ?
        ContextRelation::matching : ContextRelation::foreign;
}

bool ContextFilter::admits(ContextRelation relation) const
{
    return relation == ContextRelation::matching || _policy == ContextFilterPolicy::keep;
}

bool ContextFilter::account(ContextFilterCounts& counts, ContextRelation relation,
                            uint64_t count) const
{
    uint64_t* relation_count = relation == ContextRelation::matching ? &counts.matching :
        relation == ContextRelation::foreign ? &counts.foreign : &counts.unknown;
    uint64_t* policy_count = admits(relation) ? &counts.kept : &counts.dropped;
    constexpr auto maximum = std::numeric_limits<uint64_t>::max();
    if (*relation_count > maximum - count || *policy_count > maximum - count) return false;
    *relation_count += count;
    *policy_count += count;
    return true;
}

ContextFilterSummary ContextFilter::summary(const ContextFilterCounts& counts) const
{
    ContextLatch latch = _latch.load(std::memory_order_acquire);
    // Execution has stopped before summary(), so writing cannot persist here.
    return {_policy, latch, _source.load(std::memory_order_relaxed),
            _gate_pc.load(std::memory_order_relaxed), _cr3.load(std::memory_order_relaxed),
            _paging_root.load(std::memory_order_relaxed),
            counts.kept, counts.dropped, counts.matching, counts.foreign, counts.unknown};
}

} // namespace cpu2tensor
