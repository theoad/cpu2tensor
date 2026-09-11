// SPDX-License-Identifier: AGPL-3.0-only
#include <cpu2tensor/transition_window.hpp>

#include <cassert>
#include <atomic>
#include <cstdint>
#include <pthread.h>
#include <sched.h>

using namespace cpu2tensor;

namespace {

Result<Done> observe(TransitionWindow& window, uint32_t source, uint64_t address)
{
    return window.observe(source, address, window.admit());
}

struct AdmissionRace final {
    TransitionWindow* window = nullptr;
    BlockAdmission admission{};
    std::atomic<bool> admitted{false};
    std::atomic<bool> continue_observe{false};
};

void* observe_after_end(void* opaque)
{
    auto& race = *static_cast<AdmissionRace*>(opaque);
    race.admission = race.window->admit();
    race.admitted.store(true, std::memory_order_release);
    while (!race.continue_observe.load(std::memory_order_acquire)) sched_yield();
    assert(race.window->observe(0, 20, race.admission).ok());
    return nullptr;
}

TransitionCount find(const TransitionWindow& window, uint32_t source, uint64_t id,
                     uint64_t from, uint64_t to)
{
    for (uint32_t index = 0; index < window.row_count(source, id); ++index) {
        const auto row = window.row(source, id, index);
        if (row.from_address == from && row.destination == to) return row;
    }
    return {};
}

void repeated_two_source_windows()
{
    TransitionWindow window;
    assert(window.configure(2, 8).ok());
    assert(window.begin().ok());
    assert(observe(window, 0, 10).ok());
    assert(observe(window, 1, 90).ok());
    assert(observe(window, 0, 20).ok());
    assert(observe(window, 1, 91).ok());
    assert(observe(window, 0, 10).ok());
    assert(observe(window, 0, 20).ok());
    assert(window.end().ok());
    const auto first_result = window.close();
    assert(first_result.ok());
    const auto first = first_result.value();
    assert(first.id == 1 && first.status == WindowStatus::ended);
    assert(first.sources == 2 && first.capacity_per_source == 8);
    assert(first.distinct == 3 && first.observed == 4 && first.overflow == 0);
    assert(find(window, 0, first.id, 10, 20).count == 2);
    assert(find(window, 0, first.id, 20, 10).count == 1);
    assert(find(window, 1, first.id, 90, 91).count == 1);

    assert(window.begin().ok());
    assert(observe(window, 0, 20).ok());
    assert(observe(window, 0, 30).ok());
    assert(window.end().ok());
    const auto second_result = window.close();
    assert(second_result.ok());
    const auto second = second_result.value();
    assert(second.id == 2 && second.distinct == 1 && second.observed == 1);
    assert(find(window, 0, second.id, 20, 30).count == 1);
    assert(find(window, 0, second.id, 10, 20).count == 0);
}

void incomplete_and_abort_are_explicit()
{
    TransitionWindow window;
    assert(window.configure(2, 4).ok());
    const auto empty = window.close();
    assert(empty.ok() && empty.value().status == WindowStatus::none);
    assert(window.begin().ok());
    assert(window.open() && window.admit().included);
    assert(observe(window, 0, 1).ok());
    assert(observe(window, 0, 2).ok());
    const auto incomplete_result = window.close();
    assert(incomplete_result.ok());
    const auto incomplete = incomplete_result.value();
    assert(incomplete.status == WindowStatus::incomplete);
    assert(incomplete.observed == 1);
    assert(!window.open() && !window.admit().included);

    assert(window.begin().ok());
    assert(observe(window, 1, 3).ok());
    assert(observe(window, 1, 4).ok());
    assert(window.abort().ok());
    assert(!window.open() && !window.admit().included);
    const auto aborted_result = window.close();
    assert(aborted_result.ok());
    const auto aborted = aborted_result.value();
    assert(aborted.status == WindowStatus::aborted);
    assert(aborted.observed == 1);
}

void overflow_is_not_silent()
{
    TransitionWindow window;
    assert(window.configure(1, 2).ok());
    assert(window.begin().ok());
    for (uint64_t address = 1; address <= 5; ++address)
        assert(observe(window, 0, address).ok());
    assert(window.end().ok());
    const auto summary_result = window.close();
    assert(summary_result.ok());
    const auto summary = summary_result.value();
    assert(summary.status == WindowStatus::ended);
    assert(summary.distinct == 2);
    assert(summary.observed == 4);
    assert(summary.overflow == 2);
}

void admitted_block_finishes_after_end()
{
    TransitionWindow window;
    assert(window.configure(1, 8).ok());
    assert(window.begin().ok());
    assert(observe(window, 0, 10).ok());
    AdmissionRace race{&window};
    pthread_t thread;
    assert(pthread_create(&thread, nullptr, observe_after_end, &race) == 0);
    while (!race.admitted.load(std::memory_order_acquire)) sched_yield();
    assert(race.admission.included);
    assert(window.end().ok());
    assert(!window.admit().included);
    race.continue_observe.store(true, std::memory_order_release);
    assert(pthread_join(thread, nullptr) == 0);
    const auto closed = window.close();
    assert(closed.ok());
    assert(closed.value().observed == 1);
    assert(find(window, 0, closed.value().id, 10, 20).count == 1);
}

void invalid_markers_and_sources_fail()
{
    TransitionWindow window;
    assert(!window.configure(0, 8).ok());
    assert(window.configure(2, 8).ok());
    assert(!window.configure(2, 8).ok());
    assert(!window.end().ok());
    assert(!window.abort().ok());
    assert(window.begin().ok());
    assert(!window.begin().ok());
    assert(!window.observe(2, 1, window.admit()).ok());
    assert(window.end().ok());
    assert(!window.end().ok());
    (void)window.close();
}

void high_rate_run_stays_at_fixed_row_capacity()
{
    constexpr uint64_t blocks = 1000000;
    TransitionWindow window;
    assert(window.configure(2, 8).ok());
    assert(window.begin().ok());
    for (uint64_t index = 0; index < blocks; ++index) {
        constexpr uint64_t pattern[] = {10, 20, 30, 20};
        assert(observe(window, 0, pattern[index % 4]).ok());
        assert(observe(window, 1, 1000 + index).ok());
    }
    assert(window.end().ok());
    const auto closed = window.close();
    assert(closed.ok());
    const auto summary = closed.value();
    assert(summary.observed == 2 * (blocks - 1));
    assert(window.row_count(0, summary.id) == 4);
    assert(window.row_count(1, summary.id) == 8);
    const auto first = window.source_summary(0, summary.id);
    const auto second = window.source_summary(1, summary.id);
    assert(first.distinct == 4 && first.observed == blocks - 1 && first.overflow == 0);
    assert(second.distinct == 8 && second.observed == blocks - 1 &&
           second.overflow == blocks - 9);
    assert(find(window, 0, summary.id, 10, 20).count == blocks / 4);
    assert(find(window, 0, summary.id, 20, 30).count == blocks / 4);
    assert(find(window, 0, summary.id, 30, 20).count == blocks / 4);
    assert(find(window, 0, summary.id, 20, 10).count == blocks / 4 - 1);
}

} // namespace

int main()
{
    repeated_two_source_windows();
    incomplete_and_abort_are_explicit();
    overflow_is_not_silent();
    admitted_block_finishes_after_end();
    invalid_markers_and_sources_fail();
    high_rate_run_stays_at_fixed_row_capacity();
}
