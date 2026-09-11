// SPDX-License-Identifier: AGPL-3.0-only
#include <cpu2tensor/transition_window.hpp>

#include <climits>
#include <new>

namespace cpu2tensor {

TransitionWindow::~TransitionWindow()
{
    if (_sources != nullptr) {
        for (uint32_t source = 0; source < _source_count; ++source) {
            delete[] _sources[source].slots;
            delete[] _sources[source].indices;
        }
    }
    delete[] _sources;
}

Result<Done> TransitionWindow::configure(uint32_t sources, uint32_t capacity_per_source)
{
    if (_sources != nullptr) return Result<Done>::failure("Transition window is already configured");
    if (sources == 0 || capacity_per_source < 2 ||
        (capacity_per_source & (capacity_per_source - 1)) != 0 ||
        uint64_t{sources} * capacity_per_source > max_transition_slots) {
        return Result<Done>::failure("Transition capacity must be a power of two and fit the fixed total bound");
    }
    auto* allocated = new (std::nothrow) Source[sources];
    if (allocated == nullptr) return Result<Done>::failure("Cannot allocate transition sources");
    for (uint32_t source = 0; source < sources; ++source) {
        allocated[source].slots = new (std::nothrow) Slot[capacity_per_source];
        allocated[source].indices = new (std::nothrow) uint32_t[capacity_per_source];
        if (allocated[source].slots == nullptr || allocated[source].indices == nullptr) {
            for (uint32_t previous = 0; previous <= source; ++previous) {
                delete[] allocated[previous].slots;
                delete[] allocated[previous].indices;
            }
            delete[] allocated;
            return Result<Done>::failure("Cannot allocate transition table");
        }
    }
    _sources = allocated;
    _source_count = sources;
    _capacity = capacity_per_source;
    _mask = capacity_per_source - 1;
    return Result<Done>::success({});
}

Result<Done> TransitionWindow::begin()
{
    if (_sources == nullptr) return Result<Done>::failure("Transition window is not configured");
    State expected = State::idle;
    if (!_state.compare_exchange_strong(expected, State::opening, std::memory_order_acq_rel))
        return Result<Done>::failure("Transition window begin arrived before the previous window closed");
    const auto previous = _window.fetch_add(1, std::memory_order_acq_rel);
    if (previous == UINT64_MAX) {
        _state.store(State::idle, std::memory_order_release);
        return Result<Done>::failure("Transition window identifier overflow");
    }
    _state.store(State::recording, std::memory_order_release);
    return Result<Done>::success({});
}

Result<Done> TransitionWindow::end()
{
    State expected = State::recording;
    if (!_state.compare_exchange_strong(expected, State::ended, std::memory_order_acq_rel))
        return Result<Done>::failure("Transition window end arrived without an open window");
    return Result<Done>::success({});
}

Result<Done> TransitionWindow::abort()
{
    State expected = State::recording;
    if (!_state.compare_exchange_strong(expected, State::aborted, std::memory_order_acq_rel))
        return Result<Done>::failure("Transition window abort arrived without an open window");
    return Result<Done>::success({});
}

BlockAdmission TransitionWindow::admit() const
{
    if (_state.load(std::memory_order_acquire) != State::recording) return {};
    return {_window.load(std::memory_order_acquire), true};
}

bool TransitionWindow::open() const
{
    const State state = _state.load(std::memory_order_acquire);
    return state == State::opening || state == State::recording;
}

uint64_t TransitionWindow::hash(uint64_t source, uint64_t destination)
{
    uint64_t value = source ^ (destination + UINT64_C(0x9e3779b97f4a7c15));
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

void TransitionWindow::reset(Source& source, uint64_t window)
{
    source.window = window;
    source.previous = 0;
    source.observed = 0;
    source.overflow = 0;
    source.distinct = 0;
    source.has_previous = false;
}

Result<Done> TransitionWindow::observe(uint32_t source_index, uint64_t address,
                                       BlockAdmission admission)
{
    if (source_index >= _source_count)
        return Result<Done>::failure("Transition source is outside the configured range");
    if (!admission.included) return Result<Done>::success({});
    if (admission.window == 0 || _window.load(std::memory_order_acquire) != admission.window)
        return Result<Done>::failure("Transition admission belongs to another window");
    const uint64_t window = admission.window;
    auto& source = _sources[source_index];
    if (source.window != window) reset(source, window);
    if (!source.has_previous) {
        source.previous = address;
        source.has_previous = true;
        return Result<Done>::success({});
    }
    if (source.observed == UINT64_MAX)
        return Result<Done>::failure("Transition execution count overflow");
    ++source.observed;
    const uint64_t previous = source.previous;
    source.previous = address;
    uint32_t slot_index = static_cast<uint32_t>(hash(previous, address)) & _mask;
    for (uint32_t probe = 0; probe < _capacity; ++probe) {
        auto& slot = source.slots[slot_index];
        if (slot.window != window) {
            if (source.distinct == _capacity)
                return Result<Done>::failure("Transition index capacity is inconsistent");
            slot.window = window;
            slot.from_address = previous;
            slot.destination = address;
            slot.count = 1;
            source.indices[source.distinct++] = slot_index;
            return Result<Done>::success({});
        }
        if (slot.from_address == previous && slot.destination == address) {
            if (slot.count == UINT64_MAX)
                return Result<Done>::failure("Transition count overflow");
            ++slot.count;
            return Result<Done>::success({});
        }
        slot_index = (slot_index + 1) & _mask;
    }
    if (source.overflow == UINT64_MAX)
        return Result<Done>::failure("Transition overflow count overflow");
    ++source.overflow;
    return Result<Done>::success({});
}

Result<WindowSummary> TransitionWindow::close()
{
    const uint64_t window = _window.load(std::memory_order_acquire);
    const State state = _state.exchange(State::idle, std::memory_order_acq_rel);
    if (state == State::idle) return Result<WindowSummary>::success({});
    WindowSummary summary{window, state == State::ended ? WindowStatus::ended :
        state == State::aborted ? WindowStatus::aborted : WindowStatus::incomplete,
        _source_count, _capacity, 0, 0, 0};
    for (uint32_t index = 0; index < _source_count; ++index) {
        const auto& source = _sources[index];
        if (source.window != window) continue;
        if (summary.distinct > UINT64_MAX - source.distinct ||
            summary.observed > UINT64_MAX - source.observed ||
            summary.overflow > UINT64_MAX - source.overflow)
            return Result<WindowSummary>::failure("Transition window totals overflow");
        summary.distinct += source.distinct;
        summary.observed += source.observed;
        summary.overflow += source.overflow;
    }
    return Result<WindowSummary>::success(summary);
}

uint32_t TransitionWindow::row_count(uint32_t source_index, uint64_t window) const
{
    if (source_index >= _source_count || _sources[source_index].window != window) return 0;
    return _sources[source_index].distinct;
}

TransitionCount TransitionWindow::row(uint32_t source_index, uint64_t window, uint32_t index) const
{
    if (source_index >= _source_count) return {};
    const auto& source = _sources[source_index];
    if (source.window != window || index >= source.distinct) return {};
    const auto& slot = source.slots[source.indices[index]];
    return {slot.from_address, slot.destination, slot.count};
}

} // namespace cpu2tensor
