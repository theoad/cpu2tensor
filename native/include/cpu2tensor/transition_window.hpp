// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cpu2tensor/result.hpp>

namespace cpu2tensor {

enum class WindowStatus : uint32_t {
    none = 0,
    ended = 1,
    aborted = 2,
    incomplete = 3,
};

inline constexpr uint64_t max_transition_slots = uint64_t{1} << 20;

struct TransitionCount final {
    uint64_t from_address = 0;
    uint64_t destination = 0;
    uint64_t count = 0;
};

struct BlockAdmission final {
    uint64_t window = 0;
    bool included = false;
};

struct WindowSummary final {
    uint64_t id = 0;
    WindowStatus status = WindowStatus::none;
    uint32_t sources = 0;
    uint32_t capacity_per_source = 0;
    uint64_t distinct = 0;
    uint64_t observed = 0;
    uint64_t overflow = 0;
};

struct TransitionSourceSummary final {
    uint32_t distinct = 0;
    uint64_t observed = 0;
    uint64_t overflow = 0;
};

class TransitionWindow final {
public:
    TransitionWindow() = default;
    ~TransitionWindow();
    TransitionWindow(const TransitionWindow&) = delete;
    TransitionWindow& operator=(const TransitionWindow&) = delete;
    TransitionWindow(TransitionWindow&&) = delete;
    TransitionWindow& operator=(TransitionWindow&&) = delete;

    Result<Done> configure(uint32_t sources, uint32_t capacity_per_source);
    Result<Done> begin();
    Result<Done> end();
    Result<Done> abort();
    BlockAdmission admit() const;
    Result<Done> observe(uint32_t source, uint64_t address, BlockAdmission admission);

    // The caller must stop every producer before close and retain that stop until
    // it has read the rows. A later begin may reuse their fixed storage.
    Result<WindowSummary> close();
    uint32_t row_count(uint32_t source, uint64_t window) const;
    TransitionCount row(uint32_t source, uint64_t window, uint32_t index) const;
    TransitionSourceSummary source_summary(uint32_t source, uint64_t window) const;
    bool open() const;

private:
    enum class State : uint8_t { idle, opening, recording, ended, aborted };

    struct Slot final {
        uint64_t window = 0;
        uint64_t from_address = 0;
        uint64_t destination = 0;
        uint64_t count = 0;
    };

    struct Source final {
        Slot* slots = nullptr;
        uint32_t* indices = nullptr;
        uint64_t window = 0;
        uint64_t previous = 0;
        uint64_t observed = 0;
        uint64_t overflow = 0;
        uint32_t distinct = 0;
        bool has_previous = false;
    };

    static uint64_t hash(uint64_t source, uint64_t destination);
    void reset(Source& source, uint64_t window);

    Source* _sources = nullptr;
    uint32_t _source_count = 0;
    uint32_t _capacity = 0;
    uint32_t _mask = 0;
    std::atomic<State> _state{State::idle};
    std::atomic<uint64_t> _window{0};
};

} // namespace cpu2tensor
