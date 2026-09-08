// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cpu2tensor/result.hpp>

namespace cpu2tensor {

inline constexpr size_t frame_ring_max_bytes = 4096;
inline constexpr size_t frame_ring_cache_line_bytes = 128;

enum class FramePush { stored, full };

struct FrameView final {
    const uint8_t* data = nullptr;
    size_t size = 0;
};

// One producer owns try_push; one consumer owns front, pop, and empty.
// The caller owns waiting, cancellation, and the lifetime of both threads.
template <size_t Capacity = 8> class FrameRing final {
    static_assert(Capacity != 0 && (Capacity & (Capacity - 1)) == 0,
                  "frame ring capacity must be a power of two");
    static_assert(Capacity <= UINT64_MAX / 2);
    static_assert(std::atomic<uint64_t>::is_always_lock_free);

public:
    FrameRing() = default;
    FrameRing(const FrameRing&) = delete;
    FrameRing& operator=(const FrameRing&) = delete;
    FrameRing(FrameRing&&) = delete;
    FrameRing& operator=(FrameRing&&) = delete;

    Result<FramePush> try_push(const uint8_t* data, size_t size) {
        if (data == nullptr || size == 0 || size > frame_ring_max_bytes) {
            return Result<FramePush>::failure("frame size must be between 1 and 4096 bytes");
        }
        const uint64_t write = _producer.position.load(std::memory_order_relaxed);
        const uint64_t read = _consumer.position.load(std::memory_order_acquire);
        if (write - read == Capacity) return Result<FramePush>::success(FramePush::full);

        Slot& slot = _slots[write & (Capacity - 1)];
        std::memcpy(slot.data, data, size);
        slot.size = size;
        // Publish only after all bytes and the length are initialized.
        _producer.position.store(write + 1, std::memory_order_release);
        return Result<FramePush>::success(FramePush::stored);
    }

    // The returned bytes remain owned by the consumer until pop. In particular,
    // the collector must finish publishing them before releasing the slot.
    FrameView front() const {
        const uint64_t read = _consumer.position.load(std::memory_order_relaxed);
        if (read == _producer.position.load(std::memory_order_acquire)) return {};
        const Slot& slot = _slots[read & (Capacity - 1)];
        return {slot.data, slot.size};
    }

    bool pop() {
        const uint64_t read = _consumer.position.load(std::memory_order_relaxed);
        if (read == _producer.position.load(std::memory_order_acquire)) return false;
        // The producer cannot overwrite this slot until this release is seen.
        _consumer.position.store(read + 1, std::memory_order_release);
        return true;
    }

    // This is a consumer-side snapshot, not proof that a live producer is done.
    bool empty() const { return front().data == nullptr; }

private:
    struct alignas(frame_ring_cache_line_bytes) Cursor final {
        std::atomic<uint64_t> position{0};
    };

    // Adjacent slots can belong to different threads at the same time.
    struct alignas(frame_ring_cache_line_bytes) Slot final {
        size_t size;
        // Unused tails are never published, read, or cleared.
        uint8_t data[frame_ring_max_bytes];
    };

    Cursor _producer;
    Cursor _consumer;
    Slot _slots[Capacity];
};

} // namespace cpu2tensor
