// SPDX-License-Identifier: AGPL-3.0-only
#include <cassert>
#include <cstdint>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <cpu2tensor/frame_ring.hpp>

using namespace cpu2tensor;

static void test_slot_ownership() {
    FrameRing<2> ring;
    uint8_t first[frame_ring_max_bytes];
    std::memset(first, 0xa7, sizeof(first));
    const uint8_t second[] = {1, 2, 3};
    assert(ring.empty());
    assert(ring.front().data == nullptr);
    assert(!ring.pop());
    assert(!ring.try_push(nullptr, 1).ok());
    assert(!ring.try_push(first, 0).ok());
    assert(!ring.try_push(first, sizeof(first) + 1).ok());
    assert(ring.empty());

    assert(ring.try_push(first, sizeof(first)).value() == FramePush::stored);
    FrameView held = ring.front();
    assert(held.size == sizeof(first));
    assert(ring.try_push(second, sizeof(second)).value() == FramePush::stored);
    assert(ring.try_push(second, sizeof(second)).value() == FramePush::full);
    assert(held.data == ring.front().data);
    assert(std::memcmp(held.data, first, sizeof(first)) == 0);
    assert(ring.pop());
    // Reuse the long slot with a short frame. No old tail becomes visible.
    assert(ring.try_push(second, 1).value() == FramePush::stored);
    assert(ring.front().size == sizeof(second));
    assert(std::memcmp(ring.front().data, second, sizeof(second)) == 0);
    assert(ring.pop());
    assert(ring.front().size == 1 && ring.front().data[0] == 1);
    assert(ring.pop());
    assert(ring.empty());
    assert(!ring.pop());
}

static constexpr uint64_t stress_frames = 1000000;

static size_t frame_size(uint64_t sequence) {
    return sizeof(sequence) + sequence % (frame_ring_max_bytes - sizeof(sequence) + 1);
}

static uint8_t frame_byte(uint64_t sequence, size_t offset) {
    return static_cast<uint8_t>((sequence * 13) ^ (offset * 17));
}

struct Stress final {
    FrameRing<2> ring;
    std::atomic<bool> producer_saw_full{false};
    std::atomic<bool> producer_finished{false};
};

static void* produce(void* opaque) {
    auto& state = *static_cast<Stress*>(opaque);
    uint8_t bytes[frame_ring_max_bytes];
    for (uint64_t sequence = 0; sequence < stress_frames; ++sequence) {
        const size_t size = frame_size(sequence);
        std::memcpy(bytes, &sequence, sizeof(sequence));
        for (size_t offset = sizeof(sequence); offset < size; ++offset) {
            bytes[offset] = frame_byte(sequence, offset);
        }
        for (;;) {
            const auto pushed = state.ring.try_push(bytes, size);
            assert(pushed.ok());
            if (pushed.value() == FramePush::stored) break;
            state.producer_saw_full.store(true, std::memory_order_release);
            sched_yield();
        }
    }
    state.producer_finished.store(true, std::memory_order_release);
    return nullptr;
}

static void test_concurrent_wraparound() {
    Stress state;
    pthread_t producer;
    assert(pthread_create(&producer, nullptr, produce, &state) == 0);
    // Hold the consumer until the producer has encountered actual backpressure.
    while (!state.producer_saw_full.load(std::memory_order_acquire)) sched_yield();
    for (uint64_t expected = 0; expected < stress_frames; ++expected) {
        FrameView frame;
        do {
            frame = state.ring.front();
            if (frame.data == nullptr) sched_yield();
        } while (frame.data == nullptr);
        uint64_t actual;
        std::memcpy(&actual, frame.data, sizeof(actual));
        assert(actual == expected);
        assert(frame.size == frame_size(expected));
        for (size_t offset = sizeof(expected); offset < frame.size; ++offset) {
            assert(frame.data[offset] == frame_byte(expected, offset));
        }
        assert(state.ring.pop());
    }
    assert(pthread_join(producer, nullptr) == 0);
    assert(state.producer_finished.load(std::memory_order_acquire));
    assert(state.ring.empty());
    assert(!state.ring.pop());
}

int main() {
    test_slot_ownership();
    test_concurrent_wraparound();
}
