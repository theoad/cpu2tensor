// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <cstddef>
#include <cstdint>
#include <cpu2tensor/result.hpp>

namespace cpu2tensor {
inline constexpr uint32_t wire_magic = 0x31543243;
inline constexpr uint16_t legacy_wire_version = 2;
inline constexpr uint16_t wire_version = 3;
inline constexpr size_t header_bytes = 32;
inline constexpr uint32_t max_addresses = 256;
inline constexpr uint32_t max_sources = 256;
inline constexpr uint32_t max_registers = 512;
inline constexpr uint32_t max_register_bytes = 256;
inline constexpr size_t register_schema_bytes = 72;
inline constexpr size_t max_payload_bytes = 4064;
inline constexpr size_t max_frame_bytes = header_bytes + max_payload_bytes;
inline constexpr uint64_t feature_memory = 1 << 8;
inline constexpr uint64_t feature_registers = 1 << 9;
inline constexpr uint64_t feature_memory_values = 1 << 10;
inline constexpr uint64_t feature_stdio = 1 << 11;
inline constexpr uint64_t feature_system = 1 << 12;
inline constexpr uint64_t feature_kernel = 1 << 13;
inline constexpr uint64_t feature_window = 1 << 14;
inline constexpr uint64_t feature_system_memory = 1 << 15;
inline constexpr size_t context_bytes = 64;
inline constexpr uint64_t feature_address_context = 1 << 16;
inline constexpr uint64_t feature_executable_layout = 1 << 17;
inline constexpr uint64_t feature_mixed = 1 << 18;
inline constexpr uint64_t feature_stop = 1 << 19;
inline constexpr uint64_t feature_transition_windows = 1 << 20;
inline constexpr uint64_t feature_observation_reduction = 1 << 22;
inline constexpr uint8_t terminal_report_version = 1;
inline constexpr size_t layout_bytes = 24;
inline constexpr size_t transition_count_bytes = 24;
inline constexpr size_t transition_window_bytes = 40;
inline constexpr size_t reduced_context_bytes = 72;
inline constexpr size_t observation_summary_bytes = 72;
inline constexpr uint32_t max_action_bytes = 256;

enum class Kind : uint16_t {
    hello = 1, blocks = 2, source_end = 3, complete = 4, error = 5,
    register_schema = 6, registers = 7, memory = 8, input_request = 9,
    kernel_request = 10, guest_event = 11, address_context = 12, executable_layout = 13,
    mixed = 14, block_transitions = 15, transition_window = 16, terminal_report = 17,
    observation_summary = 19, reduced_context = 20
};
enum class Architecture : uint64_t { aarch64 = 1, x86_64 = 2 };
enum class Failure : uint64_t { capture = 1, target_killed = 2, unsupported_target = 3, transport = 4 };
enum class TerminalReason : uint8_t { max_run_deadline = 1 };
inline constexpr uint8_t terminal_hello = 1 << 0;
inline constexpr uint8_t terminal_data = 1 << 1;
inline constexpr uint8_t terminal_start_configured = 1 << 2;
inline constexpr uint8_t terminal_start_observed = 1 << 3;
inline constexpr uint8_t terminal_stop_configured = 1 << 4;
inline constexpr uint8_t terminal_stop_observed = 1 << 5;
inline constexpr uint8_t terminal_flags = terminal_hello | terminal_data |
    terminal_start_configured | terminal_start_observed |
    terminal_stop_configured | terminal_stop_observed;
inline constexpr uint64_t terminal_report_detail(TerminalReason reason, uint8_t flags) {
    return terminal_report_version | (static_cast<uint64_t>(reason) << 8) |
        (static_cast<uint64_t>(flags) << 16);
}
struct Header final {
    Kind kind{};
    uint32_t source = 0;
    uint32_t count = 0;
    uint64_t sequence = 0;
    uint64_t detail = 0;
    uint16_t version = wire_version;
};
void store_u16(uint8_t* bytes, uint16_t value);
void store_u32(uint8_t* bytes, uint32_t value);
void store_u64(uint8_t* bytes, uint64_t value);
uint16_t load_u16(const uint8_t* bytes);
uint32_t load_u32(const uint8_t* bytes);
uint64_t load_u64(const uint8_t* bytes);
void encode_header(uint8_t* bytes, const Header& header);
Result<Header> decode_header(const uint8_t* bytes, size_t size);
size_t payload_size(const Header& header);

class Stream final {
public:
    Stream() = default;
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&&) = delete;
    Stream& operator=(Stream&&) = delete;
    Result<Done> accept(const Header& header, const uint8_t* payload = nullptr);
    bool finished() const { return _finished; }
    bool memory_values() const { return (_features & feature_memory_values) != 0; }
    bool system_memory() const { return (_features & feature_system_memory) != 0; }
private:
    struct Registers final {
        uint16_t widths[max_registers]{};
        bool sampled[max_registers]{};
        char names[max_registers][64]{};
    };
    struct TransitionKey final {
        uint64_t window = 0;
        uint64_t from_address = 0;
        uint64_t destination = 0;
        uint32_t source = 0;
    };
    Result<Done> schema(const Header& header, const uint8_t* payload);
    Result<Done> registers(const Header& header, const uint8_t* payload);
    Result<Done> memory(const Header& header, const uint8_t* payload);
    Result<Done> remember_transition(uint32_t source, uint64_t from_address,
                                     uint64_t destination);
    Result<Done> grow_transition_keys(uint32_t capacity);
    Registers* _registers[max_sources]{};
    uint64_t _next[max_sources]{};
    bool _seen[max_sources]{};
    bool _data[max_sources]{};
    bool _baseline_complete[max_sources]{};
    bool _ended[max_sources]{};
    bool _has_context[max_sources]{};
    uint64_t _context_sequence[max_sources]{};
    uint64_t _features = 0;
    uint16_t _version = 0;
    uint64_t _next_window = 1;
    uint64_t _window_distinct = 0;
    uint64_t _window_observed = 0;
    TransitionKey* _window_keys = nullptr;
    uint32_t _window_key_capacity = 0;
    uint32_t _window_rows_per_source[max_sources]{};
    uint32_t _window_sources = 0;
    uint32_t _window_capacity_per_source = 0;
    uint32_t _window_max_source_plus_one = 0;
    bool _window_expected = false;
    bool _window_rows = false;
    uint64_t _observation_transition_counts[max_sources]{};
    uint64_t _observation_context_rows[max_sources]{};
    uint64_t _observation_last_block[max_sources]{};
    uint32_t _observation_sources = 0;
    uint32_t _observation_transition_capacity = 0;
    uint32_t _observation_context_capacity = 0;
    bool _observation_has_context[max_sources]{};
    bool _observation_summary[max_sources]{};
    bool _started = false;
    bool _finished = false;
    bool _layout_seen = false;
    bool _source_data_seen = false;
};
} // namespace cpu2tensor
