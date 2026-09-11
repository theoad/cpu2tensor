// SPDX-License-Identifier: AGPL-3.0-only
#include <cpu2tensor/trace.hpp>
#include <cpu2tensor/transition_window.hpp>
#include <cstring>
#include <new>

namespace cpu2tensor {
namespace {
void store(uint8_t* bytes, uint64_t value, size_t size) {
    for (size_t i = 0; i < size; ++i) bytes[i] = static_cast<uint8_t>(value >> (8 * i));
}
uint64_t load(const uint8_t* bytes, size_t size) {
    uint64_t value = 0;
    for (size_t i = 0; i < size; ++i) value |= uint64_t{bytes[i]} << (8 * i);
    return value;
}
}
void store_u16(uint8_t* bytes, uint16_t value) { store(bytes, value, 2); }
void store_u32(uint8_t* bytes, uint32_t value) { store(bytes, value, 4); }
void store_u64(uint8_t* bytes, uint64_t value) { store(bytes, value, 8); }
uint16_t load_u16(const uint8_t* bytes) { return static_cast<uint16_t>(load(bytes, 2)); }
uint32_t load_u32(const uint8_t* bytes) { return static_cast<uint32_t>(load(bytes, 4)); }
uint64_t load_u64(const uint8_t* bytes) { return load(bytes, 8); }
void encode_header(uint8_t* bytes, const Header& header) {
    store_u32(bytes, wire_magic);
    store_u16(bytes + 4, header.version);
    store_u16(bytes + 6, static_cast<uint16_t>(header.kind));
    store_u32(bytes + 8, header.source);
    store_u32(bytes + 12, header.count);
    store_u64(bytes + 16, header.sequence);
    store_u64(bytes + 24, header.detail);
}
size_t payload_size(const Header& header) {
    if (header.kind == Kind::guest_event) return static_cast<size_t>(header.detail);
    if (header.kind == Kind::blocks) return header.count * sizeof(uint64_t);
    if (header.kind == Kind::register_schema || header.kind == Kind::registers || header.kind == Kind::memory ||
        header.kind == Kind::address_context || header.kind == Kind::executable_layout || header.kind == Kind::mixed ||
        header.kind == Kind::block_transitions || header.kind == Kind::transition_window ||
        header.kind == Kind::context_filter || header.kind == Kind::observation_summary ||
        header.kind == Kind::reduced_context)
        return static_cast<size_t>(header.detail);
    return 0;
}
Result<Header> decode_header(const uint8_t* bytes, size_t size) {
    if (size != header_bytes) return Result<Header>::failure("Trace header must contain 32 bytes");
    if (load_u32(bytes) != wire_magic) return Result<Header>::failure("Not a cpu2tensor trace");
    const auto version = load_u16(bytes + 4);
    if (version != legacy_wire_version && version != wire_version)
        return Result<Header>::failure("Unsupported trace version");
    Header h{static_cast<Kind>(load_u16(bytes + 6)), load_u32(bytes + 8), load_u32(bytes + 12),
             load_u64(bytes + 16), load_u64(bytes + 24), version};
    if (h.source >= max_sources || h.count > max_addresses)
        return Result<Header>::failure("Trace exceeds the supported source or batch limit");
    switch (h.kind) {
    case Kind::mixed:
        if (h.count == 0 || h.detail < 16 || h.detail > max_payload_bytes)
            return Result<Header>::failure("Invalid mixed batch size");
        break;
    case Kind::executable_layout:
        if (h.source != 0 || h.sequence != 0 || h.count != 1 || h.detail != layout_bytes)
            return Result<Header>::failure("Invalid executable layout fields");
        break;
    case Kind::block_transitions:
        if (h.count == 0 || h.sequence == 0 || h.detail > max_payload_bytes ||
            h.detail != h.count * transition_count_bytes)
            return Result<Header>::failure("Invalid block transition batch fields");
        break;
    case Kind::transition_window:
        if (h.source != 0 || h.count != 1 || h.sequence == 0 || h.detail != transition_window_bytes)
            return Result<Header>::failure("Invalid transition window fields");
        break;
    case Kind::context_filter:
        if (h.version != wire_version || h.source != 0 || h.count != 1 ||
            h.sequence != 0 || h.detail != context_filter_bytes)
            return Result<Header>::failure("Invalid context filter summary fields");
        break;
    case Kind::observation_summary:
        if (h.count != 1 || h.sequence != 0 || h.detail != observation_summary_bytes)
            return Result<Header>::failure("Invalid reduced observation summary fields");
        break;
    case Kind::reduced_context:
        if (h.count == 0 || h.detail != h.count * reduced_context_bytes)
            return Result<Header>::failure("Invalid reduced context batch fields");
        break;
    case Kind::kernel_request:
    case Kind::guest_event:
        if (h.source != 0 || h.count != 0 || h.sequence != 0 || h.detail == 0 ||
            h.detail > (h.kind == Kind::kernel_request ? 127u : 1024u))
            return Result<Header>::failure("Invalid kernel adapter frame fields");
        break;
    case Kind::blocks:
        if (h.count == 0 || h.detail != 0) return Result<Header>::failure("Invalid block batch fields");
        break;
    case Kind::input_request:
        if (h.source != 0 || h.count != 0 || h.detail == 0 || h.detail > max_action_bytes)
            return Result<Header>::failure("Invalid stdin request fields");
        break;
    case Kind::source_end:
        if (h.count != 0 || h.detail != 0) return Result<Header>::failure("Invalid source end fields");
        break;
    case Kind::register_schema:
    case Kind::registers:
    case Kind::memory:
    case Kind::address_context:
        if (h.count == 0 || h.detail > max_payload_bytes || h.detail == 0)
            return Result<Header>::failure("Invalid signal payload size");
        if (h.kind == Kind::register_schema && (h.sequence != 0 || h.detail != h.count * register_schema_bytes))
            return Result<Header>::failure("Invalid register schema size or sequence");
        if (h.kind == Kind::registers && h.detail < h.count * 17)
            return Result<Header>::failure("Register payload is too short");
        if (h.kind == Kind::address_context && (h.count != 1 || h.detail != context_bytes))
            return Result<Header>::failure("Invalid address context size");
        if (h.kind == Kind::memory && h.detail != h.count * 24 && h.detail != h.count * 40 &&
            h.detail != h.count * 48 && h.detail != h.count * 64)
            return Result<Header>::failure("Invalid memory payload size");
        break;
    case Kind::hello:
    case Kind::complete:
    case Kind::error:
        if (h.source != 0 || h.count != 0 || h.sequence != 0)
            return Result<Header>::failure("Invalid control frame fields");
        if (h.kind == Kind::hello) {
            const auto architecture = h.detail & 255;
            constexpr auto allowed = uint64_t{255} | feature_memory | feature_registers | feature_memory_values | feature_stdio |
                feature_system | feature_kernel | feature_window | feature_system_memory | feature_address_context |
                feature_executable_layout | feature_mixed | feature_stop | feature_transition_windows |
                feature_context_filter | feature_observation_reduction;
            if ((architecture != 1 && architecture != 2) || (h.detail & ~allowed) != 0)
                return Result<Header>::failure("Unsupported target architecture or features");
            if ((h.detail & feature_memory_values) && !(h.detail & feature_memory))
                return Result<Header>::failure("Memory values require memory observations");
            if ((h.detail & feature_executable_layout) && (h.detail & feature_system))
                return Result<Header>::failure("Executable layout requires user-mode capture");
            if ((h.detail & feature_system_memory) &&
                (!(h.detail & feature_address_context) || !(h.detail & feature_memory)))
                return Result<Header>::failure("System memory context needs x86 system memory capture");
            if ((h.detail & feature_address_context) && (!(h.detail & feature_system) || architecture != 2))
                return Result<Header>::failure("Address context needs x86 system capture");
            if (architecture == 2 && (h.detail & feature_system) && (h.detail & feature_registers) &&
                !(h.detail & feature_address_context))
                return Result<Header>::failure("System x86 register capture requires address context; update the worker");
            if (((h.detail & feature_kernel) && !(h.detail & feature_system)) ||
                ((h.detail & feature_stdio) && (h.detail & feature_system)))
                return Result<Header>::failure("Incompatible system and action features");
            if ((h.detail & feature_transition_windows) && !(h.detail & feature_kernel))
                return Result<Header>::failure("Transition windows need an interactive system worker");
            if (h.detail & feature_observation_reduction) {
                if (h.version != wire_version || !(h.detail & feature_system) ||
                    !(h.detail & feature_address_context) ||
                    (h.detail & (feature_kernel | feature_memory | feature_registers |
                                 feature_transition_windows)))
                    return Result<Header>::failure("Reduced observations need v3 context-only system capture");
            }
            if ((h.detail & feature_context_filter) &&
                (h.version != wire_version || !(h.detail & feature_address_context) ||
                 !(h.detail & feature_system) || architecture != 2 ||
                 !(h.detail & (feature_memory | feature_registers))))
                return Result<Header>::failure("Context filtering needs rich x86 system capture");
        }
        if (h.kind == Kind::complete && h.detail > 255) return Result<Header>::failure("Invalid target exit code");
        if (h.kind == Kind::error && (h.detail < 1 || h.detail > 4)) return Result<Header>::failure("Unknown trace failure");
        break;
    case Kind::terminal_report: {
        if (h.version != wire_version || h.source != 0 || h.count != 0 || h.sequence != 0 ||
            (h.detail & UINT64_C(0xffffffffff000000)) != 0)
            return Result<Header>::failure("Invalid terminal report fields");
        const auto version = static_cast<uint8_t>(h.detail);
        const auto reason = static_cast<uint8_t>(h.detail >> 8);
        const auto flags = static_cast<uint8_t>(h.detail >> 16);
        if (version != terminal_report_version ||
            reason != static_cast<uint8_t>(TerminalReason::max_run_deadline) ||
            (flags & ~terminal_flags) != 0 ||
            ((flags & terminal_data) && !(flags & terminal_hello)) ||
            ((flags & terminal_start_observed) && !(flags & terminal_start_configured)) ||
            ((flags & terminal_stop_observed) && !(flags & terminal_stop_configured)))
            return Result<Header>::failure("Unknown or inconsistent terminal report");
        break;
    }
    default: return Result<Header>::failure("Unknown trace frame kind");
    }
    return Result<Header>::success(h);
}
Stream::~Stream() {
    for (auto* registers : _registers) delete registers;
    delete[] _window_keys;
}

namespace {
uint64_t transition_hash(uint32_t source, uint64_t from_address, uint64_t destination) {
    uint64_t value = from_address ^ (destination + UINT64_C(0x9e3779b97f4a7c15)) ^ source;
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}
}

Result<Done> Stream::grow_transition_keys(uint32_t capacity) {
    auto* keys = new (std::nothrow) TransitionKey[capacity];
    if (keys == nullptr) return Result<Done>::failure("Cannot allocate transition validation table");
    const uint32_t mask = capacity - 1;
    for (uint32_t index = 0; index < _window_key_capacity; ++index) {
        const auto& old = _window_keys[index];
        if (old.window != _next_window) continue;
        uint32_t destination = static_cast<uint32_t>(
            transition_hash(old.source, old.from_address, old.destination)) & mask;
        while (keys[destination].window == _next_window)
            destination = (destination + 1) & mask;
        keys[destination] = old;
    }
    delete[] _window_keys;
    _window_keys = keys;
    _window_key_capacity = capacity;
    return Result<Done>::success({});
}

Result<Done> Stream::remember_transition(uint32_t source, uint64_t from_address,
                                         uint64_t destination) {
    if (_window_distinct >= max_transition_slots)
        return Result<Done>::failure("Transition rows exceed the fixed total capacity");
    if (_window_key_capacity == 0 ||
        (_window_distinct + 1) * 2 > _window_key_capacity) {
        const uint32_t capacity = _window_key_capacity == 0 ? 512 : _window_key_capacity * 2;
        const auto grown = grow_transition_keys(capacity);
        if (!grown.ok()) return grown;
    }
    const uint32_t mask = _window_key_capacity - 1;
    uint32_t index = static_cast<uint32_t>(
        transition_hash(source, from_address, destination)) & mask;
    for (;;) {
        auto& key = _window_keys[index];
        if (key.window != _next_window) {
            key = {_next_window, from_address, destination, source};
            return Result<Done>::success({});
        }
        if (key.source == source && key.from_address == from_address &&
            key.destination == destination)
            return Result<Done>::failure("Duplicate block transition row");
        index = (index + 1) & mask;
    }
}
Result<Done> Stream::schema(const Header& h, const uint8_t* bytes) {
    if (!(_features & feature_registers) || _data[h.source])
        return Result<Done>::failure("Register schema is disabled or arrived after source data");
    auto*& state = _registers[h.source];
    if (state == nullptr) state = new (std::nothrow) Registers;
    if (state == nullptr) return Result<Done>::failure("Cannot allocate register schema");
    for (uint32_t row = 0; row < h.count; ++row) {
        const auto* item = bytes + row * register_schema_bytes;
        const auto id = load_u32(item);
        const auto width = load_u32(item + 4);
        if (id >= max_registers || width == 0 || width > max_register_bytes || state->widths[id] != 0)
            return Result<Done>::failure("Invalid or duplicate register ID or width");
        const char* name = reinterpret_cast<const char*>(item + 8);
        size_t length = 0;
        while (length < 64 && name[length] != 0) {
            if (name[length] < 33 || name[length] > 126) return Result<Done>::failure("Invalid register name");
            ++length;
        }
        if (length == 0 || length == 64) return Result<Done>::failure("Register name needs a terminator");
        for (size_t i = length; i < 64; ++i)
            if (name[i] != 0) return Result<Done>::failure("Register name padding must be zero");
        for (uint32_t other = 0; other < max_registers; ++other)
            if (state->widths[other] != 0 && std::strcmp(state->names[other], name) == 0)
                return Result<Done>::failure("Duplicate register name");
        state->widths[id] = static_cast<uint16_t>(width);
        std::memcpy(state->names[id], name, 64);
    }
    return Result<Done>::success({});
}
Result<Done> Stream::registers(const Header& h, const uint8_t* bytes) {
    auto* state = _registers[h.source];
    if (!(_features & feature_registers) || state == nullptr)
        return Result<Done>::failure("Register data needs an enabled schema");
    size_t offset = 0;
    for (uint32_t row = 0; row < h.count; ++row) {
        if (h.detail - offset < 16) return Result<Done>::failure("Truncated register record");
        const auto id = load_u32(bytes + offset + 8);
        const auto width = load_u16(bytes + offset + 12);
        const auto flags = load_u16(bytes + offset + 14);
        const auto phase = flags & 255;
        if (id >= max_registers || width == 0 || width != state->widths[id] || width > h.detail - offset - 16)
            return Result<Done>::failure("Unknown register or incorrect register width");
        if ((flags & ~0x103) != 0 || phase < 1 || phase > 3)
            return Result<Done>::failure("Invalid register checkpoint flags");
        const bool baseline = (flags & 256) != 0;
        if (baseline == state->sampled[id]) return Result<Done>::failure("Missing or repeated register baseline");
        state->sampled[id] = true;
        offset += 16 + width;
    }
    if (offset != h.detail) return Result<Done>::failure("Trailing bytes in register payload");
    return Result<Done>::success({});
}
Result<Done> Stream::memory(const Header& h, const uint8_t* bytes) {
    if (!(_features & feature_memory)) return Result<Done>::failure("Memory observations are disabled");
    const size_t prefix = system_memory() ? 48 : 24;
    const size_t stride = prefix + (memory_values() ? 16 : 0);
    if (h.detail != h.count * stride) return Result<Done>::failure("Memory value setting does not match payload");
    for (uint32_t row = 0; row < h.count; ++row) {
        const auto* item = bytes + row * stride;
        const auto width = load_u32(item + 16);
        const auto flags = load_u32(item + 20);
        if (width == 0 || (width & (width - 1)) != 0 || flags > 3)
            return Result<Done>::failure("Invalid memory access size or flags");
        if (system_memory()) {
            const auto physical = load_u64(item + 24);
            const auto mapped = load_u32(item + 32);
            const auto mapping = load_u32(item + 36);
            if (!_has_context[h.source] || load_u64(item + 40) != _context_sequence[h.source])
                return Result<Done>::failure("Memory access needs its current address context");
            const auto virtual_address = load_u64(item + 8);
            if ((mapping != 0 && mapping != 1 && mapping != 3 && mapping != 7) ||
                (mapping == 0 && (mapped != 0 || physical != 0)) ||
                ((mapping & 1) && (mapped == 0 || mapped > width ||
                mapped > 4096 - (virtual_address & 4095) || physical > UINT64_MAX - (mapped - 1))))
                return Result<Done>::failure("Invalid physical mapping coverage");
        }
        if (memory_values()) {
            if (width > 16) return Result<Done>::failure("Unsupported memory value width");
            for (uint32_t i = width; i < 16; ++i)
                if (item[prefix + i] != 0) return Result<Done>::failure("Memory value padding must be zero");
        }
    }
    return Result<Done>::success({});
}
Result<Done> Stream::accept(const Header& h, const uint8_t* bytes) {
    uint8_t encoded[header_bytes];
    encode_header(encoded, h);
    const auto checked = decode_header(encoded, sizeof(encoded));
    if (!checked.ok()) return Result<Done>::failure(checked.error());
    if (_finished) return Result<Done>::failure("Frame received after trace end");
    if (_started && h.version != _version)
        return Result<Done>::failure("Trace frame version changed after Hello");
    if (!_started) {
        if (h.kind == Kind::terminal_report) {
            _version = h.version;
            _finished = true;
            return Result<Done>::success({});
        }
        if (h.kind != Kind::hello) return Result<Done>::failure("Trace must start with Hello");
        _started = true;
        _version = h.version;
        _features = h.detail;
        return Result<Done>::success({});
    }
    if (h.kind == Kind::hello) return Result<Done>::failure("Duplicate trace Hello");
    if (h.kind == Kind::mixed) {
        if (!(_features & feature_mixed) || bytes == nullptr)
            return Result<Done>::failure("Mixed capture needs its feature and payload");
        size_t offset = 0;
        uint32_t count = 0;
        while (offset < h.detail) {
            if (h.detail - offset < 8) return Result<Done>::failure("Truncated mixed run header");
            const auto kind = static_cast<Kind>(load_u16(bytes + offset));
            const auto rows = load_u16(bytes + offset + 2);
            const auto size = load_u32(bytes + offset + 4);
            offset += 8;
            if ((kind != Kind::blocks && kind != Kind::registers && kind != Kind::memory &&
                 kind != Kind::address_context) || rows == 0 || rows > h.count - count ||
                size > h.detail - offset || (kind == Kind::blocks && size != rows * 8u))
                return Result<Done>::failure("Invalid mixed run kind, count or length");
            if (h.sequence > UINT64_MAX - count)
                return Result<Done>::failure("Mixed sequence overflow");
            const auto result = accept({kind, h.source, rows, h.sequence + count,
                                        kind == Kind::blocks ? 0 : size, h.version}, bytes + offset);
            if (!result.ok()) return result;
            count += rows;
            offset += size;
        }
        if (count != h.count) return Result<Done>::failure("Mixed event count does not match its runs");
        return Result<Done>::success({});
    }
    if (h.kind == Kind::executable_layout) {
        if (!(_features & feature_executable_layout) || _layout_seen || _source_data_seen)
            return Result<Done>::failure("Executable layout must appear once before source data");
        if (bytes == nullptr || load_u64(bytes) >= load_u64(bytes + 8))
            return Result<Done>::failure("Executable layout needs a nonempty code span");
        _layout_seen = true;
        // This describes the worker's initial process, not CPU zero. It consumes
        // no source sequence and can follow schemas emitted during vCPU init.
        return Result<Done>::success({});
    }
    if ((_features & feature_executable_layout) && !_layout_seen &&
        h.kind != Kind::register_schema && h.kind != Kind::error)
        return Result<Done>::failure("Trace data needs its initial executable layout");
    if (h.kind == Kind::kernel_request || h.kind == Kind::guest_event) {
        if (!(_features & feature_kernel))
            return Result<Done>::failure("Kernel adapter frame needs an interactive system worker");
        if (h.kind == Kind::guest_event && bytes == nullptr)
            return Result<Done>::failure("Guest event payload is missing");
        if (h.kind == Kind::kernel_request && (_features & feature_registers)) {
            for (uint32_t source = 0; source < max_sources; ++source) {
                if (!_data[source]) continue;
                const auto* registers = _registers[source];
                if (registers == nullptr) return Result<Done>::failure("Kernel boundary needs a register schema");
                bool any_sampled = false;
                for (uint32_t id = 0; id < max_registers; ++id)
                    any_sampled |= registers->sampled[id];
                if ((_features & feature_context_filter) && !any_sampled) continue;
                for (uint32_t id = 0; id < max_registers; ++id)
                    if (registers->widths[id] != 0 && !registers->sampled[id])
                        return Result<Done>::failure("Kernel boundary needs complete source register baselines");
            }
        }
        if (h.kind == Kind::kernel_request && (_features & feature_transition_windows)) {
            if (_window_expected)
                return Result<Done>::failure("Action boundary arrived without exactly one transition window");
            _window_expected = true;
        }
        // Adapter events have no vCPU attribution and consume no source sequence.
        return Result<Done>::success({});
    }
    if (h.kind == Kind::block_transitions) {
        const bool action = (_features & feature_transition_windows) != 0;
        const bool observation = (_features & feature_observation_reduction) != 0;
        if ((!action && !observation) || bytes == nullptr ||
            h.sequence != _next_window || h.source >= max_sources || _ended[h.source] ||
            (action && !_window_expected) ||
            (observation && _observation_summary[h.source]))
            return Result<Done>::failure("Block transitions need their current reduction");
        for (uint32_t row = 0; row < h.count; ++row) {
            const uint8_t* item = bytes + row * transition_count_bytes;
            const auto remembered = remember_transition(h.source, load_u64(item), load_u64(item + 8));
            if (!remembered.ok()) return remembered;
            const uint64_t count = load_u64(bytes + row * transition_count_bytes + 16);
            if (count == 0 || _window_observed > UINT64_MAX - count)
                return Result<Done>::failure("Invalid or overflowing block transition count");
            _window_observed += count;
            if (observation) {
                if (_observation_transition_counts[h.source] > UINT64_MAX - count)
                    return Result<Done>::failure("Reduced transition count overflow");
                _observation_transition_counts[h.source] += count;
            }
            ++_window_distinct;
        }
        if (_window_rows_per_source[h.source] > max_transition_slots - h.count)
            return Result<Done>::failure("Per-source transition row count overflow");
        _window_rows_per_source[h.source] += h.count;
        if (_window_max_source_plus_one <= h.source)
            _window_max_source_plus_one = h.source + 1;
        _window_rows = true;
        return Result<Done>::success({});
    }
    if (h.kind == Kind::reduced_context) {
        if (!(_features & feature_observation_reduction) || bytes == nullptr ||
            _ended[h.source] || _observation_summary[h.source] ||
            h.sequence != _next[h.source] || h.count > UINT64_MAX - h.sequence)
            return Result<Done>::failure("Reduced contexts need their current source projection");
        for (uint32_t row = 0; row < h.count; ++row) {
            const auto* item = bytes + row * reduced_context_bytes;
            const uint64_t block = load_u64(item);
            const uint64_t mode = load_u64(item + 56);
            const uint64_t known = load_u64(item + 64);
            if ((_observation_has_context[h.source] &&
                 block <= _observation_last_block[h.source]) ||
                (known != 15 && known != 63) ||
                (known == 15 && (mode != 0 || load_u64(item + 48) != 0)) ||
                (known == 63 && mode != 16 && mode != 32 && mode != 64))
                return Result<Done>::failure("Invalid reduced context position or availability");
            _observation_last_block[h.source] = block;
            _observation_has_context[h.source] = true;
        }
        if (_observation_context_rows[h.source] > max_transition_slots - h.count)
            return Result<Done>::failure("Reduced context rows exceed the fixed bound");
        _observation_context_rows[h.source] += h.count;
        _next[h.source] += h.count;
        _seen[h.source] = true;
        _source_data_seen = true;
        _data[h.source] = true;
        return Result<Done>::success({});
    }
    if (h.kind == Kind::observation_summary) {
        if (!(_features & feature_observation_reduction) || bytes == nullptr ||
            _ended[h.source] || _observation_summary[h.source])
            return Result<Done>::failure("Reduced observation summary is disabled or repeated");
        const uint32_t sources = load_u32(bytes);
        const uint32_t transition_capacity = load_u32(bytes + 4);
        const uint32_t context_capacity = load_u32(bytes + 8);
        const uint32_t reserved = load_u32(bytes + 12);
        const uint64_t blocks = load_u64(bytes + 16);
        const uint64_t transitions = load_u64(bytes + 24);
        const uint64_t distinct = load_u64(bytes + 32);
        const uint64_t transition_overflow = load_u64(bytes + 40);
        const uint64_t context_changes = load_u64(bytes + 48);
        const uint64_t retained_contexts = load_u64(bytes + 56);
        const uint64_t context_overflow = load_u64(bytes + 64);
        const uint64_t expected_transitions = blocks == 0 ? 0 : blocks - 1;
        if (sources == 0 || sources > max_sources || h.source >= sources ||
            transition_capacity < 2 ||
            (transition_capacity & (transition_capacity - 1)) != 0 ||
            context_capacity != transition_capacity || reserved != 0 ||
            uint64_t{sources} * transition_capacity > max_transition_slots ||
            transitions != expected_transitions ||
            distinct != _window_rows_per_source[h.source] ||
            distinct > transition_capacity || transition_overflow > transitions ||
            _observation_transition_counts[h.source] > transitions - transition_overflow ||
            _observation_transition_counts[h.source] + transition_overflow != transitions ||
            retained_contexts != _observation_context_rows[h.source] ||
            retained_contexts > context_capacity || context_overflow > context_changes ||
            retained_contexts + context_overflow != context_changes ||
            retained_contexts != (context_changes < context_capacity ?
                                  context_changes : context_capacity) ||
            context_changes > blocks || (blocks != 0 && context_changes == 0) ||
            (_observation_has_context[h.source] &&
             _observation_last_block[h.source] >= blocks) ||
            (_observation_sources != 0 &&
             (_observation_sources != sources ||
              _observation_transition_capacity != transition_capacity ||
              _observation_context_capacity != context_capacity)))
            return Result<Done>::failure("Reduced observation summary does not close its source");
        if (_observation_sources == 0) {
            _observation_sources = sources;
            _observation_transition_capacity = transition_capacity;
            _observation_context_capacity = context_capacity;
        }
        _observation_summary[h.source] = true;
        _seen[h.source] = true;
        _source_data_seen = true;
        return Result<Done>::success({});
    }
    if (h.kind == Kind::transition_window) {
        if (!(_features & feature_transition_windows) || bytes == nullptr ||
            h.sequence != _next_window || !_window_expected)
            return Result<Done>::failure("Transition metadata needs its current window");
        const auto status = static_cast<WindowStatus>(load_u32(bytes));
        const uint32_t sources = load_u32(bytes + 4);
        const uint32_t capacity = load_u32(bytes + 8);
        const uint32_t reserved = load_u32(bytes + 12);
        const uint64_t distinct = load_u64(bytes + 16);
        const uint64_t observed = load_u64(bytes + 24);
        const uint64_t overflow = load_u64(bytes + 32);
        const bool valid_status = status == WindowStatus::ended || status == WindowStatus::aborted ||
                                  status == WindowStatus::incomplete;
        bool rows_fit = true;
        for (uint32_t source = 0; source < max_sources; ++source)
            if (_window_rows_per_source[source] > capacity) rows_fit = false;
        if (!valid_status || sources == 0 || sources > max_sources || capacity < 2 ||
            (capacity & (capacity - 1)) != 0 || reserved != 0 ||
            uint64_t{sources} * capacity > max_transition_slots ||
            distinct > uint64_t{sources} * capacity || !rows_fit ||
            _window_max_source_plus_one > sources ||
            (_window_sources != 0 && (_window_sources != sources ||
                                      _window_capacity_per_source != capacity)) ||
            distinct != _window_distinct || overflow > observed ||
            _window_observed > observed - overflow || _window_observed + overflow != observed)
            return Result<Done>::failure("Transition window summary does not close its rows");
        if (_window_sources == 0) {
            _window_sources = sources;
            _window_capacity_per_source = capacity;
        }
        _window_expected = false;
        ++_next_window;
        if (_next_window == 0) return Result<Done>::failure("Transition window sequence overflow");
        _window_distinct = 0;
        _window_observed = 0;
        std::memset(_window_rows_per_source, 0, sizeof(_window_rows_per_source));
        _window_max_source_plus_one = 0;
        _window_rows = false;
        return Result<Done>::success({});
    }
    if (h.kind == Kind::context_filter) {
        if (!(_features & feature_context_filter) || bytes == nullptr || _context_filter_seen)
            return Result<Done>::failure("Context filter summary needs its feature and appears once");
        for (uint32_t source = 0; source < max_sources; ++source)
            if (_seen[source] && !_ended[source])
                return Result<Done>::failure("Context filter summary needs every observed source end");
        const uint32_t policy = load_u32(bytes);
        const uint32_t latch = load_u32(bytes + 4);
        const uint32_t source = load_u32(bytes + 8);
        const uint32_t reserved = load_u32(bytes + 12);
        const uint64_t cr3 = load_u64(bytes + 24);
        const uint64_t root = load_u64(bytes + 32);
        const uint64_t kept = load_u64(bytes + 40);
        const uint64_t dropped = load_u64(bytes + 48);
        const uint64_t matching = load_u64(bytes + 56);
        const uint64_t foreign = load_u64(bytes + 64);
        const uint64_t unknown = load_u64(bytes + 72);
        if ((policy != 1 && policy != 2) || (latch != 2 && latch != 3) ||
            source >= max_sources || !_seen[source] || !_ended[source] || reserved != 0 ||
            root != (cr3 & ~UINT64_C(0xfff)) || kept != _context_filter_memory_rows ||
            kept > UINT64_MAX - dropped ||
            matching > UINT64_MAX - foreign || matching + foreign > UINT64_MAX - unknown ||
            kept + dropped != matching + foreign + unknown ||
            (policy == 1 && (kept != matching || dropped != foreign + unknown)) ||
            (policy == 2 && dropped != 0))
            return Result<Done>::failure("Invalid context filter summary values");
        _context_filter_seen = true;
        return Result<Done>::success({});
    }
    if (h.kind == Kind::error || h.kind == Kind::terminal_report) {
        _finished = true;
        return Result<Done>::success({});
    }
    if (h.kind == Kind::complete) {
        if ((_features & feature_transition_windows) && (_window_rows || _window_expected))
            return Result<Done>::failure("Trace ended before its required transition window");
        for (uint32_t source = 0; source < _window_sources; ++source)
            if (!_ended[source])
                return Result<Done>::failure("Trace ended before a declared window source ended");
        for (uint32_t source = 0; source < max_sources; ++source)
            if (_seen[source] && !_ended[source]) return Result<Done>::failure("Trace ended before all sources ended");
        if (_features & feature_observation_reduction) {
            if (_observation_sources == 0)
                return Result<Done>::failure("Reduced trace ended without source summaries");
            for (uint32_t source = 0; source < _observation_sources; ++source)
                if (!_observation_summary[source] || !_ended[source])
                    return Result<Done>::failure("Reduced trace ended before every source summary");
        }
        if ((_features & feature_context_filter) && !_context_filter_seen)
            return Result<Done>::failure("Trace ended before its context filter summary");
        _finished = true;
        return Result<Done>::success({});
    }
    const auto source = h.source;
    if ((_features & feature_stdio) && source != 0)
        return Result<Done>::failure("Stdin interaction supports one vCPU only");
    if (h.kind == Kind::input_request && !(_features & feature_stdio))
        return Result<Done>::failure("Stdin request needs interactive capture");
    if ((_features & feature_observation_reduction) &&
        (h.kind == Kind::blocks || h.kind == Kind::address_context))
        return Result<Done>::failure("Reduced observations cannot contain raw block or context rows");
    if (_ended[source]) return Result<Done>::failure("Frame received after source end");
    if ((_features & feature_observation_reduction) && h.kind == Kind::source_end &&
        !_observation_summary[source])
        return Result<Done>::failure("Reduced source ended before its summary");
    if (h.kind == Kind::register_schema || h.kind == Kind::registers || h.kind == Kind::memory ||
        h.kind == Kind::address_context) {
        if (bytes == nullptr) return Result<Done>::failure("Signal payload is missing");
    }
    if (h.kind == Kind::register_schema) {
        const auto result = schema(h, bytes);
        if (result.ok()) _seen[source] = true;
        return result;
    }
    if (h.sequence != _next[source]) return Result<Done>::failure("Missing or repeated source entries");
    if (h.count > UINT64_MAX - h.sequence) return Result<Done>::failure("Source sequence overflow");
    if (h.kind == Kind::address_context) {
        if (!(_features & feature_address_context)) return Result<Done>::failure("Address context is not enabled");
        const auto mode = load_u64(bytes + 48);
        const auto known = load_u64(bytes + 56);
        if ((known != 15 && known != 63) ||
            (known == 15 && (mode != 0 || load_u64(bytes + 40) != 0)) ||
            (known == 63 && mode != 16 && mode != 32 && mode != 64))
            return Result<Done>::failure("Invalid address context availability or execution mode");
        _has_context[source] = true;
        _context_sequence[source] = h.sequence;
    }
    const bool filtered_registers = (_features & feature_context_filter) != 0;
    const bool needs_baseline = !filtered_registers &&
        (h.kind == Kind::blocks || h.kind == Kind::memory || h.kind == Kind::input_request ||
         (h.kind == Kind::source_end && _data[source]));
    if ((_features & feature_address_context) && h.kind == Kind::blocks && !_has_context[source])
        return Result<Done>::failure("System blocks need an initial address context");
    if ((_features & feature_registers) && needs_baseline && !_baseline_complete[source]) {
        const auto* registers = _registers[source];
        if (registers == nullptr) return Result<Done>::failure("Source data needs a register schema");
        for (uint32_t id = 0; id < max_registers; ++id)
            if (registers->widths[id] != 0 && !registers->sampled[id])
                return Result<Done>::failure("Source data needs a complete register baseline");
        _baseline_complete[source] = true;
    }
    if (h.kind == Kind::registers) {
        if ((_features & feature_address_context) && !_has_context[source])
            return Result<Done>::failure("System register state needs an address context");
        const auto result = registers(h, bytes);
        if (!result.ok()) return result;
    }
    if (h.kind == Kind::memory) {
        const auto result = memory(h, bytes);
        if (!result.ok()) return result;
        if ((_features & feature_context_filter) != 0) {
            if (_context_filter_memory_rows > UINT64_MAX - h.count)
                return Result<Done>::failure("Context filter memory row count overflow");
            _context_filter_memory_rows += h.count;
        }
    }
    _seen[source] = true;
    _source_data_seen = true;
    _data[source] = true;
    _next[source] += h.count;
    if (h.kind == Kind::source_end) _ended[source] = true;
    return Result<Done>::success({});
}
} // namespace cpu2tensor
