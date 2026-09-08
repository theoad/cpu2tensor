// SPDX-License-Identifier: AGPL-3.0-only
#include <cpu2tensor/trace.hpp>
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
    store_u16(bytes + 4, wire_version);
    store_u16(bytes + 6, static_cast<uint16_t>(header.kind));
    store_u32(bytes + 8, header.source);
    store_u32(bytes + 12, header.count);
    store_u64(bytes + 16, header.sequence);
    store_u64(bytes + 24, header.detail);
}
size_t payload_size(const Header& header) {
    if (header.kind == Kind::guest_event) return static_cast<size_t>(header.detail);
    if (header.kind == Kind::blocks) return header.count * sizeof(uint64_t);
    if (header.kind == Kind::register_schema || header.kind == Kind::registers || header.kind == Kind::memory)
        return static_cast<size_t>(header.detail);
    return 0;
}
Result<Header> decode_header(const uint8_t* bytes, size_t size) {
    if (size != header_bytes) return Result<Header>::failure("Trace header must contain 32 bytes");
    if (load_u32(bytes) != wire_magic) return Result<Header>::failure("Not a cpu2tensor trace");
    if (load_u16(bytes + 4) != wire_version) return Result<Header>::failure("Unsupported trace version");
    Header h{static_cast<Kind>(load_u16(bytes + 6)), load_u32(bytes + 8), load_u32(bytes + 12),
             load_u64(bytes + 16), load_u64(bytes + 24)};
    if (h.source >= max_sources || h.count > max_addresses)
        return Result<Header>::failure("Trace exceeds the supported source or batch limit");
    switch (h.kind) {
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
        if (h.count == 0 || h.detail > max_payload_bytes || h.detail == 0)
            return Result<Header>::failure("Invalid signal payload size");
        if (h.kind == Kind::register_schema && (h.sequence != 0 || h.detail != h.count * register_schema_bytes))
            return Result<Header>::failure("Invalid register schema size or sequence");
        if (h.kind == Kind::registers && h.detail < h.count * 17)
            return Result<Header>::failure("Register payload is too short");
        if (h.kind == Kind::memory && h.detail != h.count * 24 && h.detail != h.count * 40)
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
                feature_system | feature_kernel | feature_window;
            if ((architecture != 1 && architecture != 2) || (h.detail & ~allowed) != 0)
                return Result<Header>::failure("Unsupported target architecture or features");
            if ((h.detail & feature_memory_values) && !(h.detail & feature_memory))
                return Result<Header>::failure("Memory values require memory observations");
            if (((h.detail & feature_kernel) && !(h.detail & feature_system)) ||
                ((h.detail & feature_stdio) && (h.detail & feature_system)))
                return Result<Header>::failure("Incompatible system and action features");
        }
        if (h.kind == Kind::complete && h.detail > 255) return Result<Header>::failure("Invalid target exit code");
        if (h.kind == Kind::error && (h.detail < 1 || h.detail > 4)) return Result<Header>::failure("Unknown trace failure");
        break;
    default: return Result<Header>::failure("Unknown trace frame kind");
    }
    return Result<Header>::success(h);
}
Stream::~Stream() { for (auto* registers : _registers) delete registers; }
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
    const size_t stride = memory_values() ? 40 : 24;
    if (h.detail != h.count * stride) return Result<Done>::failure("Memory value setting does not match payload");
    for (uint32_t row = 0; row < h.count; ++row) {
        const auto* item = bytes + row * stride;
        const auto width = load_u32(item + 16);
        const auto flags = load_u32(item + 20);
        if (width == 0 || (width & (width - 1)) != 0 || flags > 3)
            return Result<Done>::failure("Invalid memory access size or flags");
        if (memory_values()) {
            if (width > 16) return Result<Done>::failure("Unsupported memory value width");
            for (uint32_t i = width; i < 16; ++i)
                if (item[24 + i] != 0) return Result<Done>::failure("Memory value padding must be zero");
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
    if (!_started) {
        if (h.kind != Kind::hello) return Result<Done>::failure("Trace must start with Hello");
        _started = true;
        _features = h.detail;
        return Result<Done>::success({});
    }
    if (h.kind == Kind::hello) return Result<Done>::failure("Duplicate trace Hello");
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
                for (uint32_t id = 0; id < max_registers; ++id)
                    if (registers->widths[id] != 0 && !registers->sampled[id])
                        return Result<Done>::failure("Kernel boundary needs complete source register baselines");
            }
        }
        // Adapter events have no vCPU attribution and consume no source sequence.
        return Result<Done>::success({});
    }
    if (h.kind == Kind::error) { _finished = true; return Result<Done>::success({}); }
    if (h.kind == Kind::complete) {
        for (uint32_t source = 0; source < max_sources; ++source)
            if (_seen[source] && !_ended[source]) return Result<Done>::failure("Trace ended before all sources ended");
        _finished = true;
        return Result<Done>::success({});
    }
    const auto source = h.source;
    if ((_features & feature_stdio) && source != 0)
        return Result<Done>::failure("Stdin interaction supports one vCPU only");
    if (h.kind == Kind::input_request && !(_features & feature_stdio))
        return Result<Done>::failure("Stdin request needs interactive capture");
    if (_ended[source]) return Result<Done>::failure("Frame received after source end");
    if (h.kind == Kind::register_schema || h.kind == Kind::registers || h.kind == Kind::memory) {
        if (bytes == nullptr) return Result<Done>::failure("Signal payload is missing");
    }
    if (h.kind == Kind::register_schema) {
        const auto result = schema(h, bytes);
        if (result.ok()) _seen[source] = true;
        return result;
    }
    if (h.sequence != _next[source]) return Result<Done>::failure("Missing or repeated source entries");
    if (h.count > UINT64_MAX - h.sequence) return Result<Done>::failure("Source sequence overflow");
    const bool needs_baseline = h.kind == Kind::blocks || h.kind == Kind::memory || h.kind == Kind::input_request ||
        (h.kind == Kind::source_end && _data[source]);
    if ((_features & feature_registers) && needs_baseline && !_baseline_complete[source]) {
        const auto* registers = _registers[source];
        if (registers == nullptr) return Result<Done>::failure("Source data needs a register schema");
        for (uint32_t id = 0; id < max_registers; ++id)
            if (registers->widths[id] != 0 && !registers->sampled[id])
                return Result<Done>::failure("Source data needs a complete register baseline");
        _baseline_complete[source] = true;
    }
    if (h.kind == Kind::registers) {
        const auto result = registers(h, bytes);
        if (!result.ok()) return result;
    }
    if (h.kind == Kind::memory) {
        const auto result = memory(h, bytes);
        if (!result.ok()) return result;
    }
    _seen[source] = true;
    _data[source] = true;
    _next[source] += h.count;
    if (h.kind == Kind::source_end) _ended[source] = true;
    return Result<Done>::success({});
}
} // namespace cpu2tensor
