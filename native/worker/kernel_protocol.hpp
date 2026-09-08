// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cpu2tensor/result.hpp>
#include <json-c/json.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace cpu2tensor::kernel_protocol {
constexpr size_t line_capacity = 8192;
constexpr size_t event_capacity = 1024;

// Both local channels are newline framed and bounded, with no unbounded JSON buffer.
struct Lines final {
    char bytes[line_capacity]{};
    size_t used = 0;
    bool eof = false;
    Result<Done> read_from(int fd) {
        if (used == sizeof(bytes) - 1) return Result<Done>::failure("Kernel control line is too long");
        const auto count = read(fd, bytes + used, sizeof(bytes) - 1 - used);
        if (count > 0) { used += static_cast<size_t>(count); bytes[used] = 0; }
        else if (count == 0) eof = true;
        else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
            return Result<Done>::failure("Kernel control channel failed");
        return Result<Done>::success({});
    }
    size_t line_size() const {
        const auto* end = static_cast<const char*>(std::memchr(bytes, '\n', used));
        return end == nullptr ? 0 : static_cast<size_t>(end - bytes) + 1;
    }
    void consume(size_t size) {
        used -= size;
        std::memmove(bytes, bytes + size, used);
        bytes[used] = 0;
    }
};
class Json final {
public:
    Json(const char* bytes, size_t size) {
        auto* parser = json_tokener_new_ex(16);
        if (parser == nullptr) return;
        json_tokener_set_flags(parser, JSON_TOKENER_STRICT);
        value = json_tokener_parse_ex(parser, bytes, static_cast<int>(size));
        _parse_error = json_tokener_get_error(parser);
        _parse_end = json_tokener_get_parse_end(parser);
        _type = json_type_to_name(json_object_get_type(value));
        if (_parse_error != json_tokener_success)
            _problem = _parse_error == json_tokener_continue ? "incomplete-json" : "malformed-json";
        else if (_parse_end != size) _problem = "trailing-json-bytes";
        else if (!json_object_is_type(value, json_type_object)) _problem = "non-object-json";
        else _problem = nullptr;
        if (_problem != nullptr) {
            if (value != nullptr) json_object_put(value);
            value = nullptr;
        }
        json_tokener_free(parser);
    }
    ~Json() { if (value != nullptr) json_object_put(value); }
    Json(const Json&) = delete;
    Json& operator=(const Json&) = delete;
    json_object* value = nullptr;
    json_object* get(const char* key) const {
        json_object* result = nullptr;
        if (value != nullptr) json_object_object_get_ex(value, key, &result);
        return result;
    }
    const char* event_problem() const {
        if (_problem != nullptr) return _problem;
        json_object* field = nullptr;
        if (!json_object_object_get_ex(value, "event", &field)) return "missing-event-field";
        if (!json_object_is_type(field, json_type_string)) return "non-string-event-field";
        if (std::strlen(json_object_get_string(field)) != static_cast<size_t>(json_object_get_string_len(field)))
            return "embedded-null-event-field";
        return nullptr;
    }
    const char* problem() const { return _problem; }
    const char* parse_error() const { return json_tokener_error_desc(_parse_error); }
    size_t parse_end() const { return _parse_end; }
    const char* type() const { return _type; }

private:
    const char* _problem = "parser-allocation-failed";
    const char* _type = "unparsed";
    json_tokener_error _parse_error = json_tokener_success;
    size_t _parse_end = 0;
};

inline const char* string_value(json_object* value) {
    if (!json_object_is_type(value, json_type_string)) return nullptr;
    const char* text = json_object_get_string(value);
    return std::strlen(text) == static_cast<size_t>(json_object_get_string_len(value)) ? text : nullptr;
}

// Called only on failure. Keep every byte of an in-limit payload, escape control
// characters, and bound oversized input independently of producer behavior.
inline void log_guest_failure(FILE* output, const char* reason, const char* bytes,
                              size_t size, const Json* event = nullptr) {
    char preview[4 * event_capacity + 1];
    constexpr char hex[] = "0123456789abcdef";
    const size_t shown = size < event_capacity ? size : event_capacity;
    size_t used = 0;
    for (size_t i = 0; i < shown; ++i) {
        const auto byte = static_cast<unsigned char>(bytes[i]);
        if (byte == '"' || byte == '\\') {
            preview[used++] = '\\';
            preview[used++] = static_cast<char>(byte);
        } else if (byte >= 32 && byte <= 126) preview[used++] = static_cast<char>(byte);
        else {
            preview[used++] = '\\';
            preview[used++] = 'x';
            preview[used++] = hex[byte >> 4];
            preview[used++] = hex[byte & 15];
        }
    }
    preview[used] = 0;
    std::fprintf(output, "cpu2tensor: guest-event failure reason=%s bytes=%zu shown=%zu truncated=%s",
                 reason, size, shown, shown < size ? "true" : "false");
    if (event != nullptr)
        std::fprintf(output, " parse_error=\"%s\" parse_end=%zu root_type=%s",
                     event->parse_error(), event->parse_end(), event->type());
    std::fprintf(output, " payload=\"%s\"\n", preview);
}
} // namespace cpu2tensor::kernel_protocol
