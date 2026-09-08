// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cstring>

namespace cpu2tensor {
// Colons keep an exact-name list inside one QEMU plugin option.
inline bool valid_register_selection(const char* names) {
    const size_t size = std::strlen(names);
    if (size == 0 || size > 1023 || names[0] == ':' || names[size - 1] == ':') return false;
    for (size_t i = 0; i < size; ++i) {
        const char c = names[i];
        if (c == ':' && names[i + 1] != ':') continue;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_') continue;
        return false;
    }
    return true;
}
inline bool listed_register(const char* names, const char* name) {
    const size_t length = std::strlen(name);
    while (*names != 0) {
        const char* end = std::strchr(names, ':');
        const size_t size = end == nullptr ? std::strlen(names) : static_cast<size_t>(end - names);
        if (size == length && std::memcmp(names, name, length) == 0) return true;
        if (end == nullptr) break;
        names = end + 1;
    }
    return false;
}
inline bool unavailable_x86_register(const char* name, bool system) {
    return (system && std::strcmp(name, "eflags") == 0) ||
        listed_register("ftag:fiseg:fioff:foseg:fooff:fop", name);
}
} // namespace cpu2tensor
