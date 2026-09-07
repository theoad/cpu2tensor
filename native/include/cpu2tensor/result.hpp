// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <cassert>

namespace cpu2tensor {

// Results carry small values. The caller checks the error before reading a value.
template <typename T> class Result final {
public:
    static Result success(T value) { return Result(value, nullptr); }
    static Result failure(const char* error) { return Result(T{}, error); }
    bool ok() const { return _error == nullptr; }
    const char* error() const { return _error; }
    const T& value() const { assert(ok()); return _value; }

private:
    Result(T value, const char* error) : _value(value), _error(error) {}
    T _value;
    const char* _error;
};

struct Done final {};

} // namespace cpu2tensor
