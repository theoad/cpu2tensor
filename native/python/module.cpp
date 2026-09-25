// SPDX-License-Identifier: AGPL-3.0-only
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <cpu2tensor/trace.hpp>

namespace {
using namespace cpu2tensor;
constexpr const char* stream_name = "cpu2tensor.Stream";
constexpr size_t max_context_group_frames = 32;
void delete_stream(PyObject* capsule) { delete static_cast<Stream*>(PyCapsule_GetPointer(capsule, stream_name)); }
PyObject* new_stream(PyObject*, PyObject*) {
    auto* stream = new (std::nothrow) Stream;
    if (stream == nullptr) return PyErr_NoMemory();
    PyObject* capsule = PyCapsule_New(stream, stream_name, delete_stream);
    if (capsule == nullptr) delete stream;
    return capsule;
}
PyObject* header_size(PyObject*, PyObject* argument) {
    Py_buffer view{};
    if (PyObject_GetBuffer(argument, &view, PyBUF_SIMPLE) != 0) return nullptr;
    const auto header = decode_header(static_cast<const uint8_t*>(view.buf), view.len);
    PyBuffer_Release(&view);
    if (!header.ok()) { PyErr_SetString(PyExc_ValueError, header.error()); return nullptr; }
    return PyLong_FromSize_t(payload_size(header.value()));
}

uint64_t native_u64(const uint8_t* bytes) {
    uint64_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

PyObject* raw_trace_sketch(PyObject*, PyObject* arguments) {
    PyObject* trace_object = nullptr;
    PyObject* offset_object = nullptr;
    Py_buffer trace{};
    Py_buffer offsets{};
    unsigned segments = 0;
    unsigned pair_bins = 0;
    int include_length = 0;
    if (!PyArg_ParseTuple(
            arguments, "OOIIp", &trace_object, &offset_object, &segments,
            &pair_bins, &include_length))
        return nullptr;
    if (PyObject_GetBuffer(trace_object, &trace, PyBUF_SIMPLE) != 0)
        return nullptr;
    if (PyObject_GetBuffer(offset_object, &offsets, PyBUF_SIMPLE) != 0) {
        PyBuffer_Release(&trace);
        return nullptr;
    }
    const auto release = [&]() {
        PyBuffer_Release(&offsets);
        PyBuffer_Release(&trace);
    };
    constexpr Py_ssize_t offset_width = sizeof(uint64_t);
    const bool power_of_two = pair_bins == 0 || (pair_bins & (pair_bins - 1)) == 0;
    if (segments == 0 || !power_of_two || offsets.len < 2 * offset_width ||
        offsets.len % offset_width != 0 || trace.len < 0) {
        release();
        PyErr_SetString(PyExc_ValueError, "Invalid raw trace sketch buffers or dimensions");
        return nullptr;
    }
    const size_t rows = static_cast<size_t>(offsets.len) / sizeof(uint64_t) - 1;
    if (segments > (std::numeric_limits<size_t>::max() - pair_bins - 1) / 256) {
        release();
        PyErr_SetString(PyExc_OverflowError, "Raw trace sketch dimensions overflow");
        return nullptr;
    }
    const size_t byte_dimensions = static_cast<size_t>(segments) * 256;
    const size_t dimensions =
        byte_dimensions + pair_bins + (include_length != 0 ? 1 : 0);
    if (rows > static_cast<size_t>(PY_SSIZE_T_MAX) / dimensions / sizeof(float)) {
        release();
        PyErr_SetString(PyExc_OverflowError, "Raw trace sketch output is too large");
        return nullptr;
    }
    const auto* offset_bytes = static_cast<const uint8_t*>(offsets.buf);
    uint64_t previous = native_u64(offset_bytes);
    if (previous != 0) {
        release();
        PyErr_SetString(PyExc_ValueError, "Raw trace offsets must start at zero");
        return nullptr;
    }
    for (size_t row = 0; row < rows; ++row) {
        const uint64_t next = native_u64(offset_bytes + (row + 1) * sizeof(uint64_t));
        if (next <= previous || next > static_cast<uint64_t>(trace.len)) {
            release();
            PyErr_SetString(PyExc_ValueError, "Raw trace offsets must be ordered and nonempty");
            return nullptr;
        }
        previous = next;
    }
    if (previous != static_cast<uint64_t>(trace.len)) {
        release();
        PyErr_SetString(PyExc_ValueError, "Raw trace offsets do not cover the byte buffer");
        return nullptr;
    }

    PyObject* result = PyByteArray_FromStringAndSize(
        nullptr, static_cast<Py_ssize_t>(rows * dimensions * sizeof(float)));
    if (result == nullptr) {
        release();
        return nullptr;
    }
    auto* output = reinterpret_cast<float*>(PyByteArray_AsString(result));
    std::memset(output, 0, rows * dimensions * sizeof(float));
    const auto* bytes = static_cast<const uint8_t*>(trace.buf);
    // Keep the GIL while reading caller-owned buffers. Both NumPy arrays are
    // mutable, so releasing it would let another Python thread change either
    // trace bytes or validated offsets while this loop is using them. A
    // production pipeline can parallelize reducers across processes without
    // copying the raw AUX payload merely to make this call thread-safe.
    for (size_t row = 0; row < rows; ++row) {
        const size_t start = native_u64(offset_bytes + row * sizeof(uint64_t));
        const size_t end = native_u64(offset_bytes + (row + 1) * sizeof(uint64_t));
        const size_t length = end - start;
        float* features = output + row * dimensions;
        for (size_t position = 0; position < length; ++position) {
            const size_t segment = position * segments / length;
            ++features[segment * 256 + bytes[start + position]];
            if (pair_bins != 0 && position + 1 < length) {
                const uint32_t pair_key =
                    (static_cast<uint32_t>(bytes[start + position]) << 8) |
                    bytes[start + position + 1];
                const uint32_t pair_hash =
                    (pair_key ^ (pair_key >> 7) ^ (pair_key >> 3)) & (pair_bins - 1);
                ++features[byte_dimensions + pair_hash];
            }
        }
        for (size_t segment = 0; segment < segments; ++segment) {
            float count = 0;
            for (size_t value = 0; value < 256; ++value)
                count += features[segment * 256 + value];
            if (count == 0) count = 1;
            for (size_t value = 0; value < 256; ++value)
                features[segment * 256 + value] /= count;
        }
        if (pair_bins != 0) {
            const float pairs = static_cast<float>(length > 1 ? length - 1 : 1);
            for (size_t index = 0; index < pair_bins; ++index)
                features[byte_dimensions + index] /= pairs;
        }
        if (include_length != 0)
            features[dimensions - 1] = static_cast<float>(std::log1p(length) / 16.0);
    }
    release();
    return result;
}
// The dictionary owns each buffer. Only fixed column counts cross the Python API.
Result<char*> add_buffer(PyObject* columns, const char* name, size_t size) {
    PyObject* buffer = PyByteArray_FromStringAndSize(nullptr, size);
    if (buffer == nullptr) return Result<char*>::failure("Cannot allocate tensor buffer");
    const int added = PyDict_SetItemString(columns, name, buffer);
    char* bytes = PyByteArray_AsString(buffer);
    Py_DECREF(buffer);
    if (added != 0) return Result<char*>::failure("Cannot add tensor column");
    return Result<char*>::success(bytes);
}
void native_integer(char* column, uint32_t row, uint64_t value) {
    std::memcpy(column + row * sizeof(value), &value, sizeof(value));
}
PyObject* decode_schema(const Header& h, const uint8_t* bytes) {
    PyObject* result = PyDict_New();
    if (result == nullptr) return nullptr;
    for (uint32_t row = 0; row < h.count; ++row) {
        const auto* item = bytes + row * register_schema_bytes;
        PyObject* id = PyLong_FromUnsignedLong(load_u32(item));
        PyObject* name = PyUnicode_FromString(reinterpret_cast<const char*>(item + 8));
        const bool ok = id != nullptr && name != nullptr && PyDict_SetItem(result, id, name) == 0;
        Py_XDECREF(id); Py_XDECREF(name);
        if (!ok) { Py_DECREF(result); return nullptr; }
    }
    return result;
}
PyObject* decode_signals(const Header& h, const uint8_t* bytes, bool memory_values, bool system_memory) {
    const bool registers = h.kind == Kind::registers;
    const char* names[4] = {"pc", registers ? "ids" : "addresses", registers ? "widths" : "sizes", "flags"};
    PyObject* result = PyDict_New();
    if (result == nullptr) return nullptr;
    char* columns[4];
    for (unsigned column = 0; column < 4; ++column) {
        const auto added = add_buffer(result, names[column], h.count * sizeof(uint64_t));
        if (!added.ok()) { Py_DECREF(result); return nullptr; }
        columns[column] = added.value();
    }
    size_t value_width = registers ? 0 : (memory_values ? 16 : 0);
    if (registers) {
        size_t offset = 0;
        for (uint32_t row = 0; row < h.count; ++row) {
            const auto width = load_u16(bytes + offset + 12);
            if (width > value_width) value_width = width;
            offset += 16 + width;
        }
    }
    char* values = nullptr;
    if (value_width != 0) {
        const auto added = add_buffer(result, "values", h.count * value_width);
        if (!added.ok()) { Py_DECREF(result); return nullptr; }
        values = added.value();
        std::memset(values, 0, h.count * value_width);
    } else if (PyDict_SetItemString(result, "values", Py_None) != 0) { Py_DECREF(result); return nullptr; }
    if (registers) {
        PyObject* width = PyLong_FromSize_t(value_width);
        const bool added = width != nullptr && PyDict_SetItemString(result, "value_width", width) == 0;
        Py_XDECREF(width);
        if (!added) { Py_DECREF(result); return nullptr; }
    }
    char* mapping[4]{};
    constexpr const char* mapping_names[] = {"physical_addresses", "mapped_sizes", "mapping_flags", "context_sequences"};
    if (!registers) {
        for (unsigned column = 0; column < 4; ++column) {
            if (system_memory) {
                const auto added = add_buffer(result, mapping_names[column], h.count * sizeof(uint64_t));
                if (!added.ok()) { Py_DECREF(result); return nullptr; }
                mapping[column] = added.value();
            } else if (PyDict_SetItemString(result, mapping_names[column], Py_None) != 0) {
                Py_DECREF(result); return nullptr;
            }
        }
    }
    size_t offset = 0;
    for (uint32_t row = 0; row < h.count; ++row) {
        const auto* item = bytes + offset;
        native_integer(columns[0], row, load_u64(item));
        if (registers) {
            const auto width = load_u16(item + 12);
            native_integer(columns[1], row, load_u32(item + 8));
            native_integer(columns[2], row, width);
            native_integer(columns[3], row, load_u16(item + 14));
            std::memcpy(values + row * value_width, item + 16, width);
            offset += 16 + width;
        } else {
            native_integer(columns[1], row, load_u64(item + 8));
            native_integer(columns[2], row, load_u32(item + 16));
            native_integer(columns[3], row, load_u32(item + 20));
            const size_t prefix = system_memory ? 48 : 24;
            if (system_memory) {
                native_integer(mapping[0], row, load_u64(item + 24));
                native_integer(mapping[1], row, load_u32(item + 32));
                native_integer(mapping[2], row, load_u32(item + 36));
                native_integer(mapping[3], row, load_u64(item + 40));
            }
            if (values != nullptr) std::memcpy(values + row * value_width, item + prefix, 16);
            offset += prefix + (memory_values ? 16 : 0);
        }
    }
    return result;
}
PyObject* decode_context(const Header& h, const uint8_t* bytes) {
    constexpr const char* names[] = {"pc", "cr0", "cr3", "cr4", "efer", "cs_base", "mode", "known"};
    PyObject* result = PyDict_New();
    if (result == nullptr) return nullptr;
    for (unsigned column = 0; column < 8; ++column) {
        const auto added = add_buffer(result, names[column], h.count * sizeof(uint64_t));
        if (!added.ok()) { Py_DECREF(result); return nullptr; }
        for (uint32_t row = 0; row < h.count; ++row)
            native_integer(added.value(), row, load_u64(bytes + row * context_bytes + column * 8));
    }
    return result;
}

PyObject* decode_reduced_context(const Header& h, const uint8_t* bytes) {
    constexpr const char* names[] = {
        "block_positions", "pc", "cr0", "cr3", "cr4", "efer", "cs_base", "mode", "known"
    };
    PyObject* result = PyDict_New();
    if (result == nullptr) return nullptr;
    for (unsigned column = 0; column < 9; ++column) {
        const auto added = add_buffer(result, names[column], h.count * sizeof(uint64_t));
        if (!added.ok()) { Py_DECREF(result); return nullptr; }
        for (uint32_t row = 0; row < h.count; ++row)
            native_integer(added.value(), row,
                load_u64(bytes + row * reduced_context_bytes + column * sizeof(uint64_t)));
    }
    return result;
}

PyObject* decode_transitions(const Header& h, const uint8_t* bytes) {
    constexpr const char* names[] = {"from_addresses", "destinations", "counts"};
    PyObject* result = PyDict_New();
    if (result == nullptr) return nullptr;
    for (unsigned column = 0; column < 3; ++column) {
        const auto added = add_buffer(result, names[column], h.count * sizeof(uint64_t));
        if (!added.ok()) { Py_DECREF(result); return nullptr; }
        for (uint32_t row = 0; row < h.count; ++row)
            native_integer(added.value(), row,
                load_u64(bytes + row * transition_count_bytes + column * sizeof(uint64_t)));
    }
    return result;
}

struct MixedTable final {
    size_t size = 0;
    uint32_t count = 0;
    uint8_t bytes[max_payload_bytes];
    uint64_t sequences[max_addresses];
};

bool add_sequences(PyObject* columns, const char* name, const MixedTable& table) {
    const auto added = add_buffer(columns, name, table.count * sizeof(uint64_t));
    if (!added.ok()) return false;
    for (uint32_t row = 0; row < table.count; ++row)
        native_integer(added.value(), row, table.sequences[row]);
    return true;
}

PyObject* decode_mixed(const Header& h, const uint8_t* bytes, bool memory_values, bool system_memory) {
    constexpr Kind kinds[] = {Kind::blocks, Kind::registers, Kind::memory, Kind::address_context};
    constexpr const char* names[] = {"blocks", "registers", "memory", "context"};
    MixedTable tables[4];
    size_t offset = 0;
    uint32_t event_offset = 0;
    // Stream::accept has checked every run. Gather only used bytes on the stack;
    // Python receives one owned column buffer per field, never an object per event.
    while (offset < h.detail) {
        if (h.detail - offset < 8) {
            PyErr_SetString(PyExc_ValueError, "Mixed trace run header is truncated");
            return nullptr;
        }
        const auto kind = static_cast<Kind>(load_u16(bytes + offset));
        const uint32_t count = load_u16(bytes + offset + 2);
        const uint32_t size = load_u32(bytes + offset + 4);
        offset += 8;
        unsigned index = 0;
        while (index < 4 && kinds[index] != kind) ++index;
        if (index == 4 || size > h.detail - offset ||
            size > max_payload_bytes - tables[index].size ||
            count > max_addresses - tables[index].count) {
            PyErr_SetString(PyExc_ValueError, "Mixed trace table exceeds its validated bounds");
            return nullptr;
        }
        auto& table = tables[index];
        std::memcpy(table.bytes + table.size, bytes + offset, size);
        for (uint32_t row = 0; row < count; ++row)
            table.sequences[table.count + row] = h.sequence + event_offset + row;
        table.count += count;
        table.size += size;
        event_offset += count;
        offset += size;
    }

    PyObject* result = PyDict_New();
    if (result == nullptr) return nullptr;
    for (unsigned index = 0; index < 4; ++index) {
        const auto& table = tables[index];
        if (table.count == 0) continue;
        const Header header{kinds[index], h.source, table.count, h.sequence, table.size};
        PyObject* columns = nullptr;
        if (kinds[index] == Kind::blocks) {
            columns = PyByteArray_FromStringAndSize(nullptr, table.count * sizeof(uint64_t));
            if (columns != nullptr) {
                char* destination = PyByteArray_AsString(columns);
                for (uint32_t row = 0; row < table.count; ++row)
                    native_integer(destination, row, load_u64(table.bytes + row * sizeof(uint64_t)));
            }
        } else if (kinds[index] == Kind::address_context) {
            columns = decode_context(header, table.bytes);
        } else {
            columns = decode_signals(header, table.bytes, memory_values, system_memory);
        }
        if (columns == nullptr) { Py_DECREF(result); return nullptr; }
        const bool sequences_added = kinds[index] == Kind::blocks
            ? add_sequences(result, "block_sequences", table)
            : add_sequences(columns, "sequences", table);
        const bool added = sequences_added && PyDict_SetItemString(result, names[index], columns) == 0;
        Py_DECREF(columns);
        if (!added) { Py_DECREF(result); return nullptr; }
    }
    return result;
}

struct ContextCounts final {
    uint32_t blocks = 0;
    uint32_t contexts = 0;
};

Result<ContextCounts> count_context_rows(const Header& h, const uint8_t* bytes) {
    ContextCounts counts;
    size_t offset = 0;
    uint32_t events = 0;
    while (offset < h.detail) {
        if (h.detail - offset < 8)
            return Result<ContextCounts>::failure("Truncated mixed run header");
        const auto kind = static_cast<Kind>(load_u16(bytes + offset));
        const uint32_t rows = load_u16(bytes + offset + 2);
        const uint32_t size = load_u32(bytes + offset + 4);
        offset += 8;
        const size_t row_bytes = kind == Kind::blocks ? sizeof(uint64_t) : context_bytes;
        if ((kind != Kind::blocks && kind != Kind::address_context) || rows == 0 ||
            rows > h.count - events || size != rows * row_bytes || size > h.detail - offset)
            return Result<ContextCounts>::failure("Context-only group contains another signal");
        if (kind == Kind::blocks) counts.blocks += rows;
        else counts.contexts += rows;
        events += rows;
        offset += size;
    }
    if (events != h.count)
        return Result<ContextCounts>::failure("Mixed event count does not match its runs");
    return Result<ContextCounts>::success(counts);
}

struct ContextGroup final {
    uint32_t source = 0;
    uint64_t first_sequence = 0;
    uint32_t blocks = 0;
    uint32_t contexts = 0;
};

struct ContextColumns final {
    char* blocks = nullptr;
    char* block_sequences = nullptr;
    char* context[8]{};
    char* context_sequences = nullptr;
};

PyObject* allocate_context_columns(const ContextGroup& group, ContextColumns* columns) {
    constexpr const char* context_names[] = {
        "pc", "cr0", "cr3", "cr4", "efer", "cs_base", "mode", "known"
    };
    PyObject* result = PyDict_New();
    if (result == nullptr) return nullptr;
    if (group.blocks != 0) {
        const auto blocks = add_buffer(result, "blocks", group.blocks * sizeof(uint64_t));
        const auto sequences = add_buffer(
            result, "block_sequences", group.blocks * sizeof(uint64_t));
        if (!blocks.ok() || !sequences.ok()) { Py_DECREF(result); return nullptr; }
        columns->blocks = blocks.value();
        columns->block_sequences = sequences.value();
    }
    if (group.contexts == 0) return result;
    PyObject* context = PyDict_New();
    if (context == nullptr) { Py_DECREF(result); return nullptr; }
    for (unsigned index = 0; index < 8; ++index) {
        const auto added = add_buffer(
            context, context_names[index], group.contexts * sizeof(uint64_t));
        if (!added.ok()) { Py_DECREF(context); Py_DECREF(result); return nullptr; }
        columns->context[index] = added.value();
    }
    const auto sequences = add_buffer(
        context, "sequences", group.contexts * sizeof(uint64_t));
    if (!sequences.ok() || PyDict_SetItemString(result, "context", context) != 0) {
        Py_DECREF(context); Py_DECREF(result); return nullptr;
    }
    columns->context_sequences = sequences.value();
    Py_DECREF(context);
    return result;
}

void fill_context_columns(
    const Header& h, const uint8_t* bytes, ContextColumns* columns,
    uint32_t* block_row, uint32_t* context_row
) {
    size_t offset = 0;
    uint32_t event_offset = 0;
    while (offset < h.detail) {
        const auto kind = static_cast<Kind>(load_u16(bytes + offset));
        const uint32_t rows = load_u16(bytes + offset + 2);
        offset += 8;
        if (kind == Kind::blocks) {
            for (uint32_t row = 0; row < rows; ++row) {
                native_integer(columns->blocks, *block_row, load_u64(bytes + offset + row * 8));
                native_integer(columns->block_sequences, *block_row, h.sequence + event_offset + row);
                ++*block_row;
            }
            offset += rows * sizeof(uint64_t);
        } else {
            for (uint32_t row = 0; row < rows; ++row) {
                for (unsigned column = 0; column < 8; ++column)
                    native_integer(columns->context[column], *context_row,
                        load_u64(bytes + offset + row * context_bytes + column * 8));
                native_integer(columns->context_sequences, *context_row,
                    h.sequence + event_offset + row);
                ++*context_row;
            }
            offset += rows * context_bytes;
        }
        event_offset += rows;
    }
}

PyObject* context_group_result(PyObject* frames, const char* error) {
    PyObject* result = PyTuple_New(2);
    PyObject* message = error == nullptr ? Py_None : PyUnicode_FromString(error);
    if (message == Py_None) Py_INCREF(message);
    if (result == nullptr || message == nullptr) {
        Py_XDECREF(result);
        Py_XDECREF(message);
        Py_DECREF(frames);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, frames);
    PyTuple_SET_ITEM(result, 1, message);
    return result;
}

// Context-only capture is common and its wire frames are deliberately small.
// Validate several frames with one GIL release, then publish one owned column
// set per source. Cross-source order is unspecified; each source's sequence is
// still exact. The caller places control frames at group boundaries.
PyObject* decode_context_frames(PyObject*, PyObject* arguments) {
    PyObject* capsule = nullptr;
    PyObject* argument = nullptr;
    if (!PyArg_ParseTuple(arguments, "OO", &capsule, &argument)) return nullptr;
    auto* stream = static_cast<Stream*>(PyCapsule_GetPointer(capsule, stream_name));
    if (stream == nullptr) return nullptr;
    PyObject* sequence = PySequence_Fast(argument, "frames must be a sequence");
    if (sequence == nullptr) return nullptr;
    const auto frame_count = PySequence_Fast_GET_SIZE(sequence);
    if (frame_count < 1 || frame_count > static_cast<Py_ssize_t>(max_context_group_frames)) {
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_ValueError, "Context group must contain between 1 and 32 frames");
        return nullptr;
    }

    Py_buffer views[max_context_group_frames]{};
    Header headers[max_context_group_frames]{};
    Py_ssize_t acquired = 0;
    const char* error = nullptr;
    Py_ssize_t valid_headers = 0;
    for (Py_ssize_t index = 0; index < frame_count; ++index) {
        PyObject* item = PySequence_Fast_GET_ITEM(sequence, index);
        if (PyObject_GetBuffer(item, &views[index], PyBUF_SIMPLE) != 0) break;
        ++acquired;
        const auto* bytes = static_cast<const uint8_t*>(views[index].buf);
        const auto header = decode_header(
            bytes, views[index].len >= static_cast<Py_ssize_t>(header_bytes) ? header_bytes : 0);
        if (!header.ok()) { error = header.error(); break; }
        headers[index] = header.value();
        const auto size = payload_size(headers[index]);
        if (views[index].len != static_cast<Py_ssize_t>(header_bytes + size)) {
            error = "Trace frame length does not match its header";
            break;
        }
        if (headers[index].kind != Kind::mixed) {
            error = "Context group contains a non-mixed frame";
            break;
        }
        ++valid_headers;
    }
    if (PyErr_Occurred()) {
        for (Py_ssize_t index = 0; index < acquired; ++index) PyBuffer_Release(&views[index]);
        Py_DECREF(sequence);
        return nullptr;
    }

    Py_ssize_t accepted = 0;
    const char* validation_error = nullptr;
    Py_BEGIN_ALLOW_THREADS
    for (; accepted < valid_headers; ++accepted) {
        const auto* bytes = static_cast<const uint8_t*>(views[accepted].buf);
        const auto result = stream->accept(headers[accepted], bytes + header_bytes);
        if (!result.ok()) { validation_error = result.error(); break; }
    }
    Py_END_ALLOW_THREADS
    if (validation_error != nullptr) error = validation_error;

    ContextGroup groups[max_context_group_frames]{};
    size_t group_count = 0;
    for (Py_ssize_t index = 0; index < accepted; ++index) {
        const auto* bytes = static_cast<const uint8_t*>(views[index].buf);
        const auto counts = count_context_rows(headers[index], bytes + header_bytes);
        if (!counts.ok()) { error = counts.error(); accepted = index; break; }
        size_t group = 0;
        while (group < group_count && groups[group].source != headers[index].source) ++group;
        if (group == group_count) {
            groups[group] = {headers[index].source, headers[index].sequence, 0, 0};
            ++group_count;
        }
        groups[group].blocks += counts.value().blocks;
        groups[group].contexts += counts.value().contexts;
    }

    PyObject* outputs = PyList_New(0);
    if (outputs == nullptr) error = "Cannot allocate decoded frame list";
    for (size_t group = 0; outputs != nullptr && group < group_count; ++group) {
        ContextColumns columns;
        PyObject* payload = allocate_context_columns(groups[group], &columns);
        if (payload == nullptr) { Py_DECREF(outputs); outputs = nullptr; break; }
        uint32_t block_row = 0;
        uint32_t context_row = 0;
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t index = 0; index < accepted; ++index) {
            if (headers[index].source != groups[group].source) continue;
            const auto* bytes = static_cast<const uint8_t*>(views[index].buf);
            fill_context_columns(headers[index], bytes + header_bytes, &columns,
                                 &block_row, &context_row);
        }
        Py_END_ALLOW_THREADS
        PyObject* decoded = Py_BuildValue("(IIIKKN)", static_cast<unsigned>(Kind::mixed),
            groups[group].source, groups[group].blocks + groups[group].contexts,
            static_cast<unsigned long long>(groups[group].first_sequence), 0ull, payload);
        if (decoded == nullptr || PyList_Append(outputs, decoded) != 0) {
            Py_XDECREF(decoded);
            Py_DECREF(outputs);
            outputs = nullptr;
            break;
        }
        Py_DECREF(decoded);
    }
    for (Py_ssize_t index = 0; index < acquired; ++index) PyBuffer_Release(&views[index]);
    Py_DECREF(sequence);
    if (outputs == nullptr) return nullptr;
    return context_group_result(outputs, error);
}

PyObject* decode(PyObject*, PyObject* arguments) {
    PyObject* capsule = nullptr;
    Py_buffer view{};
    if (!PyArg_ParseTuple(arguments, "Oy*", &capsule, &view)) return nullptr;
    auto* stream = static_cast<Stream*>(PyCapsule_GetPointer(capsule, stream_name));
    if (stream == nullptr) { PyBuffer_Release(&view); return nullptr; }
    const auto* bytes = static_cast<const uint8_t*>(view.buf);
    const auto header = decode_header(bytes, view.len >= static_cast<Py_ssize_t>(header_bytes) ? header_bytes : 0);
    if (!header.ok()) { PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, header.error()); return nullptr; }
    const auto& frame = header.value();
    const auto size = payload_size(frame);
    if (view.len != static_cast<Py_ssize_t>(header_bytes + size)) {
        PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, "Trace frame length does not match its header"); return nullptr;
    }
    const auto accepted = stream->accept(frame, bytes + header_bytes);
    if (!accepted.ok()) { PyBuffer_Release(&view); PyErr_SetString(PyExc_ValueError, accepted.error()); return nullptr; }
    PyObject* payload = nullptr;
    if (frame.kind == Kind::register_schema) payload = decode_schema(frame, bytes + header_bytes);
    else if (frame.kind == Kind::registers || frame.kind == Kind::memory)
        payload = decode_signals(frame, bytes + header_bytes, stream->memory_values(), stream->system_memory());
    else if (frame.kind == Kind::address_context) payload = decode_context(frame, bytes + header_bytes);
    else if (frame.kind == Kind::reduced_context)
        payload = decode_reduced_context(frame, bytes + header_bytes);
    else if (frame.kind == Kind::block_transitions)
        payload = decode_transitions(frame, bytes + header_bytes);
    else if (frame.kind == Kind::mixed)
        payload = decode_mixed(frame, bytes + header_bytes, stream->memory_values(), stream->system_memory());
    else {
        payload = PyByteArray_FromStringAndSize(nullptr, size);
        if (payload != nullptr && (frame.kind == Kind::guest_event ||
                                   frame.kind == Kind::transition_window ||
                                   frame.kind == Kind::context_filter ||
                                   frame.kind == Kind::observation_summary))
            std::memcpy(PyByteArray_AsString(payload), bytes + header_bytes, size);
        if (payload != nullptr && (frame.kind == Kind::blocks || frame.kind == Kind::executable_layout)) {
            char* destination = PyByteArray_AsString(payload);
            for (size_t row = 0; row < size / sizeof(uint64_t); ++row)
                native_integer(destination, row, load_u64(bytes + header_bytes + row * sizeof(uint64_t)));
        }
    }
    PyBuffer_Release(&view);
    if (payload == nullptr) return nullptr;
    return Py_BuildValue("(IIIKKN)", static_cast<unsigned>(frame.kind), frame.source, frame.count,
        static_cast<unsigned long long>(frame.sequence), static_cast<unsigned long long>(frame.detail), payload);
}
PyMethodDef methods[] = {
    {"new_stream", new_stream, METH_NOARGS, "Create validation state for one connection."},
    {"payload_size", header_size, METH_O, "Check a header and return its bounded payload size."},
    {"decode", decode, METH_VARARGS, "Validate a frame and return owned tensor columns."},
    {"decode_context_frames", decode_context_frames, METH_VARARGS,
     "Validate and collate a bounded context-only frame group."},
    {"raw_trace_sketch", raw_trace_sketch, METH_VARARGS,
     "Reduce concatenated undecoded trace bytes to owned float columns."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", "Native trace validation.", -1,
                      methods, nullptr, nullptr, nullptr, nullptr};
}
PyMODINIT_FUNC PyInit__native() { return PyModule_Create(&module); }
