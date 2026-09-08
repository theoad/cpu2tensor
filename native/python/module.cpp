// SPDX-License-Identifier: AGPL-3.0-only
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstring>
#include <new>
#include <cpu2tensor/trace.hpp>

namespace {
using namespace cpu2tensor;
constexpr const char* stream_name = "cpu2tensor.Stream";
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
PyObject* decode_signals(const Header& h, const uint8_t* bytes, bool memory_values) {
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
            if (values != nullptr) std::memcpy(values + row * value_width, item + 24, 16);
            offset += memory_values ? 40 : 24;
        }
    }
    return result;
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
        payload = decode_signals(frame, bytes + header_bytes, stream->memory_values());
    else {
        payload = PyByteArray_FromStringAndSize(nullptr, size);
        if (payload != nullptr && frame.kind == Kind::guest_event)
            std::memcpy(PyByteArray_AsString(payload), bytes + header_bytes, size);
        if (payload != nullptr && frame.kind == Kind::blocks) {
            char* destination = PyByteArray_AsString(payload);
            for (uint32_t row = 0; row < frame.count; ++row)
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
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", "Native trace validation.", -1,
                      methods, nullptr, nullptr, nullptr, nullptr};
}
PyMODINIT_FUNC PyInit__native() { return PyModule_Create(&module); }
