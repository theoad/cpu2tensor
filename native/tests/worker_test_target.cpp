// SPDX-License-Identifier: AGPL-3.0-only
#include <cpu2tensor/trace.hpp>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

// A lifecycle fixture, not a QEMU replacement: seal capture but never exit.
// The worker must still cancel this child when the consumer disconnects.
int main(int argc, char** argv) {
    if (argc < 3) return 1;
    const char* argument = std::strstr(argv[2], ",fd=");
    if (argument == nullptr) return 1;
    const int output = std::atoi(argument + 4);
    using namespace cpu2tensor;
    uint8_t bytes[header_bytes + sizeof(uint64_t)];
    const Header headers[] = {
        {Kind::hello, 0, 0, 0, 1}, {Kind::blocks, 0, 1, 0, 0},
        {Kind::source_end, 0, 0, 1, 0}, {Kind::complete},
    };
    for (const auto& header : headers) {
        encode_header(bytes, header);
        store_u64(bytes + header_bytes, 0x1234);
        const auto size = header_bytes + header.count * sizeof(uint64_t);
        if (write(output, bytes, size) != static_cast<ssize_t>(size)) return 1;
    }
    for (;;) pause();
}
