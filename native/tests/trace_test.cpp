// SPDX-License-Identifier: AGPL-3.0-only
#include <cassert>
#include <cstring>
#include <cpu2tensor/trace.hpp>

using namespace cpu2tensor;

int main() {
    uint8_t bytes[header_bytes];
    Header batch{Kind::blocks, 2, 2, 0, 0};
    encode_header(bytes, batch);
    assert(std::memcmp(bytes, "C2T1\2\0\2\0", 8) == 0);
    assert(decode_header(bytes, sizeof(bytes)).value().source == 2);
    bytes[4] = 3;
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    assert(!decode_header(bytes, 4).ok());
    store_u64(bytes, UINT64_MAX);
    assert(load_u64(bytes) == UINT64_MAX);
    store_u64(bytes, uint64_t{1} << 63);
    assert(load_u64(bytes) == uint64_t{1} << 63);

    Stream stream;
    assert(!stream.accept(batch).ok());
    assert(stream.accept({Kind::hello, 0, 0, 0, 1}).ok());
    assert(!stream.accept({Kind::hello, 0, 0, 0, 1}).ok());
    assert(stream.accept(batch).ok());
    assert(!stream.accept(batch).ok());
    // Source 1 can finish while source 2 is still active.
    assert(stream.accept({Kind::source_end, 1, 0, 0, 0}).ok());
    assert(!stream.accept({Kind::complete}).ok());
    assert(!stream.accept({Kind::blocks, 2, 1, 3, 0}).ok());
    assert(stream.accept({Kind::blocks, 2, 1, 2, 0}).ok());
    assert(stream.accept({Kind::source_end, 2, 0, 3, 0}).ok());
    assert(!stream.accept({Kind::source_end, 2, 0, 3, 0}).ok());
    assert(stream.accept({Kind::complete}).ok());
    assert(!stream.accept({Kind::complete}).ok());
    assert(stream.finished());

    Stream invalid;
    assert(!invalid.accept({Kind::hello, 0, 0, 0, 99}).ok());
    assert(invalid.accept({Kind::hello, 0, 0, 0, 2}).ok());
    assert(!invalid.accept({Kind::blocks, max_sources, 1, 0, 0}).ok());
    assert(!invalid.accept({Kind::blocks, 0, max_addresses + 1, 0, 0}).ok());
    assert(invalid.accept({Kind::error, 0, 0, 0, 1}).ok());
    assert(invalid.finished());
}
