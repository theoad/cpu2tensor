// SPDX-License-Identifier: AGPL-3.0-only
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int main(void)
{
    uint64_t sum = 0;
    uint64_t count = 0;
    unsigned char bytes[4096];
    size_t read_count;
    while ((read_count = fread(bytes, 1, sizeof(bytes), stdin)) != 0) {
        for (size_t index = 0; index < read_count; ++index) {
            sum += bytes[index];
        }
        count += read_count;
    }
    if (ferror(stdin)) {
        fputs("checksum: cannot read input\n", stderr);
        return 1;
    }
    if (printf("bytes=%" PRIu64 " sum=%" PRIu64 "\n", count, sum) < 0) {
        return 1;
    }
    return fflush(stdout) == 0 ? 0 : 1;
}
