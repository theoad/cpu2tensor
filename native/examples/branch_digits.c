// SPDX-License-Identifier: AGPL-3.0-only
#include "digit_paths.h"
#include <stdio.h>

int main(void) {
    char line[18];
    unsigned samples = 0;
    while (fgets(line, sizeof(line), stdin) != NULL) {
        unsigned index = 0;
        for (; index < 16 && line[index] >= '0' && line[index] <= '9'; ++index) {}
        if (index != 16 || line[16] != '\n') {
            fputs("Expected exactly sixteen decimal digits per line\n", stderr);
            return 2;
        }
        for (index = 0; index < 16; ++index) visit_digit((unsigned)(line[index] - '0'));
        sample_end();
        ++samples;
    }
    if (ferror(stdin)) return 2;
    printf("samples=%u\n", samples);
    return 0;
}
