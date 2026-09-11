// SPDX-License-Identifier: AGPL-3.0-only
#include <unistd.h>

int main(void)
{
    char first[2];
    char second[2];
    if (read(STDIN_FILENO, first, sizeof(first)) != 2) return 2;
    if (read(STDIN_FILENO, second, sizeof(second)) != 2) return 2;
    return first[0] == 'a' && first[1] == '\n' &&
           second[0] == 'b' && second[1] == '\n' ? 0 : 2;
}
