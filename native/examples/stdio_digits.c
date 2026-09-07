// SPDX-License-Identifier: AGPL-3.0-only
#include "digit_paths.h"
#include <errno.h>
#include <sys/random.h>
#include <unistd.h>

int main(void) {
    unsigned char cue;
    // Reject the remainder so every digit has equal probability.
    do {
        ssize_t count;
        do { count = getrandom(&cue, 1, 0); } while (count < 0 && errno == EINTR);
        if (count != 1) return 2;
    } while (cue >= 250);
    cue %= 10;
    visit_digit(cue);
    char action[2];
    const ssize_t count = read(STDIN_FILENO, action, sizeof(action));
    if (count != 2 || action[1] != '\n' || action[0] < '0' || action[0] > '9') return 2;
    return action[0] == (char)('0' + cue) ? 0 : 1;
}
