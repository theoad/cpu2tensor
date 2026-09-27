// SPDX-License-Identifier: AGPL-3.0-only
// Hold an arbitrary program at a stable pre-exec boundary for perf attachment.
#include <errno.h>
#include <stdio.h>
#include <unistd.h>

static int write_all(int descriptor, const char *data, size_t size) {
    while (size != 0) {
        const ssize_t written = write(descriptor, data, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        data += (size_t)written;
        size -= (size_t)written;
    }
    return 0;
}

int main(int argc, char **argv) {
    static const char ready[] = "READY\n";
    unsigned char command = 0;
    ssize_t received = 0;

    if (argc < 2) {
        fputs("usage: hardware_exec_gate PROGRAM [ARG ...]\n", stderr);
        return 2;
    }
    if (write_all(STDOUT_FILENO, ready, sizeof(ready) - 1) != 0) {
        perror("cannot publish ready boundary");
        return 3;
    }
    do {
        received = read(STDIN_FILENO, &command, 1);
    } while (received < 0 && errno == EINTR);
    if (received != 1 || command != 'x') {
        fputs("invalid execution gate command\n", stderr);
        return 4;
    }
    execv(argv[1], &argv[1]);
    perror("cannot execute target");
    return 5;
}
