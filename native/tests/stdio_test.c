// SPDX-License-Identifier: AGPL-3.0-only
#include <unistd.h>
#include <string.h>
#include <sys/syscall.h>
#include <stdint.h>
#include <signal.h>

static void continued(int number) { (void)number; }

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    if (strcmp(argv[1], "twice") == 0) {
        char data[2];
        if (read(0, data, 2) != 2) return 2;
        if (read(0, data, 2) != 2) return 2;
        return 0;
    }
    if (strcmp(argv[1], "short") == 0) {
        // Invalid destination: leave the delivered input pending, then retry.
        (void)syscall(SYS_read, 0, (void*)(uintptr_t)1, 2);
        char data[2];
        return read(0, data, 2) == 2 ? 0 : 2;
    }
    if (strcmp(argv[1], "sigcont") == 0) {
        return signal(SIGCONT, continued) == SIG_ERR ? 2 : 0;
    }
    if (strcmp(argv[1], "close") == 0) return close(0);
    return 2;
}
