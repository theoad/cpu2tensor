// SPDX-License-Identifier: AGPL-3.0-only
// Gated benign syscall families for host-kernel hardware-trace experiments.

#include <errno.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

static int gate(void) {
    const char ready[] = "READY\n";
    char start = 0;
    if (write(STDOUT_FILENO, ready, sizeof(ready) - 1) != (ssize_t)(sizeof(ready) - 1)) {
        return -1;
    }
    return read(STDIN_FILENO, &start, 1) == 1 ? 0 : -1;
}

static int run_getpid(uint64_t loops, uint64_t *result) {
    for (uint64_t index = 0; index < loops; ++index) {
        *result ^= (uint64_t)syscall(SYS_getpid);
    }
    return 0;
}

static int run_fstat(uint64_t loops, uint64_t *result) {
    struct stat state;
    for (uint64_t index = 0; index < loops; ++index) {
        if (fstat(STDIN_FILENO, &state) != 0) {
            return -1;
        }
        *result ^= (uint64_t)state.st_mode;
    }
    return 0;
}

static int run_futex(uint64_t loops, uint64_t *result) {
    int word = 0;
    for (uint64_t index = 0; index < loops; ++index) {
        long value = syscall(SYS_futex, &word, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
        if (value < 0) {
            return -1;
        }
        *result += (uint64_t)value;
    }
    return 0;
}

static int run_openat(uint64_t loops, uint64_t *result) {
    for (uint64_t index = 0; index < loops; ++index) {
        int descriptor = openat(AT_FDCWD, "/dev/null", O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            return -1;
        }
        *result ^= (uint64_t)descriptor;
        if (close(descriptor) != 0) {
            return -1;
        }
    }
    return 0;
}

static int run_pipe(uint64_t loops, uint64_t *result) {
    const char value = 'x';
    char observed = 0;
    for (uint64_t index = 0; index < loops; ++index) {
        int descriptors[2];
        if (pipe2(descriptors, O_CLOEXEC) != 0 ||
            write(descriptors[1], &value, 1) != 1 ||
            read(descriptors[0], &observed, 1) != 1 ||
            close(descriptors[0]) != 0 || close(descriptors[1]) != 0) {
            return -1;
        }
        *result += (uint64_t)(unsigned char)observed;
    }
    return 0;
}

static int run_mmap(uint64_t loops, uint64_t *result) {
    const size_t bytes = 4096;
    for (uint64_t index = 0; index < loops; ++index) {
        unsigned char *mapping = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) {
            return -1;
        }
        mapping[index % bytes] = (unsigned char)index;
        *result += mapping[index % bytes];
        if (munmap(mapping, bytes) != 0) {
            return -1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s getpid|fstat|futex|openat|pipe|mmap loops\n", argv[0]);
        return 2;
    }
    errno = 0;
    char *end = NULL;
    for (const unsigned char *cursor = (const unsigned char *)argv[2];
         *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            fprintf(stderr, "loops must be a positive integer\n");
            return 2;
        }
    }
    uint64_t loops = strtoull(argv[2], &end, 10);
    if (errno != 0 || end == argv[2] || *end != '\0' || loops == 0) {
        fprintf(stderr, "loops must be a positive integer\n");
        return 2;
    }
    if (gate() != 0) {
        return 3;
    }

    uint64_t result = 0;
    int status = -1;
    if (strcmp(argv[1], "getpid") == 0) {
        status = run_getpid(loops, &result);
    } else if (strcmp(argv[1], "fstat") == 0) {
        status = run_fstat(loops, &result);
    } else if (strcmp(argv[1], "futex") == 0) {
        status = run_futex(loops, &result);
    } else if (strcmp(argv[1], "openat") == 0) {
        status = run_openat(loops, &result);
    } else if (strcmp(argv[1], "pipe") == 0) {
        status = run_pipe(loops, &result);
    } else if (strcmp(argv[1], "mmap") == 0) {
        status = run_mmap(loops, &result);
    }
    if (status != 0) {
        perror("hardware kernel workload");
        return 4;
    }
    printf("%llu\n", (unsigned long long)result);
    return 0;
}
