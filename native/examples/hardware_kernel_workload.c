// SPDX-License-Identifier: AGPL-3.0-only
// Gated benign syscall families for host-kernel hardware-trace experiments.

#include <errno.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

static int gate(void) {
    const char ready[] = "READY\n";
    char start = 0;
    if (write(STDOUT_FILENO, ready, sizeof(ready) - 1) != (ssize_t)(sizeof(ready) - 1)) {
        return -1;
    }
    ssize_t count = read(STDIN_FILENO, &start, 1);
    return count == 1 ? 0 : count == 0 ? 1 : -1;
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

static int run_eventfd(uint64_t loops, uint64_t *result) {
    for (uint64_t index = 0; index < loops; ++index) {
        int descriptor = eventfd(0, EFD_CLOEXEC);
        uint64_t value = index + 1;
        uint64_t observed = 0;
        if (descriptor < 0 || write(descriptor, &value, sizeof(value)) != sizeof(value) ||
            read(descriptor, &observed, sizeof(observed)) != sizeof(observed) ||
            close(descriptor) != 0) {
            return -1;
        }
        *result ^= observed;
    }
    return 0;
}

static int run_epoll(uint64_t loops, uint64_t *result) {
    for (uint64_t index = 0; index < loops; ++index) {
        int event_descriptor = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        int epoll_descriptor = epoll_create1(EPOLL_CLOEXEC);
        struct epoll_event registration = {.events = EPOLLIN, .data.u64 = index};
        struct epoll_event observed = {0};
        uint64_t value = 1;
        uint64_t consumed = 0;
        if (event_descriptor < 0 || epoll_descriptor < 0 ||
            epoll_ctl(epoll_descriptor, EPOLL_CTL_ADD, event_descriptor, &registration) != 0 ||
            write(event_descriptor, &value, sizeof(value)) != sizeof(value) ||
            epoll_wait(epoll_descriptor, &observed, 1, 0) != 1 ||
            read(event_descriptor, &consumed, sizeof(consumed)) != sizeof(consumed) ||
            close(epoll_descriptor) != 0 || close(event_descriptor) != 0) {
            return -1;
        }
        *result ^= observed.data.u64 + consumed;
    }
    return 0;
}

static int run_socketpair(uint64_t loops, uint64_t *result) {
    const char value = 's';
    char observed = 0;
    for (uint64_t index = 0; index < loops; ++index) {
        int descriptors[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, descriptors) != 0 ||
            send(descriptors[0], &value, 1, 0) != 1 ||
            recv(descriptors[1], &observed, 1, 0) != 1 ||
            close(descriptors[0]) != 0 || close(descriptors[1]) != 0) {
            return -1;
        }
        *result += (uint64_t)(unsigned char)observed;
    }
    return 0;
}

static int run_getrandom(uint64_t loops, uint64_t *result) {
    uint64_t value = 0;
    for (uint64_t index = 0; index < loops; ++index) {
        if (getrandom(&value, sizeof(value), 0) != sizeof(value)) {
            return -1;
        }
        *result ^= value;
    }
    return 0;
}

static int run_memfd(uint64_t loops, uint64_t *result) {
    const size_t bytes = 4096;
    for (uint64_t index = 0; index < loops; ++index) {
        int descriptor = (int)syscall(SYS_memfd_create, "cpu2tensor", MFD_CLOEXEC);
        if (descriptor < 0 || ftruncate(descriptor, (off_t)bytes) != 0) {
            return -1;
        }
        unsigned char *mapping = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                      MAP_SHARED, descriptor, 0);
        if (mapping == MAP_FAILED) {
            close(descriptor);
            return -1;
        }
        mapping[index % bytes] = (unsigned char)index;
        *result += mapping[index % bytes];
        if (munmap(mapping, bytes) != 0 || close(descriptor) != 0) {
            return -1;
        }
    }
    return 0;
}

static int run_ioctl(uint64_t loops, uint64_t *result) {
    const char value = 'i';
    char observed = 0;
    for (uint64_t index = 0; index < loops; ++index) {
        int descriptors[2];
        int available = 0;
        if (pipe2(descriptors, O_CLOEXEC) != 0 ||
            write(descriptors[1], &value, 1) != 1 ||
            ioctl(descriptors[0], FIONREAD, &available) != 0 || available != 1 ||
            read(descriptors[0], &observed, 1) != 1 ||
            close(descriptors[0]) != 0 || close(descriptors[1]) != 0) {
            return -1;
        }
        *result += (uint64_t)available;
    }
    return 0;
}

static int run_dup(uint64_t loops, uint64_t *result) {
    int descriptor = open("/dev/null", O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        return -1;
    }
    for (uint64_t index = 0; index < loops; ++index) {
        int duplicate = fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
        if (duplicate < 0 || close(duplicate) != 0) {
            close(descriptor);
            return -1;
        }
        *result ^= (uint64_t)duplicate;
    }
    return close(descriptor);
}

static int run_yield(uint64_t loops, uint64_t *result) {
    for (uint64_t index = 0; index < loops; ++index) {
        if (sched_yield() != 0) {
            return -1;
        }
        *result += index;
    }
    return 0;
}

static int run_uname(uint64_t loops, uint64_t *result) {
    struct utsname state;
    for (uint64_t index = 0; index < loops; ++index) {
        if (uname(&state) != 0) {
            return -1;
        }
        *result += (uint64_t)(unsigned char)state.release[index % sizeof(state.release)];
    }
    return 0;
}

static int run_readlink(uint64_t loops, uint64_t *result) {
    char path[4096];
    for (uint64_t index = 0; index < loops; ++index) {
        ssize_t length = readlink("/proc/self/exe", path, sizeof(path));
        if (length <= 0) {
            return -1;
        }
        *result += (uint64_t)length;
    }
    return 0;
}

static int run_fork(uint64_t loops, uint64_t *result) {
    for (uint64_t index = 0; index < loops; ++index) {
        pid_t child = fork();
        if (child == 0) {
            _exit((int)(index & 0x7f));
        }
        int status = 0;
        if (child < 0 || waitpid(child, &status, 0) != child || !WIFEXITED(status)) {
            return -1;
        }
        *result += (uint64_t)WEXITSTATUS(status);
    }
    return 0;
}

static int run_family(const char *family, uint64_t loops, uint64_t *result) {
    if (strcmp(family, "getpid") == 0) {
        return run_getpid(loops, result);
    }
    if (strcmp(family, "fstat") == 0) {
        return run_fstat(loops, result);
    }
    if (strcmp(family, "futex") == 0) {
        return run_futex(loops, result);
    }
    if (strcmp(family, "openat") == 0) {
        return run_openat(loops, result);
    }
    if (strcmp(family, "pipe") == 0) {
        return run_pipe(loops, result);
    }
    if (strcmp(family, "mmap") == 0) {
        return run_mmap(loops, result);
    }
    if (strcmp(family, "eventfd") == 0) {
        return run_eventfd(loops, result);
    }
    if (strcmp(family, "epoll") == 0) {
        return run_epoll(loops, result);
    }
    if (strcmp(family, "socketpair") == 0) {
        return run_socketpair(loops, result);
    }
    if (strcmp(family, "getrandom") == 0) {
        return run_getrandom(loops, result);
    }
    if (strcmp(family, "memfd") == 0) {
        return run_memfd(loops, result);
    }
    if (strcmp(family, "ioctl") == 0) {
        return run_ioctl(loops, result);
    }
    if (strcmp(family, "dup") == 0) {
        return run_dup(loops, result);
    }
    if (strcmp(family, "yield") == 0) {
        return run_yield(loops, result);
    }
    if (strcmp(family, "uname") == 0) {
        return run_uname(loops, result);
    }
    if (strcmp(family, "readlink") == 0) {
        return run_readlink(loops, result);
    }
    if (strcmp(family, "fork") == 0) {
        return run_fork(loops, result);
    }
    errno = EINVAL;
    return -1;
}

static int parse_loops(const char *text, uint64_t *loops) {
    errno = 0;
    char *end = NULL;
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            return -1;
        }
    }
    *loops = strtoull(text, &end, 10);
    return errno == 0 && end != text && *end == '\0' && *loops != 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    int server = argc == 4 && strcmp(argv[1], "--server") == 0;
    if ((!server && argc != 3) || (server && argc != 4)) {
        fprintf(stderr, "usage: %s [--server] FAMILY LOOPS\n", argv[0]);
        return 2;
    }
    const char *family = argv[server ? 2 : 1];
    uint64_t loops = 0;
    if (parse_loops(argv[server ? 3 : 2], &loops) != 0) {
        fprintf(stderr, "loops must be a positive integer\n");
        return 2;
    }
    do {
        int gated = gate();
        if (server && gated == 1) {
            return 0;
        }
        if (gated != 0) {
            return 3;
        }
        uint64_t result = 0;
        if (run_family(family, loops, &result) != 0) {
            perror("hardware kernel workload");
            return 4;
        }
        if (server) {
            if (printf("DONE %llu\n", (unsigned long long)result) < 0 ||
                fflush(stdout) != 0) {
                return 5;
            }
        } else {
            printf("%llu\n", (unsigned long long)result);
        }
    } while (server);
    return 0;
}
