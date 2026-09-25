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

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s FAMILY LOOPS\n", argv[0]);
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
    } else if (strcmp(argv[1], "eventfd") == 0) {
        status = run_eventfd(loops, &result);
    } else if (strcmp(argv[1], "epoll") == 0) {
        status = run_epoll(loops, &result);
    } else if (strcmp(argv[1], "socketpair") == 0) {
        status = run_socketpair(loops, &result);
    } else if (strcmp(argv[1], "getrandom") == 0) {
        status = run_getrandom(loops, &result);
    } else if (strcmp(argv[1], "memfd") == 0) {
        status = run_memfd(loops, &result);
    } else if (strcmp(argv[1], "ioctl") == 0) {
        status = run_ioctl(loops, &result);
    } else if (strcmp(argv[1], "dup") == 0) {
        status = run_dup(loops, &result);
    } else if (strcmp(argv[1], "yield") == 0) {
        status = run_yield(loops, &result);
    } else if (strcmp(argv[1], "uname") == 0) {
        status = run_uname(loops, &result);
    } else if (strcmp(argv[1], "readlink") == 0) {
        status = run_readlink(loops, &result);
    } else if (strcmp(argv[1], "fork") == 0) {
        status = run_fork(loops, &result);
    }
    if (status != 0) {
        perror("hardware kernel workload");
        return 4;
    }
    printf("%llu\n", (unsigned long long)result);
    return 0;
}
