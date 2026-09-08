// SPDX-License-Identifier: AGPL-3.0-only
#include "kernel.hpp"
#include "kernel_protocol.hpp"
#include <cpu2tensor/trace.hpp>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace cpu2tensor;
using namespace cpu2tensor::kernel_protocol;

static void write_all(int fd, const char* bytes, size_t size) {
    while (size != 0) {
        const auto count = write(fd, bytes, size);
        if (count < 0 && errno == EINTR) continue;
        assert(count > 0);
        bytes += count;
        size -= static_cast<size_t>(count);
    }
}

// A protocol fixture, not QEMU: negotiate startup, then send one bad serial
// record. Remaining alive lets the test verify the worker owns child cleanup.
static int producer(int argc, char** argv) {
    const int trace = std::atoi(std::strstr(argv[2], ",fd=") + 4);
    int serial = -1, qmp = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::strstr(argv[i], "socket,id=c2tserial,fd=") == argv[i])
            serial = std::atoi(std::strstr(argv[i], ",fd=") + 4);
        if (std::strstr(argv[i], "socket,id=c2tqmp,fd=") == argv[i])
            qmp = std::atoi(std::strstr(argv[i], ",fd=") + 4);
    }
    assert(serial >= 0 && qmp >= 0);
    std::fprintf(stderr, "fixture-child:%d\n", getpid());
    uint8_t header[header_bytes];
    encode_header(header, {Kind::hello, 0, 0, 0, 2 | feature_system | feature_kernel});
    write_all(trace, reinterpret_cast<const char*>(header), sizeof(header));
    constexpr char greeting[] = "{\"QMP\":{}}\n";
    write_all(qmp, greeting, sizeof(greeting) - 1);
    for (int i = 0; i < 2; ++i) {
        Lines lines;
        while (lines.line_size() == 0) assert(lines.read_from(qmp).ok() && !lines.eof);
        const Json command(lines.bytes, lines.line_size());
        assert(command.value != nullptr);
        char response[64];
        const int size = std::snprintf(response, sizeof(response), "{\"return\":{},\"id\":%d}\n",
                                       json_object_get_int(command.get("id")));
        write_all(qmp, response, static_cast<size_t>(size));
    }
    const char* record = argv[argc - 1];
    if (std::strcmp(record, "oversize") == 0) {
        char bytes[event_capacity + 16];
        std::memset(bytes, 'x', sizeof(bytes));
        std::memcpy(bytes, "C2T ", 4);
        bytes[sizeof(bytes) - 1] = '\n';
        write_all(serial, bytes, sizeof(bytes));
    } else if (std::strcmp(record, "unterminated") == 0) {
        write_all(serial, "C2T {", 5);
        shutdown(serial, SHUT_WR);
    } else write_all(serial, record, std::strlen(record));
    for (;;) pause();
}

static void check_failure(const char* self, const char* record, const char* reason) {
    int logs[2];
    assert(pipe(logs) == 0);
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    assert(listener >= 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    assert(bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
    assert(listen(listener, 1) == 0);
    socklen_t length = sizeof(address);
    assert(getsockname(listener, reinterpret_cast<sockaddr*>(&address), &length) == 0);
    const pid_t worker = fork();
    assert(worker >= 0);
    if (worker == 0) {
        close(logs[0]);
        assert(dup2(logs[1], STDERR_FILENO) >= 0);
        close(logs[1]);
        char* arguments[] = {const_cast<char*>(record), nullptr};
        const KernelOptions options{self, "/unused-plugin", "none", "off", "off", "auto", nullptr,
                                    "legacy", "pipe", 2000, arguments};
        const auto result = run_kernel(options, listener);
        if (!result.ok()) std::fprintf(stderr, "worker-error:%s\n", result.error());
        _exit(result.ok() ? 0 : 1);
    }
    close(logs[1]);
    close(listener);
    const int client = socket(AF_INET, SOCK_STREAM, 0);
    assert(client >= 0);
    assert(connect(client, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
    char bytes[4096];
    ssize_t count;
    do { count = read(client, bytes, sizeof(bytes)); } while (count > 0 || (count < 0 && errno == EINTR));
    assert(count == 0);
    close(client);
    char output[8192]{};
    size_t used = 0;
    while ((count = read(logs[0], output + used, sizeof(output) - 1 - used)) > 0) {
        used += static_cast<size_t>(count);
        assert(used < sizeof(output) - 1);
    }
    assert(count == 0);
    close(logs[0]);
    int status;
    assert(waitpid(worker, &status, 0) == worker);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    if (std::strstr(output, reason) == nullptr) std::fprintf(stderr, "%s", output);
    assert(std::strstr(output, reason) != nullptr);
    assert(std::strstr(output, "payload=\"") != nullptr);
    assert(std::strstr(output, "worker-error:") != nullptr);
    if (std::strcmp(record, "unterminated") != 0)
        assert(std::strstr(output, "guest-event state hello=1") != nullptr);
    const char* child_line = std::strstr(output, "fixture-child:");
    assert(child_line != nullptr);
    const pid_t child = std::atoi(child_line + std::strlen("fixture-child:"));
    assert(child > 0);
    assert(kill(child, 0) == -1 && errno == ESRCH);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "-plugin") == 0) return producer(argc, argv);
    alarm(30);
    check_failure(argv[0], "C2T {broken}\n", "reason=malformed-json");
    check_failure(argv[0], "C2T {\n", "reason=incomplete-json");
    check_failure(argv[0], "C2T []\n", "reason=non-object-json");
    check_failure(argv[0], "C2T {}\n", "reason=missing-event-field");
    check_failure(argv[0], "C2T {\"event\":null}\n", "reason=non-string-event-field");
    check_failure(argv[0], "C2T {\"event\":\"ready\\u0000extra\"}\n", "reason=embedded-null-event-field");
    check_failure(argv[0], "C2T {\"event\":\"unknown\"}\n", "reason=unknown-event-name");
    check_failure(argv[0], "C2T {\"event\":\"ready\"}\n", "reason=unexpected-ready");
    check_failure(argv[0], "oversize", "reason=payload-size");
    check_failure(argv[0], "unterminated", "reason=unterminated-serial-line");
}
