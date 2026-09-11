// SPDX-License-Identifier: AGPL-3.0-only
#include "kernel.hpp"
#include "kernel_protocol.hpp"
#include <cpu2tensor/trace.hpp>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <poll.h>
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

static void read_all(int fd, void* output, size_t size) {
    auto* bytes = static_cast<uint8_t*>(output);
    while (size != 0) {
        const auto count = read(fd, bytes, size);
        if (count < 0 && errno == EINTR) continue;
        assert(count > 0);
        bytes += count;
        size -= static_cast<size_t>(count);
    }
}

static void answer_qmp_command(int qmp) {
    Lines lines;
    while (lines.line_size() == 0) assert(lines.read_from(qmp).ok() && !lines.eof);
    const Json command(lines.bytes, lines.line_size());
    assert(command.value != nullptr);
    char response[64];
    const int size = std::snprintf(response, sizeof(response), "{\"return\":{},\"id\":%d}\n",
                                   json_object_get_int(command.get("id")));
    write_all(qmp, response, static_cast<size_t>(size));
}

static KernelOptions test_options(const char* self, char** arguments,
                                  bool interactive = true, int maximum_ms = 0) {
    return {self, "/unused-plugin", "none", "off", "off", "auto",
            nullptr, nullptr, nullptr, nullptr, nullptr, "none", "on", 4096,
            "legacy", "pipe", 2000, maximum_ms, interactive, arguments};
}

// A protocol fixture, not QEMU: negotiate startup, then send one chosen serial
// record. Remaining alive lets the test verify the worker owns child cleanup.
static int producer(int argc, char** argv) {
    const char* control_field = std::strstr(argv[2], ",control=");
    const bool interactive = std::strstr(argv[2], ",kernel=on,") != nullptr;
    assert(interactive == (control_field != nullptr));
    const int trace = std::atoi(std::strstr(argv[2], ",fd=") + 4);
    const int control = interactive ?
        std::atoi(control_field + std::strlen(",control=")) : -1;
    const char* record = argv[argc - 1];
    int console = -1, serial = -1, qmp = -1;
    int console_index = -1, serial_index = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::strstr(argv[i], "socket,id=c2tconsole,fd=") == argv[i])
            console = std::atoi(std::strstr(argv[i], ",fd=") + 4);
        if (std::strstr(argv[i], "socket,id=c2tserial,fd=") == argv[i])
            serial = std::atoi(std::strstr(argv[i], ",fd=") + 4);
        if (std::strstr(argv[i], "socket,id=c2tqmp,fd=") == argv[i])
            qmp = std::atoi(std::strstr(argv[i], ",fd=") + 4);
        if (std::strcmp(argv[i], "chardev:c2tconsole") == 0) console_index = i;
        if (std::strcmp(argv[i], "chardev:c2tserial") == 0) serial_index = i;
    }
    assert(console >= 0 && serial >= 0 && qmp >= 0);
    assert(console_index >= 0 && console_index < serial_index);
    std::fprintf(stderr, "fixture-child:%d\n", getpid());
    constexpr char diagnostic[] = "C2T {\"event\":\"console-only-diagnostic\"}\n";
    write_all(console, diagnostic, sizeof(diagnostic) - 1);
    uint8_t header[header_bytes];
    encode_header(header, {Kind::hello, 0, 0, 0,
                           2 | feature_system | (interactive ? feature_kernel : 0)});
    write_all(trace, reinterpret_cast<const char*>(header), sizeof(header));
    constexpr char greeting[] = "{\"QMP\":{}}\n";
    write_all(qmp, greeting, sizeof(greeting) - 1);
    answer_qmp_command(qmp);
    if (std::strcmp(record, "observation") == 0) {
        assert(!interactive);
        constexpr char events[] =
            "C2T {\"event\":\"start\",\"mode\":\"observe\"}\n"
            "C2T {\"event\":\"result\",\"step\":0,\"action\":\"getpid\",\"value\":1}\n"
            "C2T {\"event\":\"complete\",\"steps\":1,\"ok\":true}\n";
        write_all(serial, events, sizeof(events) - 1);
        encode_header(header, {Kind::complete});
        write_all(trace, reinterpret_cast<const char*>(header), sizeof(header));
        shutdown(console, SHUT_WR);
        shutdown(serial, SHUT_WR);
        shutdown(qmp, SHUT_WR);
        return 0;
    }
    if (std::strcmp(record, "observation-deadline") == 0) {
        assert(!interactive);
        constexpr char start[] =
            "C2T {\"event\":\"start\",\"mode\":\"observe\"}\n";
        write_all(serial, start, sizeof(start) - 1);
        for (;;) pause();
    }
    answer_qmp_command(qmp);
    if (std::strcmp(record, "action-routing") == 0) {
        constexpr char events[] =
            "C2T {\"event\":\"start\",\"mode\":\"interactive\"}\n"
            "C2T {\"event\":\"ready\",\"step\":0}\n";
        write_all(serial, events, sizeof(events) - 1);
        answer_qmp_command(qmp);
        char drain;
        read_all(control, &drain, 1);
        assert(drain == 'D');
        encode_header(header, {Kind::kernel_request, 0, 0, 0, 127});
        write_all(trace, reinterpret_cast<const char*>(header), sizeof(header));
        pollfd channels[] = {{serial, POLLIN, 0}, {console, POLLIN, 0}};
        assert(poll(channels, 2, 2000) == 1);
        assert(channels[0].revents & POLLIN);
        assert(!(channels[1].revents & POLLIN));
        char action[7];
        read_all(serial, action, sizeof(action));
        assert(std::memcmp(action, "getpid\n", sizeof(action)) == 0);
        std::fprintf(stderr, "fixture-action:on-adapter\n");
        constexpr char result[] =
            "C2T {\"event\":\"result\",\"step\":0,\"action\":\"getpid\",\"value\":1}\n";
        write_all(serial, result, sizeof(result) - 1);
        for (;;) pause();
    }
    if (std::strstr(record, "input: ImExPS/2 Generic Explorer Mouse") != nullptr) {
        constexpr char start[] =
            "C2T {\"event\":\"start\",\"mode\":\"interactive\"}\n";
        write_all(serial, start, sizeof(start) - 1);
    }
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

static void check_failure(const char* self, const char* record, const char* reason,
                          bool interactive = true) {
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
        const KernelOptions options = test_options(self, arguments, interactive);
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

static void check_recovered_printk_suffix(const char* self) {
    constexpr char payload[] =
        "{\"event\":\"result\",\"step\":2,\"action\":\"getpid\",\"value\":1}";
    constexpr char record[] =
        "C2T {\"event\":\"result\",\"step\":2,\"action\":\"getpid\",\"value\":1}"
        "[    6.795768] input: ImExPS/2 Generic Explorer Mouse as "
        "/devices/platform/i8042/serio1/input/input3\n";
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
        const KernelOptions options = test_options(self, arguments);
        const auto result = run_kernel(options, listener);
        if (!result.ok()) std::fprintf(stderr, "worker-error:%s\n", result.error());
        _exit(result.ok() ? 0 : 1);
    }
    close(logs[1]);
    close(listener);
    const int client = socket(AF_INET, SOCK_STREAM, 0);
    assert(client >= 0);
    assert(connect(client, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);

    uint8_t header_bytes_buffer[header_bytes];
    read_all(client, header_bytes_buffer, sizeof(header_bytes_buffer));
    auto decoded = decode_header(header_bytes_buffer, sizeof(header_bytes_buffer));
    assert(decoded.ok() && decoded.value().kind == Kind::hello);
    assert(payload_size(decoded.value()) == 0);
    read_all(client, header_bytes_buffer, sizeof(header_bytes_buffer));
    decoded = decode_header(header_bytes_buffer, sizeof(header_bytes_buffer));
    assert(decoded.ok() && decoded.value().kind == Kind::guest_event);
    char start[128]{};
    assert(payload_size(decoded.value()) < sizeof(start));
    read_all(client, start, payload_size(decoded.value()));
    assert(std::strstr(start, "\"event\":\"start\"") != nullptr);
    read_all(client, header_bytes_buffer, sizeof(header_bytes_buffer));
    decoded = decode_header(header_bytes_buffer, sizeof(header_bytes_buffer));
    assert(decoded.ok() && decoded.value().kind == Kind::guest_event);
    assert(payload_size(decoded.value()) == sizeof(payload) - 1);
    char forwarded[sizeof(payload)]{};
    read_all(client, forwarded, sizeof(payload) - 1);
    assert(std::strcmp(forwarded, payload) == 0);
    close(client);

    char output[8192]{};
    size_t used = 0;
    ssize_t count;
    while ((count = read(logs[0], output + used, sizeof(output) - 1 - used)) > 0) {
        used += static_cast<size_t>(count);
        assert(used < sizeof(output) - 1);
    }
    assert(count == 0);
    close(logs[0]);
    int status;
    assert(waitpid(worker, &status, 0) == worker);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(std::strstr(output, "C2T {\"event\":\"console-only-diagnostic\"}") != nullptr);
    assert(std::strstr(output, "guest-event suffix class=kernel-printk") != nullptr);
    assert(std::strstr(output, "[    6.795768] input: ImExPS/2 Generic Explorer Mouse") !=
           nullptr);
    assert(std::strstr(output, payload) == nullptr);
    const char* child_line = std::strstr(output, "fixture-child:");
    assert(child_line != nullptr);
    const pid_t child = std::atoi(child_line + std::strlen("fixture-child:"));
    assert(child > 0);
    assert(kill(child, 0) == -1 && errno == ESRCH);
}

static void check_observation_protocol(const char* self) {
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
        char* arguments[] = {const_cast<char*>("observation"), nullptr};
        const KernelOptions options = test_options(self, arguments, false);
        const auto result = run_kernel(options, listener);
        if (!result.ok()) std::fprintf(stderr, "worker-error:%s\n", result.error());
        _exit(result.ok() ? 0 : 1);
    }
    close(logs[1]);
    close(listener);
    const int client = socket(AF_INET, SOCK_STREAM, 0);
    assert(client >= 0);
    assert(connect(client, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);

    uint8_t encoded[header_bytes];
    read_all(client, encoded, sizeof(encoded));
    auto decoded = decode_header(encoded, sizeof(encoded));
    assert(decoded.ok() && decoded.value().kind == Kind::hello);
    assert(decoded.value().detail & feature_system);
    assert(!(decoded.value().detail & feature_kernel));
    read_all(client, encoded, sizeof(encoded));
    decoded = decode_header(encoded, sizeof(encoded));
    // Observation protocol records are validated by the worker and stay out of
    // the tensor stream consumed by Pool.
    assert(decoded.ok() && decoded.value().kind == Kind::complete);
    char extra;
    assert(read(client, &extra, 1) == 0);
    close(client);

    char output[8192]{};
    size_t used = 0;
    ssize_t count;
    while ((count = read(logs[0], output + used, sizeof(output) - 1 - used)) > 0) {
        used += static_cast<size_t>(count);
        assert(used < sizeof(output) - 1);
    }
    close(logs[0]);
    int status;
    assert(waitpid(worker, &status, 0) == worker);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(std::strstr(output, "C2T {\"event\":\"console-only-diagnostic\"}") != nullptr);
    assert(std::strstr(output, "C2T {\"event\":\"complete\"") != nullptr);
    assert(std::strstr(output, "worker-error:") == nullptr);
    const char* child_line = std::strstr(output, "fixture-child:");
    assert(child_line != nullptr);
    const pid_t child = std::atoi(child_line + std::strlen("fixture-child:"));
    assert(child > 0);
    assert(kill(child, 0) == -1 && errno == ESRCH);
}

static void check_observation_deadline_report(const char* self) {
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
        char* arguments[] = {const_cast<char*>("observation-deadline"), nullptr};
        const auto result = run_kernel(test_options(self, arguments, false, 100), listener);
        if (!result.ok()) std::fprintf(stderr, "worker-error:%s\n", result.error());
        _exit(result.ok() ? 0 : 1);
    }
    close(logs[1]);
    close(listener);
    const int client = socket(AF_INET, SOCK_STREAM, 0);
    assert(client >= 0);
    assert(connect(client, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);

    uint8_t encoded[header_bytes];
    read_all(client, encoded, sizeof(encoded));
    auto decoded = decode_header(encoded, sizeof(encoded));
    assert(decoded.ok() && decoded.value().kind == Kind::hello);
    read_all(client, encoded, sizeof(encoded));
    decoded = decode_header(encoded, sizeof(encoded));
    assert(decoded.ok() && decoded.value().kind == Kind::terminal_report);
    const auto flags = static_cast<uint8_t>(decoded.value().detail >> 16);
    assert((flags & terminal_hello) != 0);
    assert((flags & terminal_data) == 0);
    char extra;
    assert(read(client, &extra, 1) == 0);
    close(client);

    char output[8192]{};
    size_t used = 0;
    ssize_t count;
    while ((count = read(logs[0], output + used, sizeof(output) - 1 - used)) > 0) {
        used += static_cast<size_t>(count);
        assert(used < sizeof(output) - 1);
    }
    close(logs[0]);
    int status;
    assert(waitpid(worker, &status, 0) == worker);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    assert(std::strstr(output, "Target exceeded --max-run-ms deadline") != nullptr);
    const char* child_line = std::strstr(output, "fixture-child:");
    assert(child_line != nullptr);
    const pid_t child = std::atoi(child_line + std::strlen("fixture-child:"));
    assert(child > 0);
    assert(kill(child, 0) == -1 && errno == ESRCH);
}

static void check_action_uses_adapter_channel(const char* self) {
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
        char* arguments[] = {const_cast<char*>("action-routing"), nullptr};
        const KernelOptions options = test_options(self, arguments);
        const auto result = run_kernel(options, listener);
        if (!result.ok()) std::fprintf(stderr, "worker-error:%s\n", result.error());
        _exit(result.ok() ? 0 : 1);
    }
    close(logs[1]);
    close(listener);
    const int client = socket(AF_INET, SOCK_STREAM, 0);
    assert(client >= 0);
    assert(connect(client, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);

    bool action_sent = false;
    bool result_seen = false;
    while (!result_seen) {
        uint8_t encoded[header_bytes];
        read_all(client, encoded, sizeof(encoded));
        const auto decoded = decode_header(encoded, sizeof(encoded));
        assert(decoded.ok());
        const auto size = payload_size(decoded.value());
        assert(size <= event_capacity);
        char payload[event_capacity + 1]{};
        if (size != 0) read_all(client, payload, size);
        if (decoded.value().kind == Kind::kernel_request) {
            assert(!action_sent && decoded.value().detail == 127);
            uint8_t action[4 + 7];
            store_u32(action, 7);
            std::memcpy(action + 4, "getpid\n", 7);
            write_all(client, reinterpret_cast<const char*>(action), sizeof(action));
            action_sent = true;
        } else if (decoded.value().kind == Kind::guest_event &&
                   std::strstr(payload, "\"event\":\"result\"") != nullptr) {
            assert(std::strstr(payload, "\"action\":\"getpid\"") != nullptr);
            result_seen = true;
        }
    }
    assert(action_sent);
    close(client);

    char output[8192]{};
    size_t used = 0;
    ssize_t count;
    while ((count = read(logs[0], output + used, sizeof(output) - 1 - used)) > 0) {
        used += static_cast<size_t>(count);
        assert(used < sizeof(output) - 1);
    }
    assert(count == 0);
    close(logs[0]);
    int status;
    assert(waitpid(worker, &status, 0) == worker);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(std::strstr(output, "fixture-action:on-adapter") != nullptr);
    const char* child_line = std::strstr(output, "fixture-child:");
    assert(child_line != nullptr);
    const pid_t child = std::atoi(child_line + std::strlen("fixture-child:"));
    assert(child > 0);
    assert(kill(child, 0) == -1 && errno == ESRCH);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "-plugin") == 0) return producer(argc, argv);
    alarm(30);
    check_recovered_printk_suffix(argv[0]);
    check_observation_protocol(argv[0]);
    check_observation_deadline_report(argv[0]);
    check_failure(argv[0], "plain text on private serial\n",
                  "reason=missing-event-prefix", false);
    check_failure(argv[0], "C2T {broken}\n", "reason=malformed-json", false);
    check_action_uses_adapter_channel(argv[0]);
    check_failure(argv[0], "C2T {broken}\n", "reason=malformed-json");
    check_failure(argv[0], "C2T {\n", "reason=incomplete-json");
    check_failure(argv[0], "C2T []\n", "reason=non-object-json");
    check_failure(argv[0], "C2T {}\n", "reason=missing-event-field");
    check_failure(argv[0], "C2T {\"event\":null}\n", "reason=non-string-event-field");
    check_failure(argv[0], "C2T {\"event\":\"ready\\u0000extra\"}\n", "reason=embedded-null-event-field");
    check_failure(argv[0], "C2T {\"event\":\"unknown\"}\n", "reason=unknown-event-name");
    check_failure(argv[0], "C2T {\"event\":\"ready\"}\n", "reason=unexpected-ready");
    check_failure(argv[0], "C2T {\"event\":\"result\"}garbage\n",
                  "reason=unexpected-event-suffix");
    check_failure(argv[0],
                  "C2T {\"event\":\"result\"}C2T {\"event\":\"ready\"}\n",
                  "reason=unexpected-event-suffix");
    check_failure(argv[0], "oversize", "reason=payload-size");
    check_failure(argv[0], "unterminated", "reason=unterminated-serial-line");
}
