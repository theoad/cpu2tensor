// SPDX-License-Identifier: AGPL-3.0-only
#include "kernel.hpp"
#include "kernel_protocol.hpp"
#include <cpu2tensor/trace.hpp>
#include <json-c/json.h>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

namespace cpu2tensor {
namespace {
using namespace kernel_protocol;
constexpr size_t argument_capacity = 256;
constexpr int socket_capacity = 65536;

int64_t now_ms() {
    timespec time{};
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * int64_t{1000} + time.tv_nsec / 1000000;
}
class Fd final {
public:
    explicit Fd(int value = -1) : value(value) {}
    ~Fd() { if (value >= 0) close(value); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    int value;
};
class Process final {
public:
    explicit Process(pid_t pid) : pid(pid) {}
    ~Process() {
        if (pid > 0) {
            kill(-pid, SIGKILL);
            kill(pid, SIGKILL);
            while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
        }
    }
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;
    pid_t pid;
};
Result<Done> nonblocking(int fd) {
    if (fcntl(fd, F_SETFL, O_NONBLOCK) < 0)
        return Result<Done>::failure("Cannot make kernel transport nonblocking");
    return Result<Done>::success({});
}
Result<bool> send_bytes(int fd, const uint8_t* bytes, size_t size, int timeout) {
    const auto deadline = now_ms() + timeout;
    size_t sent = 0;
    while (sent < size) {
        const auto count = send(fd, bytes + sent, size - sent, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (count > 0) { sent += static_cast<size_t>(count); continue; }
        if (count < 0 && errno == EINTR) continue;
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            const auto remaining = deadline - now_ms();
            if (remaining <= 0) return Result<bool>::failure("Kernel transport write timed out");
            pollfd descriptor{fd, POLLOUT, 0};
            if (poll(&descriptor, 1, static_cast<int>(remaining)) >= 0) continue;
            if (errno == EINTR) continue;
        }
        if (count == 0 || errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN)
            return Result<bool>::success(true);
        return Result<bool>::failure("Kernel transport write failed");
    }
    return Result<bool>::success(false);
}
Result<Done> qmp_command(int fd, const char* name, int id, int timeout) {
    char bytes[128];
    const int size = std::snprintf(bytes, sizeof(bytes), "{\"execute\":\"%s\",\"id\":%d}\n", name, id);
    const auto sent = send_bytes(fd, reinterpret_cast<const uint8_t*>(bytes), static_cast<size_t>(size), timeout);
    if (!sent.ok()) return Result<Done>::failure(sent.error());
    if (sent.value()) return Result<Done>::failure("QMP channel closed while sending a command");
    return Result<Done>::success({});
}
Result<Done> request_drain(int fd) {
    sigset_t blocked, previous;
    sigemptyset(&blocked);
    sigaddset(&blocked, SIGPIPE);
    if (pthread_sigmask(SIG_BLOCK, &blocked, &previous) != 0)
        return Result<Done>::failure("Cannot protect kernel control write");
    const char command = 'D';
    ssize_t written;
    do { written = write(fd, &command, 1); } while (written < 0 && errno == EINTR);
    if (written < 0 && errno == EPIPE && !sigismember(&previous, SIGPIPE)) {
        // Consume only this thread's newly generated broken-pipe signal before
        // restoring its mask, so normal error cleanup can reap QEMU.
        timespec immediate{};
        while (sigtimedwait(&blocked, nullptr, &immediate) < 0 && errno == EINTR) {}
    }
    if (pthread_sigmask(SIG_SETMASK, &previous, nullptr) != 0 || written != 1)
        return Result<Done>::failure("Cannot request kernel trace drain");
    return Result<Done>::success({});
}
Result<bool> action(int client, int serial, int timeout) {
    const auto deadline = now_ms() + timeout;
    uint8_t bytes[4 + 127];
    size_t used = 0;
    size_t needed = 4;
    while (used < needed) {
        const auto remaining = deadline - now_ms();
        if (remaining <= 0) return Result<bool>::failure("Kernel action timed out");
        pollfd descriptor{client, POLLIN | POLLRDHUP, 0};
        const int ready = poll(&descriptor, 1, static_cast<int>(remaining));
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0) return Result<bool>::failure("Cannot receive kernel action before timeout");
        const auto count = recv(client, bytes + used, needed - used, MSG_DONTWAIT);
        if (count == 0) return Result<bool>::success(true);
        if (count < 0) {
            if (errno == EAGAIN || errno == EINTR) continue;
            if (errno == ECONNRESET || errno == ENOTCONN) return Result<bool>::success(true);
            return Result<bool>::failure("Cannot receive kernel action");
        }
        used += static_cast<size_t>(count);
        if (used == 4 && needed == 4) {
            const auto length = load_u32(bytes);
            if (length < 2 || length > 127) return Result<bool>::failure("Kernel action needs 2..127 bytes");
            needed += length;
        }
    }
    if (bytes[needed - 1] != '\n') return Result<bool>::failure("Kernel command needs a newline");
    for (size_t i = 4; i + 1 < needed; ++i)
        if (bytes[i] < 32 || bytes[i] > 126)
            return Result<bool>::failure("Kernel command must be a single printable ASCII line");
    const auto sent = send_bytes(serial, bytes + 4, needed - 4, timeout);
    if (!sent.ok()) return Result<bool>::failure(sent.error());
    if (sent.value()) return Result<bool>::failure("Guest serial channel closed during action delivery");
    return Result<bool>::success(false);
}
} // namespace

Result<bool> run_kernel(const KernelOptions& options, int listener) {
    // The worker owns QMP and the serial adapter exclusively. Arbitrary machine
    // options still belong to the operator, but cannot replace these channels.
    const char* reserved[] = {"qmp", "monitor", "mon", "serial", "nographic", "daemonize", "S", "incoming", "gdb", "s"};
    for (size_t i = 0; options.arguments[i] != nullptr; ++i) {
        const char* argument = options.arguments[i];
        if (*argument != '-') continue;
        while (*argument == '-') ++argument;
        for (const char* option : reserved)
            if (std::strcmp(argument, option) == 0 ||
                (std::strncmp(argument, option, std::strlen(option)) == 0 && argument[std::strlen(option)] == '='))
                return Result<bool>::failure("Kernel adapter owns QMP/serial/startup; remove conflicting QEMU option");
    }
    int accepted;
    do { accepted = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC); } while (accepted < 0 && errno == EINTR);
    const Fd client(accepted);
    if (client.value < 0) return Result<bool>::failure("Cannot accept kernel consumer");
    setsockopt(client.value, SOL_SOCKET, SO_SNDBUF, &socket_capacity, sizeof(socket_capacity));
    int trace[2];
    if (pipe2(trace, O_CLOEXEC) != 0) return Result<bool>::failure("Cannot create kernel capture pipe");
    const Fd trace_read(trace[0]);
    Fd trace_write(trace[1]);
    int controls[2];
    if (pipe2(controls, O_CLOEXEC) != 0) return Result<bool>::failure("Cannot create kernel drain pipe");
    Fd control_read(controls[0]);
    const Fd control_write(controls[1]);
    int qmp_pair[2];
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, qmp_pair) != 0)
        return Result<bool>::failure("Cannot create QMP channel");
    const Fd qmp(qmp_pair[0]);
    Fd qmp_guest(qmp_pair[1]);
    int serial_pair[2];
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, serial_pair) != 0)
        return Result<bool>::failure("Cannot create guest serial channel");
    const Fd serial(serial_pair[0]);
    Fd serial_guest(serial_pair[1]);
    if (!nonblocking(trace_read.value).ok() || !nonblocking(qmp.value).ok() || !nonblocking(serial.value).ok())
        return Result<bool>::failure("Cannot configure kernel channels");
    char plugin[PATH_MAX + 256];
    const int length = std::snprintf(plugin, sizeof(plugin), "%s,fd=%d,control=%d,kernel=on,registers=%s,memory=%s,values=%s,context=%s,batching=%s,publication=%s%s%s",
        options.plugin, trace_write.value, control_read.value, options.registers, options.memory, options.values, options.context,
        options.batching, options.publication,
        options.start_pc == nullptr ? "" : ",start=", options.start_pc == nullptr ? "" : options.start_pc);
    if (length < 0 || static_cast<size_t>(length) >= sizeof(plugin)) return Result<bool>::failure("Kernel plugin options too long");
    char qmp_spec[128], serial_spec[128];
    std::snprintf(qmp_spec, sizeof(qmp_spec), "socket,id=c2tqmp,fd=%d", qmp_guest.value);
    std::snprintf(serial_spec, sizeof(serial_spec), "socket,id=c2tserial,fd=%d", serial_guest.value);
    char* arguments[argument_capacity]{};
    const char* fixed[] = {options.qemu, "-plugin", plugin, "-chardev", qmp_spec, "-mon", "chardev=c2tqmp,mode=control",
                          "-chardev", serial_spec, "-serial", "chardev:c2tserial", "-display", "none", "-monitor", "none", "-S"};
    size_t count = 0;
    for (const char* value : fixed) arguments[count++] = const_cast<char*>(value);
    for (size_t i = 0; options.arguments[i] != nullptr; ++i) {
        if (count + 1 == argument_capacity) return Result<bool>::failure("Too many kernel QEMU arguments");
        arguments[count++] = options.arguments[i];
    }
    const auto parent = getpid();
    const auto pid = fork();
    if (pid == 0) {
        if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent || setpgid(0, 0) != 0) _exit(126);
        const int inherited[] = {trace_write.value, control_read.value, qmp_guest.value, serial_guest.value};
        for (int fd : inherited) if (fcntl(fd, F_SETFD, 0) < 0) _exit(126);
        const int empty = open("/dev/null", O_RDONLY);
        if (empty < 0 || dup2(empty, STDIN_FILENO) < 0) _exit(126);
        execv(options.qemu, arguments);
        _exit(127);
    }
    if (pid < 0) return Result<bool>::failure("Cannot start system QEMU");
    Process process(pid);
    Fd* child_fds[] = {&trace_write, &control_read, &qmp_guest, &serial_guest};
    for (auto* fd : child_fds) { close(fd->value); fd->value = -1; }
    Stream stream;
    Lines qmp_lines, serial_lines;
    uint8_t frame[max_frame_bytes];
    size_t frame_used = 0, frame_needed = header_bytes;
    bool hello = false, greeting = false, requested = false, draining = false;
    bool seal = false, trace_eof = false, guest_complete = false, child_done = false;
    bool guest_started = false;
    int child_status = 0, pending = 0, next_id = 1;
    enum class Command { none, capabilities, resume, stop } command = Command::none;
    auto deadline = now_ms() + options.timeout_ms;
    int64_t control_deadline = 0;
    for (;;) {
        if (seal && trace_eof && child_done && serial_lines.eof) {
            if (serial_lines.used != 0 || !guest_complete)
                return Result<bool>::failure("Guest exited without a successful adapter completion");
            if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0)
                return Result<bool>::failure("System QEMU failed after capture completion");
            encode_header(frame, {Kind::complete});
            const auto sent = send_bytes(client.value, frame, header_bytes, options.timeout_ms);
            if (!sent.ok() || sent.value()) return sent;
            return Result<bool>::success(false);
        }
        const auto remaining = deadline - now_ms();
        if (remaining <= 0) return Result<bool>::failure("Kernel run timed out before capture or action boundary");
        if (control_deadline != 0 && now_ms() >= control_deadline)
            return Result<bool>::failure("Kernel QMP command or trace drain timed out");
        pollfd watches[] = {{trace_eof ? -1 : trace_read.value, POLLIN, 0}, {qmp_lines.eof ? -1 : qmp.value, POLLIN, 0},
                            {serial_lines.eof ? -1 : serial.value, POLLIN, 0}, {client.value, POLLIN | POLLRDHUP, 0}};
        const int ready = poll(watches, 4, static_cast<int>(remaining > 100 ? 100 : remaining));
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0) return Result<bool>::failure("Cannot poll kernel channels");
        if (watches[3].revents & (POLLRDHUP | POLLHUP | POLLERR)) return Result<bool>::success(true);
        if (watches[3].revents & POLLIN) return Result<bool>::failure("Kernel action arrived before a paused boundary");
        if (watches[0].revents && !trace_eof) {
            const auto read_count = read(trace_read.value, frame + frame_used, frame_needed - frame_used);
            if (read_count > 0) { frame_used += static_cast<size_t>(read_count); deadline = now_ms() + options.timeout_ms; }
            else if (read_count == 0) trace_eof = true;
            else if (errno != EAGAIN && errno != EINTR) return Result<bool>::failure("Cannot read kernel capture pipe");
            if (frame_used == header_bytes && frame_needed == header_bytes) {
                const auto header = decode_header(frame, header_bytes);
                if (!header.ok()) return Result<bool>::failure(header.error());
                frame_needed += payload_size(header.value());
            }
            if (frame_used == frame_needed) {
                if (seal) return Result<bool>::failure("Kernel capture continued after its seal");
                const auto header = decode_header(frame, header_bytes).value();
                const auto valid = stream.accept(header, frame + header_bytes);
                if (!valid.ok()) return Result<bool>::failure(valid.error());
                if (!hello) {
                    if (header.kind != Kind::hello || !(header.detail & feature_kernel))
                        return Result<bool>::failure("Kernel plugin did not negotiate the system adapter");
                    hello = true;
                }
                if (header.kind == Kind::complete) seal = true;
                else {
                    if (header.kind == Kind::kernel_request && !draining)
                        return Result<bool>::failure("Unexpected kernel drain seal");
                    const auto sent = send_bytes(client.value, frame, frame_needed, options.timeout_ms);
                    if (header.kind == Kind::error) return Result<bool>::failure("Kernel capture failed");
                    if (!sent.ok() || sent.value()) return sent;
                    if (header.kind == Kind::kernel_request) {
                        const auto received = action(client.value, serial.value, options.timeout_ms);
                        if (!received.ok() || received.value()) return received;
                        draining = false;
                        requested = false;
                        pending = next_id++;
                        command = Command::resume;
                        control_deadline = now_ms() + options.timeout_ms;
                        const auto resumed = qmp_command(qmp.value, "cont", pending, options.timeout_ms);
                        if (!resumed.ok()) return Result<bool>::failure(resumed.error());
                        deadline = now_ms() + options.timeout_ms;
                    }
                }
                frame_used = 0;
                frame_needed = header_bytes;
            }
            if (trace_eof && (!seal || frame_used != 0)) return Result<bool>::failure("Incomplete kernel trace");
        }
        if (watches[1].revents && !qmp_lines.eof) {
            const auto read_result = qmp_lines.read_from(qmp.value);
            if (!read_result.ok()) return Result<bool>::failure(read_result.error());
        }
        while (const size_t size = qmp_lines.line_size()) {
            const Json message(qmp_lines.bytes, size);
            if (message.value == nullptr) return Result<bool>::failure("Invalid bounded QMP response");
            if (message.get("QMP") != nullptr) {
                if (greeting) return Result<bool>::failure("Repeated QMP greeting");
                greeting = true;
                pending = next_id++;
                command = Command::capabilities;
                control_deadline = now_ms() + options.timeout_ms;
                const auto sent = qmp_command(qmp.value, "qmp_capabilities", pending, options.timeout_ms);
                if (!sent.ok()) return Result<bool>::failure(sent.error());
            } else if (message.get("id") != nullptr) {
                if (!json_object_is_type(message.get("id"), json_type_int) || json_object_get_int(message.get("id")) != pending || pending == 0 || message.get("error") != nullptr || message.get("return") == nullptr)
                    return Result<bool>::failure("Unexpected or failed QMP command response");
                const auto completed = command;
                command = Command::none;
                pending = 0;
                if (completed == Command::capabilities) {
                    pending = next_id++;
                    command = Command::resume;
                    control_deadline = now_ms() + options.timeout_ms;
                    const auto sent = qmp_command(qmp.value, "cont", pending, options.timeout_ms);
                    if (!sent.ok()) return Result<bool>::failure(sent.error());
                } else if (completed == Command::stop) {
                    // QMP stopped execution. The plugin's explicit drain, not
                    // STOP or pipe emptiness, establishes the captured frontier.
                    const auto drained = request_drain(control_write.value);
                    if (!drained.ok()) return Result<bool>::failure(drained.error());
                    draining = true;
                } else control_deadline = 0;
            } else if (message.get("event") == nullptr) return Result<bool>::failure("Unknown QMP message");
            qmp_lines.consume(size);
            deadline = now_ms() + options.timeout_ms;
        }
        if (watches[2].revents && !serial_lines.eof) {
            const auto read_result = serial_lines.read_from(serial.value);
            if (!read_result.ok()) {
                log_guest_failure(stderr, "serial-framing-or-read", serial_lines.bytes, serial_lines.used);
                return Result<bool>::failure(read_result.error());
            }
        }
        while (const size_t size = serial_lines.line_size()) {
            if (size >= 4 && std::memcmp(serial_lines.bytes, "C2T ", 4) == 0) {
                const char* bytes = serial_lines.bytes + 4;
                size_t length = size - 4;
                while (length > 0 && (bytes[length - 1] == '\n' || bytes[length - 1] == '\r')) --length;
                const auto fail_event = [&](const char* reason, const char* error, const Json* event = nullptr) {
                    log_guest_failure(stderr, reason, bytes, length, event);
                    std::fprintf(stderr, "cpu2tensor: guest-event state hello=%d started=%d requested=%d draining=%d complete=%d pending_qmp=%d\n",
                                 hello, guest_started, requested, draining, guest_complete, pending);
                    return Result<bool>::failure(error);
                };
                if (length == 0 || length > event_capacity)
                    return fail_event("payload-size", "Invalid guest event size or ordering");
                if (!hello) return fail_event("before-plugin-hello", "Invalid guest event size or ordering");
                const Json event(bytes, length);
                if (event.event_problem() != nullptr)
                    return fail_event(event.event_problem(), "Invalid guest adapter event", &event);
                const char* name = string_value(event.get("event"));
                if (std::strcmp(name, "start") == 0) {
                    if (guest_started) return fail_event("repeated-start", "Guest reboot or repeated adapter start is unsupported", &event);
                    guest_started = true;
                } else if (std::strcmp(name, "ready") == 0) {
                    if (!guest_started || requested || guest_complete)
                        return fail_event("unexpected-ready", "Unexpected guest action request", &event);
                    requested = true;
                } else if (std::strcmp(name, "complete") == 0) {
                    if (!json_object_is_type(event.get("ok"), json_type_boolean) || !json_object_get_boolean(event.get("ok")))
                        return fail_event("unsuccessful-complete", "Guest adapter reported an unsuccessful workload", &event);
                    guest_complete = true;
                } else if (std::strcmp(name, "result") != 0 && std::strcmp(name, "error") != 0)
                    return fail_event("unknown-event-name", "Unknown guest adapter event", &event);
                uint8_t event_frame[header_bytes + event_capacity];
                const Header header{Kind::guest_event, 0, 0, 0, length};
                encode_header(event_frame, header);
                std::memcpy(event_frame + header_bytes, bytes, length);
                const auto sent = send_bytes(client.value, event_frame, header_bytes + length, options.timeout_ms);
                if (!sent.ok() || sent.value()) return sent;
            }
            // Boot diagnostics stay in the worker log, never in model tensors.
            std::fwrite(serial_lines.bytes, 1, size, stderr);
            serial_lines.consume(size);
            deadline = now_ms() + options.timeout_ms;
        }
        if (serial_lines.eof && serial_lines.used != 0) {
            log_guest_failure(stderr, "unterminated-serial-line", serial_lines.bytes, serial_lines.used);
            return Result<bool>::failure("Guest serial channel ended inside a line");
        }
        if (requested && pending == 0 && !draining) {
            pending = next_id++;
            command = Command::stop;
            control_deadline = now_ms() + options.timeout_ms;
            const auto stopped = qmp_command(qmp.value, "stop", pending, options.timeout_ms);
            if (!stopped.ok()) return Result<bool>::failure(stopped.error());
        }
        if (!child_done) {
            const auto ended = waitpid(process.pid, &child_status, WNOHANG);
            if (ended == process.pid) { child_done = true; process.pid = -1; }
            else if (ended < 0 && errno != EINTR) return Result<bool>::failure("Cannot reap system QEMU");
        }
    }
}
} // namespace cpu2tensor
