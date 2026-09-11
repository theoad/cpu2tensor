// SPDX-License-Identifier: AGPL-3.0-only
#include <cpu2tensor/trace.hpp>
#include <cpu2tensor/register_selection.hpp>
#include "kernel.hpp"
#include <arpa/inet.h>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

namespace {
using namespace cpu2tensor;
constexpr int socket_buffer_bytes = 64 * 1024;
constexpr int default_timeout_ms = 30000;
constexpr size_t max_arguments = 256;
constexpr char run_deadline_error[] = "Target exceeded --max-run-ms deadline";

enum class Transfer : uint8_t { ready, cancelled };
enum class RunOutcome : uint8_t { completed, cancelled };
struct ChildExit final { int status = 0; bool cancelled = false; };
struct TerminalProgress final {
    bool hello_sent = false;
    bool data_sent = false;
    bool start_configured = false;
    bool start_observed = false;
    bool stop_configured = false;
    bool stop_observed = false;
    uint16_t version = wire_version;
};

bool observation_frame(Kind kind) {
    return kind == Kind::blocks || kind == Kind::registers || kind == Kind::memory ||
        kind == Kind::address_context || kind == Kind::executable_layout ||
        kind == Kind::mixed || kind == Kind::block_transitions ||
        kind == Kind::transition_window || kind == Kind::observation_summary ||
        kind == Kind::reduced_context;
}

bool execution_frame(Kind kind) {
    return kind == Kind::blocks || kind == Kind::registers || kind == Kind::memory ||
        kind == Kind::address_context || kind == Kind::mixed ||
        kind == Kind::block_transitions || kind == Kind::transition_window ||
        kind == Kind::observation_summary || kind == Kind::reduced_context;
}

void report_deadline(int client, const TerminalProgress& progress) {
    if (progress.version != wire_version) return;
    uint8_t flags = (progress.hello_sent ? terminal_hello : 0) |
        (progress.data_sent ? terminal_data : 0) |
        (progress.start_configured ? terminal_start_configured : 0) |
        (progress.start_observed ? terminal_start_observed : 0) |
        (progress.stop_configured ? terminal_stop_configured : 0) |
        (progress.stop_observed ? terminal_stop_observed : 0);
    uint8_t bytes[header_bytes];
    encode_header(bytes, {Kind::terminal_report, 0, 0, 0,
                          terminal_report_detail(TerminalReason::max_run_deadline, flags)});
    // One nonblocking attempt cannot extend the run deadline. A full socket may
    // lose this optional diagnostic; the client then reports the actual EOF or
    // truncated frame instead of inventing a worker reason.
    (void)send(client, bytes, sizeof(bytes), MSG_NOSIGNAL | MSG_DONTWAIT);
}

Result<RunOutcome> fail_before_client_frame(int client, const TerminalProgress& progress,
                                            const char* error) {
    if (std::strcmp(error, run_deadline_error) == 0)
        report_deadline(client, progress);
    return Result<RunOutcome>::failure(error);
}

int remaining_timeout(const timespec& start, int timeout_ms) {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    const int64_t elapsed = (now.tv_sec - start.tv_sec) * 1000 + (now.tv_nsec - start.tv_nsec) / 1000000;
    return elapsed >= timeout_ms ? 0 : timeout_ms - static_cast<int>(elapsed);
}

// One absolute budget for an observation run. Successful I/O never renews it.
// Disabled budgets avoid clock reads on the normal forwarding path.
class RunDeadline final {
public:
    explicit RunDeadline(int maximum_ms) : _end_ms(maximum_ms == 0 ? 0 : now_ms() + maximum_ms) {}
    Result<int> limit(int operation_ms) const {
        if (_end_ms == 0) return Result<int>::success(operation_ms);
        const int64_t remaining = _end_ms - now_ms();
        if (remaining <= 0) return Result<int>::failure(run_deadline_error);
        return Result<int>::success(remaining < operation_ms ? static_cast<int>(remaining) : operation_ms);
    }
private:
    static int64_t now_ms() {
        timespec now{};
        clock_gettime(CLOCK_MONOTONIC, &now);
        return static_cast<int64_t>(now.tv_sec) * 1000 + now.tv_nsec / 1000000;
    }
    const int64_t _end_ms;
};

struct Options final {
    const char* qemu = nullptr;
    const char* plugin = nullptr;
    const char* input = nullptr;
    const char* host = "127.0.0.1";
    const char* registers = "general";
    const char* memory = "on";
    const char* values = "off";
    const char* context = "auto";
    int port = 0;
    int timeout_ms = default_timeout_ms;
    int max_run_ms = 0;
    bool stdio = false;
    bool system = false;
    bool kernel = false;
    bool kernel_protocol = false;
    bool layout = false;
    const char* start_pc = nullptr;
    const char* stop_pc = nullptr;
    const char* window_start_pc = nullptr;
    const char* window_end_pc = nullptr;
    const char* window_abort_pc = nullptr;
    const char* reducer = "none";
    const char* blocks = "on";
    int transition_capacity = 4096;
    const char* batching = "legacy";
    const char* publication = "pipe";
    int episodes = 1;
    char** target = nullptr;
};

Result<RunOutcome> stopped_transfer(const Options& options, const Result<Transfer>& result) {
    if (!result.ok()) return Result<RunOutcome>::failure(result.error());
    if (options.stdio) return Result<RunOutcome>::success(RunOutcome::cancelled);
    return Result<RunOutcome>::failure("Consumer disconnected before trace completion");
}

Result<int> number(const char* text, int minimum, int maximum) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < minimum || value > maximum)
        return Result<int>::failure("Invalid numeric option");
    return Result<int>::success(static_cast<int>(value));
}

Result<Options> parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--") == 0) {
            if (i + 1 < argc) options.target = argv + i + 1;
            break;
        }
        if (i + 1 >= argc) return Result<Options>::failure("Option needs a value");
        const char* key = argv[i++];
        const char* value = argv[i];
        if (std::strcmp(key, "--qemu") == 0) options.qemu = value;
        else if (std::strcmp(key, "--plugin") == 0) options.plugin = value;
        else if (std::strcmp(key, "--input") == 0) options.input = value;
        else if (std::strcmp(key, "--host") == 0) options.host = value;
        else if (std::strcmp(key, "--registers") == 0) options.registers = value;
        else if (std::strcmp(key, "--memory") == 0) options.memory = value;
        else if (std::strcmp(key, "--memory-values") == 0) options.values = value;
        else if (std::strcmp(key, "--context") == 0) {
            if (std::strcmp(value, "auto") != 0 && std::strcmp(value, "on") != 0)
                return Result<Options>::failure("--context needs auto or on");
            options.context = value;
        }
        else if (std::strcmp(key, "--publication") == 0) {
            if (std::strcmp(value, "pipe") != 0 && std::strcmp(value, "ring") != 0)
                return Result<Options>::failure("--publication needs pipe or ring");
            options.publication = value;
        }
        else if (std::strcmp(key, "--batching") == 0) {
            if (std::strcmp(value, "legacy") != 0 && std::strcmp(value, "mixed") != 0)
                return Result<Options>::failure("--batching needs legacy or mixed");
            options.batching = value;
        }
        else if (std::strcmp(key, "--start-pc") == 0 || std::strcmp(key, "--stop-pc") == 0 ||
                 std::strcmp(key, "--window-start-pc") == 0 ||
                 std::strcmp(key, "--window-end-pc") == 0 ||
                 std::strcmp(key, "--window-abort-pc") == 0) {
            char* end = nullptr;
            errno = 0;
            (void)std::strtoull(value, &end, 0);
            if (end == value || errno != 0 || *end != '\0' || value[0] == '-')
                return Result<Options>::failure("Capture boundary needs a guest block address");
            if (std::strcmp(key, "--start-pc") == 0) options.start_pc = value;
            else if (std::strcmp(key, "--stop-pc") == 0) options.stop_pc = value;
            else if (std::strcmp(key, "--window-start-pc") == 0) options.window_start_pc = value;
            else if (std::strcmp(key, "--window-end-pc") == 0) options.window_end_pc = value;
            else options.window_abort_pc = value;
        } else if (std::strcmp(key, "--reducer") == 0) {
            if (std::strcmp(value, "none") != 0 && std::strcmp(value, "block-transitions") != 0)
                return Result<Options>::failure("--reducer needs none or block-transitions");
            options.reducer = value;
        } else if (std::strcmp(key, "--blocks") == 0) {
            if (std::strcmp(value, "on") != 0 && std::strcmp(value, "off") != 0)
                return Result<Options>::failure("--blocks needs on or off");
            options.blocks = value;
        } else if (std::strcmp(key, "--transition-capacity") == 0) {
            const auto parsed = number(value, 2, 65536);
            if (!parsed.ok() || (parsed.value() & (parsed.value() - 1)) != 0)
                return Result<Options>::failure("--transition-capacity needs a power of two from 2 to 65536");
            options.transition_capacity = parsed.value();
        } else if (std::strcmp(key, "--system") == 0 ||
                   std::strcmp(key, "--kernel-protocol") == 0 ||
                   std::strcmp(key, "--kernel-adapter") == 0) {
            if (std::strcmp(value, "on") != 0 && std::strcmp(value, "off") != 0)
                return Result<Options>::failure("System settings need on or off");
            if (std::strcmp(key, "--system") == 0) options.system = std::strcmp(value, "on") == 0;
            else if (std::strcmp(key, "--kernel-protocol") == 0)
                options.kernel_protocol = std::strcmp(value, "on") == 0;
            else options.kernel = std::strcmp(value, "on") == 0;
        }
        else if (std::strcmp(key, "--layout") == 0) {
            if (std::strcmp(value, "on") != 0 && std::strcmp(value, "off") != 0)
                return Result<Options>::failure("--layout needs on or off");
            options.layout = std::strcmp(value, "on") == 0;
        }
        else if (std::strcmp(key, "--stdio") == 0) {
            if (std::strcmp(value, "on") != 0 && std::strcmp(value, "off") != 0)
                return Result<Options>::failure("--stdio needs on or off");
            options.stdio = std::strcmp(value, "on") == 0;
        } else if (std::strcmp(key, "--episodes") == 0) {
            const auto parsed = number(value, 1, INT_MAX);
            if (!parsed.ok()) return Result<Options>::failure(parsed.error());
            options.episodes = parsed.value();
        } else if (std::strcmp(key, "--port") == 0) {
            const auto parsed = number(value, 0, 65535);
            if (!parsed.ok()) return Result<Options>::failure(parsed.error());
            options.port = parsed.value();
        } else if (std::strcmp(key, "--max-run-ms") == 0) {
            const auto parsed = number(value, 1, INT_MAX);
            if (!parsed.ok()) return Result<Options>::failure(parsed.error());
            options.max_run_ms = parsed.value();
        } else if (std::strcmp(key, "--timeout-ms") == 0) {
            const auto parsed = number(value, 1, INT_MAX);
            if (!parsed.ok()) return Result<Options>::failure(parsed.error());
            options.timeout_ms = parsed.value();
        } else return Result<Options>::failure("Unknown worker option");
    }
    if (std::strcmp(options.context, "on") == 0 && !options.system)
        return Result<Options>::failure("--context on requires x86 system emulation");
    if (options.max_run_ms != 0 && (options.stdio || options.kernel))
        return Result<Options>::failure("--max-run-ms requires observation-only capture");
    if (options.kernel && (!options.system || options.stdio || options.input != nullptr))
        return Result<Options>::failure("--kernel-adapter on needs --system on and owns guest input");
    if (options.kernel_protocol && (!options.system || options.stdio || options.input != nullptr))
        return Result<Options>::failure("--kernel-protocol on needs --system on and owns guest input");
    if (options.kernel) options.kernel_protocol = true;
    if (options.system && options.stdio) return Result<Options>::failure("System guests use --kernel-adapter, not --stdio");
    if (options.system && options.layout) return Result<Options>::failure("--layout on requires a user-mode target");
    if (options.stop_pc != nullptr && (options.stdio || options.kernel))
        return Result<Options>::failure("--stop-pc requires observation-only capture");
    const bool any_window = options.window_start_pc != nullptr || options.window_end_pc != nullptr ||
                            options.window_abort_pc != nullptr;
    const bool all_window = options.window_start_pc != nullptr && options.window_end_pc != nullptr &&
                            options.window_abort_pc != nullptr;
    if (any_window && !all_window)
        return Result<Options>::failure("Action windows need start, end, and abort guest block addresses");
    if (all_window && !options.kernel)
        return Result<Options>::failure("Action windows require --kernel-adapter on");
    const bool reduced = std::strcmp(options.reducer, "block-transitions") == 0;
    if (all_window && !reduced)
        return Result<Options>::failure("Action windows require block transition reduction");
    if (reduced && !all_window && (!options.system || options.kernel ||
        std::strcmp(options.blocks, "off") != 0 ||
        std::strcmp(options.registers, "none") != 0 ||
        std::strcmp(options.memory, "off") != 0 ||
        std::strcmp(options.context, "on") != 0 ||
        std::strcmp(options.batching, "legacy") != 0))
        return Result<Options>::failure(
            "Observation reduction needs context-only system capture, --blocks off, and legacy batching");
    if (std::strcmp(options.blocks, "off") == 0 && std::strcmp(options.reducer, "none") == 0)
        return Result<Options>::failure("--blocks off requires a reducer");
    if (options.system && !options.kernel_protocol && options.input == nullptr) options.input = "/dev/null";
    if (options.qemu == nullptr || options.plugin == nullptr ||
        (!options.stdio && !options.kernel_protocol && options.input == nullptr) ||
        options.target == nullptr)
        return Result<Options>::failure("Required: --qemu PATH --plugin PATH (--input FILE or --stdio on) -- TARGET [ARGS]");
    if (std::strchr(options.plugin, ',') != nullptr)
        return Result<Options>::failure("Plugin path cannot contain a comma (QEMU option separator)");
    if (!valid_register_selection(options.registers))
        return Result<Options>::failure("--registers needs none, general, all, or colon-separated names");
    const char* settings[] = {options.memory, options.values};
    for (const char* setting : settings) {
        if (std::strcmp(setting, "on") != 0 && std::strcmp(setting, "off") != 0)
            return Result<Options>::failure("Memory settings need on or off");
    }
    if (std::strcmp(options.values, "on") == 0 && std::strcmp(options.memory, "off") == 0)
        return Result<Options>::failure("Memory values require memory observations");
    if (options.stdio && options.input != nullptr)
        return Result<Options>::failure("Stdin interaction cannot use a prescribed input file");
    if (options.episodes > 1 && options.port == 0)
        return Result<Options>::failure("Repeated episodes need a fixed --port");
    return Result<Options>::success(options);
}

class Descriptor final {
public:
    explicit Descriptor(int fd) : _fd(fd) {}
    ~Descriptor() { if (_fd >= 0) close(_fd); }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    Descriptor(Descriptor&&) = delete;
    Descriptor& operator=(Descriptor&&) = delete;
    int get() const { return _fd; }
    int release() { const int fd = _fd; _fd = -1; return fd; }
private:
    int _fd;
};

class Child final {
public:
    explicit Child(pid_t pid) : _pid(pid) {}
    ~Child() {
        if (_pid > 0) {
            // Cancellation also stops guest threads and any unexpected descendants.
            kill(-_pid, SIGKILL);
            kill(_pid, SIGKILL);
            while (waitpid(_pid, nullptr, 0) < 0 && errno == EINTR) {}
        }
    }
    Child(const Child&) = delete;
    Child& operator=(const Child&) = delete;
    Child(Child&&) = delete;
    Child& operator=(Child&&) = delete;
    Result<Transfer> wait_stopped(int client, int timeout_ms) {
        timespec start{};
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (;;) {
            int status = 0;
            const auto result = waitpid(_pid, &status, WUNTRACED | WNOHANG);
            if (result == _pid) {
                if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGSTOP)
                    return Result<Transfer>::success(Transfer::ready);
                if (WIFEXITED(status) || WIFSIGNALED(status)) _pid = -1;
                return Result<Transfer>::failure("Target did not stop at its input boundary");
            }
            if (result < 0 && errno != EINTR) return Result<Transfer>::failure("Cannot wait for input stop");
            if (remaining_timeout(start, timeout_ms) == 0)
                return Result<Transfer>::failure("Target did not stop before timeout");
            pollfd connection{client, POLLRDHUP, 0};
            const int ready = poll(&connection, 1, 1);
            if (ready > 0 && (connection.revents & (POLLRDHUP | POLLHUP | POLLERR)))
                return Result<Transfer>::success(Transfer::cancelled);
            if ((ready > 0 && (connection.revents & POLLNVAL)) || (ready < 0 && errno != EINTR))
                return Result<Transfer>::failure("Cannot wait for input stop");
        }
    }
    Result<Done> resume() {
        if (kill(_pid, SIGCONT) != 0) return Result<Done>::failure("Cannot resume target");
        return Result<Done>::success({});
    }
    Result<ChildExit> finish(int client, int timeout_ms, const RunDeadline& deadline) {
        timespec start{};
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (;;) {
            const auto budget = deadline.limit(10);
            if (!budget.ok()) return Result<ChildExit>::failure(budget.error());
            int status = 0;
            const auto result = waitpid(_pid, &status, WNOHANG);
            if (result == _pid) { _pid = -1; return Result<ChildExit>::success({status, false}); }
            if (result < 0 && errno != EINTR) return Result<ChildExit>::failure("Cannot read target exit status");
            if (remaining_timeout(start, timeout_ms) == 0)
                return Result<ChildExit>::failure("Target did not exit after capture ended");
            pollfd connection{client, POLLIN | POLLRDHUP, 0};
            const int ready = poll(&connection, 1, budget.value());
            if (ready > 0 && (connection.revents & (POLLRDHUP | POLLHUP | POLLERR)))
                return Result<ChildExit>::success({0, true});
            if (ready > 0) return Result<ChildExit>::failure("Consumer sent unexpected data while target was exiting");
            if (ready < 0 && errno != EINTR) return Result<ChildExit>::failure("Cannot wait for target exit");
        }
    }
private:
    pid_t _pid;
};

Result<Transfer> wait_ready(int fd, short events, int client, int timeout_ms, const RunDeadline& deadline) {
    pollfd watches[2] = {{fd, events, 0}, {client, POLLIN | POLLRDHUP, 0}};
    int count;
    do {
        const auto budget = deadline.limit(timeout_ms);
        if (!budget.ok()) return Result<Transfer>::failure(budget.error());
        count = poll(watches, 2, budget.value());
    } while (count < 0 && errno == EINTR);
    const auto budget = deadline.limit(timeout_ms);
    if (!budget.ok()) return Result<Transfer>::failure(budget.error());
    if (count == 0) return Result<Transfer>::failure("Trace connection timed out");
    if (count < 0) return Result<Transfer>::failure("Cannot wait for trace data");
    if (watches[1].revents & (POLLRDHUP | POLLHUP | POLLERR))
        return Result<Transfer>::success(Transfer::cancelled);
    if (watches[1].revents & (POLLIN | POLLNVAL))
        return Result<Transfer>::failure("Consumer sent unexpected data or its descriptor failed");
    if (watches[0].revents & (POLLERR | POLLNVAL))
        return Result<Transfer>::failure("Trace descriptor failed");
    return Result<Transfer>::success(Transfer::ready);
}

Result<Transfer> read_exact(int pipe, int client, uint8_t* bytes, size_t size, int timeout_ms, const RunDeadline& deadline) {
    size_t offset = 0;
    while (offset < size) {
        const auto ready = wait_ready(pipe, POLLIN, client, timeout_ms, deadline);
        if (!ready.ok() || ready.value() == Transfer::cancelled) return ready;
        const auto count = read(pipe, bytes + offset, size - offset);
        if (count == 0) return Result<Transfer>::failure("Capture ended without a complete frame and seal");
        if (count < 0) {
            if (errno == EINTR) continue;
            return Result<Transfer>::failure("Cannot read capture pipe");
        }
        offset += static_cast<size_t>(count);
    }
    return Result<Transfer>::success(Transfer::ready);
}

Result<Transfer> send_exact(int client, const uint8_t* bytes, size_t size, int timeout_ms, const RunDeadline& deadline) {
    size_t offset = 0;
    while (offset < size) {
        const auto budget = deadline.limit(timeout_ms);
        if (!budget.ok()) return Result<Transfer>::failure(budget.error());
        const auto count = send(client, bytes + offset, size - offset, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (count > 0) { offset += static_cast<size_t>(count); continue; }
        if (count < 0 && errno == EINTR) continue;
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            const auto ready = wait_ready(client, POLLOUT, client, timeout_ms, deadline);
            if (!ready.ok() || ready.value() == Transfer::cancelled) return ready;
            continue;
        }
        if (count == 0 || errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN)
            return Result<Transfer>::success(Transfer::cancelled);
        return Result<Transfer>::failure("Cannot send a trace batch");
    }
    return Result<Transfer>::success(Transfer::ready);
}

Result<Done> require_empty_stdin(int input) {
    int queued = 0;
    if (ioctl(input, FIONREAD, &queued) != 0)
        return Result<Done>::failure("Cannot inspect target stdin pipe");
    if (queued != 0)
        return Result<Done>::failure("Target requested input before consuming its previous action");
    return Result<Done>::success({});
}

Result<Transfer> receive_action(int client, int target_input, int input_reader, size_t limit, int timeout_ms) {
    uint8_t action[sizeof(uint32_t) + max_action_bytes];
    size_t received = 0;
    size_t needed = sizeof(uint32_t);
    timespec start{};
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (received < needed) {
        const int remaining = remaining_timeout(start, timeout_ms);
        if (remaining == 0) return Result<Transfer>::failure("Timed out waiting for a complete stdin action");
        pollfd watch{client, POLLIN | POLLRDHUP, 0};
        const int ready = poll(&watch, 1, remaining);
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0) return Result<Transfer>::failure("Timed out or failed waiting for stdin action");
        if (watch.revents & (POLLRDHUP | POLLHUP | POLLERR))
            return Result<Transfer>::success(Transfer::cancelled);
        const auto count = recv(client, action + received, needed - received, MSG_DONTWAIT);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (count == 0 || (count < 0 && errno == ECONNRESET))
            return Result<Transfer>::success(Transfer::cancelled);
        if (count < 0) return Result<Transfer>::failure("Cannot read stdin action");
        received += static_cast<size_t>(count);
        if (received == sizeof(uint32_t) && needed == sizeof(uint32_t)) {
            const auto size = load_u32(action);
            if (size == 0 || size > limit) return Result<Transfer>::failure("Stdin action exceeds the request limit");
            needed += size;
        }
    }
    const auto empty = require_empty_stdin(input_reader);
    if (!empty.ok()) return Result<Transfer>::failure(empty.error());
    // The worker alone writes this pipe, and the single guest vCPU is stopped.
    // A nonblocking atomic write cannot hang even if that invariant is violated.
    for (;;) {
        if (remaining_timeout(start, timeout_ms) == 0)
            return Result<Transfer>::failure("Timed out before delivering stdin action");
        pollfd watch{client, POLLRDHUP, 0};
        const int ready = poll(&watch, 1, 0);
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0 || (watch.revents & POLLNVAL))
            return Result<Transfer>::failure("Cannot check consumer before stdin delivery");
        if (watch.revents & (POLLRDHUP | POLLHUP | POLLERR))
            return Result<Transfer>::success(Transfer::cancelled);
        const auto written = write(target_input, action + sizeof(uint32_t), needed - sizeof(uint32_t));
        if (written < 0 && errno == EINTR) continue;
        if (written != static_cast<ssize_t>(needed - sizeof(uint32_t)))
            return Result<Transfer>::failure("Cannot deliver a complete bounded stdin action");
        return Result<Transfer>::success(Transfer::ready);
    }
}

Result<int> listen_at(const Options& options) {
    Descriptor listener(socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (listener.get() < 0) return Result<int>::failure("Cannot create worker socket");
    const int reuse_address = 1;
    if (setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &reuse_address, sizeof(reuse_address)) != 0)
        return Result<int>::failure("Cannot enable worker address reuse");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(options.port));
    if (inet_pton(AF_INET, options.host, &address.sin_addr) != 1)
        return Result<int>::failure("--host needs an IPv4 address");
    if (bind(listener.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 || listen(listener.get(), 1) != 0)
        return Result<int>::failure("Cannot listen on the requested worker address");
    socklen_t length = sizeof(address);
    if (getsockname(listener.get(), reinterpret_cast<sockaddr*>(&address), &length) != 0)
        return Result<int>::failure("Cannot read worker address");
    std::fprintf(stderr, "cpu2tensor listening on tcp://%s:%u\n", options.host, ntohs(address.sin_port));
    return Result<int>::success(listener.release());
}

Result<RunOutcome> run(const Options& options, int listener) {
    if (options.kernel_protocol) {
        const auto result = run_kernel({options.qemu, options.plugin, options.registers, options.memory,
            options.values, options.context, options.start_pc, options.stop_pc, options.window_start_pc,
            options.window_end_pc, options.window_abort_pc, options.reducer, options.blocks,
            options.transition_capacity, options.batching, options.publication,
            options.timeout_ms, options.max_run_ms, options.kernel, options.target}, listener);
        if (!result.ok()) return Result<RunOutcome>::failure(result.error());
        return Result<RunOutcome>::success(result.value() ? RunOutcome::cancelled : RunOutcome::completed);
    }
    int input_pipe[2] = {-1, -1};
    if (options.stdio && pipe2(input_pipe, O_CLOEXEC) != 0)
        return Result<RunOutcome>::failure("Cannot create target stdin pipe");
    const Descriptor input(options.stdio ? input_pipe[0] : open(options.input, O_RDONLY | O_CLOEXEC));
    const Descriptor action_writer(input_pipe[1]);
    if (input.get() < 0) return Result<RunOutcome>::failure("Cannot open target input file");
    if (options.stdio && fcntl(action_writer.get(), F_SETFL, O_NONBLOCK) != 0)
        return Result<RunOutcome>::failure("Cannot make target stdin delivery nonblocking");
    if (access(options.qemu, X_OK) != 0 || access(options.plugin, R_OK) != 0)
        return Result<RunOutcome>::failure("QEMU or plugin is missing or inaccessible; supply compatible paths");
    int accepted;
    do { accepted = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC); } while (accepted < 0 && errno == EINTR);
    const Descriptor client(accepted);
    if (client.get() < 0) return Result<RunOutcome>::failure("Cannot accept consumer connection");
    if (setsockopt(client.get(), SOL_SOCKET, SO_SNDBUF, &socket_buffer_bytes, sizeof(socket_buffer_bytes)) != 0)
        return Result<RunOutcome>::failure("Cannot set bounded socket buffer");
    int pipes[2];
    if (pipe2(pipes, O_CLOEXEC) != 0) return Result<RunOutcome>::failure("Cannot create capture pipe");
    const Descriptor reader(pipes[0]);
    // Writer is closed by the parent immediately after fork.
    char plugin_option[PATH_MAX + 384];
    const int option_size = std::snprintf(plugin_option, sizeof(plugin_option), "%s,fd=%d,registers=%s,memory=%s,values=%s,context=%s,stdio=%s,layout=%s,batching=%s,publication=%s,blocks=%s,reducer=%s,transition_capacity=%d%s%s%s%s", options.plugin, pipes[1], options.registers, options.memory, options.values, options.context, options.stdio ? "on" : "off", options.layout ? "on" : "off", options.batching, options.publication, options.blocks, options.reducer, options.transition_capacity, options.start_pc == nullptr ? "" : ",start=", options.start_pc == nullptr ? "" : options.start_pc, options.stop_pc == nullptr ? "" : ",stop=", options.stop_pc == nullptr ? "" : options.stop_pc);
    if (option_size < 0 || static_cast<size_t>(option_size) >= sizeof(plugin_option)) {
        close(pipes[1]);
        return Result<RunOutcome>::failure("Plugin path is too long");
    }
    char* arguments[max_arguments]{};
    arguments[0] = const_cast<char*>(options.qemu);
    arguments[1] = const_cast<char*>("-plugin");
    arguments[2] = plugin_option;
    size_t index = 3;
    for (size_t i = 0; options.target[i] != nullptr; ++i) {
        if (index + 1 == max_arguments) { close(pipes[1]); return Result<RunOutcome>::failure("Too many target arguments"); }
        arguments[index++] = options.target[i];
    }
    const pid_t parent = getpid();
    const RunDeadline deadline(options.max_run_ms);
    const pid_t pid = fork();
    if (pid == 0) {
        if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent || setpgid(0, 0) != 0 ||
            dup2(input.get(), STDIN_FILENO) < 0 || fcntl(STDIN_FILENO, F_SETFD, 0) != 0 ||
            fcntl(pipes[1], F_SETFD, 0) != 0) _exit(126);
        close(reader.get());
        execv(options.qemu, arguments);
        std::fprintf(stderr, "cpu2tensor: cannot execute QEMU\n");
        _exit(127);
    }
    close(pipes[1]);
    if (pid < 0) return Result<RunOutcome>::failure("Cannot start target");
    Child child(pid);
    Stream stream;
    TerminalProgress progress{
        false, false, options.start_pc != nullptr, false,
        options.stop_pc != nullptr, false,
    };
    uint8_t frame[max_frame_bytes];
    for (;;) {
        auto read_result = read_exact(reader.get(), client.get(), frame, header_bytes, options.timeout_ms, deadline);
        if (!read_result.ok())
            return fail_before_client_frame(client.get(), progress, read_result.error());
        if (read_result.value() == Transfer::cancelled)
            return stopped_transfer(options, read_result);
        const auto decoded = decode_header(frame, header_bytes);
        if (!decoded.ok()) return Result<RunOutcome>::failure(decoded.error());
        const Header header = decoded.value();
        if (header.kind == Kind::hello && bool(header.detail & feature_system) != options.system)
            return Result<RunOutcome>::failure("QEMU mode does not match --system setting");
        const size_t payload = payload_size(header);
        read_result = read_exact(reader.get(), client.get(), frame + header_bytes, payload, options.timeout_ms, deadline);
        if (!read_result.ok())
            return fail_before_client_frame(client.get(), progress, read_result.error());
        if (read_result.value() == Transfer::cancelled)
            return stopped_transfer(options, read_result);
        const auto valid = stream.accept(header, frame + header_bytes);
        if (!valid.ok()) return Result<RunOutcome>::failure(valid.error());
        if (header.kind == Kind::input_request) {
            if (!options.stdio) return Result<RunOutcome>::failure("Unexpected stdin request");
            const auto stopped = child.wait_stopped(client.get(), options.timeout_ms);
            if (!stopped.ok() || stopped.value() == Transfer::cancelled)
                return stopped_transfer(options, stopped);
            const auto empty = require_empty_stdin(input.get());
            if (!empty.ok()) return Result<RunOutcome>::failure(empty.error());
            const auto announced = send_exact(client.get(), frame, header_bytes, options.timeout_ms, deadline);
            if (!announced.ok() || announced.value() == Transfer::cancelled)
                return stopped_transfer(options, announced);
            const auto action = receive_action(client.get(), action_writer.get(), input.get(), header.detail, options.timeout_ms);
            if (!action.ok() || action.value() == Transfer::cancelled)
                return stopped_transfer(options, action);
            const auto resumed = child.resume();
            if (!resumed.ok()) return Result<RunOutcome>::failure(resumed.error());
            continue;
        }
        if (header.kind == Kind::complete) {
            if (progress.start_configured) progress.start_observed = true;
            if (progress.stop_configured) progress.stop_observed = true;
            const auto ended = child.finish(client.get(), options.timeout_ms, deadline);
            if (!ended.ok())
                return fail_before_client_frame(client.get(), progress, ended.error());
            if (ended.value().cancelled)
                return stopped_transfer(options, Result<Transfer>::success(Transfer::cancelled));
            uint8_t extra;
            if (fcntl(reader.get(), F_SETFL, O_NONBLOCK) != 0)
                return Result<RunOutcome>::failure("Cannot check capture pipe closure");
            ssize_t tail;
            do {
                const auto budget = deadline.limit(options.timeout_ms);
                if (!budget.ok())
                    return fail_before_client_frame(client.get(), progress, budget.error());
                tail = read(reader.get(), &extra, 1);
            } while (tail < 0 && errno == EINTR);
            if (tail != 0) return Result<RunOutcome>::failure("Capture pipe stayed open or contained data after its seal");
            Header terminal{Kind::complete};
            terminal.version = header.version;
            if (WIFEXITED(ended.value().status)) terminal.detail = WEXITSTATUS(ended.value().status);
            else { terminal.kind = Kind::error; terminal.detail = static_cast<uint64_t>(Failure::target_killed); }
            encode_header(frame, terminal);
            const auto sent = send_exact(client.get(), frame, header_bytes, options.timeout_ms, deadline);
            if (!sent.ok() || sent.value() == Transfer::cancelled) return stopped_transfer(options, sent);
            return Result<RunOutcome>::success(RunOutcome::completed);
        }
        const auto sent = send_exact(client.get(), frame, header_bytes + payload, options.timeout_ms, deadline);
        // A known capture failure remains a failure if the peer closes after
        // receiving it; cancellation must not turn invalid capture into success.
        if (header.kind == Kind::error) return Result<RunOutcome>::failure("Capture reported a failure");
        if (!sent.ok() || sent.value() == Transfer::cancelled) return stopped_transfer(options, sent);
        if (header.kind == Kind::hello) {
            progress.hello_sent = true;
            progress.version = header.version;
        }
        if (observation_frame(header.kind)) {
            progress.data_sent = true;
            if (progress.start_configured && execution_frame(header.kind))
                progress.start_observed = true;
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    const auto options = parse(argc, argv);
    if (!options.ok()) { std::fprintf(stderr, "cpu2tensor: %s\n", options.error()); return 2; }
    const auto listening = listen_at(options.value());
    if (!listening.ok()) { std::fprintf(stderr, "cpu2tensor: %s\n", listening.error()); return 1; }
    const Descriptor listener(listening.value());
    for (int episode = 0; episode < options.value().episodes; ++episode) {
        const auto result = run(options.value(), listener.get());
        if (!result.ok()) { std::fprintf(stderr, "cpu2tensor: %s\n", result.error()); return 1; }
        // The run's Child has already stopped and reaped a cancelled target.
        if (result.value() == RunOutcome::cancelled) continue;
    }
    return 0;
}
