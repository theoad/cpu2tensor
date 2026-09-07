// SPDX-License-Identifier: AGPL-3.0-only
// Small target behaviors used by the real QEMU capture checks.
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr unsigned int thread_count = 2;
constexpr uint64_t loop_count = 10000;
pthread_barrier_t start;

struct Work final {
    uint64_t sum = 0;
};

void* add_numbers(void* argument)
{
    auto& work = *static_cast<Work*>(argument);
    (void)pthread_barrier_wait(&start);
    for (uint64_t number = 0; number < loop_count; ++number) {
        work.sum += number;
    }
    return nullptr;
}

int run_threads()
{
    pthread_t threads[thread_count];
    Work work[thread_count];
    if (pthread_barrier_init(&start, nullptr, thread_count) != 0) {
        return 1;
    }
    for (unsigned int index = 0; index < thread_count; ++index) {
        if (pthread_create(&threads[index], nullptr, add_numbers, &work[index]) != 0) {
            // A started thread may be waiting at the barrier. End the process
            // instead of trying to join a thread whose partner never started.
            _exit(1);
        }
    }
    for (unsigned int index = 0; index < thread_count; ++index) {
        if (pthread_join(threads[index], nullptr) != 0 ||
            work[index].sum != loop_count * (loop_count - 1) / 2) {
            return 1;
        }
    }
    return pthread_barrier_destroy(&start) == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::fputs("plugin test: choose threads, fork, exec, fault, or exit\n", stderr);
        return 1;
    }
    if (std::strcmp(argv[1], "threads") == 0) {
        return run_threads();
    }
    if (std::strcmp(argv[1], "fork") == 0) {
        const pid_t child = fork();
        if (child == 0) {
            _exit(0);
        }
        int status = 0;
        return child > 0 && waitpid(child, &status, 0) == child && status == 0 ? 0 : 1;
    }
    if (std::strcmp(argv[1], "exec") == 0) {
        execl("/bin/true", "true", static_cast<char*>(nullptr));
        return 1;
    }
    if (std::strcmp(argv[1], "fault") == 0) {
        (void)raise(SIGSEGV);
        return 1;
    }
    if (std::strcmp(argv[1], "exit") == 0) {
        return 7;
    }
    std::fputs("plugin test: unknown target behavior\n", stderr);
    return 1;
}
