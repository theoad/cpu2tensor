// SPDX-License-Identifier: AGPL-3.0-only
// Deterministic benign canaries for aligned PT, PEBS, PMU, and CPU timing capture.

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

static const uint64_t MAX_LOOPS = 100000000ULL;
static const size_t HOT_POINTER_BYTES = 4096;
static const size_t LARGE_POINTER_BYTES = 64ULL * 1024ULL * 1024ULL;
static const uint32_t POINTER_STRIDE = 17;
static const uint32_t PHASE_DELAY_STEPS = 1024;

struct counter_cell {
    _Alignas(64) _Atomic uint64_t value;
};

struct thread_gate {
    _Atomic int start;
    _Atomic unsigned ready;
};

struct contention_worker {
    struct thread_gate *gate;
    struct counter_cell *counter;
    uint64_t loops;
    int cpu;
    int status;
};

struct phase_state {
    struct thread_gate gate;
    _Atomic uint64_t phase;
    _Atomic uint64_t acknowledged;
    uint64_t payload;
};

struct phase_worker {
    struct phase_state *state;
    uint64_t loops;
    uint64_t result;
    volatile uint64_t delay_digest;
    int cpu;
    int producer;
    int shifted;
    int status;
};

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s MODE VARIANT LOOPS [CPU_A [CPU_B]]\n"
            "  flow       ordered|permuted\n"
            "  memory     hot|large\n"
            "  contention private|shared\n"
            "  phase      normal|shifted\n",
            program);
}

static int parse_u64(const char *text, uint64_t *value) {
    if (text[0] == '\0') {
        return -1;
    }
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            return -1;
        }
    }
    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return -1;
    }
    *value = (uint64_t)parsed;
    return 0;
}

static int parse_cpu(const char *text, int *cpu) {
    uint64_t parsed = 0;
    if (parse_u64(text, &parsed) != 0 || parsed >= CPU_SETSIZE) {
        return -1;
    }
    *cpu = (int)parsed;
    return 0;
}

static int pin_current_thread(int cpu) {
    if (cpu < 0) {
        return 0;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static int gate(void) {
    const char ready[] = "READY\n";
    char start = 0;
    if (write(STDOUT_FILENO, ready, sizeof(ready) - 1) !=
        (ssize_t)(sizeof(ready) - 1)) {
        return -1;
    }
    return read(STDIN_FILENO, &start, 1) == 1 ? 0 : -1;
}

static uint64_t triangular(uint64_t value) {
    if ((value & 1U) == 0) {
        return (value / 2) * (value + 1);
    }
    return value * ((value + 1) / 2);
}

static int print_result(const char *mode, const char *variant, uint64_t loops,
                        uint64_t value, uint64_t expected, int cpu_a, int cpu_b) {
    const char *oracle = value == expected ? "pass" : "fail";
    printf("RESULT mode=%s variant=%s loops=%llu value=%llu expected=%llu "
           "cpu_a=%d cpu_b=%d oracle=%s\n",
           mode, variant, (unsigned long long)loops, (unsigned long long)value,
           (unsigned long long)expected, cpu_a, cpu_b, oracle);
    return value == expected ? 0 : 5;
}

static NOINLINE void flow_one(volatile uint64_t *value, uint64_t input) {
    *value += input;
}

static NOINLINE void flow_two(volatile uint64_t *value, uint64_t input) {
    *value += input * 2;
}

static NOINLINE void flow_three(volatile uint64_t *value, uint64_t input) {
    *value += input * 3;
}

static NOINLINE void flow_four(volatile uint64_t *value, uint64_t input) {
    *value += input * 4;
}

static int run_flow(const char *variant, uint64_t loops, int cpu) {
    int status = pin_current_thread(cpu);
    if (status != 0) {
        fprintf(stderr, "cannot pin flow worker to CPU %d: %s\n", cpu,
                strerror(status));
        return 4;
    }
    if (gate() != 0) {
        return 3;
    }
    volatile uint64_t result = 0;
    const int permuted = strcmp(variant, "permuted") == 0;
    for (uint64_t index = 1; index <= loops; ++index) {
        if (permuted) {
            flow_three(&result, index);
            flow_one(&result, index);
            flow_four(&result, index);
            flow_two(&result, index);
        } else {
            flow_one(&result, index);
            flow_two(&result, index);
            flow_three(&result, index);
            flow_four(&result, index);
        }
    }
    return print_result("flow", variant, loops, result,
                        triangular(loops) * 10, cpu, -1);
}

static int run_memory(const char *variant, uint64_t loops, int cpu) {
    const int large = strcmp(variant, "large") == 0;
    const size_t active_bytes = large ? LARGE_POINTER_BYTES : HOT_POINTER_BYTES;
    const uint32_t stride = POINTER_STRIDE;
    const size_t count = active_bytes / sizeof(uint32_t);
    const size_t allocated_count = LARGE_POINTER_BYTES / sizeof(uint32_t);
    uint32_t *links = malloc(LARGE_POINTER_BYTES);
    if (links == NULL) {
        fprintf(stderr, "cannot allocate %zu-byte pointer-chase table\n",
                LARGE_POINTER_BYTES);
        return 4;
    }
    const uint32_t mask = (uint32_t)(count - 1);
    for (uint32_t index = 0; index < (uint32_t)allocated_count; ++index) {
        links[index] = index < count ? (index + stride) & mask : index;
    }
    int status = pin_current_thread(cpu);
    if (status != 0) {
        fprintf(stderr, "cannot pin memory worker to CPU %d: %s\n", cpu,
                strerror(status));
        free(links);
        return 4;
    }
    if (gate() != 0) {
        free(links);
        return 3;
    }
    volatile uint32_t *observed_links = links;
    uint32_t current = 0;
    for (uint64_t index = 0; index < loops; ++index) {
        current = observed_links[current];
    }
    const uint64_t expected = (loops * stride) & mask;
    free(links);
    return print_result("memory", variant, loops, current, expected, cpu, -1);
}

static void wait_for_workers(struct thread_gate *thread_gate, unsigned count) {
    while (atomic_load_explicit(&thread_gate->ready, memory_order_acquire) != count) {
        sched_yield();
    }
}

static int wait_for_start(struct thread_gate *thread_gate) {
    int start = 0;
    while ((start = atomic_load_explicit(&thread_gate->start,
                                          memory_order_acquire)) == 0) {
        sched_yield();
    }
    return start;
}

static void *contention_worker_main(void *opaque) {
    struct contention_worker *worker = opaque;
    worker->status = pin_current_thread(worker->cpu);
    atomic_fetch_add_explicit(&worker->gate->ready, 1, memory_order_release);
    if (wait_for_start(worker->gate) < 0 || worker->status != 0) {
        return NULL;
    }
    for (uint64_t index = 0; index < worker->loops; ++index) {
        atomic_fetch_add_explicit(&worker->counter->value, 1,
                                  memory_order_relaxed);
    }
    return NULL;
}

static int run_contention(const char *variant, uint64_t loops, int cpu_a,
                          int cpu_b) {
    struct counter_cell counters[2];
    atomic_init(&counters[0].value, 0);
    atomic_init(&counters[1].value, 0);
    struct thread_gate thread_gate;
    atomic_init(&thread_gate.start, 0);
    atomic_init(&thread_gate.ready, 0);
    const int shared = strcmp(variant, "shared") == 0;
    struct contention_worker workers[2] = {
        {.gate = &thread_gate, .counter = &counters[0], .loops = loops,
         .cpu = cpu_a, .status = 0},
        {.gate = &thread_gate, .counter = shared ? &counters[0] : &counters[1],
         .loops = loops, .cpu = cpu_b, .status = 0},
    };
    pthread_t threads[2];
    int created = 0;
    for (; created < 2; ++created) {
        int status = pthread_create(&threads[created], NULL,
                                    contention_worker_main, &workers[created]);
        if (status != 0) {
            fprintf(stderr, "cannot create contention worker: %s\n",
                    strerror(status));
            atomic_store_explicit(&thread_gate.start, -1, memory_order_release);
            for (int index = 0; index < created; ++index) {
                pthread_join(threads[index], NULL);
            }
            return 4;
        }
    }
    wait_for_workers(&thread_gate, 2);
    for (int index = 0; index < 2; ++index) {
        if (workers[index].status != 0) {
            fprintf(stderr, "cannot pin contention worker to CPU %d: %s\n",
                    workers[index].cpu, strerror(workers[index].status));
            atomic_store_explicit(&thread_gate.start, -1, memory_order_release);
            for (int joined = 0; joined < 2; ++joined) {
                pthread_join(threads[joined], NULL);
            }
            return 4;
        }
    }
    if (gate() != 0) {
        atomic_store_explicit(&thread_gate.start, -1, memory_order_release);
        for (int index = 0; index < 2; ++index) {
            pthread_join(threads[index], NULL);
        }
        return 3;
    }
    atomic_store_explicit(&thread_gate.start, 1, memory_order_release);
    for (int index = 0; index < 2; ++index) {
        pthread_join(threads[index], NULL);
    }
    uint64_t result = atomic_load_explicit(&counters[0].value,
                                            memory_order_relaxed);
    if (!shared) {
        result += atomic_load_explicit(&counters[1].value,
                                       memory_order_relaxed);
    }
    return print_result("contention", variant, loops, result, loops * 2,
                        cpu_a, cpu_b);
}

static NOINLINE uint64_t phase_delay(uint64_t seed) {
    volatile uint64_t value = seed | 1U;
    for (uint32_t index = 0; index < PHASE_DELAY_STEPS; ++index) {
        value = value * 2862933555777941757ULL + 3037000493ULL;
    }
    return value;
}

static void *phase_worker_main(void *opaque) {
    struct phase_worker *worker = opaque;
    worker->status = pin_current_thread(worker->cpu);
    atomic_fetch_add_explicit(&worker->state->gate.ready, 1,
                              memory_order_release);
    if (wait_for_start(&worker->state->gate) < 0 || worker->status != 0) {
        return NULL;
    }
    for (uint64_t round = 1; round <= worker->loops; ++round) {
        if (worker->producer) {
            if (!worker->shifted) {
                worker->delay_digest ^= phase_delay(round);
            }
            worker->state->payload = round;
            atomic_store_explicit(&worker->state->phase, round,
                                  memory_order_release);
            if (worker->shifted) {
                worker->delay_digest ^= phase_delay(round);
            }
            while (atomic_load_explicit(&worker->state->acknowledged,
                                        memory_order_acquire) != round) {
                sched_yield();
            }
        } else {
            if (!worker->shifted) {
                worker->delay_digest ^= phase_delay(round ^ UINT64_C(0x5555));
            }
            while (atomic_load_explicit(&worker->state->phase,
                                        memory_order_acquire) != round) {
                sched_yield();
            }
            if (worker->shifted) {
                worker->delay_digest ^= phase_delay(round ^ UINT64_C(0x5555));
            }
            worker->result += worker->state->payload;
            atomic_store_explicit(&worker->state->acknowledged, round,
                                  memory_order_release);
        }
    }
    return NULL;
}

static int run_phase(const char *variant, uint64_t loops, int cpu_a, int cpu_b) {
    struct phase_state state;
    atomic_init(&state.gate.start, 0);
    atomic_init(&state.gate.ready, 0);
    atomic_init(&state.phase, 0);
    atomic_init(&state.acknowledged, 0);
    state.payload = 0;
    const int shifted = strcmp(variant, "shifted") == 0;
    struct phase_worker workers[2] = {
        {.state = &state, .loops = loops, .result = 0, .delay_digest = 0,
         .cpu = cpu_a, .producer = 1, .shifted = shifted, .status = 0},
        {.state = &state, .loops = loops, .result = 0, .delay_digest = 0,
         .cpu = cpu_b, .producer = 0, .shifted = shifted, .status = 0},
    };
    pthread_t threads[2];
    int created = 0;
    for (; created < 2; ++created) {
        int status = pthread_create(&threads[created], NULL, phase_worker_main,
                                    &workers[created]);
        if (status != 0) {
            fprintf(stderr, "cannot create phase worker: %s\n", strerror(status));
            atomic_store_explicit(&state.gate.start, -1, memory_order_release);
            for (int index = 0; index < created; ++index) {
                pthread_join(threads[index], NULL);
            }
            return 4;
        }
    }
    wait_for_workers(&state.gate, 2);
    for (int index = 0; index < 2; ++index) {
        if (workers[index].status != 0) {
            fprintf(stderr, "cannot pin phase worker to CPU %d: %s\n",
                    workers[index].cpu, strerror(workers[index].status));
            atomic_store_explicit(&state.gate.start, -1, memory_order_release);
            for (int joined = 0; joined < 2; ++joined) {
                pthread_join(threads[joined], NULL);
            }
            return 4;
        }
    }
    if (gate() != 0) {
        atomic_store_explicit(&state.gate.start, -1, memory_order_release);
        for (int index = 0; index < 2; ++index) {
            pthread_join(threads[index], NULL);
        }
        return 3;
    }
    atomic_store_explicit(&state.gate.start, 1, memory_order_release);
    for (int index = 0; index < 2; ++index) {
        pthread_join(threads[index], NULL);
    }
    const uint64_t final_phase = atomic_load_explicit(&state.phase,
                                                       memory_order_relaxed);
    const uint64_t final_ack = atomic_load_explicit(&state.acknowledged,
                                                     memory_order_relaxed);
    if (final_phase != loops || final_ack != loops) {
        fprintf(stderr, "phase protocol did not reach its exact terminal state\n");
        return 5;
    }
    return print_result("phase", variant, loops, workers[1].result,
                        triangular(loops), cpu_a, cpu_b);
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        usage(argv[0]);
        return 2;
    }
    uint64_t loops = 0;
    if (parse_u64(argv[3], &loops) != 0 || loops == 0 || loops > MAX_LOOPS) {
        fprintf(stderr, "loops must be an integer from 1 through %llu\n",
                (unsigned long long)MAX_LOOPS);
        return 2;
    }
    int cpu_a = -1;
    int cpu_b = -1;
    if (argc >= 5 && parse_cpu(argv[4], &cpu_a) != 0) {
        fprintf(stderr, "CPU_A must be a valid nonnegative Linux CPU number\n");
        return 2;
    }
    if (argc == 6 && parse_cpu(argv[5], &cpu_b) != 0) {
        fprintf(stderr, "CPU_B must be a valid nonnegative Linux CPU number\n");
        return 2;
    }

    const char *mode = argv[1];
    const char *variant = argv[2];
    if (strcmp(mode, "flow") == 0) {
        if ((strcmp(variant, "ordered") != 0 &&
             strcmp(variant, "permuted") != 0) || argc == 6) {
            usage(argv[0]);
            return 2;
        }
        return run_flow(variant, loops, cpu_a);
    }
    if (strcmp(mode, "memory") == 0) {
        if ((strcmp(variant, "hot") != 0 && strcmp(variant, "large") != 0) ||
            argc == 6) {
            usage(argv[0]);
            return 2;
        }
        return run_memory(variant, loops, cpu_a);
    }
    if (strcmp(mode, "contention") == 0) {
        if ((strcmp(variant, "private") != 0 &&
             strcmp(variant, "shared") != 0) || argc == 5) {
            usage(argv[0]);
            return 2;
        }
        return run_contention(variant, loops, cpu_a, cpu_b);
    }
    if (strcmp(mode, "phase") == 0) {
        if ((strcmp(variant, "normal") != 0 &&
             strcmp(variant, "shifted") != 0) || argc == 5) {
            usage(argv[0]);
            return 2;
        }
        return run_phase(variant, loops, cpu_a, cpu_b);
    }
    usage(argv[0]);
    return 2;
}
