// SPDX-License-Identifier: AGPL-3.0-only
#include "digit_paths.h"

// A volatile side effect keeps ten distinct callable paths in optimized builds.
static volatile unsigned visits[10];
__attribute__((noinline)) void digit_0(void) { ++visits[0]; }
__attribute__((noinline)) void digit_1(void) { ++visits[1]; }
__attribute__((noinline)) void digit_2(void) { ++visits[2]; }
__attribute__((noinline)) void digit_3(void) { ++visits[3]; }
__attribute__((noinline)) void digit_4(void) { ++visits[4]; }
__attribute__((noinline)) void digit_5(void) { ++visits[5]; }
__attribute__((noinline)) void digit_6(void) { ++visits[6]; }
__attribute__((noinline)) void digit_7(void) { ++visits[7]; }
__attribute__((noinline)) void digit_8(void) { ++visits[8]; }
__attribute__((noinline)) void digit_9(void) { ++visits[9]; }
__attribute__((noinline)) void sample_end(void) { __asm__ volatile ("" ::: "memory"); }
void visit_digit(unsigned digit) {
    switch (digit) {
    case 0: digit_0(); break;
    case 1: digit_1(); break;
    case 2: digit_2(); break;
    case 3: digit_3(); break;
    case 4: digit_4(); break;
    case 5: digit_5(); break;
    case 6: digit_6(); break;
    case 7: digit_7(); break;
    case 8: digit_8(); break;
    case 9: digit_9(); break;
    }
}
