// SPDX-License-Identifier: AGPL-3.0-only
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

// Named data and instruction labels let tests compare the trace with the ELF.
uint8_t signals_u8;
uint16_t signals_u16;
uint32_t signals_u32;
uint64_t signals_u64;
_Alignas(16) uint64_t signals_u128[2];
_Alignas(16) const uint64_t signals_vector_input[2] = {
    UINT64_C(0x0123456789abcdef), UINT64_C(0xfedcba9876543210)
};

__attribute__((noinline)) void signals_run(void)
{
#if defined(__aarch64__)
    __asm__ volatile (
        "adrp x9, signals_u8\n"
        "add x9, x9, :lo12:signals_u8\n"
        "mov w10, #0xa5\n"
        ".global signals_store_u8\n"
        "signals_store_u8: strb w10, [x9]\n"
        ".global signals_load_u8\n"
        "signals_load_u8: ldrb w11, [x9]\n"
        ".global signals_load_signed_u8\n"
        "signals_load_signed_u8: ldrsb x12, [x9]\n"

        "adrp x9, signals_u16\n"
        "add x9, x9, :lo12:signals_u16\n"
        "mov w10, #0xb6c7\n"
        ".global signals_store_u16\n"
        "signals_store_u16: strh w10, [x9]\n"
        ".global signals_load_u16\n"
        "signals_load_u16: ldrh w11, [x9]\n"

        "adrp x9, signals_u32\n"
        "add x9, x9, :lo12:signals_u32\n"
        "movz w10, #0xfa0b\n"
        "movk w10, #0xd8e9, lsl #16\n"
        ".global signals_store_u32\n"
        "signals_store_u32: str w10, [x9]\n"
        ".global signals_load_u32\n"
        "signals_load_u32: ldr w11, [x9]\n"

        "adrp x9, signals_u64\n"
        "add x9, x9, :lo12:signals_u64\n"
        "movz x10, #0x3210\n"
        "movk x10, #0x7654, lsl #16\n"
        "movk x10, #0xba98, lsl #32\n"
        "movk x10, #0xfedc, lsl #48\n"
        ".global signals_store_u64\n"
        "signals_store_u64: str x10, [x9]\n"
        ".global signals_load_u64\n"
        "signals_load_u64: ldr x11, [x9]\n"

        "adrp x9, signals_vector_input\n"
        "add x9, x9, :lo12:signals_vector_input\n"
        "ldr q0, [x9]\n"
        "adrp x9, signals_u128\n"
        "add x9, x9, :lo12:signals_u128\n"
        ".global signals_store_u128\n"
        "signals_store_u128: str q0, [x9]\n"
        ".global signals_load_u128\n"
        "signals_load_u128: ldr q0, [x9]\n"

        // Each branch forces a new entry sample; the middle two keep x0 equal.
        "mov x0, #0\n"
        "mov x1, #0x8000000000000000\n"
        "cmp x0, x0\n"
        "b signals_register_zero\n"
        ".global signals_register_zero\n"
        "signals_register_zero: mov x0, x10\n"
        "b signals_register_high\n"
        ".global signals_register_high\n"
        "signals_register_high: nop\n"
        "b signals_register_unchanged\n"
        ".global signals_register_unchanged\n"
        "signals_register_unchanged: mov x0, #0\n"
        "b signals_register_zero_again\n"
        ".global signals_register_zero_again\n"
        "signals_register_zero_again: nop\n"
        : : : "x0", "x1", "x9", "x10", "x11", "x12", "v0", "cc", "memory"
    );
#elif defined(__x86_64__)
    __asm__ volatile (
        "movb $0xa5, %%al\n"
        ".global signals_store_u8\n"
        "signals_store_u8: movb %%al, signals_u8(%%rip)\n"
        ".global signals_load_u8\n"
        "signals_load_u8: movzbq signals_u8(%%rip), %%rdx\n"
        ".global signals_load_signed_u8\n"
        "signals_load_signed_u8: movsbq signals_u8(%%rip), %%rdx\n"

        "movw $0xb6c7, %%ax\n"
        ".global signals_store_u16\n"
        "signals_store_u16: movw %%ax, signals_u16(%%rip)\n"
        ".global signals_load_u16\n"
        "signals_load_u16: movzwq signals_u16(%%rip), %%rdx\n"

        "movl $0xd8e9fa0b, %%eax\n"
        ".global signals_store_u32\n"
        "signals_store_u32: movl %%eax, signals_u32(%%rip)\n"
        ".global signals_load_u32\n"
        "signals_load_u32: movl signals_u32(%%rip), %%edx\n"

        "movabs $0xfedcba9876543210, %%rax\n"
        ".global signals_store_u64\n"
        "signals_store_u64: movq %%rax, signals_u64(%%rip)\n"
        ".global signals_load_u64\n"
        "signals_load_u64: movq signals_u64(%%rip), %%rdx\n"

        "movdqa signals_vector_input(%%rip), %%xmm0\n"
        ".global signals_store_u128\n"
        "signals_store_u128: movdqa %%xmm0, signals_u128(%%rip)\n"
        ".global signals_load_u128\n"
        "signals_load_u128: movdqa signals_u128(%%rip), %%xmm0\n"

        // RCX remains fixed; RAX follows the same checkpoints as ARM's X0.
        "mov $0, %%rax\n"
        "movabs $0x8000000000000000, %%rcx\n"
        "cmp %%rax, %%rax\n"
        "jmp signals_register_zero\n"
        ".global signals_register_zero\n"
        "signals_register_zero: movabs $0xfedcba9876543210, %%rax\n"
        "jmp signals_register_high\n"
        ".global signals_register_high\n"
        "signals_register_high: nop\n"
        "jmp signals_register_unchanged\n"
        ".global signals_register_unchanged\n"
        "signals_register_unchanged: mov $0, %%rax\n"
        "jmp signals_register_zero_again\n"
        ".global signals_register_zero_again\n"
        "signals_register_zero_again: nop\n"
        : : : "rax", "rcx", "rdx", "xmm0", "cc", "memory"
    );
#else
#error "The signal fixture needs AArch64 or x86-64."
#endif
}

int main(void)
{
    signals_run();
    if (signals_u8 != UINT8_C(0xa5) || signals_u16 != UINT16_C(0xb6c7) ||
        signals_u32 != UINT32_C(0xd8e9fa0b) ||
        signals_u64 != UINT64_C(0xfedcba9876543210) ||
        signals_u128[0] != signals_vector_input[0] ||
        signals_u128[1] != signals_vector_input[1]) {
        fputs("signals: stored values differ\n", stderr);
        return 1;
    }
    const uint64_t checksum = signals_u8 ^ signals_u16 ^ signals_u32 ^
        signals_u64 ^ signals_u128[0] ^ signals_u128[1];
    if (printf("signals: ok checksum=%016" PRIx64 "\n", checksum) < 0) {
        return 1;
    }
    return fflush(stdout) == 0 ? 0 : 1;
}
