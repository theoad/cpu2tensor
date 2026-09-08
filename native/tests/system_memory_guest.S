/* SPDX-License-Identifier: AGPL-3.0-only */
/* A standalone, benign paging oracle. Build instructions are in docs/. */
.section .text
.global _start
.code16
_start:
    cli
    xor %ax, %ax
    mov %ax, %ds
    mov %ax, %es
    mov %ax, %ss
    mov $0x7c00, %sp
    lgdt gdt_pointer
    mov %cr0, %eax
    or $1, %eax
    mov %eax, %cr0
    ljmp $8, $protected
.code32
protected:
    mov $16, %ax
    mov %ax, %ds
    mov %ax, %es
    mov %ax, %ss
    mov $0x1000, %edi
    xor %eax, %eax
    mov $5120, %ecx
    rep stosl
    movl $0x2003, 0x1000
    movl $0x5003, 0x4000
    mov $3, %eax
    xor %ecx, %ecx
fill_pages:
    mov %eax, 0x2000(,%ecx,4)
    mov %eax, 0x5000(,%ecx,4)
    add $4096, %eax
    inc %ecx
    cmp $1024, %ecx
    jne fill_pages
    /* Adjacent virtual pages use deliberately nonadjacent physical pages. */
    movl $0x20003, 0x2020
    movl $0x30003, 0x2024
    movl $0x20003, 0x2028
    movl $0xfee00003, 0x202c
    movl $0x40003, 0x5020
    movl $0x50003, 0x5024
    mov $0x1000, %eax
    mov %eax, %cr3
    mov %cr0, %eax
    or $0x80000000, %eax
    mov %eax, %cr0
    movl $0x11223344, 0x8ffe
    mov 0x8ffe, %eax
    movw 0xaffe, %ax
    mov $0x4000, %eax
    mov %eax, %cr3
    movl $0x55667788, 0x8ffe
    mov 0x8ffe, %eax
    mov $0x1000, %eax
    mov %eax, %cr3
    mov 0x8ffe, %eax
    movl $0x40003, 0x2020
    invlpg (0x8000)
    mov 0x8ffe, %eax
    mov $0x80, %eax
    mov %eax, %cr4
    mov 0x8ffe, %eax
    mov $0xc0000080, %ecx
    mov $0x800, %eax
    xor %edx, %edx
    wrmsr
    mov 0x8ffe, %eax
    /* Read-only local APIC version register, to distinguish RAM from MMIO. */
    mov 0xb030, %eax
    /* Enter long mode, then return to a 32-bit code segment. The optional
       state hook must retain raw GPR high bits in that compatibility segment. */
    mov %cr0, %eax
    and $0x7fffffff, %eax
    mov %eax, %cr0
    movl $0x2003, 0x1000
    movl $0x3003, 0x2000
    movl $0, 0x2004
    movl $0x83, 0x3000
    mov $0xa0, %eax
    mov %eax, %cr4
    mov $0x900, %eax
    wrmsr
    mov %cr0, %eax
    or $0x80000000, %eax
    mov %eax, %cr0
    ljmp $24, $long_mode
.code64
long_mode:
    movabs $0x1122334455667788, %rax
    movabs $0x8877665544332211, %r8
    mov %rax, 0x6000
    pushq $8
    /* Encode push imm32 explicitly: this file is linked as ELF32. */
    .byte 0x68
    .long compatibility_mode
    lretq
.code32
compatibility_mode:
    mov %eax, 0x6008
    mov $0x10, %eax
    out %al, $0xf4
    hlt
.balign 8
gdt:
    .quad 0
    .quad 0x00cf9a000000ffff
    .quad 0x00cf92000000ffff
    .quad 0x00af9a000000ffff
gdt_pointer:
    .word 31
    .long gdt
.org 510
.word 0xaa55
