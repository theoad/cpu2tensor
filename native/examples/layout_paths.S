/* SPDX-License-Identifier: AGPL-3.0-only */

/*
 * Read and echo digits 0..3. Every valid byte executes the same number of
 * blocks; only the selected path address differs. All code references are
 * relative, so linking the same object elsewhere preserves its text bytes.
 */
.text
.balign 4
.global _start
.global layout_path_0
.global layout_path_1
.global layout_path_2
.global layout_path_3
.global sample_end
.type _start, %function

_start:
    sub sp, sp, #16

read_byte:
    mov x0, #0
    mov x1, sp
    mov x2, #1
    mov x8, #63
    svc #0
    cbz x0, exit_success
    cmp x0, #1
    b.ne exit_failure

    ldrb w9, [sp]
    sub w9, w9, #'0'
    cmp w9, #3
    b.hi exit_failure

    /* Four eight-byte paths keep the dispatch and trace length identical. */
    adr x11, layout_path_0
    add x11, x11, x9, lsl #3
    br x11

layout_path_0:
    mov x10, #0
    b sample_end
layout_path_1:
    mov x10, #1
    b sample_end
layout_path_2:
    mov x10, #2
    b sample_end
layout_path_3:
    mov x10, #3
    b sample_end

sample_end:
    mov x0, #1
    mov x1, sp
    mov x2, #1
    mov x8, #64
    svc #0
    cmp x0, #1
    b.ne exit_failure
    b read_byte

exit_success:
    mov x0, #0
    b exit_process
exit_failure:
    mov x0, #2
exit_process:
    mov x8, #93
    svc #0
.size _start, .-_start

.section .note.GNU-stack,"",%progbits
