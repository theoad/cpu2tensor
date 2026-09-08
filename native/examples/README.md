# Native examples

`checksum.c` is the small observation fixture. `branch_digits.c` and
`digit_paths.c` supply the [trace learning example](../../docs/learn-trace.md).
`stdio_digits.c` supplies the [stdin learning example](../../docs/stdio-example.md).
`kernel_init.c` is the [benign kernel guest](../../docs/kernel-examples.md).
`layout_paths.S` is the freestanding AArch64 static PIE for the
[normalization notebook](../../docs/tutorials/normalization.ipynb), built only on Linux AArch64.

All targets are built through the single native CMake project. The optional
static guest target uses `CPU2TENSOR_BUILD_KERNEL_EXAMPLE=ON` on Linux.
