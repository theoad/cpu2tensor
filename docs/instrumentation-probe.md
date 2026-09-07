# Register and memory API probe

Checked on 2026-09-07 against the operator's QEMU 11.0.3 on `trail-arm`.
This is a source/header review, not proof that the new capture path passes its
runtime checks. No QEMU installation or build was performed by this probe.

The inspected source root is
`~/.cache/cpu2tensor/qemu-aarch64/11.0.3-20260907/source`; the matching public
header is in the adjacent `install/include/qemu-plugin.h`. Paths and line numbers
below refer to that source root, unless an AlphaFlow path is given.

## Registers

- `GArray *qemu_plugin_get_registers(void)` belongs in the vCPU-init callback.
  Descriptors contain a handle, name, optional feature name, and read-only flag.
  Free the returned array, not the constant strings. Discover each vCPU's handles
  in its own context; do not assume a global profile without checking it.
- `bool qemu_plugin_read_register(handle, GByteArray *buffer)` returns bytes in
  target byte order. The GDB reader appends to the buffer: set its length to zero
  before every read. Allocate and establish supported sizes at initialization;
  retain the allocation for block callbacks. A false result, changed width, or
  unsupported width must be explicit, never interpreted as a zero register.
- A null handle can be valid. `plugins/api.c:420` encodes register indices as
  pointers, including zero. Match the descriptor name with a separate found flag.
- Request `QEMU_PLUGIN_CB_R_REGS` on block-entry execution callbacks. Register
  reads are also allowed in callbacks with no flags argument except atexit and
  flush. Source `plugins/core.c:307` sets read/write access for vCPU exit.
- A block-entry snapshot describes state before that block executes. Differences
  between two snapshots are sampled changes; a register changed and restored
  within a block is invisible. Do not call these every instruction's writes.
- ARM's core profile is `x0` through `x30`, `sp`, and `pc` at eight bytes, with
  `cpsr` at four bytes. See `target/arm/gdbstub64.c:35` and
  `gdbstub/gdb-xml/aarch64-core.xml`. Vector and SVE/SME descriptors can be much
  wider. An explicitly named scalar profile is safer than truncating every
  descriptor to 64 bits. The first snapshot must emit an initial value for every
  selected register so a consumer can reconstruct later sampled state.
- The installed header also contains `qemu_plugin_read_x86_64_regs`, explicitly
  documented as a private extension. The new portable implementation should use
  the public descriptor/read API rather than depend on this AlphaFlow extension.

## Memory

Register `qemu_plugin_register_vcpu_mem_cb(insn, callback,
QEMU_PLUGIN_CB_NO_REGS, QEMU_PLUGIN_MEM_RW, userdata)` for each translated
instruction. Its callback receives the source index, memory info, virtual address,
and user data. Attach the instruction address, not just the containing block
address, if the event promises an instruction PC.

The public queries provide size shift, store/load direction, endianness, and
sign-extension metadata. Width is `1 << size_shift`. Memory callbacks run after
successful transactions; faulting accesses do not emit memory callbacks. One
instruction may emit several transactions, including vector and atomic operations.
Do not promise one callback per instruction or a global memory order.
These semantics are stated in `docs/devel/tcg-plugins.rst:89–124` and the header.

`qemu_plugin_mem_get_value(info)` returns a tagged unsigned value for 1, 2, 4, 8,
or 16 bytes. Integer fields are already host-order numeric values; the 128-bit
case has numeric `low` and `high` halves. Encode these fields explicitly for the
wire. Do not byte-swap a numeric value again merely because the guest access is
big-endian. Retain access byte order if clients need to reconstruct memory bytes.
Sign extension describes the load, while the value type remains the transaction
width. Reading memory again is not a substitute for the transaction value.

Guard the width before calling the value API: `plugins/api.c:349` asserts for
size shifts outside zero through four. Keep a validity field distinct from a zero
value. The runtime saves actual operation values in per-CPU fields before the
callback (`tcg/tcg-op-ldst.c:194–238`), so the getter needs no buffer allocation.
Not calling the getter avoids plugin-side value work, but does not prove QEMU
eliminates all underlying value instrumentation; measure the two modes later.

Other vCPU threads keep running during a memory callback. Per-source buffers and
progress preserve that concurrency. No callback should lock across all source
streams or reread shared memory to manufacture an access ordering.

For future system emulation, `qemu_plugin_get_hwaddr(info, address)` returns a
callback-lifetime handle for physical address and MMIO queries. Linux-user returns
null. Physical addresses may belong to different address spaces. DMA and host
syscall writes are outside the CPU load/store callback guarantee.

## Completion and faults

Normal thread destruction can invoke vCPU exit (`hw/core/cpu-common.c:280`;
`linux-user/syscall.c:9721`). However, process exit follows
`qemu_plugin_user_exit` (`plugins/core.c:798`), which unregisters all callbacks
except atexit before quiescing and flushing. It does not provide a final register
snapshot for every source. Atexit may flush captured records, but cannot read
registers. Therefore stream completion and the last sampled register state are
different facts; do not promise complete final-block register effects.

Block/instruction entry also does not prove all translated instructions completed:
a synchronous exception can stop execution within the block. Preserve worker exit
status and incomplete-stream errors independently of any last recorded block.

## Narrow AlphaFlow review

Reference inspected:
`AlphaFlow/tasks/execution-exploration/experiments/reachability-v0/plugin/block_plugin.c`
in the sibling checkout. These are limitations relevant to reuse, not a complete
audit. No source was copied and reference licensing was not established.

- The trace mode reads a private x86 snapshot (`af_trace_block`, line 1036).
  Its fixed 17-register profile and little-endian public-register decoder do not
  generalize to ARM, vector widths, or arbitrary byte order.
- Its first register read establishes a private baseline but emits no initial
  register values. A consumer of only its delta records cannot reconstruct all
  initial state. The new stream should include the initial selected profile.
- It assigns changes at the next in-scope block to the previous transition.
  With code filtering, intervening out-of-scope code can contribute those changes.
  Preserve sampling semantics rather than claim exclusive attribution.
- Memory values use the actual QEMU getter and distinguish invalid values, which
  is useful. However the width check before calling it allows shifts far above
  the getter's actual limit of four; this is not a portable unsupported-size path.
- The ring's full-buffer loop waits forever without a cancellation check
  (`af_reserve_trace_record`, line 952). Do not inherit this disconnect behavior.
- Translation descriptors remain on a list until process exit. Translation is not
  execution, repeated translations are normal, and long-running code can grow
  this storage. Avoid allocating one permanent descriptor per translation when
  a scalar address can be used safely instead.

## Available checks and proposed fixtures

`trail-arm` also has `x86_64-linux-gnu-gcc` and
`~/.cache/alphaflow/qemu/11.0.3/install/bin/qemu-x86_64` (reported version 11.0.3),
with its header under that installation's `include/`. This permits an additional
x86 user-mode correctness fixture on ARM; it is not x86-host performance evidence.
On `trail-x86`, `nvidia-smi` is not on PATH. CUDA execution remains unverified.

Use small benign fixtures with independently known results:

1. Branch across visible symbols while setting one scalar register to a high-bit
   value, leaving it unchanged for a block, then changing it. Check the initial
   snapshot, delta suppression, width, source, and following entry address.
2. Store and load known byte, halfword, word, doubleword, and vector patterns in
   a named static array. Check address, instruction address, direction, width,
   low/high bits, and values-off behavior. Check signed loads separately.
3. Exit immediately after a final block and verify final partial batches without
   claiming an unavailable exit register snapshot. Trigger a benign memory fault
   in a separate fixture to verify fault status and absence of a successful
   transaction record for that access.
4. Run independent pthread work over separate arrays and check per-source
   sequences, initial state, and completion. Retain existing slow-consumer and
   disconnect checks with the larger signal stream and retained tensor batches.

Actual runtime results belong in the integration evidence after implementation.

### Controlled fixture output

`native/tests/signals_target.c` now supplies named instruction labels and data for
the first two proposed checks. On `trail-arm`, strict C17 builds with `-O2 -Wall
-Wextra -Wpedantic -Werror -fno-pie -no-pie` passed native ARM, operator AArch64
QEMU, and operator x86 QEMU execution. The x86 build additionally used `-static`.
All three executions printed `signals: ok checksum=0123456751428186`. Builds are
under `~/.cache/cpu2tensor/signals-fixture-check`, off the shared source mount.
This verifies the fixture's stored values and output, not the new trace decoder.

The ELF exports `signals_store_u8`, `signals_load_u8`, and equivalent labels for
16, 32, 64, and 128 bits. Data symbols are `signals_u8` through `signals_u128`.
Expected unsigned values are `a5`, `b6c7`, `d8e9fa0b`, `fedcba9876543210`, and
128-bit low/high halves `0123456789abcdef` / `fedcba9876543210`. A separate
`signals_load_signed_u8` loads the first byte with sign extension. Tests should
check actual callback transaction sizes for the vector instruction rather than
assume that every ISA implements it as one 16-byte transaction.

Branches expose four entry labels: `signals_register_zero`,
`signals_register_high`, `signals_register_unchanged`, and
`signals_register_zero_again`. At these entries, ARM `x0` or x86 `rax` is zero,
`fedcba9876543210`, the same value, and zero. ARM `x1` or x86 `rcx` remains
`8000000000000000`. A compare before the first branch establishes flags that the
remaining moves and branches preserve: ARM NZCV mask `f0000000` should equal
`60000000`; x86 arithmetic flag mask `8d5` should equal `44`. Other status bits
are intentionally unspecified. All assembly operands are properly aligned,
scratch register changes and condition-code changes are declared, and the C
caller verifies every stored value before reporting success.
