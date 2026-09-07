# Register and memory observations

This slice extends block tracing with general-register changes and successful
memory transactions. AlphaFlow is a reviewed reference, not copied code. See
[investigation](instrumentation-probe.md) for API and semantic evidence.

## Capture settings

Worker defaults: `--registers general --memory on --memory-values off`.
`--registers none` disables register sampling; `--registers all` requests all
registers QEMU exposes, subject to explicit supported width/count limits.
`--memory off` disables memory callbacks. Values require memory to be enabled.
No memory value lookup or payload is produced when values are disabled.

Registers have names and byte widths discovered per vCPU using the public API.
The general profile selects AArch64 x0–x30, sp, pc, cpsr; x86-64 integer general
registers, rip, eflags, segment selectors and fs_base/gs_base where exposed.
All mode supports at most 512 registers, each at most 256 bytes. Unsupported or
changing widths fail explicitly, never truncate. Wide SVE/SME configurations may
exceed this limit. General mode is the portable default.

The first sample emits every selected register, including zero values. Later
samples emit only changed values. Sample at block entry and syscall entry; a
vCPU-exit callback can provide an additional sample. Atexit only drains buffers:
QEMU does not permit register reads there. Consequently, stream completion is
not a guarantee of the final register state. Changes that happen and reverse
between checkpoints are not observed. Registers are sampled before the new block;
these deltas must not be labeled as effects of that block.

Memory callbacks report successful emulated accesses, not faulting accesses,
syscall copy buffers, DMA, or all other writers to guest memory. One instruction
can produce several transactions. Memory values are actual callback transaction
values up to 16 bytes; a wider access with values enabled fails explicitly.
Numeric values are encoded in little-endian significance order. A flag retains
whether the guest access was big endian. No global counter or per-event lock is
added. Source sequence order interleaves block, register, and memory events from
that source only; it does not establish cross-vCPU memory order.

## Version 2 framing

Keep the 32-byte header and magic from the first [batch contract](batches.md),
with version now 2. Maximum payload is 4064 bytes, so each full frame fits in
Linux PIPE_BUF. Block frames still hold at most 256 eight-byte addresses.
Hello detail: low byte architecture (1 ARM, 2 x86); bit 8 memory, bit 9 registers,
bit 10 memory values. Reject unknown bits and values without memory.

New kinds: RegisterSchema=6, Registers=7, Memory=8. Their header detail is payload
byte count. Header count is row count. Registers and Memory advance the same
per-source sequence as Blocks. Schema frames carry sequence zero and do not
advance it; they must precede data for that source. SourceEnd uses the next event
sequence, including register and memory events. Old clients reject version 2.

Schema rows are 72 bytes: id uint32, width uint32, name 64 zero-padded ASCII bytes
with a required terminator. IDs are 0..511, widths 1..256, names nonempty and
unique within a source. Several frames can describe a source. Limits apply per
source, never assume another source has identical handles or schema.

Register rows have a 16-byte prefix: checkpoint PC uint64, id uint32, width uint16,
flags uint16; then exactly width value bytes in little-endian significance order.
Flags: checkpoint in low byte (1 block entry, 2 syscall entry, 3 vCPU exit),
bit 8 initial baseline. A register's first observation must be a baseline.

All declared registers need baselines before a block or memory event, and before
ending a nonempty source. The baseline can span several register frames. A source
with no events may end without a baseline, with or without a schema: QEMU provides
no legal final register read for a source that never reached a checkpoint.

Memory rows: instruction PC uint64, address uint64, size uint32, flags uint32,
then optionally 16 value bytes when Hello enabled values. Flags: bit 0 store,
bit 1 big-endian guest access. Size is a power of two. Value padding is zero.
Source attribution and instruction PC come from the actual memory callback.

The native decoder validates schemas, row bounds, widths, flags, and sequences,
and returns owned column buffers. It performs per-event transcode in native code;
Python only constructs a fixed number of tensors per batch.

## Python objects

`Pool.read()` remains a synchronous iterator. `Batch.addresses` contains block
entry addresses for block batches and is empty for other signal batches.
`Batch.registers` and `Batch.memory` are optional typed tables, populated for the
corresponding batch. No Python object per event is created.

RegisterChanges: `pc` int64[N], `ids` int64[N], `widths` int64[N], `flags` int64[N],
`values` uint8[N,W], with zero padding to the widest row W. `names` maps IDs to
names in that source's schema. MemoryAccesses: `pc`, `addresses`, `sizes`, `flags`
int64[N], and `values` uint8[N,16] or None. Bit patterns remain exact on CPU/MPS.
A batch's first_sequence plus row index identifies its event sequence. Each batch
owns its storage independently; schema metadata is small per-source state.

The default worker now produces all three batch types. Address-only clients can
explicitly select `--registers none --memory off` to retain the minimal path.
EOF without completion, source gaps, invalid records, and unsupported capabilities
remain explicit errors. CUDA support will use the same typed buffers; actual
CUDA validation requires an operator-provided device.


## Cost and coverage

Frames currently contain one signal kind and flush when kind changes. Default
capture therefore often produces small frames. This is a correctness baseline,
not an optimized aggregate trace rate. Block-only capture still batches up to 256
addresses. Register reads are public per-register calls with reusable storage;
they avoid AlphaFlow's private x86-only shortcut. There is no Python object per
event inside a batch, but small batches still incur Python and device dispatch
overhead. Measure and improve batching before claiming GPU-bound scale.

## Stdin interaction extension

Optional Hello bit 11 selects single-vCPU stdin interaction. Kind 9 is an input
request: source 0, count 0, sequence equal to the next source event, and detail
holding the maximum action byte count (1..256). It has no payload and does not
advance source event sequence. The ordinary Pool rejects this feature; use
StdioEnv. Older decoders reject the newly recognized feature rather than hang.

The plugin flushes existing samples before the stdin-read boundary. The worker
waits for a real stopped QEMU child before publishing the request. Actions are a
little-endian uint32 byte count followed by those exact bytes. Delivery and wait
have bounded timeouts; the worker resumes QEMU after delivery. See the
[stdin contract](stdio-example.md) for single-thread and signal/descriptor limits.
