# Register and memory observations

This slice extends block tracing with general-register changes and successful
memory transactions. AlphaFlow is a reviewed reference, not copied code. See
[investigation](instrumentation-probe.md) for API and semantic evidence.

## Capture settings

Worker defaults: `--registers general --memory on --memory-values off`.
`--registers none` disables register sampling; `--registers all` requests all
trustworthy registers the backend exposes, subject to width/count limits.
Use `--registers rax:xmm0` for exact names; the program counter is included
so checkpoints remain interpretable. Unknown or unavailable explicit names fail.
See [capture benchmarks](benchmark-capture.md) to measure a selection's cost.
`--memory off` disables memory callbacks. Values require memory to be enabled.
No memory value lookup or payload is produced when values are disabled.

Registers have names and byte widths discovered per vCPU using the public API.
The general profile selects AArch64 x0–x30, sp, pc, cpsr; x86-64 integer general
registers, rip, eflags, segment selectors and fs_base/gs_base where exposed.
Full-system x86 capture omits `eflags` in both profiles: upstream QEMU returns
stale lazy arithmetic flags inside block callbacks. The worker reports this,
and the schema is authoritative. See [the measured API failure](kernel-qemu-build.md).
The x86 `all` profile also omits six constant-zero GDB fields: `ftag`, `fiseg`,
`fioff`, `foseg`, `fooff`, and `fop`, with visible diagnostics. Exact x86 system
register capture requires the operator's [optional state hook](qemu-state-hook.md).
Without it the runtime explains the missing capability; `--registers none` still
permits system memory capture through public APIs. User-mode capture and
block-only system capture do not require that hook.

The hook supplies raw full-width GPR storage, including bits inaccessible in
16/32-bit code. AddressContext supplies execution width and CS base. Register
EIP/RIP is an offset in CS; instrumentation PCs are linear addresses. Interpret
them with that context rather than assuming they are interchangeable.

All mode supports at most 512 registers, each at most 256 bytes. Unsupported or
changing widths fail explicitly, never truncate. Wide SVE/SME configurations may
exceed this limit. General mode is the portable default.

The first sample emits every selected register, including zero values. Later
samples emit only changed values. Sample at block entry and, in Linux-user mode, syscall entry; a
vCPU-exit callback can provide an additional sample. Atexit only drains buffers:
QEMU does not permit register reads there. Consequently, stream completion is
not a guarantee of the final register state. Changes that happen and reverse
between checkpoints are not observed. Registers are sampled before the new block;
these deltas must not be labeled as effects of that block.

Memory callbacks report successful emulated accesses, not faulting accesses,
host syscall copy buffers in Linux-user mode, DMA, or all other writers to guest
memory. In a system guest, emulated kernel instructions are captured, including
their successful memory accesses. x86 system capture includes raw virtual
addresses, ordered paging context, and the supported physical prefix described
below. A vCPU stream can span guest processes: CR3 is not a PID, and neither VA
nor GPA alone identifies storage across every machine address space. ASLR is
not normalized. User-mode memory keeps its process-relative virtual addresses. One instruction
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
bit 10 memory values, bit 11 stdin actions, bit 12 system emulation, bit 13
kernel adapter, bit 14 explicit capture start window, bit 15 system memory
mapping, bit 16 address context, bit 17 initial executable layout. Bit 17 is
opt-in (`--layout on`) and requires user mode. Bit 15 requires memory and context; bit 16
requires x86 system mode. Older version-2 decoders reject these new feature bits.
System x86 register streams without address context are rejected: update a legacy
worker rather than interpret its GDB-view samples as raw architectural storage.
Reject unknown bits and
values without memory. Kernel actions require system mode; stdin actions cannot
be combined with system mode.

New kinds: RegisterSchema=6, Registers=7, Memory=8. Their header detail is payload
byte count. Header count is row count. Registers and Memory advance the same
per-source sequence as Blocks. Schema frames carry sequence zero and do not
advance it; they must precede data for that source. SourceEnd uses the next event
sequence, including register, memory and context events. Version-1 clients reject version 2.

Schema rows are 72 bytes: id uint32, width uint32, name 64 zero-padded ASCII bytes
with a required terminator. IDs are 0..511, widths 1..256, names nonempty and
unique within a source. Several frames can describe a source. Limits apply per
source, never assume another source has identical handles or schema.

### Initial executable layout

Kind 13 is a worker-wide record: source/sequence zero sentinels, count 1 and
detail 24. Its body contains three little-endian uint64 values: `code_start`,
`code_end` (exclusive) and `initial_entry`. The span is QEMU's nominal main
executable segment span, not ELF load bias, complete mappings or binary identity.
The initial entry may be in a dynamic interpreter outside that span. System mode
is unsupported; no zero-filled record is substituted.

Capture once at first translation after loading, before execution observations.
Register schemas may precede the record. The decoder requires it before source
data, rejects duplicates and never advances a vCPU sequence for it. In Python,
`Batch.layout.values` is one owned int64 tensor with shape `(3,)`; named properties
are views. `Batch.source` and `first_sequence` are None for this event. Pool and
StdioEnv deliver it through ordinary iteration; it is off by default.

This describes the initial user process only. Later loads, remaps, libraries,
kernel layout and cross-worker synchronization need their own contracts. See the
[notebook](tutorials/normalization.ipynb) for implementation and a measured example.

### Register rows

Register rows have a 16-byte prefix: checkpoint PC uint64, id uint32, width uint16,
flags uint16; then exactly width value bytes in little-endian significance order.
Flags: checkpoint in low byte (1 block entry, 2 syscall entry, 3 vCPU exit),
bit 8 initial baseline. A register's first observation must be a baseline.

All declared registers need baselines before a block or memory event, and before
ending a nonempty source. The baseline can span several register frames. A source
with no events may end without a baseline, with or without a schema: QEMU provides
no legal final register read for a source that never reached a checkpoint.

Memory rows start with instruction PC uint64, address uint64, size uint32,
flags uint32. With Hello bit 15, append physical address uint64, mapped-prefix
size uint32, mapping flags uint32, and context sequence uint64. Then append
16 value bytes only when Hello enabled values. Prefix size is 24 bytes without
mapping and 48 bytes with mapping; total row sizes are 24/40 or 48/64 bytes. Flags: bit 0 store,
bit 1 big-endian guest access. Size is a power of two. Value padding is zero.
Source attribution and instruction PC come from the actual memory callback.

### Ordered address context and mapping coverage

AddressContext=12 is a sparse event with count 1 and detail 64. Its eight uint64
fields are: linear checkpoint/instruction PC, raw CR0, raw CR3, raw CR4, raw EFER,
CS base, execution width, known-field mask. Bits 0..5 of the mask describe those
six state fields after PC. Mask 15 means only paging controls are available;
mask 63 means all fields are available, with width 16, 32 or 64. Unavailable
CS base/mode are zero with validity clear, never a claimed architectural zero.

Context advances the normal per-vCPU sequence. Emit it before the first captured
block/register checkpoint, and whenever its values change. Memory context is
sampled at the actual memory callback: some helpers change paging state within
an instruction, so a block-entry cache alone is insufficient. A memory row's
context sequence must reference the latest context on the same source. The
decoder checks these requirements and rejects stale or cross-source references.

Mapping flags: bit 0 means physical prefix known, bit 1 means QEMU dispatch flag
known, bit 2 means the access used the I/O dispatch path. Valid combinations are 0 (unavailable), 1 (physical
prefix only), 3 (direct dispatch), and 7 (I/O dispatch). A missing
mapping has zero address/size and flags 0. GPA zero with validity set is a real
mapping, not a missing value. The dispatch flag describes the first
address only, not every byte of a potentially mixed subpage region. It is not
storage classification: subpage wrappers can route RAM through I/O dispatch,
and ROMD reads can use the direct path.

The captured physical prefix ends no later than the next 4 KiB boundary:
`min(access_size, 4096 - (VA & 4095))`. A cross-page tail remains explicitly
unmapped; do not extend the first GPA over it or discard the known transaction
value bytes. This describes address translation, not a universal storage ID.
Unmodified upstream 11.0.3 misclassifies a verified APIC access; the dispatch flag
therefore remains unknown without the validated hook/MMIO patch.

No page-table walk or later guest-memory reread is performed by the collector.
The [paging probe](system-memory-probe.md) and [state checks](state-correctness-results.md)
record what was verified. Initial RAM, implicit MMU writes and DMA remain outside
the capture contract.

The native decoder validates schemas, row bounds, widths, flags, and sequences,
and returns owned column buffers. It performs per-event transcode in native code;
Python only constructs a fixed number of tensors per batch.

## Python objects

`Pool.read()` remains a synchronous iterator. `Batch.addresses` contains block
entry addresses. `Batch.registers`, `Batch.memory` and `Batch.context` are optional
typed tables. Legacy batches populate one table; opt-in mixed batches can populate
several tables together. No Python object per event is created.

RegisterChanges: `pc` int64[N], `ids` int64[N], `widths` int64[N], `flags` int64[N],
`values` uint8[N,W], with zero padding to the widest row W. `names` maps IDs to
names in that source's schema. MemoryAccesses: `pc`, `addresses`, `sizes`, `flags`
int64[N], and `values` uint8[N,16] or None. Bit patterns remain exact on CPU/MPS.
Mapping columns are `physical_addresses`, `mapped_sizes`, `mapping_flags` and
`context_sequences`, each int64[N], or None in streams without system mapping.
AddressContext fields are int64[N]. In legacy batches, `first_sequence` plus row
index identifies an event. In mixed batches, use `block_sequences` and each
table's `sequences` int64[N] column: rows in one table need not be consecutive.
The union of those positions spans the batch's consecutive original events.
Each batch owns its storage independently; schema metadata is small per-source state.
`Batch.worker` is the endpoint index within this Pool. Per-CPU state is keyed by
`(worker, source)`; neither value is a process ID or global trajectory identity.

System capture can produce all four batch types. Address-only clients can
explicitly select `--registers none --memory off` to retain the minimal path.
EOF without completion, source gaps, invalid records, and unsupported capabilities
remain explicit errors. CUDA support will use the same typed buffers; actual
CUDA validation requires an operator-provided device.


## Cost and coverage

Default `--batching legacy` frames contain one signal kind and flush when kind
changes. `--batching mixed` keeps adjacent kinds in one bounded frame, reducing
outer framing and Python batch dispatch. See [capture performance](capture-performance.md)
for its wire contract, optional rings, and measurement limits. Block-only capture still batches up to 256
addresses. Public register reads reuse scratch storage. The optional x86 hook
uses one bulk read at a register checkpoint and a smaller context read at each
memory callback; remaining explicitly selected registers use public readers.
Without the hook, system memory context costs four public reads per callback.
No CR8/APIC read is hidden in the bulk path; CR8 is read only when selected. There is no Python object per
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

## Kernel interaction and capture windows

`--system on` selects the system worker contract. `--kernel-adapter on` adds
exclusive QMP/serial ownership and the named guest adapter. Observation-only
system runs have neither an action loop nor the plugin control thread.

KernelRequest=10 has source/count/sequence zero, detail 1..127, and no payload.
It is published only after QMP confirms all vCPUs stopped and the plugin drains
every captured source through its private control pipe. GuestEvent=11 has
source/count/sequence zero and detail equal to its 1..1024-byte UTF-8 JSON payload.
These adapter frames consume no vCPU sequence numbers. Every source that emitted
register data must have its complete selected baseline at the action boundary.
This fence is not a fresh paused-world register snapshot. KernelEnv's Gym info
reports `boundary_register_snapshot="unavailable"`; do not relabel the most recent
block checkpoint as final state.
`Pool` rejects kernel action workers; `KernelEnv` consumes these control frames
and exposes bounded event/result metadata alongside streaming tensor batches.

Actions retain the stdin transport framing: uint32 length followed by one
printable ASCII command and newline. The worker forwards one action while the
guest is paused, then resumes QEMU. The [kernel guide](kernel-examples.md)
explains the stop/drain handoff, limits, and completion rules.

`--start-pc ADDRESS` is optional. The exact basic-block entry activates capture
for that source and for other vCPUs at their next block entry. Without a stop marker, memory callbacks belong to their owning block's capture decision. The first captured block on each
source receives a full selected-register baseline. This transition neither
stops nor schedules CPUs and establishes no global memory order. A target that
exits without executing the requested block fails explicitly. Use the marker
from the actual guest binary; this option does not compensate for ASLR.


`--stop-pc ADDRESS` optionally closes an observation-only window once. The stop
block itself is excluded. A stop before the start is ignored, and later start
blocks cannot reopen a stopped window. Block and memory callbacks check admission
without a global event lock; another CPU's already admitted callback may finish
appending after the stop. This is not a simultaneous world snapshot or total
memory-order boundary. Buffered events drain normally. QEMU continues to exit;
only then can the worker report successful completion. A missing requested stop
is an explicit capture failure. Stop markers are incompatible with action
adapters in this slice. See [window acceptance](kernel-window-results.md).
