# Conditional memory callbacks for filtered capture

Paging-context filtering needs to decide at run time whether a vCPU should emit
rich memory rows. Registering a normal C callback for every rejected access made
a two-vCPU Linux boot exceed the 120-second acceptance deadline. Deciding only
when a block is translated is incorrect because QEMU can reuse that translated
block after the target context migrates to another vCPU.

The [QEMU 11.0.3 patch](../native/qemu/patches/qemu-11.0.3-memory-condition.patch)
adds one optional, versioned API:

```c
void qemu_plugin_cpu2tensor_register_vcpu_mem_cond_cb_v1(
    struct qemu_plugin_insn *insn,
    qemu_plugin_vcpu_mem_cb_t callback,
    enum qemu_plugin_cb_flags flags,
    enum qemu_plugin_mem_rw rw,
    enum qemu_plugin_cond condition,
    qemu_plugin_u64 entry,
    uint64_t immediate,
    void *userdata);
```

It has the same callback contract as `qemu_plugin_register_vcpu_mem_cb`, but
reads `entry` for the executing vCPU at each memory access and calls the plugin
only when the unsigned comparison with `immediate` succeeds. The check exists in
both generated TCG code and QEMU's helper-memory dispatcher. Updating the
scoreboard with `qemu_plugin_u64_set` therefore changes already translated blocks
without a translation-cache flush.

The generated condition adds a branch and scoreboard load to every registered memory
access. Because that branch splits an existing translated basic block, the patch first
promotes that block's `TEMP_EBB` values to `TEMP_TB`; QEMU's normal liveness pass
demotes values that do not cross the new boundary. Values that do cross it can increase
host register pressure or spills. This work occurs only in translation blocks carrying
a conditional memory callback, but its cost must be measured against the callback cost
on the named x86 host.

cpu2tensor discovers this exact symbol once with `dlsym`. An ordinary QEMU 11
build still loads the plugin for every mode that does not request filtered memory.
Requesting `--rich-context-start-pc` with memory enabled fails during plugin
installation with a direct dependency diagnostic when the symbol is absent. The
extension does not change QEMU's numeric plugin API version.

## Accounting and ordering

Each translated memory operation first performs QEMU's existing inline
`ADD_U64` on a per-vCPU total scoreboard. A conditional callback then emits the
rich row only when that vCPU's admission scoreboard is one. At each basic-block
entry cpu2tensor:

1. reads the current source's total and attributes the delta to the relation of
   its previous block;
2. samples CR3 and, for the gate block, publishes the one-shot paging root;
3. classifies the new block and updates only that vCPU's admission entry.

The last delta is read at vCPU exit. This gives exact matching, foreign, unknown,
kept, and dropped callback counts without entering cpu2tensor C++ for rejected
accesses. The two scoreboards have independent entries per vCPU; the filter adds
no cross-vCPU lock.

A source retains the relation selected at its latest block entry. If another
vCPU publishes the latch while that block runs, the source keeps its earlier
relation until its next block entry. This is deterministic within each source and
does not invent a total order between concurrent vCPUs. The gate source counts
its prior block before publishing the latch, then treats the gate block itself as
matching. A source that observes the short `writing` publication state records
that next block as unknown and never waits.

## Apply and validate separately

Apply the memory patch to pristine QEMU 11.0.3. Rich register capture also needs
the separate [x86 state hook](qemu-state-hook.md). Apply that state patch first;
the memory patch then applies cleanly. The package never builds, installs, or
bundles QEMU.

```sh
patch --dry-run -p1 < /path/to/qemu-11.0.3-memory-condition.patch
patch -p1 < /path/to/qemu-11.0.3-memory-condition.patch
ninja -C /path/to/configured/build -j 4 qemu-system-x86_64
nm -D /path/to/configured/build/qemu-system-x86_64 \
  | grep qemu_plugin_cpu2tensor_register_vcpu_mem_cond_cb_v1
```

Local development on 2026-09-11 used official tag `v11.0.3`, commit
`aeec49e8170de7846f476124602cf7acd400c3df`. The patch applies cleanly alone and
after the state patch. The complete `qemu-system-x86_64` target builds with Clang
17 on macOS AArch64, and the executable exports the versioned symbol. QEMU's
`checkpatch.pl` reports no source-style warning; its only finding is the absent
email sign-off on this repository-local raw patch.

A 512-byte x86 boot-sector fixture translated a memory block before setting its
admission entry, then reused that block twice after admission. Its unconditional
scoreboard delta was two and its callback count was two. A second fixture used
`lock incw`, which follows QEMU's atomic memory-helper path: the first read/write
pair was rejected and the two later pairs produced a delta and callback count of
four. Both guests exited with their expected status. These fixtures establish
dynamic admission for existing direct and helper-generated code locally; they do
not substitute for the full Linux acceptance below.

The earlier real Linux fixture has not been rerun with this extension. Completion,
exact guest values, runtime reduction, and capture-size reduction remain required
acceptance evidence before the context-filter issue is ready to close.
