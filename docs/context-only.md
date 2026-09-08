# Observe blocks with paging context

`--context on` enables the existing x86 system AddressContext signal even when
register and memory observation are disabled:

```sh
cpu2tensor-worker --qemu /path/to/qemu-system-x86_64 \
  --plugin /path/to/libcpu2tensor_plugin.so --system on \
  --registers none --memory off --context on \
  --batching mixed --publication pipe --max-run-ms 60000 \
  -- -kernel /path/to/kernel -initrd /path/to/guest.cpio.gz OTHER_QEMU_OPTIONS
```

Supply the machine arguments and matching start/stop PCs for your own target.
Neither QEMU nor guest artifacts are installed by the package. The default
`--context auto` preserves earlier behavior: x86 system register/memory profiles
include context automatically, while blocks-only does not. `on` requires x86
system emulation; unsupported backends fail explicitly. There is no `off` setting
that could strip context required by an existing rich profile.

## Meaning and ordering

The first observed block on each vCPU has a preceding context event. Later events
are emitted only when sampled fields change. They retain the original per-source
event sequence and appear before blocks using the newly sampled state. Mixed
frames and tensor collation can place several changes in one batch, so do not
apply the batch's last context to all its blocks. Order is local to
`(batch.worker, batch.source)`; there is no ordering between workers or vCPUs.

The row contains raw CR0, CR3, CR4, EFER, CS base and execution width, plus its
checkpoint PC and a validity mask. **Raw CR3 is paging context, not a PID.** It
can contain PCID/control bits; equal values do not establish process identity,
and values are not portable identities across worker launches. Page tables can
change without CR3 changing. With memory disabled this profile does not capture
page-table writes, reconstruct mappings or provide physical addresses.

Context is sampled at block entry in this profile. It does not enumerate every
intermediate register write or promise a fresh all-vCPU final state. The `known`
mask describes field availability, not whether a paging mode makes every stored
bit meaningful. Interpret CR3 together with the other paging controls.

The optional [x86-state hook](qemu-state-hook.md) supplies all six fields
(`known=63`). Without it, supported public control-register readers supply the
four paging controls (`known=15`); execution width and CS base remain explicitly
unknown. Missing required readers fail. Registers remain disabled in both cases.
The hook path needs no register descriptors/baselines and reads eight control/mode
words, rather than the full 28-word register state. Memory callbacks are absent.
Sampling still has a cost even when unchanged fields produce no event.

## Consume the tensors

The installed example associates raw CR3 with each block using sequence positions:

```python
from cpu2tensor import Pool
from cpu2tensor.examples.observe_context import cr3_for_blocks

latest = {}
with Pool(endpoints, device="cpu", batch_bytes=65536) as pool:
    for batch in pool.read():
        if batch.source is None:
            continue
        key = (batch.worker, batch.source)
        cr3, latest[key] = cr3_for_blocks(batch, latest.get(key))
        # batch.addresses and cr3 have the same row count.
        # Feed these integer tensors to your own feature preparation or model.
```

The helper uses vectorized sequence lookup, rejects unknown/missing CR3 and
retains only a compact last-context tensor per source. Returned int64 values
preserve all raw bits, including values represented as negative signed integers.
This is example code clients can modify; it adds no trajectory registry or public
asynchronous API. Clear the dictionary when starting a different Pool/run.

For a command-line check after starting a worker:

```sh
python -m cpu2tensor.examples.observe_context --endpoint tcp://WORKER:PORT --batch-bytes 65536
```

## Acceptance and measured cost

The benign standalone paging oracle changes CR3 from `0x1000` to `0x4000` and
back and switches among 16/32/64-bit code. Real captures validated those contexts
and their following blocks on both the optional-hook and public-fallback builds.
The fallback correctly marked mode/CS base unknown. Neither capture contained
register or memory rows. Synthetic CPU/MPS tests additionally cover two sources,
context changes inside mixed/collated batches, ownership and missing context.

A real two-vCPU Linux four-action workload passed through the public Pool with
mixed batches and collation: context preceded every block, both sources completed,
no register/memory tables appeared, and the guest reported all four actions
successful. ARM user-mode worker/plugin rejection and existing rich/stdin
regressions also passed. These checks use benign paging and program workloads.

The [12-run x86 matrix](context-only-performance.md) measured blocks-only,
blocks/context, rich values and vanilla under the same prescribed guest inputs.
The context profile retained a median 1.213 MiB versus 169.261 MiB for rich values.
Block counts differ under natural scheduling; these are profile totals, not a
matched-event compression ratio. Context and blocks-only elapsed ranges overlap;
no zero-cost or speedup claim is made. The public-reader fallback has correctness
evidence but was not part of that timing comparison.

## Compatible development build

On `trail-x86`, source and build are isolated under
`/home/user/.cache/cpu2tensor/context-only/{source,build}`. Use the paired worker
and plugin; older builds reject the new option. The optional-hook QEMU is unchanged.

| Binary | SHA-256 |
| --- | --- |
| cpu2tensor-worker | `998e50c39b19d8f70dad077c0f6574777346303b0ef1e9dee931b5ede9ef6e7a` |
| libcpu2tensor_plugin.so | `8c68053ecfe4085305cb0eed8fc3dc63c06fd9bc5dcf6bd47be2a496af0306c7` |

The source manifest is [context-source.sha256](context-source.sha256). Paging
artifacts live under `~/.cache/cpu2tensor/context-only/paging-captures` on the x86
host and Mac; all benchmark runs and command hashes are retained in the linked
performance report. Prior state/deadline/performance builds remain preserved.


The final Python suite passed 167 tests with 66 explicit unconfigured-worker or
unavailable-backend skips, both in the editable checkout and from a copied test
directory against a separately installed wheel. The installed context example
resolved from that wheel. Its SHA-256 is
`5baf3e544b55a9911fca856b400f31f6af95669d5feaeb80da6a7cba60b41109`.
Native builds passed on ARM/x86 Linux; portable native CTest passed on Mac and ARM.
CUDA and AWS remain unvalidated for this profile.


## Independent client acceptance

AlphaFlow reported independent integration acceptance for commit
`feb05f687675339fdbc65f927975c1dd6ace15d0` on 2026-09-08. Its binary hashes matched
the paired worker/plugin above. A benign getpid window completed with 10,537
blocks, two context rows, no memory/register rows, both vCPUs, a clean exit and
86,248 wire bytes. Its normal-workload corpus completed 26/26 captures with
24,049,805 blocks. These are client-reported results, not reruns by the package
maintainers or a matched performance comparison. They establish integration
acceptance; they do not establish learner accuracy or treat CR3 as a PID.
