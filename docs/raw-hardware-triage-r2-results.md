# Raw hardware triage R2 results

R2 is a capture-attribution and throughput **GO**, a familiar-benign calibration
**GO**, and an unseen-family anomaly-quality **NO-GO**. It replaces CPU-global
kernel PT with kernel-only PT attributed to the gated target process, preserves
the undecoded raw corpus, and falsifies pair hashes, trace length, and larger PCA
capacity as useful additions to the first triage model.

## Subject and protocol

- Host: `trail-x86`, Intel Core i7-10510U, Linux `7.0.0-31-generic`.
- Boot identity: `2ba4b210-4c31-48ce-92d5-4b306e16d26c`; microcode `0x100`.
- Running kernel BTF and notes hashes:
  `c0c2d95eb84d04b2332c7a456b9d6cf4d18974ee439abd1b44c20d9f738bcd38`
  and `764d1c9de60154c47117ba8d6889a52f50e5f79cc4b38b25fa1683da0d81011e`.
- Capture: `scope="process_kernel"`, target pinned to CPU 2 and controller to CPU
  3, branch-enabled Intel PT, 64-page data ring and 2,048-page AUX ring. Perf
  excluded user execution and attributed kernel execution to the target threads;
  no PT packets were decoded.
- Corpus: 17 benign syscall families, one warm-up and 512 measured executions per
  family. Each familiar family contributed 64 training, 128 calibration, and 320
  validation rows. `epoll`, `memfd`, and `fork` were entirely hidden from the
  fixed model. Leave-one-family-out evaluation independently hid every family.
- Threshold: calibration quantile targeting 1,000 reviews per million. Scores use
  strict `score > threshold` admission.

The two full process-attributed replicates used seeds `20260925` and `20260926`.
They collected 370,717,968 and 372,040,352 PT bytes in 29.39 and 30.44 seconds,
or 296.20 and 285.94 finite captures per second. Both observed exactly 3/4,480
familiar validation flags, or 669.64 reviews per million. This is below the
configured budget but has wide finite-sample uncertainty; it is not a claim of a
sub-1,000/million population rate.

The retained seed-`20260926` evidence identities are:

- raw corpus: `67c4c91c99fa314cffc8fbeacd9e9db3671d856743b9778eb267d5eefef32b7a`;
- R1-feature checkpoint: `133db6dc726a190a85be96c1ddd55b40ee5ce1d41b0b8d0948568b9fb09a36f2`;
- report: `1c65f2795c6333241fe9e03b83f61208627eaa9e2f661f7fb4e19063f1fae6d7`;
- workload executable: `d3290444d0ce16c4060985b9297d009245fbaab34ba9c5442aa6b2de027f70bd`;
- segment/capacity ablation:
  `0d36ebe24455ec6672fb8e3cf25eda8e72123e964b09f683fdb326ad68aa5180`.

## Attribution result

The CPU-global R2 repeat was not stable. On the same boot and nominal protocol,
one run admitted no fixed-family holdouts at threshold 16.90; the preserved repeat
used threshold 2.85 and admitted 327,474 benign holdouts per million. CPU scope
includes unrelated tasks and interrupt work on the selected CPU.

Target-attributed kernel PT removed much of that tail. Its two thresholds were
10.77 and 9.82, and familiar validation produced the same 3/4,480 admission count
in both replicates. Median raw trace sizes also fell, for example from roughly
16.5 KiB to 7.9 KiB for `getpid`. Perf therefore can provide the intended real
kernel signal without QEMU while excluding unrelated CPU traffic. Child processes
and threads created after attachment remain outside the current contract.

## Representation and anomaly result

The R1 sketch concatenated ordered byte histograms, hashed adjacent-byte counts,
and log trace length. Leave-one-family-out evaluation showed that pair-only and
global-byte variants classified workload identity: 14--15 of 17 hidden benign
families exceeded 100,000 reviews per million. Removing pair hashes and trace
length was better than adding capacity.

Across segment counts 2, 4, 8, and 16 and latent widths 8, 32, and 64, the smallest
effective choice was four ordered byte-histogram segments with an eight-dimensional
PCA. Its median held-out-family rate was zero and median familiar-validation rate
was 390.63 per million; 15/17 hidden families stayed below 10,000 per million.
However every `readlink` and `yield` execution was still admitted. The raw-byte
linear model therefore learns some reusable kernel regularity but still confuses
two legitimate unseen behaviors with invariant breaks. It has not earned LLM
escalation or a vulnerability-detection claim.

## Throughput

On one pinned `trail-x86` core, the simplified native reducer processed the exact
8,704-trace, 372,040,352-byte corpus at 140.62 MB/s or 3,289.86 executions/s. The
frozen 1,024-to-8 PCA scorer processed 247,907 executions/s on the same core with
one PyTorch thread. These component rates exceed the initial 1,000/s/core target.
Finite perf setup remains the end-to-end bottleneck at 286--296 captures/s, so
long-lived AUX draining and execution-boundary offsets are still required.

## Minimal AWS replication

Commit `22041af` was deployed without AlphaFlow, QEMU, a GPU, or an online
controller to one temporary `c5.metal` host in `us-east-1a`. The host exposed an
Intel Xeon Platinum 8275CL, Intel PT PMU type 11, Linux `7.0.0-1012-aws`, boot ID
`3d93e746-92f8-4bde-af6b-580ff378957d`, and microcode `0x5003901`. The instance
had an encrypted disposable 80 GiB root disk, terminate-on-shutdown behavior, and
a four-hour guest hard stop.

The same 17-family, 8,704-execution protocol collected 364,021,232 undecoded PT
bytes in 34.88 seconds, or 249.53 finite captures per second. The frozen model
admitted 1/4,480 familiar validation traces, or 223.21 reviews per million, and
none of the three fixed `epoll`, `memfd`, and `fork` holdouts. Checkpoint reload
produced bit-identical validation scores. Leave-one-family-out remained a NO-GO:
every `openat`, `readlink`, and `yield` trace and 509/512 `socketpair` traces were
admitted. This independently confirms the deployment mechanics and familiar-tail
calibration while again rejecting unseen-family quality.

On one pinned core of that named host, the exact corpus reducer processed 3,640.13
executions/s or 152.24 MB/s and the frozen scorer processed 170,116.94
executions/s. Peak process RSS while loading and reducing the retained corpus was
729,980 KiB. These are component rates; finite event creation still limits the
complete path to 249.53 executions/s.

Evidence identities are:

- report: `edd7622cb597e31a9802a09906e7d6156b1cbeb5ef92e724eecfae261dc0f149`;
- checkpoint: `d9e4a1f9b192e9a559501aa901773f83caf16760d66f2404c452b81fcd359cca`;
- corpus: `56e7de2aa0b3d1ab9ffc6c4adfbc93d8fd3befe502e6e66eb2c8612e496fe978`.

The clean deployment also exposed that the native reducer's Torch-to-buffer path
requires NumPy although it was not declared. [Issue 26](https://github.com/theoad/cpu2tensor/issues/26)
records the failure; the package now declares NumPy explicitly.

## Decision

Retain target-attributed kernel PT, the preserved raw-corpus contract, four ordered
byte-histogram segments, and the eight-dimensional frozen PCA as the minimum
baseline. Do not add a transformer merely to improve benign-family reconstruction.
The next scientific gate is a label-hidden known-vulnerability experiment on one
exact vulnerable kernel: pretrain on broad benign benchmarks including lawful
sibling activity in the affected subsystem, freeze and calibrate once, then rank
the unseen trigger. The next systems gate remains continuous AUX draining. Neither
gate should wait for the other, but no zero-day campaign is justified until hidden
trigger sensitivity is measured.
