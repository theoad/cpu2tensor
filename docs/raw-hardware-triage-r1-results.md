# Raw hardware triage R1 results

R1 is a mechanics success and an anomaly-quality **NO-GO**. It demonstrates a
loss-free kernel-only Intel PT path into a frozen raw-byte latent model, but the
small benign training corpus does not support deployment or vulnerability claims.

## Subject and protocol

- Host: `trail-x86`, Intel Core i7-10510U, Linux `7.0.0-31-generic`.
- Boot identity: `2ba4b210-4c31-48ce-92d5-4b306e16d26c`; microcode
  `0x100`; kernel build
  `#31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2`.
  The running kernel BTF and notes hashes were respectively
  `c0c2d95eb84d04b2332c7a456b9d6cf4d18974ee439abd1b44c20d9f738bcd38`
  and `764d1c9de60154c47117ba8d6889a52f50e5f79cc4b38b25fa1683da0d81011e`.
- Capture: kernel-only branch-enabled Intel PT on CPU 2; controller on CPU 3;
  64-page perf data ring and 2,048-page (8 MiB) AUX ring.
- Workload: a gated C17 target, already pinned before capture, executing `getpid`,
  `fstat`, `futex`, `openat`, `pipe`, or `mmap`. One warm-up per family preceded
  eight randomized measured repetitions per family. `getpid`, `fstat`, and
  `futex` ran 250 loops per execution; `openat`, `pipe`, and `mmap` ran 100.
  Python `random.Random(20260925)` shuffled the measured family order.
- Training: six traces from each family except `mmap`; two further traces from
  each known family were validation; all eight `mmap` traces were hidden benign
  family holdout. The reducer parsed no PT packets or symbols.
- Model: standardized eight-dimensional PCA, equivalent to a frozen linear
  autoencoder. The threshold was the training-score 99th percentile. Fitting
  was deterministic for the captured tensor and remained in memory; R1 wrote no
  model checkpoint, so there is no checkpoint artifact identity to preserve.

The measured source identities were:

- perf capture Python: `cec4b25e52c4b4b068aec626983826fa5c9899e543b5e2de2e8e2d1ea2489474`;
- triage Python: `27a920c2074c7e84b9b08952303c0a258c5b899e5131b9ea09c2be69f41b7a7e`;
- x86 native extension: `c55446dd774b0f464d0069690a027064312728f91418710b36c903c872f10b42`;
- gated workload executable: `0983a8c51f8e2f6b2100001dcc8197c91bef359564daf70b7259c50e71a10aa3`.

Those hashes identify the measured R1 implementation. Review subsequently found
that its native reducer divided empty segment bins by zero on traces shorter than
eight bytes and reread mutable buffers after releasing the GIL. R1's real traces
were all at least 17,408 bytes and its single-threaded measurements did not hit
either defect, but the candidate implementation was corrected before merge. The
corrected x86 extension hash is
`2b7fb6763034c01d2dd22cab4034af3ed89004f6fb8ed28d736c9935ee97a815`;
native/fallback equivalence was checked for every trace length from one through
seven bytes on `trail-x86`.

## Throughput

The 48 measured executions produced 1,797,168 raw PT bytes. Median trace sizes
ranged from 17,408 bytes (`getpid`) to 70,808 bytes (`mmap`). The finite capture
loop completed 285.15 executions/s. Median capture-window times ranged from
1.46 to 1.81 ms. Opening, mapping, disabling and destroying perf events on every
execution therefore misses the 1,000 executions/s/core milestone.

The fused native raw-byte reducer processed the exact 48-trace batch at 93.49
MB/s, or 2,496.99 executions/s. The earlier tensor-only implementation managed
about 1.9 MB/s and 49 executions/s on the same host; per-byte tensor temporaries,
not the sketch arithmetic, were the bottleneck.

A separate same-host component check pinned the scorer to one CPU and used one
PyTorch inference thread. For batches of 48 rows with the same 3,073 feature
dimensions, the frozen PCA scorer reached 111,389 executions/s. Configuring four
PyTorch threads on that one pinned CPU oversubscribed it and collapsed throughput
to about 535 executions/s. These are component rates, not an end-to-end fuzzer
rate. The reducer and scorer still need a bounded continuous capture pipeline.

## Anomaly result

At the training 99th-percentile threshold:

- 7/10 validation executions from familiar benign families were flagged;
- 8/8 executions from the hidden benign `mmap` family were flagged;
- hidden-benign median score was 10.58 versus familiar-family medians of
  approximately 0.18–0.25.

The model learned the narrow training support, not a useful general kernel-normal
distribution. Treating these flags as candidate bugs would create an unacceptable
review burden. R1 does not evaluate a known CVE, detect a bug, establish causal
attention, or justify LLM escalation.

## Decision

Retain the hardware path, fused reducer, frozen-model boundary and explicit loss
semantics. Do not retain the R1 calibration or claim the eight-dimensional PCA is
sufficient. The next gate must pretrain on a much broader benchmark mixture, use
family-held-out benign validation, and report false reviews per million at a fixed
review budget before revealing known-vulnerability labels. Continuous AUX draining
with execution offsets is the next performance implementation because per-execution
perf setup is already the measured systems bottleneck.
