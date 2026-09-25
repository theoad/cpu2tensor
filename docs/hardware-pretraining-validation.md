# Hardware pretraining validation

The first multimodal experiment validates whether a frozen model notices controlled
changes in flow, memory behavior, counter pressure, and cross-CPU timing. It does
not claim vulnerability detection. Intel PT, PEBS, and PMU observations are
measurement channels with different sampling, loss, and timing semantics; the
capture manifest must preserve those differences instead of manufacturing a total
order.

There are two deliberately separate scopes. The canary below runs with
`scope=process` and qualifies whether the sensors and fused model can retain known
flow, memory, and timing differences. It includes application execution and is
therefore an acquisition control, not evidence about a kernel-only model. The
pretraining subject runs with `scope=process_kernel` against the gated benign
syscall families in `hardware_kernel_workload`; user instructions are excluded.
Its required tests are whole-family holdouts, modality/time corruptions, and the
later matched vulnerable/fixed conditions. A process-scope canary pass cannot
satisfy a kernel-only go/no-go gate.

## Deterministic canary target

Build the Linux-only target with the existing hardware-example option:

```bash
cmake -S native -B build/hardware-validation -G Ninja \
  -DCPU2TENSOR_BUILD_HARDWARE_EXAMPLE=ON \
  -DBUILD_TESTING=ON
cmake --build build/hardware-validation --target hardware_multimodal_canary
```

The target initializes memory and worker threads, writes `READY`, and waits for one
input byte. Capture starts before that byte is written and stops after the process
exits. Every successful run prints its observed value, independently calculated
expected value, selected CPUs, and `oracle=pass`.

```bash
printf x | build/hardware-validation/hardware_multimodal_canary \
  flow ordered 1000000 2
printf x | build/hardware-validation/hardware_multimodal_canary \
  contention shared 1000000 2 3
```

For the two-thread modes, supply two distinct isolated logical CPUs in real
captures. Omitting CPU arguments is useful only for portability smoke tests; it
allows scheduler migration and is not validation evidence.

| Mode | Familiar variant | Injected variant | Primary observation | Exact oracle |
| --- | --- | --- | --- | --- |
| `flow` | `ordered` | `permuted` | PT call order; branch PMU secondarily | Both orders add the same four terms |
| `memory` | `hot` | `large` | PEBS load address/latency and cache/TLB PMU | Final pointer-chase index |
| `contention` | `private` | `shared` | Cache-line contention in PEBS/PMU and cross-CPU time | Exactly two atomic increments per iteration |
| `phase` | `normal` | `shifted` | Relative cross-CPU and cross-modal timing | Ordered payload sum and terminal handshake |

The memory variants allocate and pre-touch the same 64 MiB backing before `READY`;
their active pointer-chase rings are 4 KiB and 64 MiB. This keeps allocation and
exit cleanup from identifying the variant. Both rings use the same odd-stride
full-cycle permutation, so active working-set size is the intended difference.
The phase variants execute the same
deterministic delay on both threads but place it on opposite sides of publication
and observation. The canaries are finite,
unprivileged, and do not modify files or kernel state. Large loop counts consume
CPU and the shared contention mode deliberately creates cache traffic; they must
not run on production cores. They are not real-time timing standards.

Timestamp offsets, missing PEBS samples, modality dropout, and CPU-ID permutation
are acquisition-integrity fixtures applied to retained traces. They must not be
implemented in the target: doing so would mix sensor corruption with program
behavior. Sensor-integrity results are reported separately from anomaly
sensitivity.

## Frozen blinded protocol

An execution inherits a session identifier. A session fixes the boot, kernel and
microcode, binary hash, KASLR epoch, physical CPU assignment, frequency policy,
capture configuration, PT AUX size, PEBS event and period, PMU event group and
period, and ambient-load policy. Record loss and completeness for each modality.
Randomize condition order inside a session and balance cache-warm and thermal
epochs.

Split whole sessions, boots, seeds, and workload families. Never distribute
windows from one execution across training, calibration, and test. Train on benign
familiar variants only. The process-scope sensor gate freezes a separately named
threshold at the calibration 99th percentile. The evaluator retains the map
from opaque condition IDs to variants until scoring finishes. Unseen benign
families are test data and never adjust the threshold.

Tonight's process-scope sensor gate is:

- all requested modalities are present, attributable to the recorded CPU, and
  complete or explicitly rejected for loss;
- at least 1,000 familiar benign calibration executions and at least 100 held-out
  injections per canary, across at least three collection sessions;
- at least 80% held-out detection for each of flow, memory, contention, and phase
  canaries at no more than 1% familiar-benign execution alerts; and
- checkpoint reload reproduces scores and the label-hidden report.

That 1% sensor threshold is not the operational triage threshold. The initial
operational target is 1,000 reviews per million executions, the 99.9th
percentile. Estimate it from at least 100,000 familiar-benign calibration
executions and verify it on at least 100,000 independent benign executions,
split by whole sessions and families. Never reuse the small sensor calibration
to claim an operational review rate.

Report familiar and unseen-family alerts separately as executions per thousand and
alerts per hour. Also report the worst family, time or executions to first alert,
PT-only, PEBS-only, PMU-only and fused ablations, and localization against the
known canary interval with clock uncertainty. Confidence intervals resample
sessions or boots, not correlated windows. AUROC is descriptive, not the primary
gate.

Passing this gate establishes sensitivity to controlled hardware-observable
changes. It does not establish that the model detects bugs, distinguishes causes,
or operates at an acceptable prospective false-alert rate.

The separate kernel-only PoC gate uses the exact same boot and hardware but
`scope=process_kernel`. It must capture all three modalities without loss or
PEBS/PMU multiplexing; beat the marginal and PT-only baselines on cross-modal matching;
retain the frozen familiar-family alert budget; and report every held-out syscall
family rather than averaging them away. No application-scope sample may enter
kernel-only training, calibration, or evaluation.

## Unknown-alert handling

An unexpected alert is an unresolved anomaly, not automatically a false positive.
Retain its raw modalities, completeness metadata, target output, manifest, model
checkpoint, and score. Re-run the same seed, a new randomized seed, and a new boot;
compare its lawful sibling and any matched fixed build. Inspect semantic output and
kernel logs, then use KASAN, KCSAN, or UBSAN only in a separate diagnostic replay
because instrumentation changes the distribution. Minimize a reproducible case
before adjudication.

Every reviewed alert uses the
[hardware anomaly evidence bundle](hardware-anomaly-evidence.md). Review cost is
reported both per execution and per previously unseen stable evidence cluster.
Accepting one explained-benign cluster may suppress later members from expensive
LLM review, but it does not change their archived anomaly scores or retrain the
frozen model. A cluster assignment is part of the report and must not be used to
erase unresolved singleton trajectories.

The report has three disjoint buckets: expected positives, explained-benign
alerts, and unresolved anomalies. The conservative operational alert rate includes
unresolved anomalies. A second explained-benign rate and the unresolved rate may
be shown, but unresolved cases are never silently removed or relabeled to improve
specificity. A reproducible semantic failure becomes a discovery candidate and
requires independent review.

## Later CVE gates

Each known-vulnerability experiment is a blinded $2 \times 2$ factorial:

| | Lawful matched sibling | Exact trigger |
| --- | --- | --- |
| Fixed build | expected negative | trigger-novelty control |
| Vulnerable build | workload control | candidate manifestation |

Build the pair from one source tree, configuration, compiler, and boot manifest,
changing only the security fix. Include lawful siblings from both builds in benign
training so build identity is not the anomaly. Freeze the model and threshold
before revealing outcomes. The primary effect is

$$
(s_{\mathrm{vulnerable,trigger}}-s_{\mathrm{vulnerable,sibling}})
-(s_{\mathrm{fixed,trigger}}-s_{\mathrm{fixed,sibling}}).
$$

If both trigger cells score highly, the model recognized an unusual syscall
sequence rather than the defect. For probabilistic races, record an additional
vulnerable-trigger cell where the semantic or diagnostic oracle says the race did
not fire.

The first candidates are deliberately different. Version ranges name upstream
stable releases; distribution backports must be checked by fix commit:

- [CVE-2022-0847, Dirty Pipe](https://dirtypipe.cm4all.com/): vulnerable from
  5.8; fixed in 5.16.11, 5.15.25, and 5.10.102. Use a disposable-file page-cache
  mutation with matched ordinary pipe, splice, offset, and fixed-kernel siblings.
- [CVE-2022-0185](https://ubuntu.com/security/CVE-2022-0185): introduced by
  `3e1aeb0` in 5.1 and fixed by `722d948`; use the exact 5.16.1/5.16.2 boundary
  or a single-patch pair. Exercise only a minimal `fs_context` bounds violation
  in an isolated disposable host, without heap spray or a privilege payload,
  with below-boundary and rejected-parameter siblings.
- [CVE-2022-29582](https://www.openwall.com/lists/oss-security/2022/04/22/3):
  the public advisory scopes 5.10 and later; the timeout-race fix first appears
  in 5.17.3, 5.16.20, 5.15.34, and 5.10.111. Admit the `io_uring` linked-timeout
  race only after a non-weaponized ground-truth oracle is demonstrated, with
  serialized and single-cancellation siblings.

No CVE trigger or exploit is part of the canary target. CVE runs require disposable
isolated infrastructure, exact fix-commit verification rather than version-string
assumptions, and a separately approved harness. KASAN-confirmed traces are
diagnostics, not samples from the uninstrumented scoring distribution.
