# Multimodal hardware pretraining results

This log separates measured evidence from the proposed
[validation](hardware-pretraining-validation.md) and
[24-hour procedure](hardware-pretraining-scale.md). The current result qualifies
capture and a small model implementation. It does not yet establish useful
kernel anomaly detection.

## Exact subject

The capture qualification ran on `trail-x86` (`iseeyou`), an Intel Core
i7-10510U host,
with target CPU 2 and controller CPU 3. The retained subject was Linux
`7.0.0-31-generic`, boot `2ba4b210-4c31-48ce-92d5-4b306e16d26c`, microcode
`0x100`. Its BTF SHA-256 is
`c0c2d95eb84d04b2332c7a456b9d6cf4d18974ee439abd1b44c20d9f738bcd38`,
kernel-notes SHA-256 is
`764d1c9de60154c47117ba8d6889a52f50e5f79cc4b38b25fa1683da0d81011e`,
and workload SHA-256 is
`d3290444d0ce16c4060985b9297d009245fbaab34ba9c5442aa6b2de027f70bd`.
The first qualification artifact is cached as
`~/.cache/cpu2tensor/multimodal-poc/qualification-r12-v2.json`; its SHA-256 is
`655981233a19e8c4f89e30c66b6b7fb838bd78eb8c14fefab137c8889f8e59b8`.

Twelve randomized repeats of each perturbation arm used the gated `mmap`
kernel workload with 1,000 iterations:

| Arm | Median target window | Ratio to baseline |
| --- | ---: | ---: |
| No hardware events | 2.955 ms | 1.000 |
| Boundary PMUs | 2.903 ms | 0.983 |
| PT + boundary PMUs | 3.072 ms | 1.040 |
| PT + PEBS + boundary PMUs | 3.086 ms | 1.045 |

All 48 executions retained the exact output. Every requested source was present,
no perf loss was reported, and `time_enabled == time_running` for every PMU
group. Tri-modal runs had a 602,456-byte median PT trace and six exact-IP PEBS
samples at period 10,000. Every PEBS sample named the target TID, CPU 2, a
nonzero address, and a timestamp inside the shared `CLOCK_MONOTONIC_RAW`
envelope. Median source-control uncertainty was 44.103 microseconds at arm and
14.107 microseconds at stop; maxima were 106.599 and 37.474 microseconds.

## PEBS period decision

A follow-up used 5,000 iterations, six randomized repeats per arm, and three
kernel workload families on the same boot. Its six JSON artifacts and verified
hash manifest are cached below
`~/.cache/cpu2tensor/multimodal-poc/family-period-matrix-loops5000-r6/`.
The `SHA256SUMS` file hash is
`30f827aec6693578045411e4ee97d724dc51bde99d27668555b280843f8a850f`.

| Family | PEBS period | Median PT bytes | Median PEBS | PEBS/million PT bytes | Tri-modal overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| `openat` | 10,000 | 1,002,800 | 10 | 10.00 | +5.30% |
| `mmap` | 10,000 | 2,991,952 | 34 | 11.36 | +4.20% |
| `socketpair` | 10,000 | 5,490,176 | 45.5 | 8.31 | +4.66% |
| `openat` | 50,000 | 1,003,832 | 2 | 1.99 | +6.00% |
| `mmap` | 50,000 | 2,979,144 | 6 | 2.01 | +3.58% |
| `socketpair` | 50,000 | 5,408,592 | 9 | 1.66 | +4.46% |

All 144 executions kept their exact outputs and remained lossless and
non-multiplexed. A separate short `openat` attempt at period 50,000 produced an
empty PEBS execution. Period **10,000** is therefore frozen for the first PoC:
it gives roughly five times the sample density without a consistent overhead
penalty in this matrix. These six repeats qualify a choice; they do not establish
a universal perturbation bound.

The initial matrix inferred PEBS scheduling from source presence. Commit
`70cbfb5` subsequently pinned the precise event and made perf's own
`time_enabled` and `time_running` values part of the accepted evidence. A fresh
three-repeat `mmap` replay on the same exact subject produced six exact-IP,
nonzero-address samples in every tri-modal execution and equal nonzero scheduling
times in every case. Its artifact is cached as
`~/.cache/cpu2tensor/multimodal-poc/pebs-schedule-qualification.json`, SHA-256
`139308b7b4667548f4a4dc791b93bae7453707a4c36d4c48eaecc3932a15b1ff`.
The tri-modal median was 1.092 times its matched baseline in these three repeats;
the wider earlier 4.20--5.30% matrix and this 9.2% short replay are reported
separately rather than combined into a universal perturbation estimate.

## Small-model control

The initial masked bidirectional transformer has 171,143 parameters for four
PEBS and three PMU fixture features. On `mac.local` it scored 964 executions/s
on one CPU thread and 1,996 executions/s on MPS for batch 128 and one CPU lane.
Checkpoint reload reproduced scores bit-for-bit. On a deliberately correlated
synthetic fixture, whole-modality masking increased the modality-swap to clean
score ratio from 6.74 to 10.83. That supports the architectural choice but is
not evidence of hardware-signal learning. The real featurization and
kernel-family experiment remain the next gate.
