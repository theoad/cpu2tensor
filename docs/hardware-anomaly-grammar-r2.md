# Raw-grammar R2 and gated two-session data protocol

This is an offline representation/data-quality ablation, not a trained detector,
known-bug sensitivity result, or live capture. The input is the immutable
102,000-execution `trail-x86` collection's 525 preregistered raw audit shards.
The R2 probe verifies their SHA-256 values against sealed manifest
`0cf77ff5c64106598e20873cede98401fd7293ab6a59b295b15633389616b37d`
and checks the two referenced exact-session decode-state hashes. No canary label
or known-effect trace is read.

## Lightweight per-lane grammar

`cpu2tensor.examples.hardware_raw_grammar_probe_r2` divides each lane's raw PT
AUX stream at candidate 16-byte PSB signatures. It retains the ordered byte
start/stop of every candidate span and forms a compact diagnostic sketch:
16 equal *span-ordinal* groups, each with 128 hashed byte-bigram frequencies.
These groups are not time bins. A PSB signature scan is not full packet decode;
saved perf AUX records and loss status remain the completeness authority. No
branch target, instruction count, or precise packet timestamp is inferred.

Separately, each PEBS store token retains its original
`CLOCK_MONOTONIC_RAW` time, raw unsigned IP, raw virtual address and page offset,
raw data-source bits, weight, sample period, exact-IP flag, and CPU/TID. A
core-text IP additionally gets an offset
from this boot's runtime `_text`; module/non-core IPs keep raw identity and a
missing core offset. A 16-time-bin by 256-hash-site/address sketch is used only
for stability measurement; tokens keep the exact fields. PMU remains an
enabled-interval counter group, not a token timestamp. Lanes remain separate;
there is no synthetic PT-to-PEBS interleave or cross-CPU memory order.

The token schema is an experiment-local diagnostic, not a shared model ABI.
Full decoded branch paths and KASLR slide values are not assumed. A numeric
slide still lacks a verified unrelocated `_text` anchor, and the saved PT event
has TSC/MTC/CYC disabled. Thus PT byte order is available but fine PT-to-PEBS
timing remains unidentified. Sampled virtual store addresses also do not name
kernel data objects or unsampled writes.

## Retained-audit result

On `mac.local` (Apple Silicon, Python 3.12, Torch 2.14 CPU, NumPy 2.5.2), all
525 raw hashes verified. They contain 572,193,760 PT bytes, 279,580 candidate
PSB markers, and 512,201 PEBS stores. All 525 traces begin with a candidate
PSB; median candidate spans per trace are 232. Every audited store has exact IP
and nonzero virtual address; 512,156 have a runtime core-text-relative IP.
Samples report CPU 2, with no timestamp inversions or equal-time ties in this
single-lane corpus. These are quality facts for retained raw, not proof of
complete memory-write coverage or multi-lane ordering.

Each comparison uses two retained benign executions chosen deterministically
from the sealed raw IDs. “Same” matches family and actual loop count;
cross-intensity uses each family's lowest and highest retained loop count;
cross-family matches loop count. Cosine similarity is a descriptor diagnostic,
not a frozen anomaly threshold or test-set performance estimate.

| Median cosine | Pairs | PT span-ordinal bigrams | PEBS time/site/address |
| --- | ---: | ---: | ---: |
| Same family and intensity | 51 | 0.98548 | 0.21281 |
| Same family, cross intensity | 17 | 0.96570 | 0.18276 |
| Cross family, same intensity | 48 | 0.78704 | 0.06565 |

The PT descriptor retains repeatable family structure and some intensity
robustness, though its 0.020 similarity drop across intensity still matters at
a stringent review budget. The sampled PEBS site/address sequence is highly
variable even between matched benign runs. It should remain an evidence channel
and potential conditional prediction target, **not** be installed as an
unqualified stand-alone anomaly score. One boot/session cannot measure
cross-session, KASLR, or ambient-load stability.

The verified hash+load+reduce pass took 4.94 s for 525 shards (106.2 rows/s),
with 3.77 s inside the grammar reducer (139.1 rows/s) and 0.861 GB peak
process RSS. The 170 MB JSON manifest dominates that process's materialized
metadata. A separate `--stream-benchmark` pass, run only after the verified
pass, skipped manifest parsing and discarded per-shard raw/token objects after
keeping the compact sketches: 4.36 s for 525 (120.4 load+reduce rows/s),
3.47 s in reduction (151.5 rows/s), and 0.380 GB peak RSS. Its raw hashes are
*not* independently checked in that second pass; the prior verified pass is
the custody evidence. These are Mac offline reduction rates, not `trail-x86`
capture rates or a measured production scorer. They are below the original
physical host's 13.07 sealed executions/s only in separate environments, so
no cross-ISA speedup ratio is claimed. The compact two `float32` grids use
24,576 bytes per lane before token/offset/index storage; the exact raw AUX
bytes remain in custody, not in the grid.

## Collection GO checklist at the reported 6 GiB free: **not authorized to run**

The 102k physical-host reference sealed 13.07 executions/s and 15.42 MB/s of PT
under `no_turbo=1`, with zero retry, loss, missing source, or multiplexing. It
does **not** establish perturbation overhead against an untraced arm. For data
quality, its 51 family×loop strata had roughly 1,906--2,078 rows each: the
median within-stratum coefficient of variation (CV) was 2.21% for PT bytes,
12.85% for PEBS count, and 3.46% for target elapsed time; the respective
90th-percentile stratum CVs were 8.14%, 27.37%, and 10.51%. On 51 pairs from
the preregistered raw audit selection matched for family, loop count, PT-byte
exposure, and PEBS count, the *existing v3 derived* tensor cosine medians were
0.9883 PT and 0.9303 PEBS (PMU 0.999997). Exposure matching makes those an
optimistic shape-noise floor; it is not a cross-session result and PMU cosine
alone hides rate changes.

All of the following must be checked and approved before a smoke, then frozen
before Session B. “Bugless” means high-integrity, low-noise, efficiently
collected data, not a rule excluding future safely approved effect examples.

| Gate | Quantitative GO condition |
| --- | --- |
| Exact subject and inputs | Same boot ID, signed kernel/package, binary and event hashes; CPU 2 target/CPU 3 controller; `no_turbo=1` and CPU 2 cap 1,800,000 kHz throughout. Every row has a real target-consumed seed frame, canonical invocation hash, output oracle, and session/stratum ID. Shared-seed cross-session rows must have identical invocation hashes. Verify nontrivial effective input variation in the eight seeded-variation families; label the other nine fixed-workload controls. |
| Capture integrity and yield | Zero **admitted** PT/AUX gap/overflow, lost/missing lane, PEBS/PMU multiplexing, oracle mismatch, or unplanned migration; an explicit-plan rejection stops the run with its failure ledger and partial artifact preserved. The 51-row smoke must be 51/51 first-attempt clean. Across sessions target at least 99.5% first-attempt clean yield; exceeding 0.5% rejection pauses collection for diagnosis, never relabels bad data as clean. Failed-attempt raw windows are not yet guaranteed to be serializable; that limits any later continuous-run custody claim. |
| Matched-repeat noise | Use the same exact input seeds in both sessions for 10 of 20 seeds per family/intensity stratum (510 paired repeats: 240 seeded-variation and 270 fixed-workload). Report both groups separately, including per-stratum and worst-family relative paired difference $|A-B|/((A+B)/2)$ for PT bytes, PEBS count, and elapsed time. Provisional macro-median/90th-percentile ceilings are 5%/20% PT, 20%/60% PEBS, and 7%/25% elapsed; fail or investigate if exceeded. On the 102 preregistered raw repeat pairs, require median v3 exposure-matched cosine at least 0.98 PT and 0.90 PEBS, while reporting the full distribution by group. These are acquisition-noise gates, not detector sensitivity. |
| Pilot throughput and backpressure | For these 1,020-row sessions, require collector-sealed rate at least 10 executions/s over each balanced block (about 76% of the 13.07/s physical reference) **and** final off-host hash-acknowledged end-to-end rate at least 10/s. The unacknowledged backlog must stay below the 896 MiB tmpfs bound and drain fully before Session B. Report producer and custody rates separately, PT MB/s, phase times, queue high-water, first-attempt yield, and family/intensity mix; do not compare Mac reduction speed as host capture speed. |
| Million-row scale throughput | This pilot floor is not a 24-hour GO: 10/s takes 27.8 hours for one million. A later million-row/24-hour plan needs at least 11.57/s sustained end-to-end custody, with independently demonstrated headroom above that mathematical minimum and a separately approved capacity/thermal plan. |
| Disk, custody, temperature | New spool only, **at most 896 MiB tmpfs including partial files and logs**, root free at least 5.5 GiB before smoke and 5 GiB while running, and no writes, mounts, or transfers to `/dev/sda` or any child filesystem. Start below 70°C package; conservatively stop this pilot at 80°C. Record policy and temperature at least every second in the supervisor; restore pre-run turbo policy on every exit. |

These ceilings are conservative proposals from one historical session, not a
guarantee that a new seeded workload will pass. The smoke may expose a
predeclared contract error; any changed gate requires a new documented protocol
*before* Session B, not threshold tuning on Session B or effect labels.

## Two new randomized same-boot sessions: proposed, **not authorized to run**

Purpose: measure seeded input variation, sensor quality/noise, repeatability,
and custody across sessions on the *same* physical boot. The two sessions are
benign acquisition controls for this ablation; “bugless collection” is a data
quality requirement, not a restriction on later safely approved effect
validation. No effect family or label is needed to choose this protocol.

Before launch, the coordinator must accept and verify the opt-in seed-aware
target/runner change owned by the validation track. At the start of this R2
audit, the existing target accepted only `FAMILY LOOPS`, and its READY stdin
byte was ignored. The old `--seed` shuffled run order and selected loop
divisors but did **not** vary target input per execution. The in-progress
seed-aware input contract must pass a deterministic seed frame into the target,
retain a separately checked output oracle, and record the exact seed, argv,
stdin bytes, binary SHA-256, and canonical invocation SHA-256 per execution.
The current patch materially changes bounded lawful workload inputs/body/path
in **eight** families. Those are the seeded-variation strata: distinct seeds
must produce verified distinct effective target inputs. The other **nine**
consume distinct seed frames but keep the same effective workload; they are
fixed-workload session/repeat controls, not evidence of 17-family input
invariance. Freeze this classification, mapping, and allowed ranges before
collection.

Proposed size after that gate: in each of two sessions, 17 existing benign
acquisition families × 3 preregistered loop intensities × 20 input seeds = 1,020
executions: 480 seeded-variation rows and 540 fixed-workload control rows per
session. Within each stratum, 10 seeds are identical across Sessions A/B for
matched-repeat noise; 10 are session-unique. Only the eight variation families
test lawful effective input variation; the other nine test fixed-workload
repeat/session stability under distinct consumed seed frames.
Balance and independently shuffle family/intensity order in 51-row blocks,
while keeping every exact input hash and its output oracle. Session B is a
session-held-out stability set: do not tune R2 sketch widths, thresholds, or a
future model on it. The opt-in exact plan retains four of 20 raw captures per
stratum by fixed slots 0, 1, 10, and 11, independent of trace content/score:
the same two shared seeds in
both sessions plus two session-unique seeds (204 raw shards/session, 102 raw
matched-repeat pairs: 48 variation, 54 fixed-workload). On rejection, stop and
preserve the failure ledger and partial artifact; do not quietly retry or
replace the row. Every admitted execution retains its derived record, raw SHA-256, oracle,
loss/coverage status, actual seed/input
hash, and exact subject identity. If exceptional retention threatens the cap,
abort rather than silently discard evidence.

The existing 102k run averaged about 1.18 MB PT/execution, about 1.24 MB per
retained raw shard, and about 30 kB derived/execution. At those observed sizes,
204 raw + 1,020 derived rows are roughly 0.28--0.31 GB/session; both sessions
are projected below 0.7 GB before logs, manifests, temporary files, and safety
margin. This projection is **not** a capacity guarantee. Use a dedicated
896 MiB tmpfs hard cap and a 20-minute hard stop per session. The historical
13.07/s rate implies about 78 s of capture per 1,020-row session; the 10/s GO
floor implies 102 s, excluding segment sealing, transfer, and checks. Use a new
explicit worker-local spool directory, never the old 102k corpus directory.
Seal at most 128 MiB of new data per segment, copy it to the off-host Mac,
verify segment/manifest and per-shard hashes there, and acknowledge custody
before removing **only that new spooled segment**. For these two small
sessions, prefer no deletion at all while capturing; abort at 768 MiB
allocated high-water and preserve the tmpfs for diagnosis/transfer. The
896 MiB filesystem limit bounds an in-flight write. A later rolling campaign
requires a separately reviewed resume/delete mechanism. Preserve the old
physical and Mac corpora in place. Neither the spool
nor any transfer/mount operation may target the removable `/dev/sda` or its
partitions. This benign-audit selection is not a prospective alert policy: a
later scoring campaign must score before eviction and retain every alert's
original raw evidence.

The read-only host check reports the required boot ID
`31179aea-9a43-4c5d-8bf6-1205735e42c4` and production kernel
`5.13.0-30-generic`, but only **6.0 GiB free on a 98%-full root disk**.
It also reports `no_turbo=0` restored. That is not the 102k subject's frozen
frequency policy (`no_turbo=1`, CPU 2 maximum 1,800,000 kHz). Do not launch
until read-only path/mount checks prove the *new* spool mountpoint is intended
for a dedicated tmpfs, is not a symlink into `/dev/sda`, and starts with at
least **5.5 GiB** root available. The measured 6,441,705,472 free bytes are
0.71 MiB below exact 6 GiB; choosing 5.5 GiB here is a predeclared gate,
not an adjustment made after seeing capture quality. Reject a path whose
parent mount source is `/dev/sda` or a child. With the 896 MiB tmpfs hard cap,
maintain at least 5 GiB root free; stop immediately
if concurrent system use consumes that reserve. No old corpus deletion or
removable-device write is part of this plan. The coordinator must explicitly
approve reinstating/recording the exact no-turbo policy and an automatic
restore after both sessions. Recheck boot ID, kernel/package hashes, microcode,
CPU 2/3 affinity, perf event identity/period, ambient load, spool allocation,
CPU policy, temperature, and output path before each session. No frequency
change is part of this document's execution.

Thermal and quality stop gates should be stricter than the past 97--100°C
turbo-enabled pilots: start only after an idle package below 70°C, monitor at
least every 30 s, pause at 80°C and abort at 85°C or a second pause, and
restore the pre-run CPU policy on every exit path. Reject any PT/AUX gap or
overflow, PEBS/PMU multiplexing,
missing lane, target oracle failure, unplanned CPU migration, boot/subject
change, repeated capture retry, unverified shard, disk free below 5 GiB, or
capture rate below 10 sealed executions/s over a full balanced block without
an explained benign workload change. Record first-attempt admission, PEBS
density/exact-IP/nonzero-address coverage, family/intensity distributions,
phase times, total throughput, temperature, and retry causes. A matched
untraced timing arm is still required before claiming capture *perturbation*
overhead; these two sessions alone measure throughput and stability.

The launch sequence is deliberately gated: (1) seed-aware target and oracle
tests; (2) read-only exact-subject/disk/thermal/policy checks; (3) coordinator
approval of cap, host reservation, frequency restoration, and off-host path;
(4) a 51-execution balanced smoke with hash-verified transfer; (5) approval
to run the two 1,020-row sessions. Neither step 4 nor 5 has been authorized or
run. The current `NO-GO` for a 24-hour or million-execution campaign remains.

### Reviewable preflight and smoke command contract; **do not execute yet**

The opt-in `hardware_seeded_capture_plan_r2` module and collector
`--execution-plan` flag now define the exact row schedule and raw-selection
set. `smoke` has 51 balanced rows and retains all 51 raw shards; `session-a`
and `session-b` each have 1,020 rows and exactly 204 selected raw shards. Each
51-row block contains one of every family/intensity pair. The same cohort seed
gives 10 identical input seeds per stratum across A/B; the remaining 10 differ.
The loaded plan must match the collector's *full* `identity_sha256`, including
boot, kernel, CPU policy, build, capture code, and perf event parameters. The
collector seals the accepted plan in `execution-plan.json` before any perf
capture and refuses a nonempty artifact directory.
If the new host source is copied without portable `.git` metadata,
`subject_manifest` legitimately records `revision=None` and
`dirty_diff_sha256=None`; do **not** describe that as Git revision custody.
Do not copy this Mac worktree's `.git` pointer file to the host: it refers to
Mac-local Git administration and would make Git identity discovery fail.
Instead, the supervisor requires a preapproved portable source-bundle SHA-256
over the Python/native source tree plus package metadata, pins its own and the
runner/planner file hashes separately, and seals `source-bundle.json` in tmpfs.
The executable has its own SHA-256. Mac acknowledgement must verify that bundle
manifest and hash too.

The proposed new, disjoint deployment paths on `trail-x86` are
`/home/user/.cache/cpu2tensor/seeded-r2-source-20260926` for the frozen
checkout, `/home/user/.cache/cpu2tensor/seeded-r2-build-20260926/hardware_kernel_workload`
for the rebuilt seeded target, and `/run/cpu2tensor-seeded-r2-smoke` for a
dedicated 896 MiB tmpfs mount. **None exists by authorization from this note;**
their creation/build/mount need exact-path review. Do not alter the old host
checkout or `/home/user/.cache/cpu2tensor/vulnerable-kernel-5.13.0-30/store-v3-scale100k-r2`.
The first commands below are read-only **after** that deployment, before any
CPU policy change or capture. Inspect source and mount paths, root free bytes,
package-temperature sensor, and hashes. The currently restored `no_turbo=0`
is expected; the supervisor records it and installs/restores the capture policy.

```bash
set -euo pipefail
C2T_SOURCE=/home/user/.cache/cpu2tensor/seeded-r2-source-20260926
C2T_BINARY=/home/user/.cache/cpu2tensor/seeded-r2-build-20260926/hardware_kernel_workload
C2T_NEW_ROOT=/run/cpu2tensor-seeded-r2-smoke
test -d "$C2T_SOURCE"
test -f "$C2T_BINARY"
test -d "$C2T_NEW_ROOT"
test "$(cat /proc/sys/kernel/random/boot_id)" = 31179aea-9a43-4c5d-8bf6-1205735e42c4
test "$(uname -r)" = 5.13.0-30-generic
findmnt -T "$C2T_NEW_ROOT" -no SOURCE,FSTYPE,TARGET
findmnt -T / -no SOURCE,FSTYPE,TARGET
df -B1 --output=avail /
cat /sys/devices/system/cpu/intel_pstate/no_turbo
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
sha256sum "$C2T_BINARY"
sha256sum "$C2T_SOURCE/python/cpu2tensor/examples/hardware_multimodal_experiment.py"
sha256sum "$C2T_SOURCE/python/cpu2tensor/examples/hardware_seeded_capture_plan_r2.py"
sha256sum "$C2T_SOURCE/python/cpu2tensor/examples/hardware_seeded_capture_supervisor_r2.py"
PYTHONPATH="$C2T_SOURCE/python" /usr/bin/python3.12 -c '
from pathlib import Path
from cpu2tensor.examples.hardware_seeded_capture_supervisor_r2 import source_bundle_manifest
print(source_bundle_manifest(Path("/home/user/.cache/cpu2tensor/seeded-r2-source-20260926"))["digest_sha256"])
'
test "$(cd "$C2T_SOURCE" && PYTHONPATH="$C2T_SOURCE/python" /usr/bin/python3.12 -c 'import cpu2tensor.examples.hardware_multimodal_experiment as e; print(e.__file__)')" = \
  "$C2T_SOURCE/python/cpu2tensor/examples/hardware_multimodal_experiment.py"
for sensor in /sys/class/hwmon/hwmon*/name; do
  printf '%s ' "$sensor"
  cat "$sensor"
done
```

The supervisor computes the exact identity *after* it installs the approved
frequency policy, generates/seals the 51-row plan on tmpfs, and invokes the
collector. The following is the reviewable **unit `ExecStart` command shape**,
not a standalone or authorized launch. The three SHA-256 values and package
temperature input must come from the read-only preflight of the frozen host
checkout; the spool must already be a separately approved, empty 896 MiB
tmpfs mount. A systemd `ExecStopPost` must call the same module's `restore`
subcommand, with an independent restore fallback as described below.

```bash
# NOT APPROVED: for an inspected systemd unit only; never run standalone.
cd "$C2T_SOURCE"
PYTHONPATH="$C2T_SOURCE/python" /usr/bin/python3.12 -m \
  cpu2tensor.examples.hardware_seeded_capture_supervisor_r2 smoke \
  --source-root "$C2T_SOURCE" \
  --binary "$C2T_BINARY" --binary-sha256 "$C2T_BINARY_SHA256" \
  --runner-sha256 "$C2T_RUNNER_SHA256" \
  --planner-sha256 "$C2T_PLANNER_SHA256" \
  --supervisor-sha256 "$C2T_SUPERVISOR_SHA256" \
  --source-bundle-sha256 "$C2T_SOURCE_BUNDLE_SHA256" \
  --spool "$C2T_NEW_ROOT" --temperature-input "$C2T_PACKAGE_TEMP_INPUT" \
  --cohort-seed 2026092601
```

The supervisor refuses a changed source, boot, policy, mount, disk reserve, or
hot package before writing the `/run` restore state or changing CPU policy.
Its read-only preflight spawns the *same root* `/usr/bin/python3.12` child and
requires `cpu2tensor`, runner, planner, and supervisor imports to resolve under
the reviewed new source path, with `torch` import resolving to a real file; it
prints the actual Python/torch/package paths. A direct read-only `sudo -n`
probe was denied (`sudo: a password is required`), so root interpreter/package
resolution is **unverified** until an approved root unit performs that check.
It does **not** implement Mac transfer/acknowledgement; the collector's local
seal is explicitly unadmitted until that independent custody check. The
collector still stops on the first rejected attempt without serializing that
attempt's failed raw window; the failure ledger/partial artifact must be
preserved and reviewed. Confirm the exact binary path, interpreter, package
sensor, destination mount, and restore unit in the approval review. This is
**NO-GO for live smoke** until then.

### Smallest outer supervisor to clear the live-capture gate

The offline supervisor is implemented in
`python/cpu2tensor/examples/hardware_seeded_capture_supervisor_r2.py`, but it
is **not installed or tested on `trail-x86`**. This is its required host
contract for review:

1. After the read-only mount check, create one new dedicated **at most 896 MiB
   tmpfs** mount containing the plan, artifact, temporary files, and supervisor
   logs. Its filesystem limit, unlike a periodic `du` check, enforces the
   under-1 GiB cap during an in-flight raw write. On the reported host, 14 GiB
   RAM was available, but tmpfs is volatile: an interrupted capture is
   unadmitted until off-host hash custody. Require at least 5.5 GiB root free
   before launch and at least 5 GiB thereafter; never mount or write
   `/dev/sda`. The host's read-only available-root result was 6,441,705,472
   bytes, 0.71 MiB short of exact 6 GiB, so the **predeclared** start reserve
   is 5.5 GiB; the running stop remains 5 GiB. A smoke needs about 63 MiB
   retained raw. The two sessions
   together project below 0.7 GB, so **no spool deletion during capture** is
   necessary if sizes remain in range. Leave the tmpfs mounted after the
   supervisor exits, until Mac acknowledgement and explicit cleanup approval.
2. The supervisor records pre-run `no_turbo` and CPU 2 maximum frequency in a
   root-only `/run` state file, installs the exact `no_turbo=1`/1.8 GHz policy,
   and restores both in `finally`. An approved systemd unit must additionally
   run the supervisor's idempotent `restore` subcommand in `ExecStopPost`, plus
   an independent bounded restore timer as a fallback for SIGKILL/interpreter
   failure. Validate this failure path on a non-capture rehearsal before the
   smoke; Python `finally` alone cannot handle SIGKILL or power loss.
3. The supervisor launches the collector in its own process group, with a
   1,200-second hard stop. Every second inspect package temperature from a
   verified `coretemp` package sensor, root free bytes, tmpfs headroom, and
   child liveness. At 80°C stop the pilot conservatively (no resume); at
   85°C, less than 5 GiB free, or a spool/quota fault, abort immediately.
   On *every* stop send TERM to the whole process group, wait at most five
   seconds, then KILL the group and reap it. Preserve incomplete artifacts
   for diagnosis; never treat an interrupted capture as clean.
4. After a clean 51/51 seal, transfer only the new spool's sealed plan,
   manifest, derived rows, raw shards, and hashes to a reviewed Mac directory.
   The Mac independently verifies every file SHA-256, manifest identity/plan
   hash, row count and 51/51 raw retention, then emits an acknowledgement
   containing those hashes. Do not delete anything locally before that exact
   acknowledgement. For the bounded two-session run, prefer retaining the
   entire new spool even after acknowledgement; if any segment is later
   removed, require its own acknowledged hash list and remove only paths
   under this newly created spool. The old 102k corpus is never a deletion
   target.

The unit review must include at least `WorkingDirectory=` set to the exact
new source root, `Environment=PYTHONPATH=` set to that root's `python/`,
`RuntimeMaxSec=21min` as a backup beyond the supervisor's 20-minute child
deadline, `KillMode=control-group`, logs directed inside the bounded tmpfs,
and `ExecStopPost=/usr/bin/python3.12 -m cpu2tensor.examples.hardware_seeded_capture_supervisor_r2 restore` under the
same source environment. No such unit/timer has been installed. The proposed
`/run` tmpfs mount must persist after unit exit until independent Mac hash
acknowledgement; an ephemeral unit-private mount would discard unadmitted
evidence.

This code guards the **51-row smoke only**; it does not yet launch Sessions A/B.
Those sessions need a separately reviewed supervisor extension and Mac custody
gate after the smoke passes. Until the supervisor's Linux-host preflight,
restore unit, tmpfs mount, Mac acknowledger, and failure tests receive
coordinator approval,
the collection decision remains **NO-GO for any live capture** despite passing offline plan
tests.

## Reproduction

With this checkout's `cpu2tensor` package resolved:

```bash
python3.12 -m cpu2tensor.examples.hardware_raw_grammar_probe_r2 \
  /path/to/store-v3-scale100k-r2
python3.12 -m cpu2tensor.examples.hardware_raw_grammar_probe_r2 \
  --stream-benchmark /path/to/store-v3-scale100k-r2
python3.12 -m pytest -q python/tests/test_hardware_raw_grammar_probe_r2.py
python3.12 -m pytest -q \
  python/tests/test_hardware_seeded_capture_plan_r2.py \
  python/tests/test_hardware_seeded_capture_supervisor_r2.py
```

The first command verifies the 525 raw hashes; the second is a memory/throughput
isolation run on the same already-verified files, not replacement custody
verification. The focused synthetic tests check PSB span offsets, byte-order
discrimination, exact and non-core PEBS sites, separate lanes, and rejection of
misattributed or multiplexed inputs.
