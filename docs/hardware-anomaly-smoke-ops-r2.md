# Seeded hardware smoke R2: review-only operations contract

**Status: NO-GO for live capture.** These files are templates, not installed
units. Nothing in this note authorizes host writes, a tmpfs mount, perf
collection, or changes to the immutable 102k corpus. `/dev/sda` and its child
filesystems are excluded. The same-boot/kernel requirement remains
`31179aea-9a43-4c5d-8bf6-1205735e42c4` / `5.13.0-30-generic`.

## Exact new paths and unit inputs

- Source: `/home/user/.cache/cpu2tensor/seeded-r2-source-20260926`.
- Seeded binary: `/home/user/.cache/cpu2tensor/seeded-r2-build-20260926/hardware_kernel_workload`.
- Spool: `/run/cpu2tensor-seeded-r2-smoke`, an independently mounted, empty,
  exact 896 MiB tmpfs that must persist until Mac custody acknowledgement.
  Its mount root must be owned by uid 0 and the primary gid of host account
  `user`, with mode `0750`. The root service uses `Group=user` and
  `UMask=0027` so children create group-readable files/directories without
  group write access. Do not broaden permissions with `chmod -R`.
- Restore state: `/run/cpu2tensor-seeded-r2-policy.json`, fixed in code; no
  CLI override.
- Unit templates: `ops/systemd/cpu2tensor-seeded-smoke-r2.service.in`,
  `ops/systemd/cpu2tensor-seeded-restore-r2.service`, and
  `ops/systemd/cpu2tensor-seeded-restore-r2.timer`.

The smoke template's five `@...@` SHA-256 placeholders and
`@PACKAGE_TEMP_INPUT@` must be substituted with inspected exact host values,
then the rendered unit inspected and hashed. The source and binary are mounted
read-only within the service. The supervisor independently hashes the binary,
runner, planner, supervisor, and complete portable source bundle both before
the policy change and after a successful child exit, before admission. A
source-only deployment does **not** provide a Git revision/dirty-diff hash;
`source-bundle.json` and per-module hashes are its custody identity.

Supervisor stdout/stderr goes to rate-limited journald, not to the tmpfs:
systemd would otherwise create `supervisor.log` before Python starts and make
the deliberately empty-spool preflight fail. The collector's `collector.log`
is created inside tmpfs only *after* preflight; that file remains in Mac
custody. No journal file is created by this unit inside the spool. The ordinary
root-free-space stop also protects against external journal growth.

The unit is required, not a suggested wrapper. Direct invocation fails closed
unless `INVOCATION_ID`, the expected systemd cgroup, an active independent
restore timer, a `control-group` kill mode, a bounded `RuntimeMaxSec`, and the
active `ExecStopPost` restore hook are present. Preflight verifies mount
ownership/mode, effective group, process umask, and actual read/execute access
as `user`; local sealing verifies every new file
and directory remains group-readable and non-group-writable for `user` before
calling it transferable. The restore CLI itself is
idempotent and always reads the fixed state path. The timer checks once per
minute and restores only if that state exists and the smoke service state is
explicitly `inactive` or `failed`; it skips `active`, `activating`, and
`deactivating` states. It backs up `ExecStopPost`, not the normal `finally`
path. A power loss
still loses volatile tmpfs data and `/run` state; no capture is admitted from
that partial run.

## Read-only approval preflight, before any install

Run only after exact-path deployment has been separately approved. No command
below changes host state:

```bash
test "$(cat /proc/sys/kernel/random/boot_id)" = 31179aea-9a43-4c5d-8bf6-1205735e42c4
test "$(uname -r)" = 5.13.0-30-generic
findmnt -n -o SOURCE,FSTYPE,TARGET --target /run/cpu2tensor-seeded-r2-smoke
df -B1 --output=avail /
cat /proc/meminfo | rg '^MemAvailable:'
cat /proc/self/cgroup
cat /sys/devices/system/cpu/cpu2/topology/physical_package_id
cat /sys/devices/system/cpu/intel_pstate/no_turbo
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq
sha256sum /home/user/.cache/cpu2tensor/seeded-r2-build-20260926/hardware_kernel_workload
sha256sum /home/user/.cache/cpu2tensor/seeded-r2-source-20260926/python/cpu2tensor/examples/hardware_multimodal_experiment.py
sha256sum /home/user/.cache/cpu2tensor/seeded-r2-source-20260926/python/cpu2tensor/examples/hardware_seeded_capture_plan_r2.py
sha256sum /home/user/.cache/cpu2tensor/seeded-r2-source-20260926/python/cpu2tensor/examples/hardware_seeded_capture_supervisor_r2.py
```

Select a `coretemp` `tempN_input` whose `tempN_label` is *exactly* `Package id
<CPU2 physical_package_id>`. The supervisor rejects a merely similar sensor.
The root service uses the discovered
`/home/user/.cache/cpu2tensor/hardware-pretraining-go/venv/bin/python3.12`
and must resolve `torch` with
`weights_only` support, and `cpu2tensor` under the new source. This remains
unverified **as root** on `trail-x86`: a read-only `sudo -n` Python probe was denied.
The host's `/usr/bin/python3.12` and default user Python lack torch; the
virtualenv imports it. The
supervisor requires at least 5.5 GiB root free and 4 GiB `MemAvailable` at
start, then 5 GiB and 3 GiB during capture. A finite cgroup limit must leave
at least 2 GiB at start and 1 GiB throughout. The supervisor reads either
cgroup v2 or v1 memory accounting; unreadable accounting fails closed.

## Unit-bound, non-capture failure rehearsal

Only after reviewing the rendered unit and timer, use the *same new smoke
service* with its `C2T_SEEDED_REHEARSAL=1` environment value. This mode runs
the exact preflight and CPU policy state/restore path, then starts `/bin/sleep`
for 30 seconds; it never calls the target, perf, or collector and writes no
capture spool artifact. An operator must inspect the unit with
`systemctl cat`, `systemctl show -p ExecStopPost,RuntimeMaxUSec,KillMode`, and
`systemctl is-active ...timer` before start. In a first rehearsal let the
service exit normally, then verify `no_turbo`, CPU2 max frequency, and removal
of the fixed state file. In a second rehearsal, after the state file exists,
send `SIGKILL` **only to the main process of the new smoke service** using
`systemctl kill --kill-whom=main --signal=SIGKILL
cpu2tensor-seeded-smoke-r2.service`. Confirm the child is gone, `ExecStopPost`
restored both values, and the one-minute timer would restore a deliberately
stranded state. Never kill another service or change the old capture checkout.
After rehearsal, render/review a final unit with `C2T_SEEDED_REHEARSAL=0`;
check its exact bytes and hashes again. **Do not run this procedure yet.**

The supervisor's 20-minute internal deadline and the unit's 20-minute
`RuntimeMaxSec` are independent stops. Package temperature must be below
70°C to start and stops at 80°C. The 896 MiB tmpfs cap includes plan,
artifacts, logs, and partial files; a 128 MiB remaining-space stop is checked
each second. A clean smoke requires 51/51 planned, first-attempt,
raw-retained rows; any loss, unavailable modality, multiplexing, manifest
disagreement, unexpected source, or source drift leaves the spool unadmitted.

## Independent Mac custody acknowledgement

Copy the **entire** newly created spool to a reviewed Mac directory without
removing anything on the host. On the host, obtain the exact SHA-256 of
`smoke/capture-manifest.json` and the already pinned source-bundle and binary
digests. On the Mac, from the matching reviewed checkout, run:

```bash
PYTHONPATH=/Users/theoad/.codex/worktrees/cpu2tensor-hardware-pretraining-go/python \
  python3.12 -m cpu2tensor.examples.hardware_seeded_mac_custody_r2 \
  /ABSOLUTE/NEW/MAC/COPY \
  --expected-host-manifest-sha256 HOST_MANIFEST_SHA256 \
  --expected-source-bundle-sha256 SOURCE_BUNDLE_SHA256 \
  --expected-binary-sha256 BINARY_SHA256 \
  --ack-output /ABSOLUTE/NEW/MAC/ACK.json
```

The verifier reads but does not modify the copy; it rejects symlinks,
partials, unexpected files, missing or nonregular sealed evidence, excess
bytes, differing source modules, sideband hashes, subject/plan/row hashes,
and sensor status/count disagreement. It creates the ack outside the copy
with exclusive creation only after all checks pass. An ack is custody of
these exact bytes, not authorization to delete the host spool. Any deletion
would need separate exact-path approval and cannot target the old archive.

## Offline retained-schema cross-check

On `mac.local`, the supervisor sensor verifier was run against all 525 retained
raw audit shards and corresponding derived shards of the immutable 102k
corpus. Every raw/derived SHA-256 and every per-entry sensor-count dictionary
matched: 572,193,760 PT bytes and 512,201 PEBS samples. The old event is
`memory_stores`; a diagnostic-only keyword allowed checking that event, while
the smoke verifier still requires `memory_loads`. Raw tensors use int64
PEBS exact-IP flags, uint8 PT bytes, bool derived availability, and counters
with positive equal enabled/running time; these match the verifier contract.
This does not establish the new seeded smoke's input/oracle/identity quality,
and is not a new capture or throughput claim.
