#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# One prospective benign session for frozen R4 latent-geometry confirmation.
set -Eeuo pipefail

trap 'status=$?; printf "failure line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR

readonly SOURCE=/home/user/.cache/cpu2tensor/heldout-effect-source-92618e9
readonly PYTHON=/home/user/.cache/cpu2tensor/hardware-pretraining-go/venv/bin/python3.12
readonly BINARY="$SOURCE/build/hardware_kernel_workload"
readonly OUTPUT=/home/user/.cache/cpu2tensor/hardware-latent-confirmation-r5-92618e9
readonly OUTPUT_PARENT=/home/user/.cache/cpu2tensor
readonly PLAN=/home/user/.cache/cpu2tensor/hardware-latent-confirmation-r5-plan.json
readonly POLICY=/run/cpu2tensor-hardware-latent-confirmation-r5-policy
readonly NO_TURBO=/sys/devices/system/cpu/intel_pstate/no_turbo
readonly MAX_FREQ=/sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq
readonly EXPECTED_BOOT=31179aea-9a43-4c5d-8bf6-1205735e42c4
readonly EXPECTED_REVISION=92618e9bee83da121e0f92fc69b347d3ca3518d4
readonly EXPECTED_BINARY=6995c34a887e4509aefe484aef87cd816b6dfedf18f8fa1d4c7679377016b93f
readonly EXPECTED_RUNNER=cbfc22e3d9a153617a387d0e160cd84e95e34889bc50034d830356030a476688
readonly EXPECTED_HARDWARE=de97437305f3251a2cbad837189cc8abec03c273b1349ab29eb1a0c1858c1dd7
readonly EXPECTED_NATIVE=ccaed962f4730922d7f5f1d36c60afbf077fd9350a6eefa4834f6ecb7a35d3ba
readonly EXPECTED_PLAN=0724fa230f8913fc1037be11b2a929e6857f47fba52e6553bad23a5fce556863
readonly SERVICE=cpu2tensor-hardware-latent-confirmation-r5.service
readonly MAX_RUNTIME_SECONDS=600
readonly SCRIPT="$(readlink -f "$0")"
readonly RUNNER="$SOURCE/python/cpu2tensor/examples/hardware_multimodal_experiment.py"
readonly HARDWARE="$SOURCE/python/cpu2tensor/hardware.py"
readonly NATIVE=/home/user/.cache/cpu2tensor/hardware-pretraining-go/venv/lib/python3.12/site-packages/cpu2tensor/_native.cpython-312-x86_64-linux-gnu.so
readonly MIN_START_FREE=$((4 * 1024 * 1024 * 1024))
readonly MIN_RUNNING_FREE=$((3 * 1024 * 1024 * 1024))

restore_policy() {
    if [[ ! -f "$POLICY" ]]; then
        return
    fi
    local original_no_turbo original_maximum
    read -r original_no_turbo original_maximum < "$POLICY"
    printf '%s\n' "$original_no_turbo" > "$NO_TURBO"
    printf '%s\n' "$original_maximum" > "$MAX_FREQ"
    [[ "$(<"$NO_TURBO")" == "$original_no_turbo" ]]
    [[ "$(<"$MAX_FREQ")" == "$original_maximum" ]]
    rm -f "$POLICY"
}

if [[ "${1:-run}" == restore ]]; then
    restore_policy
    exit
fi

[[ $EUID -eq 0 ]]
[[ -n "${INVOCATION_ID:-}" ]]
grep -q "$SERVICE" /proc/self/cgroup
runtime_usec="$(systemctl show "$SERVICE" --property=RuntimeMaxUSec --value)"
[[ "$runtime_usec" == 10min || "$runtime_usec" == 600s ||
   "$runtime_usec" == $((MAX_RUNTIME_SECONDS * 1000000)) ]]
[[ "$(systemctl show "$SERVICE" --property=KillMode --value)" == control-group ]]
[[ "$(uname -r)" == 5.13.0-30-generic ]]
[[ "$(</proc/sys/kernel/random/boot_id)" == "$EXPECTED_BOOT" ]]
[[ "$(<"$SOURCE/source-revision.txt")" == "$EXPECTED_REVISION" ]]
[[ "$(sha256sum "$BINARY" | cut -d' ' -f1)" == "$EXPECTED_BINARY" ]]
[[ "$(sha256sum "$RUNNER" | cut -d' ' -f1)" == "$EXPECTED_RUNNER" ]]
[[ "$(sha256sum "$HARDWARE" | cut -d' ' -f1)" == "$EXPECTED_HARDWARE" ]]
[[ "$(sha256sum "$NATIVE" | cut -d' ' -f1)" == "$EXPECTED_NATIVE" ]]
[[ "$(sha256sum "$PLAN" | cut -d' ' -f1)" == "$EXPECTED_PLAN" ]]
[[ ! -e "$OUTPUT" && ! -e "$POLICY" ]]
[[ $(df -PB1 "$OUTPUT_PARENT" | awk 'NR==2 {print $4}') -ge "$MIN_START_FREE" ]]
[[ $(awk '/MemAvailable:/ {printf "%.0f\n", $2 * 1024}' /proc/meminfo) -ge $((4 * 1024 * 1024 * 1024)) ]]
! pgrep -f '[/]hardware_kernel_workload' >/dev/null
! pgrep -f '[h]ardware_multimodal_experiment' >/dev/null

temperature_input=
for label in /sys/class/hwmon/hwmon*/temp*_label; do
    if [[ "$(<"$label")" == 'Package id 0' ]]; then
        temperature_input="${label%_label}_input"
        break
    fi
done
read_temperature() {
    local value
    value="$(<"$temperature_input")"
    [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 && "$value" -le 125000 ]]
    printf '%s\n' "$value"
}

[[ -n "$temperature_input" ]]
[[ "$(read_temperature)" -lt 70000 ]]

original_no_turbo="$(<"$NO_TURBO")"
original_maximum="$(<"$MAX_FREQ")"
printf '%s %s\n' "$original_no_turbo" "$original_maximum" > "$POLICY"
chmod 600 "$POLICY"
trap restore_policy EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

systemd-run --quiet --unit=cpu2tensor-hardware-latent-confirmation-r5-restore \
    --on-active=11m --timer-property=AccuracySec=1s "$SCRIPT" restore

printf '1\n' > "$NO_TURBO"
printf '1800000\n' > "$MAX_FREQ"
[[ "$(<"$NO_TURBO")" == 1 && "$(<"$MAX_FREQ")" == 1800000 ]]

export PYTHONPATH="$SOURCE/python"
mapfile -t imported < <("$PYTHON" - <<'PY'
from importlib.util import find_spec
from pathlib import Path
for name in (
    "cpu2tensor", "cpu2tensor.hardware",
    "cpu2tensor.examples.hardware_multimodal_experiment", "cpu2tensor._native",
):
    spec = find_spec(name)
    if spec is None or spec.origin is None:
        raise SystemExit(f"cannot resolve {name}")
    print(Path(spec.origin).resolve())
PY
)
[[ "${imported[0]}" == "$SOURCE/python/cpu2tensor/__init__.py" ]]
[[ "${imported[1]}" == "$HARDWARE" ]]
[[ "${imported[2]}" == "$RUNNER" ]]
[[ "${imported[3]}" == "$NATIVE" ]]
[[ "$(read_temperature)" -lt 80000 ]]

mkdir -m 0750 "$OUTPUT"
cp "$SOURCE/source-revision.txt" "$OUTPUT/source-revision.txt"
sha256sum "$BINARY" "$SOURCE/native/examples/hardware_kernel_workload.c" \
    "$RUNNER" "$HARDWARE" "$NATIVE" "$SCRIPT" "$PLAN" \
    > "$OUTPUT/source-hashes.txt"

"$PYTHON" -m cpu2tensor.examples.hardware_multimodal_experiment \
    "$OUTPUT/benign-d" --binary "$BINARY" --collect-only \
    --families getpid --target-cpu 2 --controller-cpu 3 \
    --data-pages 1024 --aux-pages 8192 \
    --pebs-signal memory_loads --pebs-period 10000 \
    --capture-retries 0 --retain-raw-fraction 1 \
    --execution-plan "$PLAN" \
    --repetitions 1 --training-rows 1 --calibration-rows 1 \
    --heldout-family-count 1 --seed 1 \
    > "$OUTPUT/benign-d.log" 2>&1 &
pid=$!
status=0
deadline=$((SECONDS + 300))
while kill -0 "$pid" 2>/dev/null; do
    temperature="$(read_temperature)" || status=1
    if [[ "$status" -ne 0 || "$SECONDS" -ge "$deadline" ||
          "$temperature" -ge 80000 ||
          $(df -PB1 "$OUTPUT_PARENT" | awk 'NR==2 {print $4}') -lt "$MIN_RUNNING_FREE" ||
          "$(<"$NO_TURBO")" != 1 || "$(<"$MAX_FREQ")" != 1800000 ]]; then
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" || true
        exit 1
    fi
    sleep 1
done
wait "$pid" || status=$?
[[ "$status" -eq 0 ]]
[[ -f "$OUTPUT/benign-d/capture-manifest.json" ]]
[[ "$(read_temperature)" -lt 80000 ]]
[[ $(df -PB1 "$OUTPUT_PARENT" | awk 'NR==2 {print $4}') -ge "$MIN_RUNNING_FREE" ]]

chown -R user:user "$OUTPUT"
restore_policy
trap - EXIT INT TERM
systemctl stop cpu2tensor-hardware-latent-confirmation-r5-restore.timer
