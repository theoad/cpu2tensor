#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Exact-host, bounded collection for the lawful futex sensor/transfer proxy.
set -euo pipefail

readonly SOURCE=/home/user/.cache/cpu2tensor/futex-effect-source-bccf7a8
readonly PYTHON=/home/user/.cache/cpu2tensor/hardware-pretraining-go/venv/bin/python3.12
readonly BINARY="$SOURCE/build/hardware_kernel_workload"
readonly OUTPUT=/home/user/.cache/cpu2tensor/futex-effect-r1-bccf7a8
readonly OUTPUT_PARENT=/home/user/.cache/cpu2tensor
readonly POLICY=/run/cpu2tensor-futex-effect-r1-policy
readonly NO_TURBO=/sys/devices/system/cpu/intel_pstate/no_turbo
readonly MAX_FREQ=/sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq
readonly EXPECTED_BOOT=31179aea-9a43-4c5d-8bf6-1205735e42c4
readonly EXPECTED_REVISION=bccf7a89981907cd6888baa6214c0feeebb624b4
readonly EXPECTED_BINARY=760e5c5bde97989c5f26890b9fa1c121284d36b2cc23fa5801b05e30fc48daf5
readonly EXPECTED_RUNNER=d880989dd45f4edfc5a0b18649e17f35b80211047ab6374e8666e704aaeb3948
readonly EXPECTED_HARDWARE=de97437305f3251a2cbad837189cc8abec03c273b1349ab29eb1a0c1858c1dd7
readonly EXPECTED_NATIVE=ccaed962f4730922d7f5f1d36c60afbf077fd9350a6eefa4834f6ecb7a35d3ba
readonly SERVICE=cpu2tensor-futex-effect-r1.service
readonly MAX_RUNTIME_SECONDS=600
readonly SCRIPT="$(readlink -f "$0")"
readonly RUNNER="$SOURCE/python/cpu2tensor/examples/hardware_multimodal_experiment.py"
readonly HARDWARE="$SOURCE/python/cpu2tensor/hardware.py"
readonly NATIVE=/home/user/.cache/cpu2tensor/hardware-pretraining-go/venv/lib/python3.12/site-packages/cpu2tensor/_native.cpython-312-x86_64-linux-gnu.so

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
[[ ! -e "$OUTPUT" && ! -e "$POLICY" ]]
[[ $(df -PB1 "$OUTPUT_PARENT" | awk 'NR==2 {print $4}') -ge $((5 * 1024 * 1024 * 1024)) ]]
[[ $(awk '/MemAvailable:/ {printf "%.0f\n", $2 * 1024}' /proc/meminfo) -ge $((4 * 1024 * 1024 * 1024)) ]]
! pgrep -f '[/]hardware_kernel_workload' >/dev/null
! pgrep -f '[h]ardware_multimodal_experiment' >/dev/null
export PYTHONPATH="$SOURCE/python"
mapfile -t imported < <("$PYTHON" - <<'PY'
from pathlib import Path
import cpu2tensor
from cpu2tensor import hardware
from cpu2tensor.examples import hardware_multimodal_experiment as experiment
print(Path(cpu2tensor.__file__).resolve())
print(Path(hardware.__file__).resolve())
print(Path(experiment.__file__).resolve())
PY
)
[[ "${imported[0]}" == "$SOURCE/python/cpu2tensor/__init__.py" ]]
[[ "${imported[1]}" == "$HARDWARE" ]]
[[ "${imported[2]}" == "$RUNNER" ]]

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
    if [[ ! "$value" =~ ^[0-9]+$ || "$value" -le 0 || "$value" -gt 125000 ]]; then
        return 1
    fi
    printf '%s\n' "$value"
}

[[ -n "$temperature_input" ]]
temperature="$(read_temperature)"
[[ "$temperature" -lt 70000 ]]

original_no_turbo="$(<"$NO_TURBO")"
original_maximum="$(<"$MAX_FREQ")"
printf '%s %s\n' "$original_no_turbo" "$original_maximum" > "$POLICY"
chmod 600 "$POLICY"
trap restore_policy EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

systemd-run --quiet --unit=cpu2tensor-futex-effect-r1-restore \
    --on-active=11m --timer-property=AccuracySec=1s \
    "$SCRIPT" restore

printf '1\n' > "$NO_TURBO"
printf '1800000\n' > "$MAX_FREQ"
[[ "$(<"$NO_TURBO")" == 1 && "$(<"$MAX_FREQ")" == 1800000 ]]

mkdir -m 0750 "$OUTPUT"
cp "$SOURCE/source-revision.txt" "$OUTPUT/source-revision.txt"
sha256sum "$BINARY" \
    "$SOURCE/native/examples/hardware_kernel_workload.c" \
    "$RUNNER" "$HARDWARE" "$NATIVE" "$SCRIPT" \
    > "$OUTPUT/source-hashes.txt"

collect_session() {
    local name=$1 seed=$2
    local pid status=0 temperature deadline=$((SECONDS + 240))
    "$PYTHON" -m cpu2tensor.examples.hardware_multimodal_experiment \
        "$OUTPUT/$name" --binary "$BINARY" --collect-only \
        --families getpid futex_wake futex_mismatch \
        --target-cpu 2 --controller-cpu 3 \
        --data-pages 1024 --aux-pages 8192 \
        --pebs-signal memory_loads --pebs-period 10000 \
        --capture-retries 0 --retain-raw-fraction 1 \
        --loop-divisors 1 --vary-input-seed \
        --repetitions 16 --training-rows 8 --calibration-rows 4 \
        --heldout-family-count 1 --seed "$seed" \
        > "$OUTPUT/$name.log" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        temperature="$(read_temperature)" || {
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" || true
            return 1
        }
        if [[ "$SECONDS" -ge "$deadline" || "$temperature" -ge 80000 ||
              $(df -PB1 "$OUTPUT_PARENT" | awk 'NR==2 {print $4}') -lt $((5 * 1024 * 1024 * 1024)) ||
              "$(<"$NO_TURBO")" != 1 || "$(<"$MAX_FREQ")" != 1800000 ]]; then
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" || true
            return 1
        fi
        sleep 1
    done
    wait "$pid" || status=$?
    [[ "$status" -eq 0 ]]
    [[ -f "$OUTPUT/$name/capture-manifest.json" ]]
    temperature="$(read_temperature)"
    [[ "$temperature" -lt 80000 ]]
    [[ $(df -PB1 "$OUTPUT_PARENT" | awk 'NR==2 {print $4}') -ge $((5 * 1024 * 1024 * 1024)) ]]
}

collect_session session-a 2026092701
collect_session session-b 2026092702

chown -R user:user "$OUTPUT"
restore_policy
trap - EXIT INT TERM
systemctl stop cpu2tensor-futex-effect-r1-restore.timer
