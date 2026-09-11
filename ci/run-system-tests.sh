#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

readonly root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly work="$(mktemp -d /tmp/cpu2tensor-system.XXXXXX)"
trap 'rm -rf "$work"' EXIT

if [[ "${CPU2TENSOR_SYSTEM_DEADLINE:-}" != active ]]; then
    readonly started="$(date +%s)"
    set +e
    timeout --signal=TERM --kill-after=10s 590s \
        env CPU2TENSOR_SYSTEM_DEADLINE=active "$0"
    status=$?
    set -e
    if [[ $status -eq 124 || $status -eq 137 ]]; then
        echo "System tests exceeded the 590 second execution limit." >&2
    elif [[ $status -eq 0 ]]; then
        echo "System tests completed in $(( $(date +%s) - started )) seconds (limit: 600)."
    fi
    exit "$status"
fi

export CMAKE_BUILD_PARALLEL_LEVEL=2
export TORCH_NUM_THREADS=1
export OMP_NUM_THREADS=1

cmake -S "$root/native" -B "$work/native" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_TESTING=ON \
    -DCPU2TENSOR_BUILD_WORKER=ON \
    -DCPU2TENSOR_QEMU_INCLUDE_DIR=/opt/qemu/include
cmake --build "$work/native" --parallel 2
ctest --test-dir "$work/native" --output-on-failure
aarch64-linux-gnu-gcc -O2 -static \
    "$root/native/examples/checksum.c" -o "$work/checksum-aarch64"
python -m pip install --quiet --no-deps --editable "$root" \
    --config-settings="build-dir=$work/python-build"
export CPU2TENSOR_CI_BUILD="$work/native"
export CPU2TENSOR_CI_ARM_TARGET="$work/checksum-aarch64"
export CPU2TENSOR_CI_QEMU_X86=/opt/qemu/bin/qemu-x86_64
export CPU2TENSOR_CI_QEMU_ARM=/opt/qemu/bin/qemu-aarch64
cd "$root"
python -m pytest python/system_tests --quiet
