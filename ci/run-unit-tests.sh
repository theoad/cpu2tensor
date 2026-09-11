#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

readonly root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly work="$(mktemp -d /tmp/cpu2tensor-unit.XXXXXX)"
trap 'rm -rf "$work"' EXIT

if [[ "${CPU2TENSOR_UNIT_DEADLINE:-}" != active ]]; then
    readonly started="$(date +%s)"
    set +e
    timeout --signal=TERM --kill-after=5s 55s \
        env CPU2TENSOR_UNIT_DEADLINE=active "$0"
    status=$?
    set -e
    if [[ $status -eq 124 || $status -eq 137 ]]; then
        echo "Unit tests exceeded the 55 second execution limit." >&2
    elif [[ $status -eq 0 ]]; then
        echo "Unit tests completed in $(( $(date +%s) - started )) seconds (limit: 60)."
    fi
    exit "$status"
fi

export CMAKE_BUILD_PARALLEL_LEVEL=2
export TORCH_NUM_THREADS=1
export OMP_NUM_THREADS=1

cmake -S "$root/native" -B "$work/native" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_TESTING=ON \
    -DCMAKE_C_FLAGS='--coverage -O0 -g' \
    -DCMAKE_CXX_FLAGS='--coverage -O0 -g'
cmake --build "$work/native" --parallel 2
ctest --test-dir "$work/native" --output-on-failure

CFLAGS='--coverage -O0 -g' CXXFLAGS='--coverage -O0 -g' LDFLAGS='--coverage' \
    python -m pip install --quiet --no-deps --editable "$root" \
    --config-settings="build-dir=$work/python-build"
cd "$root"
python -m coverage erase
python -m coverage run --source=cpu2tensor \
    -m pytest python/tests --quiet
python -m coverage report --show-missing --fail-under=95
if [[ -n "${CPU2TENSOR_COVERAGE_DIR:-}" ]]; then
    mkdir -p "$CPU2TENSOR_COVERAGE_DIR"
    python -m coverage xml -o "$CPU2TENSOR_COVERAGE_DIR/python.xml"
fi

gcovr --root "$root" "$work/native" "$work/python-build" \
    --filter "$root/native/core/" \
    --filter "$root/native/include/cpu2tensor/" \
    --exclude-unreachable-branches \
    --print-summary --fail-under-line 95
if [[ -n "${CPU2TENSOR_COVERAGE_DIR:-}" ]]; then
    gcovr --root "$root" "$work/native" "$work/python-build" \
        --filter "$root/native/core/" \
        --filter "$root/native/include/cpu2tensor/" \
        --exclude-unreachable-branches \
        --xml "$CPU2TENSOR_COVERAGE_DIR/native.xml" --xml-pretty
    readonly python_covered="$(sed -n 's/.*lines-covered="\([0-9]*\)".*/\1/p' "$CPU2TENSOR_COVERAGE_DIR/python.xml" | head -1)"
    readonly python_valid="$(sed -n 's/.*lines-valid="\([0-9]*\)".*/\1/p' "$CPU2TENSOR_COVERAGE_DIR/python.xml" | head -1)"
    readonly native_covered="$(sed -n 's/.*lines-covered="\([0-9]*\)".*/\1/p' "$CPU2TENSOR_COVERAGE_DIR/native.xml" | head -1)"
    readonly native_valid="$(sed -n 's/.*lines-valid="\([0-9]*\)".*/\1/p' "$CPU2TENSOR_COVERAGE_DIR/native.xml" | head -1)"
    awk -v covered="$((python_covered + native_covered))" \
        -v valid="$((python_valid + native_valid))" \
        'BEGIN { printf "%.1f\n", covered * 100 / valid }' \
        > "$CPU2TENSOR_COVERAGE_DIR/total.txt"
fi
