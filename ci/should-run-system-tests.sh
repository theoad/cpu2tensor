#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 EVENT_NAME COMMIT OUTPUT_FILE" >&2
    exit 2
fi

readonly event_name="$1"
readonly commit="$2"
readonly output_file="$3"
readonly title="$(git show --no-patch --format=%s "$commit")"
readonly words="$(git rev-list --parents --max-count=1 "$commit" | wc -w)"

run=false
reason="ordinary commit"
if [[ "$title" == *'[TESTME]'* ]]; then
    run=true
    reason='commit title contains [TESTME]'
elif [[ "$event_name" == push && "$words" -gt 2 ]]; then
    run=true
    reason='pushed merge commit'
fi

echo "run=$run" >> "$output_file"
echo "System tests: $run ($reason; $title)"
