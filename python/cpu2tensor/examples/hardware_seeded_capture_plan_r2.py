# SPDX-License-Identifier: AGPL-3.0-only
"""Exact, subject-bound plans for a bounded seeded hardware-capture audit.

Plan generation is offline. Loading a plan never opens perf or changes CPU policy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

from cpu2tensor.examples.hardware_multimodal_experiment import WORKLOAD_LOOPS


SCHEMA = "cpu2tensor-seeded-capture-plan-r2"
KINDS = ("smoke", "session-a", "session-b")
INTENSITY_DIVISORS = (1, 2, 5)
RAW_SLOTS = frozenset((0, 1, 10, 11))
IDENTITY_RE = re.compile(r"[0-9a-f]{64}\Z")


def _seed(*parts: object) -> int:
    material = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def make_plan(kind: str, identity_sha256: str, cohort_seed: int) -> dict[str, object]:
    """Build 17×3 smoke or 17×3×20 session rows without trace-dependent choices."""
    if kind not in KINDS:
        raise ValueError("unknown seeded capture plan kind")
    if not IDENTITY_RE.fullmatch(identity_sha256):
        raise ValueError("plan requires a lowercase 64-digit subject identity hash")
    if type(cohort_seed) is not int or not 0 <= cohort_seed < 1 << 64:
        raise ValueError("cohort seed must be an unsigned 64-bit integer")
    rows: list[dict[str, object]] = []
    slots = range(1 if kind == "smoke" else 20)
    for slot in slots:
        block: list[dict[str, object]] = []
        for family, base_loops in WORKLOAD_LOOPS.items():
            for intensity_index, divisor in enumerate(INTENSITY_DIVISORS):
                scope = "shared" if slot < 10 else kind
                input_seed = _seed(
                    "seeded-capture-input-r2", cohort_seed, family,
                    intensity_index, slot, scope,
                )
                block.append({
                    "execution_id": f"{family}-i{intensity_index}-s{slot:02d}",
                    "family": family,
                    "repetition": intensity_index * 20 + slot,
                    "partition": "collection",
                    "intensity_index": intensity_index,
                    "loops": max(1, base_loops // divisor),
                    "input_seed": input_seed,
                    "retain_raw": kind == "smoke" or slot in RAW_SLOTS,
                })
        random.Random(_seed("seeded-capture-order-r2", cohort_seed, kind, slot)).shuffle(block)
        rows.extend(block)
    return {
        "schema": SCHEMA,
        "kind": kind,
        "identity_sha256": identity_sha256,
        "cohort_seed": cohort_seed,
        "rows": rows,
    }


def load_plan(path: Path, identity_sha256: str) -> dict[str, object]:
    """Reject changed subject, malformed rows, and non-preregistered selection."""
    data = path.read_bytes()
    if len(data) > 1_000_000:
        raise ValueError("capture plan is too large")
    def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate capture plan JSON key")
            value[key] = item
        return value
    try:
        plan = json.loads(data, object_pairs_hook=unique_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid capture plan JSON") from error
    if type(plan) is not dict or plan.get("identity_sha256") != identity_sha256:
        raise ValueError("capture plan subject identity changed")
    expected = make_plan(plan.get("kind"), identity_sha256, plan.get("cohort_seed"))
    if json.dumps(plan, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("capture plan differs from exact deterministic protocol")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--identity-sha256", required=True)
    parser.add_argument("--cohort-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(args.kind, args.identity_sha256, args.cohort_seed)
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing capture plan")
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"{args.kind}: {len(plan['rows'])} rows, "
          f"{sum(row['retain_raw'] for row in plan['rows'])} retained raw")


if __name__ == "__main__":
    main()
