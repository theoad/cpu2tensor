# SPDX-License-Identifier: AGPL-3.0-only
"""Offline exact-plan checks; these tests never open perf or change CPU policy."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from cpu2tensor.examples import hardware_multimodal_experiment as experiment
from cpu2tensor.examples import hardware_seeded_capture_plan_r2 as plans


IDENTITY = "a" * 64


class HardwareSeededCapturePlanR2Tests(unittest.TestCase):
    def test_smoke_and_sessions_have_exact_balance_and_shared_inputs(self) -> None:
        smoke = plans.make_plan("smoke", IDENTITY, 71)
        a = plans.make_plan("session-a", IDENTITY, 71)
        b = plans.make_plan("session-b", IDENTITY, 71)
        self.assertEqual(len(smoke["rows"]), 51)
        self.assertEqual(len(a["rows"]), 1020)
        self.assertEqual(len(b["rows"]), 1020)
        self.assertEqual(sum(row["retain_raw"] for row in smoke["rows"]), 51)
        self.assertEqual(sum(row["retain_raw"] for row in a["rows"]), 204)
        self.assertEqual(sum(row["retain_raw"] for row in b["rows"]), 204)
        for plan, expected in ((smoke, 1), (a, 20), (b, 20)):
            counts = Counter((row["family"], row["intensity_index"])
                             for row in plan["rows"])
            self.assertEqual(len(counts), 51)
            self.assertEqual(set(counts.values()), {expected})
            self.assertEqual(len({row["execution_id"] for row in plan["rows"]}),
                             len(plan["rows"]))
            for offset in range(0, len(plan["rows"]), 51):
                block = plan["rows"][offset:offset + 51]
                self.assertEqual(len({(row["family"], row["intensity_index"])
                                      for row in block}), 51)
            for row in plan["rows"]:
                divisor = plans.INTENSITY_DIVISORS[row["intensity_index"]]
                self.assertEqual(row["loops"],
                                 max(1, experiment.WORKLOAD_LOOPS[row["family"]] // divisor))
        a_by_id = {row["execution_id"]: row for row in a["rows"]}
        b_by_id = {row["execution_id"]: row for row in b["rows"]}
        self.assertNotEqual([row["execution_id"] for row in a["rows"]],
                            [row["execution_id"] for row in b["rows"]])
        for execution_id, first in a_by_id.items():
            second = b_by_id[execution_id]
            slot = first["repetition"] % 20
            self.assertEqual(first["input_seed"] == second["input_seed"], slot < 10)
            self.assertEqual(first["retain_raw"], second["retain_raw"])

    def test_plan_loading_fails_closed_on_identity_or_any_row_change(self) -> None:
        plan = plans.make_plan("session-a", IDENTITY, 71)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(plans.load_plan(path, IDENTITY), plan)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                plans.load_plan(path, "b" * 64)
            plan["rows"][0]["retain_raw"] = 1
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "deterministic protocol"):
                plans.load_plan(path, IDENTITY)
            path.write_text('{"schema":"x","schema":"y"}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                plans.load_plan(path, IDENTITY)

    def test_explicit_plan_cannot_enter_training_path(self) -> None:
        args = experiment.parser().parse_args([
            "/tmp/artifact", "--binary", "/tmp/workload",
            "--execution-plan", "/tmp/plan.json",
        ])
        with self.assertRaisesRegex(ValueError, "collect-only"):
            experiment.run(args)


if __name__ == "__main__":
    unittest.main()
