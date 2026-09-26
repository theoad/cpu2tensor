# SPDX-License-Identifier: AGPL-3.0-only
"""Frozen-score review budget tests; no capture or model dependency."""

import math
import unittest
from dataclasses import replace

from cpu2tensor.examples.hardware_review_budget_r1 import (
    ScoredExecution,
    evaluate_review_budget,
    exact_binomial_upper_bound,
)


class HardwareReviewBudgetR1Tests(unittest.TestCase):
    def test_exact_one_sided_bound(self) -> None:
        self.assertAlmostEqual(
            exact_binomial_upper_bound(0, 10), 1.0 - 0.05 ** 0.1
        )
        self.assertEqual(exact_binomial_upper_bound(10, 10), 1.0)
        self.assertAlmostEqual(
            exact_binomial_upper_bound(83, 1_000_000) * 1_000_000,
            99.622,
            places=2,
        )

    def test_low_ranked_effect_and_concentrated_bad_family(self) -> None:
        rows = [
            ScoredExecution("b-1", 0.95, "benign", "s1", "bad", True, cluster_id="noise"),
            ScoredExecution("b-2", 0.90, "benign", "s2", "bad", True, cluster_id="noise"),
            ScoredExecution("b-3", 0.40, "benign", "s1", "normal", True, cluster_id="normal"),
            ScoredExecution("b-4", 0.30, "benign", "s2", "normal", True, cluster_id="normal"),
            ScoredExecution("e-1", 0.99, "effect", "s1", "primary", True, "p1", "effect-a"),
            ScoredExecution("p1-b", 0.85, "benign", "s1", "primary", True, "p1", "sibling-a"),
            ScoredExecution("e-2", 0.70, "effect", "s2", "primary", True, "p2", "effect-b"),
            ScoredExecution("p2-b", 0.60, "benign", "s2", "primary", True, "p2", "sibling-b"),
        ]
        report = evaluate_review_budget(rows, threshold=0.80, top_k=2)
        self.assertEqual(report["selected_effects"], 1)
        self.assertEqual(report["raw_top_k_recall"], 0.5)
        self.assertEqual(report["benign_by_family"]["bad"]["alerts"], 2)
        self.assertEqual(report["benign_by_family"]["bad"]["alerts_per_million"], 1_000_000)
        self.assertEqual(report["benign_by_family"]["normal"]["alerts"], 0)
        self.assertEqual(report["paired"]["effect_win_rate"], 1.0)
        self.assertEqual(report["effect"]["recall_at_threshold"], 0.5)
        capped = evaluate_review_budget(rows, threshold=0.80, top_k=4, cluster_cap=1)
        self.assertEqual(capped["raw_top_k_recall"], 0.5)
        self.assertEqual(capped["cluster_capped_diagnostic"]["raw_top_k_recall"], 1.0)
        self.assertEqual(capped["cluster_capped_diagnostic"]["total_budget"], 4)
        self.assertEqual(capped["cluster_capped_diagnostic"]["selected"], 4)
        self.assertTrue(capped["cluster_capped_diagnostic"]["retrospective_only"])
        relabeled = [replace(row, label="benign", pair_id=None) for row in rows]
        relabeled_capped = evaluate_review_budget(
            relabeled, threshold=0.80, top_k=4, cluster_cap=1
        )
        self.assertEqual(
            capped["cluster_capped_diagnostic"]["selected_ids"],
            relabeled_capped["cluster_capped_diagnostic"]["selected_ids"],
        )

    def test_refuses_censored_nonfinite_and_partial_pairs(self) -> None:
        good = ScoredExecution("b", 0.1, "benign", "s", "f", True)
        for bad in (
            ScoredExecution("x", math.nan, "benign", "s", "f", True),
            ScoredExecution("x", math.inf, "effect", "s", "f", True),
            ScoredExecution("x", 0.2, "benign", "s", "f", False),
            ScoredExecution("x", 0.2, "effect", "s", "f", True, "missing"),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evaluate_review_budget([good, bad], threshold=0.5)


if __name__ == "__main__":
    unittest.main()
