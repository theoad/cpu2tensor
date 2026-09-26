# SPDX-License-Identifier: AGPL-3.0-only
"""Offline review-budget metrics for frozen, fully observed execution scores."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence


@dataclass(frozen=True)
class ScoredExecution:
    execution_id: str
    score: float
    label: Literal["benign", "effect"]
    session_id: str
    family: str
    complete: bool
    pair_id: str | None = None
    cluster_id: str | None = None


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for step in range(1, 1001):
        twice = 2 * step
        numerator = step * (b - step) * x / ((qam + twice) * (a + twice))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        numerator = -(a + step) * (qab + step) * x / ((a + twice) * (qap + twice))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 3e-14:
            return result
    raise ArithmeticError("incomplete-beta fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b


def exact_binomial_upper_bound(alerts: int, trials: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on an execution-alert probability."""
    if trials <= 0 or alerts < 0 or alerts > trials:
        raise ValueError("require 0 <= alerts <= trials and trials > 0")
    if not 0.0 < confidence < 1.0 or not math.isfinite(confidence):
        raise ValueError("confidence must be finite and in (0, 1)")
    if alerts == trials:
        return 1.0
    if alerts == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / trials)

    # P(X > alerts | p) = I_p(alerts + 1, trials - alerts).
    # At the upper confidence bound this tail equals confidence.
    low, high = 0.0, 1.0
    for _ in range(70):
        middle = (low + high) / 2.0
        tail = _regularized_beta(middle, alerts + 1.0, trials - alerts)
        if tail < confidence:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def evaluate_review_budget(
    rows: Sequence[ScoredExecution], *, threshold: float, top_k: int = 100,
    confidence: float = 0.95, cluster_cap: int | None = None,
) -> dict[str, object]:
    """Evaluate raw admissions without using labels to select or tune a threshold.

    Every supplied row must be complete and finite. Censored rows must be kept
    in a separate capture-yield ledger, never passed here as non-alerts.
    """
    if not rows:
        raise ValueError("at least one execution is required")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if cluster_cap is not None and cluster_cap <= 0:
        raise ValueError("cluster_cap must be positive")

    seen_ids: set[str] = set()
    benign: list[ScoredExecution] = []
    effects: list[ScoredExecution] = []
    pairs: dict[str, dict[str, ScoredExecution]] = {}
    by_family: dict[str, list[ScoredExecution]] = {}
    by_session: dict[str, list[ScoredExecution]] = {}
    for row in rows:
        if not row.complete:
            raise ValueError(f"censored capture: {row.execution_id}")
        if not math.isfinite(row.score):
            raise ValueError(f"non-finite score: {row.execution_id}")
        if not row.execution_id or not row.session_id or not row.family:
            raise ValueError("execution_id, session_id, and family are required")
        if cluster_cap is not None and not row.cluster_id:
            raise ValueError(f"cluster_id required for capped ranking: {row.execution_id}")
        if row.execution_id in seen_ids:
            raise ValueError(f"duplicate execution_id: {row.execution_id}")
        seen_ids.add(row.execution_id)
        if row.label == "benign":
            benign.append(row)
            by_family.setdefault(row.family, []).append(row)
            by_session.setdefault(row.session_id, []).append(row)
        elif row.label == "effect":
            effects.append(row)
        else:
            raise ValueError(f"invalid label: {row.label}")
        if row.pair_id is not None:
            arm = pairs.setdefault(row.pair_id, {})
            if row.label in arm:
                raise ValueError(f"duplicate {row.label} in pair {row.pair_id}")
            arm[row.label] = row

    if not benign:
        raise ValueError("at least one benign execution is required")
    if any(set(arm) != {"benign", "effect"} for arm in pairs.values()):
        raise ValueError("every pair_id must have one benign and one effect row")
    if any(arm["benign"].session_id != arm["effect"].session_id for arm in pairs.values()):
        raise ValueError("paired rows must come from one session")

    def admission(group: Sequence[ScoredExecution]) -> dict[str, int]:
        alerts = sum(row.score > threshold for row in group)
        return {"executions": len(group), "alerts": alerts}

    def benign_summary(group: Sequence[ScoredExecution]) -> dict[str, int | float]:
        counts = admission(group)
        return {
            **counts,
            "alerts_per_million": counts["alerts"] * 1_000_000 / counts["executions"],
        }

    ranked = sorted(rows, key=lambda row: (-row.score, row.execution_id))
    selected = ranked[:top_k]
    benign_admission = benign_summary(benign)
    effect_admission = admission(effects)
    benign_alerts = benign_admission["alerts"]
    wins = sum(arm["effect"].score > arm["benign"].score for arm in pairs.values())
    ties = sum(arm["effect"].score == arm["benign"].score for arm in pairs.values())
    report: dict[str, object] = {
        "threshold": threshold,
        "top_k": top_k,
        "selected": len(selected),
        "selected_effects": sum(row.label == "effect" for row in selected),
        "raw_top_k_recall": (
            sum(row.label == "effect" for row in selected) / len(effects)
            if effects else None
        ),
        "benign": {
            **benign_admission,
            "upper_confidence": confidence,
            "upper_confidence_per_million": (
                exact_binomial_upper_bound(benign_alerts, len(benign), confidence)
                * 1_000_000
            ),
        },
        "effect": {
            **effect_admission,
            "recall_at_threshold": (
                effect_admission["alerts"] / len(effects) if effects else None
            ),
        },
        "benign_by_family": {
            family: benign_summary(group) for family, group in sorted(by_family.items())
        },
        "benign_by_session": {
            session: benign_summary(group) for session, group in sorted(by_session.items())
        },
        "paired": {
            "pairs": len(pairs),
            "wins": wins,
            "ties": ties,
            "effect_win_rate": wins / len(pairs) if pairs else None,
        },
    }
    if cluster_cap is not None:
        cluster_counts: dict[str, int] = {}
        capped: list[ScoredExecution] = []
        for row in ranked:
            cluster = row.cluster_id
            assert cluster is not None
            if cluster_counts.get(cluster, 0) >= cluster_cap:
                continue
            capped.append(row)
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            if len(capped) == top_k:
                break
        if top_k == 0:
            capped = []
        capped_effects = sum(row.label == "effect" for row in capped)
        report["cluster_capped_diagnostic"] = {
            "retrospective_only": True,
            "cluster_cap": cluster_cap,
            "total_budget": top_k,
            "selected": len(capped),
            "selected_ids": [row.execution_id for row in capped],
            "selected_effects": capped_effects,
            "raw_top_k_recall": capped_effects / len(effects) if effects else None,
        }
    return report
