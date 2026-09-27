# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from cpu2tensor.examples.hardware_effect_calibration_pilot_r1 import (
    RAW_PER_COHORT,
    ROWS_PER_COHORT,
    validate_capture_quality,
)


def _manifest() -> dict[str, object]:
    entries = []
    for index in range(ROWS_PER_COHORT):
        entries.append({
            "raw_retained": index < RAW_PER_COHORT,
            "admission": {"attempt": 1},
            "capture": {
                "lost_sources": 0,
                "missing_sources": 0,
                "multiplexed_sources": 0,
            },
        })
    return {
        "entries": entries,
        "collection": {
            "executions": ROWS_PER_COHORT,
            "loss_count": 0,
            "admission": {"rejected_attempts": 0},
        },
    }


def test_calibration_quality_accepts_exact_lossless_cohort() -> None:
    validate_capture_quality(_manifest())


@pytest.mark.parametrize("mutation", ("retry", "loss", "raw"))
def test_calibration_quality_rejects_invalid_cohort(mutation: str) -> None:
    manifest = _manifest()
    if mutation == "retry":
        manifest["entries"][0]["admission"]["attempt"] = 2
    elif mutation == "loss":
        manifest["entries"][0]["capture"]["lost_sources"] = 1
    else:
        manifest["entries"][RAW_PER_COHORT]["raw_retained"] = True
    with pytest.raises(ValueError, match="calibration cohort"):
        validate_capture_quality(manifest)
