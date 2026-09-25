# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic health policy for the proposed 24-hour hardware campaign.

This module validates one JSON object per JSONL line and decides policy from
only that record plus immutable caller-owned state.  It performs no capture,
training, provisioning, cloud access, checkpoint I/O, or background work.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
import re
from typing import ClassVar, Mapping


SCHEMA_VERSION = 1
CADENCES = (60, 300, 900, 1800, 7200)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_STATUSES = frozenset(("running", "recovering", "aborted", "completed"))

COMMON_FIELDS = frozenset(
    (
        "schema_version",
        "run_id",
        "subject_sha256",
        "capture_sha256",
        "lineage_id",
        "timestamp_utc",
        "elapsed_seconds",
        "cadence_seconds",
        "status",
        "reasons",
    )
)

CADENCE_FIELDS = {
    60: frozenset(
        (
            "liveness",
            "boot_sha256",
            "disk_free_bytes",
            "ram_available_bytes",
            "hbm_free_bytes",
            "sealed_bytes",
            "uploaded_bytes",
            "backlog_bytes",
            "oldest_unverified_shard_age_seconds",
            "gpu_utilization_fraction",
            "data_wait_fraction",
            "disk_runway_seconds",
            "shard_hash_ok",
            "spend_usd",
            "spend_forecast_usd",
            "spot_interruption_active",
            "spot_restore_age_seconds",
        )
    ),
    300: frozenset(
        (
            "executions",
            "rejections",
            "pt_bytes",
            "pt_lost_packets",
            "aux_bad",
            "pebs_samples",
            "pebs_lost_samples",
            "pebs_density_ratio",
            "pebs_running_ratio",
            "pmu_running_ratio",
            "cpu_migrations",
            "oracle_failures",
        )
    ),
    900: frozenset(
        (
            "global_step",
            "unique_executions",
            "mask_views_used",
            "replay_ratio",
            "total_loss",
            "pt_loss",
            "pebs_loss",
            "pmu_loss",
            "gradient_norm",
            "parameter_norm",
            "nonfinite_count",
            "tokens_per_second",
            "remote_checkpoint_age_seconds",
            "checkpoint_sha256",
            "checkpoint_hash_ok",
            "remote_checkpoint_verified",
            "oom_events",
            "fresh_data_available",
        )
    ),
    1800: frozenset(
        (
            "validation_total_loss",
            "validation_pt_loss",
            "validation_pebs_loss",
            "validation_pmu_loss",
            "retrieval_recall",
            "swap_residual",
            "family_alert_rates",
            "input_drift_within_bounds",
        )
    ),
    7200: frozenset(
        (
            "capacity_projection_seconds",
            "familiar_benign_alert_rate",
            "unfamiliar_benign_alert_rate",
            "development_canary_score",
            "raw_storage_forecast_bytes",
            "storage_forecast_bytes",
            "spend_forecast_usd",
        )
    ),
}

_BOOLEAN_FIELDS = frozenset(
    (
        "liveness",
        "shard_hash_ok",
        "spot_interruption_active",
        "aux_bad",
        "checkpoint_hash_ok",
        "remote_checkpoint_verified",
        "fresh_data_available",
        "input_drift_within_bounds",
    )
)
_INTEGER_FIELDS = frozenset(
    (
        "schema_version",
        "cadence_seconds",
        "disk_free_bytes",
        "ram_available_bytes",
        "hbm_free_bytes",
        "sealed_bytes",
        "uploaded_bytes",
        "backlog_bytes",
        "executions",
        "rejections",
        "pt_bytes",
        "pt_lost_packets",
        "pebs_samples",
        "pebs_lost_samples",
        "cpu_migrations",
        "oracle_failures",
        "global_step",
        "unique_executions",
        "mask_views_used",
        "nonfinite_count",
        "oom_events",
        "raw_storage_forecast_bytes",
        "storage_forecast_bytes",
    )
)
_FRACTION_FIELDS = frozenset(
    (
        "gpu_utilization_fraction",
        "data_wait_fraction",
        "pebs_running_ratio",
        "pmu_running_ratio",
        "replay_ratio",
        "retrieval_recall",
        "familiar_benign_alert_rate",
        "unfamiliar_benign_alert_rate",
    )
)
_NONNEGATIVE_NUMBER_FIELDS = frozenset().union(
    _INTEGER_FIELDS - {"schema_version", "cadence_seconds"},
    (
        "elapsed_seconds",
        "oldest_unverified_shard_age_seconds",
        "disk_runway_seconds",
        "spend_usd",
        "spend_forecast_usd",
        "spot_restore_age_seconds",
        "pebs_density_ratio",
        "pebs_running_ratio",
        "pmu_running_ratio",
        "total_loss",
        "pt_loss",
        "pebs_loss",
        "pmu_loss",
        "gradient_norm",
        "parameter_norm",
        "tokens_per_second",
        "remote_checkpoint_age_seconds",
        "validation_total_loss",
        "validation_pt_loss",
        "validation_pebs_loss",
        "validation_pmu_loss",
        "retrieval_recall",
        "swap_residual",
        "capacity_projection_seconds",
        "familiar_benign_alert_rate",
        "unfamiliar_benign_alert_rate",
        "development_canary_score",
    ),
)


class HealthRecordError(ValueError):
    """A schema-v1 record is incomplete or ambiguous."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class HealthPolicy:
    """Preregistered bounds that are not learned from campaign observations."""

    approved_spend_ceiling_usd: float
    starved_gpu_utilization_fraction: float = 0.5
    starved_data_wait_fraction: float = 0.2
    maximum_elapsed_seconds: ClassVar[float] = 26 * 60 * 60
    minimum_disk_runway_seconds: ClassVar[float] = 2 * 60 * 60
    maximum_remote_checkpoint_age_seconds: ClassVar[float] = 30 * 60
    minimum_running_ratio: ClassVar[float] = 1.0
    minimum_pebs_density_ratio: ClassVar[float] = 0.5
    low_density_windows_to_abort: ClassVar[int] = 2
    maximum_loader_workers: ClassVar[int] = 8
    maximum_prefetch_factor: ClassVar[int] = 8
    loader_adjustment_interval_seconds: ClassVar[float] = 15 * 60
    maximum_mask_views: ClassVar[int] = 2
    maximum_spot_restore_seconds: ClassVar[float] = 30 * 60

    def __post_init__(self) -> None:
        if (
            not _is_number(self.approved_spend_ceiling_usd)
            or self.approved_spend_ceiling_usd <= 0
        ):
            raise ValueError("approved spend ceiling must be a finite positive number")
        for value in (
            self.starved_gpu_utilization_fraction,
            self.starved_data_wait_fraction,
        ):
            if not _is_number(value) or not 0 <= value <= 1:
                raise ValueError("starvation thresholds must be in [0, 1]")


@dataclass(frozen=True)
class HealthState:
    """The minimal durable controller state required for bounded actions."""

    subject_sha256: str
    boot_sha256: str
    capture_sha256: str
    lineage_id: str
    loader_workers: int
    prefetch_factor: int
    microbatch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    mask_views_used: int = 1
    oom_recoveries: int = 0
    numerical_recoveries: int = 0
    consecutive_low_pebs_density_windows: int = 0
    last_loader_adjustment_elapsed_seconds: float | None = None
    latest_verified_checkpoint_sha256: str | None = None
    spot_restore_requested: bool = False

    def __post_init__(self) -> None:
        for name in ("subject_sha256", "boot_sha256", "capture_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.lineage_id:
            raise ValueError("lineage_id must not be empty")
        if (
            min(
                self.loader_workers,
                self.prefetch_factor,
                self.microbatch_size,
                self.gradient_accumulation_steps,
                self.mask_views_used,
            )
            < 1
        ):
            raise ValueError("state sizes and counts must be positive")
        if (
            self.loader_workers > 8
            or self.prefetch_factor > 8
            or self.mask_views_used > 2
        ):
            raise ValueError(
                "state exceeds loader, prefetch, or mask-view campaign bounds"
            )
        if self.oom_recoveries not in (0, 1) or self.numerical_recoveries not in (0, 1):
            raise ValueError(
                "recovery counters may only represent zero or one recovery"
            )
        if self.consecutive_low_pebs_density_windows < 0:
            raise ValueError("low-density window count must not be negative")
        if self.last_loader_adjustment_elapsed_seconds is not None and (
            not _is_number(self.last_loader_adjustment_elapsed_seconds)
            or self.last_loader_adjustment_elapsed_seconds < 0
        ):
            raise ValueError(
                "last loader adjustment time must be finite and nonnegative"
            )
        if not _is_number(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.latest_verified_checkpoint_sha256 is not None and not _is_sha256(
            self.latest_verified_checkpoint_sha256
        ):
            raise ValueError("latest verified checkpoint must be a SHA-256 digest")


@dataclass(frozen=True)
class PolicyAction:
    """One caller-executed action with explicit lineage and rationale."""

    kind: str
    reason: str
    prior_lineage_id: str
    lineage_id: str
    parameters: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class HealthDecision:
    """Pure policy output.  An abort decision never contains actions."""

    status: str
    reasons: tuple[str, ...]
    actions: tuple[PolicyAction, ...]
    next_state: HealthState


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0


def validate_health_record(record: Mapping[str, object]) -> None:
    """Validate one decoded JSONL object against schema version 1.

    All failures are reported together through :class:`HealthRecordError`.
    Unknown fields are retained for forward-compatible producer metadata, but
    the schema version itself must match exactly.
    """
    errors: list[str] = []
    cadence = record.get("cadence_seconds")
    if not _is_integer(cadence) or cadence not in CADENCES:
        errors.append("cadence_seconds must be one of 60, 300, 900, 1800, or 7200")
        required = COMMON_FIELDS
    else:
        required = COMMON_FIELDS | CADENCE_FIELDS[cadence]
    missing = sorted(required - record.keys())
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))

    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must equal {SCHEMA_VERSION}")
    for name in ("run_id", "lineage_id"):
        value = record.get(name)
        if not isinstance(value, str) or not value:
            errors.append(f"{name} must be a nonempty string")
    for name in ("subject_sha256", "capture_sha256"):
        if not _is_sha256(record.get(name)):
            errors.append(f"{name} must be a lowercase SHA-256 digest")
    if cadence == 60 and not _is_sha256(record.get("boot_sha256")):
        errors.append("boot_sha256 must be a lowercase SHA-256 digest")
    if cadence == 900 and not _is_sha256(record.get("checkpoint_sha256")):
        errors.append("checkpoint_sha256 must be a lowercase SHA-256 digest")
    if not _is_utc_timestamp(record.get("timestamp_utc")):
        errors.append("timestamp_utc must be an RFC 3339 UTC timestamp ending in Z")
    if record.get("status") not in _STATUSES:
        errors.append("status must be running, recovering, aborted, or completed")
    reasons = record.get("reasons")
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) for reason in reasons
    ):
        errors.append("reasons must be a JSON array of strings")

    for name in sorted(required & _BOOLEAN_FIELDS):
        if not isinstance(record.get(name), bool):
            errors.append(f"{name} must be a boolean")
    for name in sorted(required & _INTEGER_FIELDS):
        if not _is_integer(record.get(name)):
            errors.append(f"{name} must be an integer")
    for name in sorted(required & _NONNEGATIVE_NUMBER_FIELDS):
        value = record.get(name)
        if not _is_number(value) or value < 0:
            errors.append(f"{name} must be a finite nonnegative number")
    for name in sorted(required & _FRACTION_FIELDS):
        value = record.get(name)
        if not _is_number(value) or not 0 <= value <= 1:
            errors.append(f"{name} must be in [0, 1]")
    family_alert_rates = record.get("family_alert_rates")
    if cadence == 1800 and (
        not isinstance(family_alert_rates, dict)
        or not family_alert_rates
        or any(
            not isinstance(name, str)
            or not name
            or not _is_number(value)
            or not 0 <= value <= 1
            for name, value in family_alert_rates.items()
        )
    ):
        errors.append("family_alert_rates must be a nonempty object of [0, 1] values")
    if errors:
        raise HealthRecordError(tuple(errors))


def parse_health_record_jsonl(line: str) -> dict[str, object]:
    """Decode and validate exactly one nonblank JSONL record."""
    if not isinstance(line, str) or not line.strip() or "\n" in line.rstrip("\r\n"):
        raise HealthRecordError(("expected exactly one nonblank JSONL record",))
    try:
        decoded = json.loads(line)
    except (json.JSONDecodeError, TypeError) as error:
        raise HealthRecordError(("invalid JSON",)) from error
    if not isinstance(decoded, dict):
        raise HealthRecordError(("health record must be a JSON object",))
    validate_health_record(decoded)
    return decoded


def _lineage_action(
    state: HealthState,
    *,
    kind: str,
    reason: str,
    suffix: str,
    parameters: tuple[tuple[str, object], ...],
    **state_changes: object,
) -> tuple[PolicyAction, HealthState]:
    lineage_id = f"{state.lineage_id}/{suffix}"
    action = PolicyAction(
        kind=kind,
        reason=reason,
        prior_lineage_id=state.lineage_id,
        lineage_id=lineage_id,
        parameters=parameters,
    )
    return action, replace(state, lineage_id=lineage_id, **state_changes)


def decide_health(
    record: Mapping[str, object],
    state: HealthState,
    policy: HealthPolicy,
) -> HealthDecision:
    """Apply hard-abort gates, then at most one bounded automatic action.

    Precedence is fixed: all applicable hard aborts are returned in policy order
    and suppress every action.  Otherwise Spot restore, numerical recovery, OOM
    recovery, replay, and loader tuning are considered in that order.
    """
    validate_health_record(record)
    cadence = int(record["cadence_seconds"])
    elapsed = float(record["elapsed_seconds"])
    next_state = state
    aborts: list[str] = []

    # Identity and integrity precede financial/time gates and every recovery.
    if record["subject_sha256"] != state.subject_sha256:
        aborts.append("subject_hash_changed")
    if record["capture_sha256"] != state.capture_sha256:
        aborts.append("capture_hash_changed")
    if record["lineage_id"] != state.lineage_id:
        aborts.append("unexpected_lineage")
    if cadence == 60 and record["boot_sha256"] != state.boot_sha256:
        aborts.append("boot_hash_changed")
    if cadence == 60 and not record["liveness"]:
        aborts.append("liveness_failed")
    if cadence == 60 and not record["shard_hash_ok"]:
        aborts.append("corrupt_shard_hash")
    if cadence == 900 and not record["checkpoint_hash_ok"]:
        aborts.append("corrupt_checkpoint_hash")
    if cadence == 900 and not record["remote_checkpoint_verified"]:
        aborts.append("remote_checkpoint_unverified")

    if (
        cadence in (60, 7200)
        and record["spend_forecast_usd"] > policy.approved_spend_ceiling_usd
    ):
        aborts.append("spend_forecast_exceeds_ceiling")
    if elapsed >= policy.maximum_elapsed_seconds:
        aborts.append("elapsed_time_reached_26_hours")
    if (
        cadence == 60
        and record["disk_runway_seconds"] < policy.minimum_disk_runway_seconds
    ):
        aborts.append("disk_runway_below_two_hours")
    if (
        cadence == 900
        and record["remote_checkpoint_age_seconds"]
        > policy.maximum_remote_checkpoint_age_seconds
    ):
        aborts.append("remote_checkpoint_older_than_30_minutes")

    if cadence == 300:
        if record["oracle_failures"]:
            aborts.append("oracle_failure")
        if record["cpu_migrations"]:
            aborts.append("cpu_migration")
        if record["pt_lost_packets"]:
            aborts.append("pt_loss")
        if record["pebs_lost_samples"]:
            aborts.append("pebs_loss")
        if record["aux_bad"]:
            aborts.append("aux_bad")
        if record["pebs_running_ratio"] < policy.minimum_running_ratio:
            aborts.append("pebs_running_ratio_below_one")
        if record["pmu_running_ratio"] < policy.minimum_running_ratio:
            aborts.append("pmu_running_ratio_below_one")
        low_density = record["pebs_density_ratio"] < policy.minimum_pebs_density_ratio
        windows = state.consecutive_low_pebs_density_windows + 1 if low_density else 0
        next_state = replace(next_state, consecutive_low_pebs_density_windows=windows)
        if windows >= policy.low_density_windows_to_abort:
            aborts.append("pebs_density_below_half_for_two_windows")

    if cadence == 900:
        if record["mask_views_used"] > policy.maximum_mask_views:
            aborts.append("mask_view_cap_exceeded")
        if record["nonfinite_count"] and state.numerical_recoveries:
            aborts.append("second_numerical_recovery")
        if record["oom_events"] and state.oom_recoveries:
            aborts.append("second_oom_recovery")
        if record["oom_events"] and state.microbatch_size % 2:
            aborts.append("oom_recovery_cannot_preserve_effective_batch")
        if record["checkpoint_hash_ok"] and record["remote_checkpoint_verified"]:
            next_state = replace(
                next_state,
                latest_verified_checkpoint_sha256=record["checkpoint_sha256"],
            )

    if cadence == 1800 and not record["input_drift_within_bounds"]:
        aborts.append("input_drift_outside_preregistered_bounds")

    if cadence == 60 and record["spot_interruption_active"]:
        if record["spot_restore_age_seconds"] > policy.maximum_spot_restore_seconds:
            aborts.append("spot_restore_exceeded_30_minutes")
        if state.latest_verified_checkpoint_sha256 is None:
            aborts.append("spot_restore_has_no_verified_checkpoint")
    elif cadence == 60 and state.spot_restore_requested:
        next_state = replace(next_state, spot_restore_requested=False)

    if aborts:
        return HealthDecision("abort", tuple(aborts), (), next_state)

    # One action per record keeps recovery precedence and lineage unambiguous.
    if (
        cadence == 60
        and record["spot_interruption_active"]
        and not state.spot_restore_requested
    ):
        action = PolicyAction(
            kind="restore_spot_trainer",
            reason="spot_interruption_active",
            prior_lineage_id=state.lineage_id,
            lineage_id=state.lineage_id,
            parameters=(
                ("checkpoint_sha256", state.latest_verified_checkpoint_sha256),
            ),
        )
        return HealthDecision(
            "adjust",
            (action.reason,),
            (action,),
            replace(next_state, spot_restore_requested=True),
        )

    if cadence == 900 and record["nonfinite_count"]:
        checkpoint = str(record["checkpoint_sha256"])
        action, next_state = _lineage_action(
            next_state,
            kind="restore_checkpoint_and_halve_learning_rate",
            reason="first_nan_or_inf",
            suffix="numerical-recovery-1",
            parameters=(
                ("checkpoint_sha256", checkpoint),
                ("learning_rate", state.learning_rate / 2),
            ),
            learning_rate=state.learning_rate / 2,
            numerical_recoveries=1,
        )
        return HealthDecision("adjust", (action.reason,), (action,), next_state)

    if cadence == 900 and record["oom_events"]:
        action, next_state = _lineage_action(
            next_state,
            kind="reduce_microbatch_preserve_effective_batch",
            reason="first_out_of_memory",
            suffix="oom-recovery-1",
            parameters=(
                ("microbatch_size", state.microbatch_size // 2),
                ("gradient_accumulation_steps", state.gradient_accumulation_steps * 2),
                ("effective_batch_unchanged", True),
            ),
            microbatch_size=state.microbatch_size // 2,
            gradient_accumulation_steps=state.gradient_accumulation_steps * 2,
            oom_recoveries=1,
        )
        return HealthDecision("adjust", (action.reason,), (action,), next_state)

    if cadence == 900 and not record["fresh_data_available"]:
        observed_views = max(state.mask_views_used, int(record["mask_views_used"]))
        if observed_views < policy.maximum_mask_views:
            action = PolicyAction(
                kind="reuse_training_shards_with_new_masks",
                reason="fresh_data_temporarily_unavailable",
                prior_lineage_id=state.lineage_id,
                lineage_id=state.lineage_id,
                parameters=(
                    ("mask_views_used", observed_views + 1),
                    ("preserve_sample_order", True),
                ),
            )
            return HealthDecision(
                "adjust",
                (action.reason,),
                (action,),
                replace(next_state, mask_views_used=observed_views + 1),
            )
        action = PolicyAction(
            kind="hold_for_fresh_data",
            reason="two_mask_view_cap_reached",
            prior_lineage_id=state.lineage_id,
            lineage_id=state.lineage_id,
        )
        return HealthDecision("adjust", (action.reason,), (action,), next_state)

    if cadence == 60:
        starved = (
            record["backlog_bytes"] > 0
            and record["gpu_utilization_fraction"]
            < policy.starved_gpu_utilization_fraction
            and record["data_wait_fraction"] >= policy.starved_data_wait_fraction
        )
        previous = state.last_loader_adjustment_elapsed_seconds
        cooled_down = (
            previous is None
            or elapsed - previous >= policy.loader_adjustment_interval_seconds
        )
        loader_workers = min(state.loader_workers + 1, policy.maximum_loader_workers)
        prefetch_factor = min(state.prefetch_factor * 2, policy.maximum_prefetch_factor)
        if (
            starved
            and cooled_down
            and (
                loader_workers != state.loader_workers
                or prefetch_factor != state.prefetch_factor
            )
        ):
            action = PolicyAction(
                kind="increase_loader_capacity",
                reason="gpu_starved_with_sealed_backlog",
                prior_lineage_id=state.lineage_id,
                lineage_id=state.lineage_id,
                parameters=(
                    ("loader_workers", loader_workers),
                    ("prefetch_factor", prefetch_factor),
                    ("preserve_sample_order", True),
                ),
            )
            next_state = replace(
                next_state,
                loader_workers=loader_workers,
                prefetch_factor=prefetch_factor,
                last_loader_adjustment_elapsed_seconds=elapsed,
            )
            return HealthDecision("adjust", (action.reason,), (action,), next_state)

    return HealthDecision("continue", (), (), next_state)


__all__ = [
    "CADENCES",
    "CADENCE_FIELDS",
    "COMMON_FIELDS",
    "SCHEMA_VERSION",
    "HealthDecision",
    "HealthPolicy",
    "HealthRecordError",
    "HealthState",
    "PolicyAction",
    "decide_health",
    "parse_health_record_jsonl",
    "validate_health_record",
]
