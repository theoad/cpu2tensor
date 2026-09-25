# SPDX-License-Identifier: AGPL-3.0-only
"""The 24-hour pretraining health policy is finite and deterministic."""

import json
import unittest

from cpu2tensor.examples.hardware_pretraining_health import (
    CADENCE_FIELDS,
    COMMON_FIELDS,
    SCHEMA_VERSION,
    HealthPolicy,
    HealthRecordError,
    HealthState,
    decide_health,
    parse_health_record_jsonl,
    validate_health_record,
)


SUBJECT = "1" * 64
BOOT = "2" * 64
CAPTURE = "3" * 64
CHECKPOINT = "4" * 64


def _record(cadence: int) -> dict[str, object]:
    common: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "campaign-1",
        "subject_sha256": SUBJECT,
        "capture_sha256": CAPTURE,
        "lineage_id": "lineage-0",
        "timestamp_utc": "2026-09-25T12:00:00Z",
        "elapsed_seconds": 3_600.0,
        "cadence_seconds": cadence,
        "status": "running",
        "reasons": [],
    }
    cadence_values: dict[int, dict[str, object]] = {
        60: {
            "liveness": True,
            "boot_sha256": BOOT,
            "disk_free_bytes": 2_000_000,
            "ram_available_bytes": 1_000_000,
            "hbm_free_bytes": 500_000,
            "sealed_bytes": 1_000,
            "uploaded_bytes": 900,
            "backlog_bytes": 100,
            "oldest_unverified_shard_age_seconds": 10.0,
            "gpu_utilization_fraction": 0.9,
            "data_wait_fraction": 0.01,
            "disk_runway_seconds": 10_800.0,
            "shard_hash_ok": True,
            "spend_usd": 20.0,
            "spend_forecast_usd": 200.0,
            "spot_interruption_active": False,
            "spot_restore_age_seconds": 0.0,
        },
        300: {
            "executions": 100,
            "rejections": 2,
            "pt_bytes": 10_000,
            "pt_lost_packets": 0,
            "aux_bad": False,
            "pebs_samples": 20,
            "pebs_lost_samples": 0,
            "pebs_density_ratio": 1.0,
            "pebs_running_ratio": 1.0,
            "pmu_running_ratio": 1.0,
            "cpu_migrations": 0,
            "oracle_failures": 0,
        },
        900: {
            "global_step": 100,
            "unique_executions": 1_000,
            "mask_views_used": 1,
            "replay_ratio": 0.0,
            "total_loss": 1.0,
            "pt_loss": 0.4,
            "pebs_loss": 0.3,
            "pmu_loss": 0.3,
            "gradient_norm": 1.0,
            "parameter_norm": 10.0,
            "nonfinite_count": 0,
            "tokens_per_second": 1_000.0,
            "remote_checkpoint_age_seconds": 60.0,
            "checkpoint_sha256": CHECKPOINT,
            "checkpoint_hash_ok": True,
            "remote_checkpoint_verified": True,
            "oom_events": 0,
            "fresh_data_available": True,
        },
        1800: {
            "validation_total_loss": 1.0,
            "validation_pt_loss": 0.4,
            "validation_pebs_loss": 0.3,
            "validation_pmu_loss": 0.3,
            "retrieval_recall": 0.8,
            "swap_residual": 0.5,
            "family_alert_rates": {"read": 0.001, "write": 0.002},
            "input_drift_within_bounds": True,
        },
        7200: {
            "capacity_projection_seconds": 50_000.0,
            "familiar_benign_alert_rate": 0.001,
            "unfamiliar_benign_alert_rate": 0.002,
            "development_canary_score": 1.2,
            "raw_storage_forecast_bytes": 1_000_000,
            "storage_forecast_bytes": 2_000_000,
            "spend_forecast_usd": 200.0,
        },
    }
    common.update(cadence_values[cadence])
    return common


def _state(**changes: object) -> HealthState:
    values: dict[str, object] = {
        "subject_sha256": SUBJECT,
        "boot_sha256": BOOT,
        "capture_sha256": CAPTURE,
        "lineage_id": "lineage-0",
        "loader_workers": 2,
        "prefetch_factor": 2,
        "microbatch_size": 16,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-3,
    }
    values.update(changes)
    return HealthState(**values)  # type: ignore[arg-type]


POLICY = HealthPolicy(approved_spend_ceiling_usd=250.0)


class HealthRecordValidationTests(unittest.TestCase):
    def test_every_cadence_round_trips_one_jsonl_record(self) -> None:
        for cadence in CADENCE_FIELDS:
            with self.subTest(cadence=cadence):
                record = _record(cadence)
                validate_health_record(record)
                self.assertEqual(parse_health_record_jsonl(json.dumps(record)), record)

    def test_every_common_and_cadence_field_is_required(self) -> None:
        for cadence, cadence_fields in CADENCE_FIELDS.items():
            for field in COMMON_FIELDS | cadence_fields:
                with self.subTest(cadence=cadence, field=field):
                    record = _record(cadence)
                    del record[field]
                    with self.assertRaises(HealthRecordError) as caught:
                        validate_health_record(record)
                    self.assertTrue(
                        any(
                            "missing required fields" in error
                            for error in caught.exception.errors
                        )
                    )

    def test_version_types_ranges_and_one_line_contract_are_enforced(self) -> None:
        invalid = _record(60)
        invalid.update(
            {
                "schema_version": 2,
                "subject_sha256": "not-a-hash",
                "gpu_utilization_fraction": 1.1,
                "disk_free_bytes": -1,
                "timestamp_utc": "2026-09-25T12:00:00+03:00",
            }
        )
        with self.assertRaises(HealthRecordError) as caught:
            validate_health_record(invalid)
        self.assertGreaterEqual(len(caught.exception.errors), 5)
        with self.assertRaises(HealthRecordError):
            parse_health_record_jsonl("{}\n{}")
        with self.assertRaises(HealthRecordError):
            parse_health_record_jsonl("[]")


class HealthAbortTests(unittest.TestCase):
    def assert_abort(
        self,
        cadence: int,
        reason: str,
        changes: dict[str, object],
        *,
        state: HealthState | None = None,
    ) -> None:
        record = _record(cadence)
        record.update(changes)
        decision = decide_health(record, state or _state(), POLICY)
        self.assertEqual(decision.status, "abort")
        self.assertIn(reason, decision.reasons)
        self.assertEqual(decision.actions, ())

    def test_identity_integrity_resource_and_time_aborts(self) -> None:
        cases = (
            (60, "subject_hash_changed", {"subject_sha256": "a" * 64}, None),
            (60, "capture_hash_changed", {"capture_sha256": "a" * 64}, None),
            (60, "unexpected_lineage", {"lineage_id": "unrecorded"}, None),
            (60, "boot_hash_changed", {"boot_sha256": "a" * 64}, None),
            (60, "liveness_failed", {"liveness": False}, None),
            (60, "corrupt_shard_hash", {"shard_hash_ok": False}, None),
            (900, "corrupt_checkpoint_hash", {"checkpoint_hash_ok": False}, None),
            (
                900,
                "remote_checkpoint_unverified",
                {"remote_checkpoint_verified": False},
                None,
            ),
            (
                60,
                "spend_forecast_exceeds_ceiling",
                {"spend_forecast_usd": 250.01},
                None,
            ),
            (
                7200,
                "spend_forecast_exceeds_ceiling",
                {"spend_forecast_usd": 251.0},
                None,
            ),
            (60, "elapsed_time_reached_26_hours", {"elapsed_seconds": 26 * 3600}, None),
            (60, "disk_runway_below_two_hours", {"disk_runway_seconds": 7199.0}, None),
            (
                900,
                "remote_checkpoint_older_than_30_minutes",
                {"remote_checkpoint_age_seconds": 1801.0},
                None,
            ),
            (
                1800,
                "input_drift_outside_preregistered_bounds",
                {"input_drift_within_bounds": False},
                None,
            ),
        )
        for cadence, reason, changes, state in cases:
            with self.subTest(reason=reason):
                self.assert_abort(cadence, reason, changes, state=state)

    def test_capture_contract_aborts(self) -> None:
        cases = (
            ("oracle_failure", {"oracle_failures": 1}),
            ("cpu_migration", {"cpu_migrations": 1}),
            ("pt_loss", {"pt_lost_packets": 1}),
            ("pebs_loss", {"pebs_lost_samples": 1}),
            ("aux_bad", {"aux_bad": True}),
            ("pebs_running_ratio_below_one", {"pebs_running_ratio": 0.999}),
            ("pmu_running_ratio_below_one", {"pmu_running_ratio": 0.999}),
        )
        for reason, changes in cases:
            with self.subTest(reason=reason):
                self.assert_abort(300, reason, changes)

    def test_density_aborts_only_after_two_consecutive_low_windows(self) -> None:
        low = _record(300)
        low["pebs_density_ratio"] = 0.49
        first = decide_health(low, _state(), POLICY)
        self.assertEqual(first.status, "continue")
        self.assertEqual(first.next_state.consecutive_low_pebs_density_windows, 1)
        second = decide_health(low, first.next_state, POLICY)
        self.assertEqual(second.status, "abort")
        self.assertIn("pebs_density_below_half_for_two_windows", second.reasons)

        healthy = _record(300)
        reset = decide_health(healthy, first.next_state, POLICY)
        self.assertEqual(reset.next_state.consecutive_low_pebs_density_windows, 0)

    def test_exhausted_recoveries_and_replay_cap_abort(self) -> None:
        self.assert_abort(
            900,
            "second_numerical_recovery",
            {"nonfinite_count": 1},
            state=_state(numerical_recoveries=1),
        )
        self.assert_abort(
            900,
            "second_oom_recovery",
            {"oom_events": 1},
            state=_state(oom_recoveries=1),
        )
        self.assert_abort(
            900,
            "oom_recovery_cannot_preserve_effective_batch",
            {"oom_events": 1},
            state=_state(microbatch_size=3),
        )
        self.assert_abort(900, "mask_view_cap_exceeded", {"mask_views_used": 3})

    def test_spot_timeout_and_missing_checkpoint_abort(self) -> None:
        self.assert_abort(
            60,
            "spot_restore_exceeded_30_minutes",
            {"spot_interruption_active": True, "spot_restore_age_seconds": 1801.0},
            state=_state(latest_verified_checkpoint_sha256=CHECKPOINT),
        )
        self.assert_abort(
            60,
            "spot_restore_has_no_verified_checkpoint",
            {"spot_interruption_active": True},
        )


class HealthActionTests(unittest.TestCase):
    def test_loader_and_prefetch_action_is_bounded_and_rate_limited(self) -> None:
        record = _record(60)
        record.update(
            {
                "gpu_utilization_fraction": 0.1,
                "data_wait_fraction": 0.8,
                "backlog_bytes": 1_000,
            }
        )
        first = decide_health(record, _state(), POLICY)
        self.assertEqual(first.status, "adjust")
        self.assertEqual(first.actions[0].kind, "increase_loader_capacity")
        self.assertEqual(
            dict(first.actions[0].parameters),
            {
                "loader_workers": 3,
                "prefetch_factor": 4,
                "preserve_sample_order": True,
            },
        )
        self.assertEqual(first.actions[0].lineage_id, "lineage-0")

        record["elapsed_seconds"] = 3_600.0 + 899
        limited = decide_health(record, first.next_state, POLICY)
        self.assertEqual(limited.status, "continue")
        record["elapsed_seconds"] = 3_600.0 + 900
        second = decide_health(record, first.next_state, POLICY)
        self.assertEqual(dict(second.actions[0].parameters)["loader_workers"], 4)

        capped = decide_health(
            record,
            _state(loader_workers=8, prefetch_factor=8),
            POLICY,
        )
        self.assertEqual(capped.status, "continue")

    def test_first_oom_halves_microbatch_and_preserves_effective_batch(self) -> None:
        record = _record(900)
        record["oom_events"] = 1
        decision = decide_health(record, _state(), POLICY)
        self.assertEqual(decision.status, "adjust")
        self.assertEqual(
            decision.actions[0].kind, "reduce_microbatch_preserve_effective_batch"
        )
        self.assertEqual(
            dict(decision.actions[0].parameters),
            {
                "microbatch_size": 8,
                "gradient_accumulation_steps": 8,
                "effective_batch_unchanged": True,
            },
        )
        self.assertEqual(decision.actions[0].reason, "first_out_of_memory")
        self.assertEqual(decision.actions[0].prior_lineage_id, "lineage-0")
        self.assertEqual(decision.next_state.lineage_id, "lineage-0/oom-recovery-1")

    def test_first_nonfinite_restores_checkpoint_and_halves_lr(self) -> None:
        record = _record(900)
        record["nonfinite_count"] = 1
        decision = decide_health(record, _state(), POLICY)
        self.assertEqual(decision.status, "adjust")
        self.assertEqual(
            decision.actions[0].kind,
            "restore_checkpoint_and_halve_learning_rate",
        )
        self.assertEqual(
            dict(decision.actions[0].parameters),
            {
                "checkpoint_sha256": CHECKPOINT,
                "learning_rate": 5e-4,
            },
        )
        self.assertEqual(
            decision.next_state.lineage_id, "lineage-0/numerical-recovery-1"
        )
        self.assertEqual(decision.next_state.numerical_recoveries, 1)

    def test_numerical_action_precedes_oom_action(self) -> None:
        record = _record(900)
        record.update({"nonfinite_count": 1, "oom_events": 1})
        decision = decide_health(record, _state(), POLICY)
        self.assertEqual(len(decision.actions), 1)
        self.assertEqual(
            decision.actions[0].kind,
            "restore_checkpoint_and_halve_learning_rate",
        )

    def test_replay_uses_second_view_then_holds_without_a_third(self) -> None:
        record = _record(900)
        record["fresh_data_available"] = False
        second_view = decide_health(record, _state(), POLICY)
        self.assertEqual(
            second_view.actions[0].kind, "reuse_training_shards_with_new_masks"
        )
        self.assertEqual(second_view.next_state.mask_views_used, 2)
        self.assertTrue(
            dict(second_view.actions[0].parameters)["preserve_sample_order"]
        )

        record["mask_views_used"] = 2
        held = decide_health(record, second_view.next_state, POLICY)
        self.assertEqual(held.actions[0].kind, "hold_for_fresh_data")
        self.assertEqual(held.next_state.mask_views_used, 2)

    def test_spot_restore_action_uses_verified_checkpoint_and_keeps_lineage(
        self,
    ) -> None:
        record = _record(60)
        record.update(
            {"spot_interruption_active": True, "spot_restore_age_seconds": 60.0}
        )
        state = _state(latest_verified_checkpoint_sha256=CHECKPOINT)
        decision = decide_health(record, state, POLICY)
        self.assertEqual(decision.actions[0].kind, "restore_spot_trainer")
        self.assertEqual(
            dict(decision.actions[0].parameters)["checkpoint_sha256"],
            CHECKPOINT,
        )
        self.assertEqual(decision.actions[0].lineage_id, "lineage-0")
        self.assertTrue(decision.next_state.spot_restore_requested)
        restored = decide_health(_record(60), decision.next_state, POLICY)
        self.assertFalse(restored.next_state.spot_restore_requested)

    def test_hard_abort_suppresses_recovery_actions(self) -> None:
        record = _record(900)
        record.update(
            {"checkpoint_hash_ok": False, "nonfinite_count": 1, "oom_events": 1}
        )
        decision = decide_health(record, _state(), POLICY)
        self.assertEqual(decision.status, "abort")
        self.assertIn("corrupt_checkpoint_hash", decision.reasons)
        self.assertEqual(decision.actions, ())
        self.assertEqual(decision.next_state.lineage_id, "lineage-0")

        spending = _record(60)
        spending.update(
            {
                "spend_forecast_usd": 251.0,
                "gpu_utilization_fraction": 0.1,
                "data_wait_fraction": 0.8,
            }
        )
        spend_decision = decide_health(spending, _state(), POLICY)
        self.assertEqual(spend_decision.status, "abort")
        self.assertEqual(spend_decision.reasons[0], "spend_forecast_exceeds_ceiling")
        self.assertEqual(spend_decision.actions, ())

    def test_exact_resource_boundaries_continue(self) -> None:
        minute = _record(60)
        minute.update(
            {
                "spend_forecast_usd": 250.0,
                "disk_runway_seconds": 2 * 3600,
                "elapsed_seconds": 26 * 3600 - 1,
            }
        )
        self.assertEqual(decide_health(minute, _state(), POLICY).status, "continue")

        checkpoint = _record(900)
        checkpoint["remote_checkpoint_age_seconds"] = 30 * 60
        self.assertEqual(decide_health(checkpoint, _state(), POLICY).status, "continue")

        spot = _record(60)
        spot.update(
            {"spot_interruption_active": True, "spot_restore_age_seconds": 30 * 60}
        )
        self.assertEqual(
            decide_health(
                spot,
                _state(latest_verified_checkpoint_sha256=CHECKPOINT),
                POLICY,
            ).status,
            "adjust",
        )

    def test_healthy_record_continues_and_refreshes_verified_checkpoint(self) -> None:
        decision = decide_health(_record(900), _state(), POLICY)
        self.assertEqual(decision.status, "continue")
        self.assertEqual(decision.actions, ())
        self.assertEqual(
            decision.next_state.latest_verified_checkpoint_sha256, CHECKPOINT
        )


if __name__ == "__main__":
    unittest.main()
