# SPDX-License-Identifier: AGPL-3.0-only
"""Anomaly bundles retain replay, localization, and honest symbol evidence."""

from pathlib import Path
import tempfile
import unittest

import torch

from cpu2tensor.examples.hardware_anomaly_bundle import (
    KernelSymbolTable,
    _pebs_samples,
    _pt_windows,
    decode_perf_mem_data_source,
)


class HardwareAnomalyBundleTests(unittest.TestCase):
    def test_perf_memory_data_source_is_semantic_and_lossless(self) -> None:
        value = (
            (1 << 1)
            | (1 << (5 + 1))
            | (3 << 33)
            | (1 << 37)
            | (2 << 43)
            | (1 << 60)
        )

        decoded = decode_perf_mem_data_source(value)

        self.assertEqual(decoded["operation"], ["load"])
        self.assertEqual(decoded["level"], ["hit"])
        self.assertEqual(decoded["level_number"], {"raw": 3, "name": "l3"})
        self.assertTrue(decoded["remote"])
        self.assertEqual(decoded["hops"], {"raw": 2, "name": "same_socket"})
        self.assertEqual(decoded["unknown_bits"], "0x1000000000000000")

    def test_symbol_table_rejects_redaction_and_resolves_exact_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kallsyms"
            path.write_text("0000000000001000 T first\n0000000000002000 t second [mod]\n")
            symbols = KernelSymbolTable.load(path)
            redacted = Path(directory) / "redacted"
            redacted.write_text("0000000000000000 T hidden\n")

            self.assertEqual(symbols.resolve(0x1005), {
                "name": "first", "kind": "T", "module": None, "offset": 5,
            })
            self.assertEqual(symbols.resolve(0x2007), {
                "name": "second", "kind": "t", "module": "mod", "offset": 7,
            })
            with self.assertRaisesRegex(ValueError, "no visible addresses"):
                KernelSymbolTable.load(redacted)

    def test_localization_maps_model_tokens_back_to_raw_windows(self) -> None:
        raw = {
            "batches": [{
                "tid": 7,
                "envelope": {
                    "arm_before_ns": 100,
                    "arm_after_ns": 110,
                    "stop_before_ns": 1_090,
                    "stop_after_ns": 1_100,
                },
                "pt": {"trace_bytes": torch.arange(32, dtype=torch.uint8)},
                "pebs": {
                    "time": torch.tensor([150, 1_050]),
                    "ip": torch.tensor([0x1005, 0x2007]),
                    "address": torch.tensor([0x3000, 0x4000]),
                    "cpu": torch.tensor([2, 2]),
                    "weight": torch.tensor([4, 8]),
                    "period": torch.tensor([10_000, 10_000]),
                    "exact_ip": torch.tensor([True, True]),
                    "data_source": torch.tensor([0x42, 0x42]),
                },
            }],
        }
        token_error = torch.zeros((1, 16))
        token_error[0, 15] = 3.0
        token_error[0, 0] = 2.0
        feature_error = torch.zeros((1, 16, 256))
        feature_error[0, 15, 9] = 7.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kallsyms"
            path.write_text("0000000000001000 T first\n0000000000002000 T second\n")
            symbols = KernelSymbolTable.load(path)

            timing_error = torch.arange(16, dtype=torch.float32).view(1, 16)
            pt = _pt_windows(raw, token_error, feature_error, timing_error, 2)
            pebs = _pebs_samples(raw, token_error, timing_error, symbols, 2)

        self.assertEqual((pt[0]["segment"], pt[0]["raw_byte_start"],
                          pt[0]["raw_byte_stop"]), (15, 30, 32))
        self.assertEqual(pt[0]["top_byte_residuals"][0], {
            "feature": "0x09", "residual": 7.0,
        })
        self.assertEqual(pt[0]["timing_residual"], 15.0)
        self.assertEqual(pebs[0]["segment"], 15)
        self.assertEqual(pebs[0]["ip_symbol"]["name"], "second")
        self.assertEqual(pebs[0]["timing_residual"], 15.0)
        self.assertEqual(pebs[1]["segment"], 0)
        self.assertEqual(pebs[1]["ip_symbol"]["offset"], 5)


if __name__ == "__main__":
    unittest.main()
