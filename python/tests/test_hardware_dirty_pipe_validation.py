# SPDX-License-Identifier: AGPL-3.0-only
"""Dirty Pipe validation keeps the effect oracle outside model scoring."""

import unittest

from cpu2tensor.examples.hardware_dirty_pipe_validation import _auc, _parse_output


class HardwareDirtyPipeValidationTests(unittest.TestCase):
    def test_auc_counts_wins_and_ties(self) -> None:
        self.assertEqual(_auc([2.0, 3.0], [1.0, 2.0]), 0.875)

    def test_output_requires_the_safe_expected_mutation(self) -> None:
        self.assertEqual(_parse_output(b"mode=effect loops=4 mutations=4\n", "effect", 4), 4)
        self.assertEqual(_parse_output(b"mode=neutral loops=4 mutations=0\n", "neutral", 4), 0)
        with self.assertRaisesRegex(RuntimeError, "expected 4"):
            _parse_output(b"mode=effect loops=4 mutations=0\n", "effect", 4)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            _parse_output(b"mode=neutral loops=3 mutations=0\n", "neutral", 4)


if __name__ == "__main__":
    unittest.main()
