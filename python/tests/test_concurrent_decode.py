# SPDX-License-Identifier: AGPL-3.0-only
"""Independent context-only Pools decode concurrently without semantic loss."""

import unittest

from concurrent_decode import expected_digest, replay


class ConcurrentDecodeTests(unittest.TestCase):
    def test_one_four_and_sixteen_independent_pools(self) -> None:
        for worker_count in (1, 4, 16):
            with self.subTest(worker_count=worker_count):
                results, errors, elapsed = replay(worker_count, 8)
                self.assertLess(elapsed, 15)
                self.assertEqual(errors, [])
                self.assertEqual(
                    sorted(results, key=lambda result: result.identity),
                    [expected_digest(identity, 8) for identity in range(worker_count)],
                )

    def test_incomplete_stream_does_not_hide_other_results(self) -> None:
        results, errors, elapsed = replay(4, 4, incomplete=2)
        self.assertLess(elapsed, 15)
        self.assertEqual(
            sorted(results, key=lambda result: result.identity),
            [expected_digest(identity, 4) for identity in (0, 1, 3)],
        )
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("Incomplete trace", str(errors[0]))


if __name__ == "__main__":
    unittest.main()
