# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded observation reductions keep exact totals without raw block rows."""

from collections import Counter
import struct
import unittest

from cpu2tensor import Pool
from test_consumer import frame, worker


FEATURES = (1 << 12) | (1 << 16) | (1 << 22)


def transition_frame(source: int, rows: list[tuple[int, int, int]]) -> bytes:
    payload = b"".join(struct.pack("<QQQ", *row) for row in rows)
    return (struct.pack("<IHHIIQQ", 0x31543243, 3, 15, source, len(rows), 1,
                        len(payload)) + payload)


def context_frame(source: int, rows: list[tuple[int, ...]]) -> bytes:
    payload = b"".join(struct.pack("<9Q", *row) for row in rows)
    return (struct.pack("<IHHIIQQ", 0x31543243, 3, 20, source, len(rows), 0,
                        len(payload)) + payload)


def summary_frame(
    source: int,
    *,
    sources: int,
    capacity: int,
    blocks: int,
    distinct: int,
    transition_overflow: int,
    context_changes: int,
    retained_contexts: int,
    context_overflow: int,
) -> bytes:
    transitions = max(0, blocks - 1)
    payload = struct.pack(
        "<IIIIQQQQQQQ", sources, capacity, capacity, 0, blocks, transitions,
        distinct, transition_overflow, context_changes, retained_contexts,
        context_overflow,
    )
    return struct.pack("<IHHIIQQ", 0x31543243, 3, 19, source, 1, 0, len(payload)) + payload


def reduced_run(addresses: list[int], *, context_changes: int = 1) -> bytes:
    counts = Counter(zip(addresses, addresses[1:]))
    rows = [(start, end, count) for (start, end), count in counts.items()]
    retained = min(context_changes, 4)
    contexts = [
        (index * 7, 0x1000 + index, 1, 0x2000 + index, 3, 4, 0, 64, 63)
        for index in range(retained)
    ]
    data = frame(1, version=3, detail=2 | FEATURES)
    if contexts:
        data += context_frame(0, contexts)
    if rows:
        data += transition_frame(0, rows)
    data += summary_frame(
        0, sources=1, capacity=4, blocks=len(addresses), distinct=len(rows),
        transition_overflow=0, context_changes=context_changes,
        retained_contexts=retained, context_overflow=context_changes - retained,
    )
    return data + frame(3, version=3, source=0, sequence=retained) + frame(4, version=3)


class ObservationReductionTests(unittest.TestCase):
    def test_high_rate_result_matches_raw_prefix_and_is_bounded(self) -> None:
        addresses = [10, 20, 30, 20] * 25000
        expected = Counter(zip(addresses, addresses[1:]))
        data = reduced_run(addresses, context_changes=10)
        self.assertLess(len(data), 1024, "wire bytes must depend on capacity, not event count")

        # Reduction rows are already producer bounded. They bypass the raw
        # occurrence collator even when a client requests byte batching.
        with worker(data, fragment=113) as endpoint, Pool(
            [endpoint], batch_bytes=64,
        ) as pool:
            batches = list(pool.read())

        transitions = Counter()
        for batch in batches:
            self.assertEqual(batch.addresses.numel(), 0)
            if batch.observation_transitions is not None:
                table = batch.observation_transitions
                for start, end, count in zip(
                    table.from_addresses.tolist(), table.destinations.tolist(),
                    table.counts.tolist(), strict=True,
                ):
                    transitions[(start, end)] = count
        self.assertEqual(transitions, expected)
        contexts = next(batch.observation_context for batch in batches
                        if batch.observation_context is not None)
        self.assertEqual(contexts.block_positions.tolist(), [0, 7, 14, 21])
        summary = next(batch.observation_summary for batch in batches
                       if batch.observation_summary is not None)
        self.assertEqual(summary.blocks, 100000)
        self.assertEqual(summary.transitions, 99999)
        self.assertEqual(summary.context_changes, 10)
        self.assertEqual((summary.retained_contexts, summary.context_overflow), (4, 6))
        self.assertFalse(summary.exact)
        self.assertTrue(summary.upstream_complete)

    def test_multiworker_keeps_worker_and_source_identity(self) -> None:
        first = reduced_run([1, 2, 1, 2])
        second = reduced_run([7, 8, 9])
        with worker(first) as one, worker(second) as two, Pool([one, two]) as pool:
            batches = list(pool.read())
        summaries = {batch.worker: batch.observation_summary for batch in batches
                     if batch.observation_summary is not None}
        self.assertEqual(set(summaries), {0, 1})
        self.assertEqual((summaries[0].blocks, summaries[1].blocks), (4, 3))

    def test_summary_is_not_published_before_upstream_completion(self) -> None:
        complete = reduced_run([1, 2, 1])
        truncated = complete[:-64]  # Remove source_end and Complete.
        observed = []
        with worker(truncated) as endpoint, Pool([endpoint]) as pool:
            batches = pool.read()
            with self.assertRaisesRegex(RuntimeError, "Incomplete trace"):
                while True:
                    observed.append(next(batches))
        self.assertTrue(any(batch.observation_transitions is not None for batch in observed))
        self.assertTrue(all(batch.observation_summary is None for batch in observed))

    def test_raw_rows_and_inconsistent_summaries_are_rejected(self) -> None:
        hello = frame(1, version=3, detail=2 | FEATURES)
        raw = hello + frame(2, version=3, addresses=(1,))
        bad_summary = hello + summary_frame(
            0, sources=1, capacity=8, blocks=3, distinct=0,
            transition_overflow=0, context_changes=0, retained_contexts=0,
            context_overflow=0,
        )
        for data in (raw, bad_summary):
            with self.subTest(size=len(data)), worker(data) as endpoint, Pool([endpoint]) as pool:
                with self.assertRaises(ValueError):
                    list(pool.read())


if __name__ == "__main__":
    unittest.main()
