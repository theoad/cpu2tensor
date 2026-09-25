# SPDX-License-Identifier: AGPL-3.0-only
"""Build a replayable, localized hardware-anomaly evidence bundle."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from cpu2tensor.examples.hardware_multimodal import (
    empirical_tail_probability,
    load_frozen_multimodal_model,
    multimodal_anomaly_evidence,
    multimodal_anomaly_score,
)
from cpu2tensor.examples.hardware_multimodal_experiment import (
    RAW_SCHEMA,
    _concatenate,
    _partition,
    _sha256,
    load_dataset,
)
from cpu2tensor.examples.hardware_multimodal_features import (
    PEBS_FEATURES,
    PMU_FEATURES,
)


ANOMALY_BUNDLE_SCHEMA = "cpu2tensor-hardware-anomaly-bundle-v3"
_LEGACY_RAW_SCHEMAS = {
    "cpu2tensor-kernel-multimodal-raw-v1",
    "cpu2tensor-kernel-multimodal-raw-v2",
}
_SEGMENTS = 16

_BIT_FIELDS = {
    "operation": (0, ("na", "load", "store", "prefetch", "execute")),
    "level": (
        5,
        (
            "na", "hit", "miss", "l1", "line_fill_buffer", "l2", "l3",
            "local_ram", "remote_ram_1_hop", "remote_ram_2_hops",
            "remote_cache_1_hop", "remote_cache_2_hops", "io", "uncached",
        ),
    ),
    "snoop": (19, ("na", "none", "hit", "miss", "hit_modified")),
    "lock": (24, ("na", "locked")),
    "tlb": (26, ("na", "hit", "miss", "l1", "l2", "walker", "os_fault")),
    "snoop_extension": (38, ("forward", "peer")),
    "blocked": (40, ("na", "data_forwarding", "address_conflict")),
}
_LEVEL_NUMBERS = {
    1: "l1", 2: "l2", 3: "l3", 4: "l4", 8: "uncached", 9: "cxl",
    10: "io", 11: "any_cache", 12: "line_fill_buffer", 13: "ram",
    14: "persistent_memory", 15: "na",
}
_HOPS = {0: "unspecified", 1: "same_node", 2: "same_socket", 3: "same_board",
         4: "remote_board"}
_KNOWN_DATA_SOURCE_BITS = (1 << 46) - 1


@dataclass(frozen=True)
class KernelSymbol:
    address: int
    kind: str
    name: str
    module: str | None


@dataclass(frozen=True)
class KernelSymbolTable:
    """One exact-boot symbol table; no relocation is guessed."""

    symbols: tuple[KernelSymbol, ...]
    addresses: tuple[int, ...]

    @classmethod
    def load(cls, path: Path) -> "KernelSymbolTable":
        symbols = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"invalid symbol line {line_number}")
            try:
                address = int(fields[0], 16)
            except ValueError as error:
                raise ValueError(f"invalid symbol address on line {line_number}") from error
            module = fields[3].strip("[]") if len(fields) >= 4 else None
            symbols.append(KernelSymbol(address, fields[1], fields[2], module))
        symbols.sort(key=lambda symbol: symbol.address)
        if not symbols or symbols[-1].address == 0:
            raise ValueError("symbol table contains no visible addresses")
        return cls(tuple(symbols), tuple(symbol.address for symbol in symbols))

    def resolve(self, address: int) -> dict[str, object] | None:
        index = bisect_right(self.addresses, address) - 1
        if index < 0:
            return None
        symbol = self.symbols[index]
        return {
            "name": symbol.name,
            "kind": symbol.kind,
            "module": symbol.module,
            "offset": address - symbol.address,
        }


def decode_perf_mem_data_source(value: int) -> dict[str, object]:
    """Decode Linux ``perf_mem_data_src`` without discarding unknown bits."""
    unsigned = value & ((1 << 64) - 1)
    result: dict[str, object] = {"raw": f"0x{unsigned:016x}"}
    for field, (shift, names) in _BIT_FIELDS.items():
        bits = (unsigned >> shift) & ((1 << len(names)) - 1)
        result[field] = [name for bit, name in enumerate(names) if bits & (1 << bit)]
    level_number = (unsigned >> 33) & 0xF
    hops = (unsigned >> 43) & 0x7
    result["level_number"] = {
        "raw": level_number,
        "name": _LEVEL_NUMBERS.get(level_number, "reserved"),
    }
    result["remote"] = bool((unsigned >> 37) & 1)
    result["hops"] = {"raw": hops, "name": _HOPS.get(hops, "reserved")}
    result["unknown_bits"] = f"0x{unsigned & ~_KNOWN_DATA_SOURCE_BITS:016x}"
    return result


def _top_features(error: torch.Tensor, names: Sequence[str], limit: int) -> list[dict[str, object]]:
    values, indices = error.flatten().topk(min(limit, error.numel()))
    return [
        {"feature": names[int(index)], "residual": float(value)}
        for value, index in zip(values, indices)
    ]


def _pt_windows(raw: dict[str, object], token_error: torch.Tensor,
                feature_error: torch.Tensor, timing_error: torch.Tensor,
                limit: int) -> list[dict[str, object]]:
    result = []
    for lane, batch in enumerate(raw["batches"]):
        trace = batch["pt"]["trace_bytes"]
        count = trace.numel()
        values, indices = token_error[lane].topk(min(limit, _SEGMENTS))
        for value, segment_tensor in zip(values, indices):
            segment = int(segment_tensor)
            start = count * segment // _SEGMENTS
            stop = count * (segment + 1) // _SEGMENTS
            window = bytes(trace[start:stop].tolist())
            result.append({
                "lane": lane,
                "tid": batch["tid"],
                "segment": segment,
                "raw_byte_start": start,
                "raw_byte_stop": stop,
                "raw_bytes_base64": base64.b64encode(window).decode("ascii"),
                "raw_bytes_sha256": hashlib.sha256(window).hexdigest(),
                "residual": float(value),
                "timing_residual": float(timing_error[lane, segment]),
                "top_byte_residuals": _top_features(
                    feature_error[lane, segment],
                    tuple(f"0x{value:02x}" for value in range(256)),
                    min(limit, 8),
                ),
            })
    return sorted(result, key=lambda row: row["residual"], reverse=True)[:limit]


def _pebs_samples(raw: dict[str, object], token_error: torch.Tensor,
                  timing_error: torch.Tensor,
                  symbols: KernelSymbolTable | None,
                  limit: int | None = None) -> list[dict[str, object]]:
    windows = []
    for lane, batch in enumerate(raw["batches"]):
        pebs = batch["pebs"]
        if pebs is None or pebs["time"].numel() == 0:
            continue
        envelope = batch["envelope"]
        start = envelope["arm_before_ns"]
        stop = envelope["stop_after_ns"]
        duration = stop - start
        for row, timestamp in enumerate(pebs["time"].tolist()):
            segment = min(_SEGMENTS - 1, (timestamp - start) * _SEGMENTS // duration)
            ip = int(pebs["ip"][row]) & ((1 << 64) - 1)
            address = int(pebs["address"][row]) & ((1 << 64) - 1)
            windows.append({
                "lane": lane,
                "tid": batch["tid"],
                "sample_index": row,
                "segment": segment,
                "residual": float(token_error[lane, segment]),
                "timing_residual": float(timing_error[lane, segment]),
                "time_from_capture_start_ns": timestamp - start,
                "cpu": int(pebs["cpu"][row]),
                "ip": f"0x{ip:016x}",
                "ip_symbol": None if symbols is None else symbols.resolve(ip),
                "address": f"0x{address:016x}",
                "weight": int(pebs["weight"][row]),
                "period": int(pebs["period"][row]),
                "exact_ip": bool(pebs["exact_ip"][row]),
                "data_source": decode_perf_mem_data_source(int(pebs["data_source"][row])),
            })
    ranked = sorted(windows, key=lambda row: row["residual"], reverse=True)
    return ranked if limit is None else ranked[:limit]


def _pmu_lanes(raw: dict[str, object], feature_error: torch.Tensor,
               timing_error: torch.Tensor) -> list[dict[str, object]]:
    """Retain every boundary counter with its lane and scored residual.

    Boundary counters have no instruction address, so assigning them a symbol
    would fabricate precision. PT and PEBS provide the address-bearing context.
    """
    result = []
    for lane, batch in enumerate(raw["batches"]):
        counters = batch["counters"]
        if counters is None:
            continue
        names = list(counters["names"])
        values = counters["values"].tolist()
        residuals = feature_error[lane, 0].tolist()
        result.append({
            "lane": lane,
            "tid": batch["tid"],
            "cpu": counters["cpu"],
            "time_enabled_ns": counters["time_enabled_ns"],
            "time_running_ns": counters["time_running_ns"],
            "timing_residual": float(timing_error[lane, 0]),
            "counters": [
                {"event": name, "delta": int(value)}
                for name, value in zip(names, values)
            ],
            "derived_feature_residuals": [
                {"feature": name, "residual": float(residual)}
                for name, residual in zip(PMU_FEATURES, residuals)
            ],
            "address_semantics": "boundary_delta_has_no_instruction_address",
        })
    return result


def _atomic_json(path: Path, value: dict[str, object]) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def build_bundle(artifact: Path, execution_id: str, checkpoint: Path,
                 *, symbols: KernelSymbolTable | None = None,
                 top_windows: int = 8) -> dict[str, object]:
    manifest, rows = load_dataset(artifact)
    entries = [entry for entry in manifest["entries"]
               if entry["execution_id"] == execution_id]
    if len(entries) != 1:
        raise ValueError("execution id must identify exactly one retained row")
    entry = entries[0]
    if not entry.get("raw_retained", True):
        raise ValueError(
            "raw evidence was not retained by the preregistered pretraining sample"
        )
    raw_path = artifact / entry["raw_path"]
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    if raw.get("schema") != RAW_SCHEMA and raw.get("schema") not in _LEGACY_RAW_SCHEMAS:
        raise ValueError("unsupported raw capture schema")
    kernel_decode_source = None
    if raw.get("schema") == RAW_SCHEMA:
        try:
            kernel_decode_source = raw["decode_sideband"]["kernel_decode_state"]
            kernel_decode_path = artifact / kernel_decode_source["path"]
        except (KeyError, TypeError) as error:
            raise ValueError("raw capture omitted exact-session decode state") from error
        if _sha256(kernel_decode_path) != kernel_decode_source.get("sha256"):
            raise ValueError("kernel decode state custody hash mismatch")
        kernel_decode = torch.load(
            kernel_decode_path, map_location="cpu", weights_only=True
        )
        if (
            kernel_decode.get("schema") != "cpu2tensor-kernel-decode-state-v1"
            or kernel_decode.get("kernel_state_sha256")
            != kernel_decode_source.get("kernel_state_sha256")
        ):
            raise ValueError("kernel decode state identity mismatch")
    if raw.get("execution") != {
        name: entry[name] for name in (
            "execution_id", "family", "repetition", "partition"
        )
    }:
        raise ValueError("raw capture identity differs from the manifest")
    model, threshold = load_frozen_multimodal_model(checkpoint, device="cpu")
    batch = rows[execution_id]
    evidence = multimodal_anomaly_evidence(model, batch)
    calibration = _partition(manifest, rows, "calibration")
    calibration_scores = multimodal_anomaly_score(model, calibration)
    tail = empirical_tail_probability(evidence.score, calibration_scores)
    invocation = raw.get("invocation") or {
        "argv": [entry["family"], str(raw["loops"])],
        "stdin": torch.empty(0, dtype=torch.uint8),
    }
    stdin = bytes(invocation["stdin"].tolist())
    stdout = bytes(raw["stdout"].tolist())
    pebs_features = evidence.pebs_feature_error[0].amax((0, 1))
    pmu_features = evidence.pmu_feature_error[0].amax((0, 1))
    pebs_samples = _pebs_samples(
        raw, evidence.pebs_token_error[0],
        evidence.timing_token_error[0, :, 16:32], symbols,
    )
    score = float(evidence.score[0])
    return {
        "schema": ANOMALY_BUNDLE_SCHEMA,
        "subject": manifest["subject"],
        "event": manifest["event"],
        "build": manifest["build"],
        "source": {
            "manifest_sha256": _sha256(artifact / "capture-manifest.json"),
            "raw_path": entry["raw_path"],
            "raw_sha256": entry["raw_sha256"],
            "derived_path": entry["derived_path"],
            "derived_sha256": entry["derived_sha256"],
            "kernel_decode_state": kernel_decode_source,
        },
        "execution": {
            "id": execution_id,
            "family": entry["family"],
            "repetition": entry["repetition"],
            "partition": entry["partition"],
            "loops": raw["loops"],
            "argv": list(invocation["argv"]),
            "stdin_base64": base64.b64encode(stdin).decode("ascii"),
            "stdin_sha256": hashlib.sha256(stdin).hexdigest(),
            "stdout_base64": base64.b64encode(stdout).decode("ascii"),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        },
        "model": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "threshold": threshold,
            "score": score,
            "alert": score > threshold,
            "threshold_margin": score - threshold,
            "empirical_tail_probability": float(tail[0]),
            "calibration_executions": calibration_scores.numel(),
            "modality_scores": {
                name: float(evidence.modality_scores[0, index])
                for index, name in enumerate(("pt", "pebs", "pmu"))
                if bool(evidence.modality_present[0, index])
            },
            "explanation_signal": {
                "method": "masked_reconstruction_residuals_used_by_score",
                "attention_exported": False,
                "anomaly_token_exported": False,
            },
        },
        "localization": {
            "pt_windows": _pt_windows(
                raw, evidence.pt_token_error[0], evidence.pt_feature_error[0],
                evidence.timing_token_error[0, :, :16],
                top_windows,
            ),
            "pebs_sample_count": len(pebs_samples),
            "pebs_samples": pebs_samples[:top_windows * 8],
            "pmu_lanes": _pmu_lanes(
                raw, evidence.pmu_feature_error[0],
                evidence.timing_token_error[0, :, 32:],
            ),
            "top_pebs_feature_residuals": _top_features(
                pebs_features, PEBS_FEATURES, top_windows
            ),
            "top_pmu_feature_residuals": _top_features(
                pmu_features, PMU_FEATURES, len(PMU_FEATURES)
            ),
            "pmu_timing_residual": float(
                evidence.timing_token_error[0, :, 32:].amax()
            ),
            "symbols_available": symbols is not None,
        },
        "full_evidence": {
            "pebs_samples": pebs_samples,
            "pt_note": (
                "Ranked PT windows retain the original AUX bytes. Decoded branches "
                "require exact-session perf sideband and are not inferred here."
            ),
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("artifact", type=Path)
    result.add_argument("execution_id")
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--kallsyms", type=Path)
    result.add_argument("--top-windows", type=int, default=8)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.top_windows <= 0:
        raise SystemExit("--top-windows must be positive")
    symbols = None if args.kallsyms is None else KernelSymbolTable.load(args.kallsyms)
    bundle = build_bundle(
        args.artifact.resolve(), args.execution_id, args.checkpoint.resolve(),
        symbols=symbols, top_windows=args.top_windows,
    )
    digest = _atomic_json(args.output.resolve(), bundle)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": digest}))


if __name__ == "__main__":
    main()
