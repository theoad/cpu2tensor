# SPDX-License-Identifier: AGPL-3.0-only
"""Learn the most frequent executed digit from real block-entry counts."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import json
from pathlib import Path
import random

import torch

from cpu2tensor import Batch, Pool


CLASSES = 10
INPUT_DIGITS = 16
FEATURE_NAMES = [f"digit_{digit}" for digit in range(CLASSES)]


def parse_symbols(text: str) -> dict[str, int]:
    """Read the example's code addresses from ordinary `nm -n` output."""
    wanted = set(FEATURE_NAMES + ["sample_end"])
    symbols: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[-1] not in wanted:
            continue
        name = fields[-1]
        if fields[-2].lower() != "t" or name in symbols:
            raise ValueError(f"Expected one code symbol named {name}")
        symbols[name] = int(fields[0], 16)
    missing = [name for name in FEATURE_NAMES if name not in symbols]
    if missing:
        raise ValueError(f"Missing digit code symbols: {', '.join(missing)}")
    if len(set(symbols.values())) != len(symbols):
        raise ValueError("Digit entries and the sample marker must have distinct addresses")
    # The interactive digit target has the same digit entries but no sample marker.
    return symbols


def make_inputs(per_class: int = 30, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Create balanced, unique inputs with a strictly most frequent digit."""
    if per_class < 5:
        raise ValueError("Use at least five inputs per class for a held-out split")
    random_source = random.Random(seed)
    rows: list[tuple[int, ...]] = []
    labels: list[int] = []
    seen: set[tuple[int, ...]] = set()
    seen_counts: set[tuple[int, ...]] = set()
    for label in range(CLASSES):
        others = [digit for digit in range(CLASSES) if digit != label]
        for _ in range(per_class):
            while True:
                # The dominant count has the same distribution in every class.
                count = random_source.randint(6, 10)
                digits = [label] * count
                digits.extend(random_source.choices(others, k=INPUT_DIGITS - count))
                random_source.shuffle(digits)
                row = tuple(digits)
                histogram = tuple(digits.count(digit) for digit in range(CLASSES))
                if (row not in seen and histogram not in seen_counts
                        and all(histogram[other] < count for other in others)):
                    break
            seen.add(row)
            seen_counts.add(histogram)
            rows.append(row)
            labels.append(label)
    order = list(range(len(rows)))
    random_source.shuffle(order)
    return (torch.tensor([rows[index] for index in order], dtype=torch.uint8),
            torch.tensor([labels[index] for index in order], dtype=torch.int64))


def count_entries(batches: Iterable[Batch], entry_addresses: Sequence[int]) -> torch.Tensor:
    """Reduce a complete, single-source trace to ten function-entry counts.

    Only captured block addresses enter the features. Symbols select code locations;
    input bytes, labels, stdout and event counts outside those locations do not.
    This example uses a non-PIE target, so its ELF entry addresses are stable.
    """
    if len(entry_addresses) != CLASSES or len(set(entry_addresses)) != CLASSES:
        raise ValueError("Provide ten distinct digit function entry addresses")
    entries = torch.tensor(entry_addresses, dtype=torch.int64)
    counts = torch.zeros(CLASSES, dtype=torch.int64)
    source: int | None = None
    for batch in batches:
        if source is None:
            source = batch.source
        if batch.source != source:
            raise ValueError("The digit example requires a single-vCPU trace")
        if batch.addresses.numel():
            addresses = batch.addresses.cpu()
            counts += (addresses[:, None] == entries[None, :]).sum(dim=0)
    if counts.sum().item() != INPUT_DIGITS:
        raise ValueError("Expected exactly sixteen captured digit function entries")
    return counts.to(torch.float32)


def count_samples(
    batches: Iterable[Batch], entry_addresses: Sequence[int], sample_end_address: int,
) -> torch.Tensor:
    """Count digit entries per input line, ending each sample at its marker.

    A marker is an ordinary target function. It gives this example explicit sample
    boundaries without exposing the input or label to the feature reducer.
    Tensor operations scan each batch; Python only appends completed sample groups.
    """
    locations = [*entry_addresses, sample_end_address]
    if len(entry_addresses) != CLASSES or len(set(locations)) != CLASSES + 1:
        raise ValueError("Provide ten distinct digit entries and a distinct sample marker")
    entries = torch.tensor(locations, dtype=torch.int64)
    pending = torch.zeros(CLASSES, dtype=torch.int64)
    samples = []
    source: int | None = None
    for batch in batches:
        if source is None:
            source = batch.source
        if batch.source != source:
            raise ValueError("The digit example requires a single-vCPU trace")
        if not batch.addresses.numel():
            continue
        matches = batch.addresses.cpu()[:, None] == entries[None, :]
        events = matches.nonzero(as_tuple=True)[1]
        if not events.numel():
            continue
        markers = events == CLASSES
        marker_counts = markers.to(torch.int64)
        groups = marker_counts.cumsum(0) - marker_counts
        completed = int(markers.sum())
        counts = torch.bincount((groups * CLASSES + events)[~markers],
                                minlength=(completed + 1) * CLASSES).reshape(-1, CLASSES)
        counts[0] += pending
        if completed:
            if not bool((counts[:completed].sum(1) == INPUT_DIGITS).all()):
                raise ValueError("Each sample must contain exactly sixteen digit entries")
            samples.append(counts[:completed].clone())
        pending = counts[-1].clone()
    if bool(pending.any()) or not samples:
        raise ValueError("Trace ended without a sample marker or contained no samples")
    return torch.cat(samples).to(torch.float32)


def validate_dataset(dataset: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check the independent label and trace oracle before any model is trained."""
    if dataset.get("format") != 1 or dataset.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Unsupported digit dataset format or feature names")
    features, labels, inputs = (dataset.get(name) for name in ("features", "labels", "inputs"))
    if not all(isinstance(value, torch.Tensor) for value in (features, labels, inputs)):
        raise ValueError("Dataset features, labels and inputs must be tensors")
    features, labels, inputs = features.cpu(), labels.cpu(), inputs.cpu()
    if (features.dtype != torch.float32 or labels.dtype != torch.int64
            or inputs.dtype != torch.uint8 or labels.ndim != 1):
        raise ValueError("Expected float32 features, int64 labels and uint8 input digits")
    rows = labels.numel()
    if features.shape != (rows, CLASSES) or inputs.shape != (rows, INPUT_DIGITS):
        raise ValueError("Dataset tensor shapes do not match")
    if rows == 0 or not bool(((inputs < CLASSES).all()) & ((labels >= 0) & (labels < CLASSES)).all()):
        raise ValueError("Input digits and labels must lie between zero and nine")
    if torch.unique(inputs, dim=0).shape[0] != rows:
        raise ValueError("Duplicate input would contaminate the held-out split")
    expected = torch.zeros(rows, CLASSES, dtype=torch.float32)
    expected.scatter_add_(1, inputs.to(torch.int64), torch.ones(rows, INPUT_DIGITS))
    if not torch.equal(features, expected):
        raise ValueError("Captured entry counts do not match the independent input histogram")
    if torch.unique(features, dim=0).shape[0] != rows:
        raise ValueError("Duplicate count vector would contaminate the held-out split")
    largest = expected.topk(2, dim=1).values
    if bool((largest[:, 0] == largest[:, 1]).any()) or not torch.equal(expected.argmax(1), labels):
        raise ValueError("Labels must identify the strictly most frequent input digit")
    class_counts = torch.bincount(labels, minlength=CLASSES)
    if int(class_counts.min()) < 5 or not bool((class_counts == class_counts[0]).all()):
        raise ValueError("Use at least five examples per class with balanced labels")
    return features, labels, inputs


def save_dataset(
    path: str | Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    inputs: torch.Tensor,
    *,
    host: str,
    capture: str,
) -> None:
    """Save captured features and independent validation data, with provenance."""
    dataset = {"format": 1, "features": features.cpu(), "labels": labels.cpu(),
               "inputs": inputs.cpu(), "feature_names": FEATURE_NAMES,
               "host": host, "capture": capture}
    validate_dataset(dataset)
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output)


def collect(
    endpoint: str, manifest_path: str | Path, symbols_path: str | Path,
    output_path: str | Path, *, host: str,
) -> None:
    """Consume one operator-started worker; host provisioning stays outside Python."""
    manifest = json.loads(Path(manifest_path).expanduser().read_text())
    symbols = json.loads(Path(symbols_path).expanduser().read_text())
    if "sample_end" not in symbols:
        raise ValueError("The observation target needs a sample_end code symbol")
    inputs = torch.tensor([[int(digit) for digit in row["input"]] for row in manifest],
                          dtype=torch.uint8)
    labels = torch.tensor([row["label"] for row in manifest], dtype=torch.int64)
    addresses = [int(symbols[name]) for name in FEATURE_NAMES]
    with Pool([endpoint]) as pool:
        features = count_samples(pool.read(), addresses, int(symbols["sample_end"]))
    save_dataset(output_path, features, labels, inputs, host=host,
                 capture="Real block-entry features from the non-PIE branch_digits target")


def split_indices(labels: torch.Tensor, seed: int = 19) -> tuple[torch.Tensor, torch.Tensor]:
    """Reserve one fifth of each class before fitting the model."""
    generator = torch.Generator().manual_seed(seed)
    train, test = [], []
    for label in range(CLASSES):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        test_count = max(1, indices.numel() // 5)
        test.append(indices[:test_count])
        train.append(indices[test_count:])
    return torch.cat(train), torch.cat(test)


def fit(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str,
    epochs: int,
    seed: int,
) -> tuple[torch.nn.Linear, list[float]]:
    """A full-batch classifier keeps the learning loop short and inspectable."""
    torch.manual_seed(seed)
    # Initialize on CPU so CPU and MPS runs begin with identical parameters.
    model = torch.nn.Linear(CLASSES, CLASSES).to(device)
    features, labels = features.to(device), labels.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_function = torch.nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(features), labels)
        loss.backward()
        if not all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise RuntimeError("Training produced a non-finite gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, losses


def accuracy(model: torch.nn.Module, features: torch.Tensor, labels: torch.Tensor, device: str) -> float:
    with torch.no_grad():
        predictions = model(features.to(device)).argmax(dim=1).cpu()
    return float((predictions == labels).to(torch.float32).mean())


def train(
    dataset_path: str | Path,
    output_directory: str | Path,
    *,
    device: str = "cpu",
    epochs: int = 120,
    seed: int = 19,
) -> dict:
    """Train on capture data, save weights, and run leakage controls."""
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    selected_device = torch.device(device)
    if selected_device.type not in ("cpu", "mps", "cuda"):
        raise ValueError("Choose CPU, MPS or CUDA")
    if selected_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this Python environment")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    dataset = torch.load(Path(dataset_path).expanduser(), map_location="cpu", weights_only=True)
    features, labels, _ = validate_dataset(dataset)
    train_rows, test_rows = split_indices(labels, seed)
    # Fixed input length supplies the scale; no test statistics are fitted.
    features = features / INPUT_DIGITS
    training, testing = features[train_rows], features[test_rows]
    train_labels, test_labels = labels[train_rows], labels[test_rows]
    model, losses = fit(training, train_labels, device=device, epochs=epochs, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    wrong_labels = train_labels[torch.randperm(train_labels.numel(), generator=generator)]
    shuffled_model, _ = fit(training, wrong_labels, device=device, epochs=epochs, seed=seed)
    shuffled_test = testing[torch.randperm(test_rows.numel(), generator=generator)]
    majority_label = int(torch.bincount(train_labels).argmax())
    metrics = {
        "device": str(selected_device), "torch_version": str(torch.__version__),
        "seed": seed, "epochs": epochs, "train_examples": train_rows.numel(),
        "test_examples": test_rows.numel(), "parameters": sum(p.numel() for p in model.parameters()),
        "capture_host": dataset.get("host", "unspecified"),
        "capture": dataset.get("capture", "unspecified"),
        "first_train_loss": losses[0], "last_train_loss": losses[-1],
        "train_accuracy": accuracy(model, training, train_labels, device),
        "test_accuracy": accuracy(model, testing, test_labels, device),
        "majority_accuracy": float((test_labels == majority_label).float().mean()),
        "shuffled_label_accuracy": accuracy(shuffled_model, testing, test_labels, device),
        "shuffled_trace_accuracy": accuracy(model, shuffled_test, test_labels, device),
    }
    passed = (metrics["test_accuracy"] >= 0.9 and metrics["last_train_loss"] < losses[0] * 0.2
              and metrics["shuffled_label_accuracy"] < 0.3 and metrics["shuffled_trace_accuracy"] < 0.3)
    metrics["passed"] = passed
    output = Path(output_directory).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "feature_names": FEATURE_NAMES, "input_scale": INPUT_DIGITS,
                "train_rows": train_rows, "test_rows": test_rows, "metrics": metrics},
               output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "loss.json").write_text(json.dumps(losses) + "\n")
    if not passed:
        raise RuntimeError(f"Learning validation failed; inspect {output / 'metrics.json'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    symbols_parser = commands.add_parser("symbols", help="Convert target nm output to JSON addresses")
    symbols_parser.add_argument("--input", type=Path, required=True)
    symbols_parser.add_argument("--output", type=Path, required=True)
    inputs_parser = commands.add_parser("inputs", help="Write reproducible target inputs")
    inputs_parser.add_argument("--output", type=Path, required=True)
    inputs_parser.add_argument("--per-class", type=int, default=30)
    inputs_parser.add_argument("--seed", type=int, default=7)
    inputs_parser.add_argument("--input-file", type=Path,
                               help="Target stdin file; defaults to the manifest path with .txt suffix")
    collect_parser = commands.add_parser("collect", help="Collect one operator-started target run")
    collect_parser.add_argument("endpoint")
    collect_parser.add_argument("--manifest", type=Path, required=True)
    collect_parser.add_argument("--symbols", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--host", required=True, help="Record the worker host and guest ISA")
    train_parser = commands.add_parser("train", help="Train on a captured dataset")
    train_parser.add_argument("dataset", type=Path)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    train_parser.add_argument("--epochs", type=int, default=120)
    train_parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if args.command == "symbols":
        symbols = parse_symbols(args.input.expanduser().read_text())
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(json.dumps(symbols, indent=2) + "\n")
        print(f"Wrote {len(symbols)} code addresses to {args.output}")
    elif args.command == "inputs":
        inputs, labels = make_inputs(args.per_class, args.seed)
        rows = [{"input": "".join(str(digit) for digit in row), "label": label}
                for row, label in zip(inputs.tolist(), labels.tolist())]
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(json.dumps(rows, indent=2) + "\n")
        input_file = (args.input_file or args.output.with_suffix(".txt")).expanduser()
        input_file.parent.mkdir(parents=True, exist_ok=True)
        input_file.write_text("".join(row["input"] + "\n" for row in rows))
        print(f"Wrote {len(rows)} balanced target inputs to {args.output}")
        print(f"Target stdin: {input_file}")
    elif args.command == "collect":
        collect(args.endpoint, args.manifest, args.symbols, args.output, host=args.host)
        print(f"Saved validated trace features to {args.output}")
    else:
        print(json.dumps(train(args.dataset, args.output, device=args.device,
                               epochs=args.epochs, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
