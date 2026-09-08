# SPDX-License-Identifier: AGPL-3.0-only
"""Learn four fixed paths across executable locations, using launch metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


LAYOUT_FIELDS = ["code_start", "code_end", "initial_entry"]
MODES = ("raw", "normalized", "metadata_only", "shuffled_labels")


def validate_dataset(dataset: dict) -> dict:
    """Keep locations disjoint and labels balanced before fitting a vocabulary."""
    if dataset.get("format") != 1:
        raise ValueError("Expected layout dataset format 1")
    names = ("pc", "mask", "layout", "labels", "train")
    if any(not isinstance(dataset.get(name), torch.Tensor) for name in names):
        raise ValueError("Expected pc, mask, layout, labels and train tensors")
    data = {name: dataset[name].cpu() for name in names}
    pc, mask, layout, labels, train = (data[name] for name in names)
    if (pc.dtype != torch.int64 or layout.dtype != torch.int64
            or labels.dtype != torch.int64 or mask.dtype != torch.bool
            or train.dtype != torch.bool):
        raise ValueError("Addresses and labels need int64; mask and train need bool")
    if pc.ndim != 2 or not pc.shape[0] or not pc.shape[1]:
        raise ValueError("Expected a nonempty padded matrix of block addresses")
    rows = pc.shape[0]
    if (mask.shape != pc.shape or layout.shape != (rows, 3)
            or labels.shape != (rows,) or train.shape != (rows,)):
        raise ValueError("Dataset tensor shapes do not match")
    if (not bool(mask.any(1).all()) or not bool(((labels >= 0) & (labels < 4)).all())
            or not bool((layout[:, 0] >= 0).all())
            or not bool((layout[:, 1] > layout[:, 0]).all())
            or not bool(((layout[:, 2] >= layout[:, 0])
                         & (layout[:, 2] < layout[:, 1])).all())):
        raise ValueError("Expected nonempty traces, four labels and a static executable layout")
    inside = (pc >= layout[:, :1]) & (pc < layout[:, 1:2])
    if not bool((inside | ~mask).all()):
        raise ValueError("Every captured block must lie in its executable text span")
    locations = layout[:, 0].unique()
    for start in locations:
        selected = layout[:, 0] == start
        if train[selected].unique().numel() != 1:
            raise ValueError("Training and held-out executable locations must be disjoint")
        if layout[selected].unique(dim=0).shape[0] != 1:
            raise ValueError("One executable location must have one consistent layout")
        counts = torch.bincount(labels[selected], minlength=4)
        if not bool((counts == counts[0]).all()) or not int(counts[0]):
            raise ValueError("Each executable location must contain all four labels equally")
    spans = layout.unique(dim=0)
    ordered = spans[spans[:, 0].argsort()]
    if bool((ordered[:-1, 1] > ordered[1:, 0]).any()):
        raise ValueError("Executable text spans at different locations must not overlap")
    for selected in (train, ~train):
        if layout[selected, 0].unique().numel() < 2:
            raise ValueError("Use at least two locations in each split for the metadata control")
    return data


def address_features(pc: torch.Tensor, layout: torch.Tensor, normalized: bool) -> torch.Tensor:
    """Subtract in integer space, before embedding or transfer to an accelerator."""
    return pc - layout[:, :1] if normalized else pc


def fit_vocabulary(values: torch.Tensor, mask: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    """Only training addresses enter the vocabulary. Token zero means unknown."""
    return values[train][mask[train]].unique(sorted=True)


def encode(values: torch.Tensor, vocabulary: torch.Tensor) -> torch.Tensor:
    positions = torch.searchsorted(vocabulary, values.contiguous())
    matched = (positions < vocabulary.numel()) & (
        vocabulary[positions.clamp(max=vocabulary.numel() - 1)] == values)
    return torch.where(matched, positions + 1, 0)


class PathModel(torch.nn.Module):
    """Average learned block embeddings, then predict one of four paths."""

    def __init__(self, vocabulary_size: int, width: int = 16) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, width)
        self.classifier = torch.nn.Linear(width, 4)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1)
        pooled = (self.embedding(tokens) * weights).sum(1) / weights.sum(1)
        return self.classifier(pooled)


def wrong_layout(layout: torch.Tensor) -> torch.Tensor:
    """Replace each location with the next distinct location, independent of labels."""
    locations, inverse = layout.unique(dim=0, return_inverse=True)
    if locations.shape[0] < 2:
        raise ValueError("Need two different layouts to supply wrong metadata")
    return locations[(inverse + 1) % locations.shape[0]]


def require_finite(values: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError(f"Nonfinite {name}; stop before saving learning results")


def train_one(data: dict, mode: str, seed: int, epochs: int = 80,
              device: str = "cpu") -> tuple[dict, dict]:
    """Run a fixed budget; held-out labels only score the unchanged final model."""
    if mode not in MODES or epochs < 1:
        raise ValueError("Choose a supported comparison and at least one epoch")
    pc, mask, layout, labels, train = (data[name] for name in
                                      ("pc", "mask", "layout", "labels", "train"))
    raw_vocabulary = fit_vocabulary(pc, mask, train)
    if mode == "metadata_only":
        values, mask = layout[:, :1], torch.ones_like(layout[:, :1], dtype=torch.bool)
    else:
        values = address_features(pc, layout, mode != "raw")
    vocabulary = fit_vocabulary(values, mask, train)
    tokens = encode(values, vocabulary).to(device)
    # Every comparison has identical parameter count and initial weights per seed.
    capacity = raw_vocabulary.numel() + 1
    if vocabulary.numel() + 1 > capacity:
        raise ValueError("Comparison vocabulary exceeds the common model capacity")
    torch.manual_seed(seed)
    model = PathModel(capacity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    mask, labels, train = mask.to(device), labels.to(device), train.to(device)
    training_labels = labels[train].clone()
    if mode == "shuffled_labels":
        permutation = torch.randperm(training_labels.numel(), generator=torch.Generator().manual_seed(seed))
        training_labels = training_labels[permutation.to(device)]
    history = []
    for epoch in range(epochs + 1):
        model.eval()
        with torch.no_grad():
            logits = model(tokens, mask)
            require_finite(logits, "logits")
            loss = torch.nn.functional.cross_entropy(logits[train], training_labels)
            accuracy = (logits[~train].argmax(1) == labels[~train]).float().mean()
            heldout_loss = torch.nn.functional.cross_entropy(logits[~train], labels[~train])
            require_finite(torch.stack((loss, heldout_loss)), "loss")
            probability = logits[~train].softmax(1).gather(1, labels[~train, None]).mean()
        history.append({"epoch": epoch, "train_loss": float(loss),
                        "train_accuracy": float((logits[train].argmax(1) == training_labels).float().mean()),
                        "heldout_accuracy": float(accuracy), "heldout_loss": float(heldout_loss),
                        "heldout_correct_probability": float(probability)})
        if epoch == epochs:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(tokens[train], mask[train]), training_labels)
        require_finite(loss, "training loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                require_finite(parameter.grad, "gradient")
        optimizer.step()
    result = {"mode": mode, "seed": seed, "history": history,
              "vocabulary_size": vocabulary.numel(), "parameter_capacity": capacity,
              "heldout_unknown_fraction": float(((tokens[~train] == 0) & mask[~train]).sum()
                                                 / mask[~train].sum())}
    if mode == "normalized":
        heldout = ~data["train"]
        wrong = address_features(pc[heldout], wrong_layout(layout[heldout]), True)
        wrong_tokens = encode(wrong, vocabulary).to(device)
        with torch.no_grad():
            wrong_logits = model(wrong_tokens, mask[~train])
            require_finite(wrong_logits, "wrong-metadata logits")
            prediction = wrong_logits.argmax(1)
        result["wrong_metadata_accuracy"] = float((prediction == labels[~train]).float().mean())
    checkpoint = {"state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
                  "vocabulary": vocabulary, "capacity": capacity, "mode": mode, "seed": seed,
                  "layout_fields": LAYOUT_FIELDS,
                  "heldout_predictions": logits[~train].argmax(1).cpu()}
    return result, checkpoint


def run(dataset: dict, output: Path, device: str = "cpu", epochs: int = 80,
        seeds: tuple[int, ...] = (7, 17, 29)) -> dict:
    """Write learning curves and reload every final checkpoint to verify predictions."""
    data = validate_dataset(dataset)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        for mode in MODES:
            result, checkpoint = train_one(data, mode, seed, epochs, device)
            path = output / f"{mode}-{seed}.pt"
            torch.save(checkpoint, path)
            restored = torch.load(path, weights_only=True)
            model = PathModel(restored["capacity"]).to(device)
            model.load_state_dict(restored["state_dict"])
            heldout = ~data["train"]
            values = address_features(data["pc"][heldout], data["layout"][heldout], mode != "raw")
            mask = data["mask"][heldout]
            if mode == "metadata_only":
                values, mask = data["layout"][heldout, :1], torch.ones_like(mask[:, :1])
            with torch.no_grad():
                prediction = model(encode(values, restored["vocabulary"]).to(device), mask.to(device)).argmax(1).cpu()
            if not torch.equal(prediction, restored["heldout_predictions"]):
                raise RuntimeError("Reloaded checkpoint predictions changed")
            result["checkpoint_reload"] = True
            results.append(result)
    report = {"format": 1, "device": device, "epochs": epochs, "seeds": list(seeds),
              "samples": int(data["labels"].numel()), "runs": results,
              "scope": "Four fixed paths at held-out executable locations; not unseen programs"}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    report = run(torch.load(args.dataset, weights_only=True), args.output, args.device, args.epochs)
    for result in report["runs"]:
        print(result["mode"], result["seed"], result["history"][-1])


if __name__ == "__main__":
    main()
