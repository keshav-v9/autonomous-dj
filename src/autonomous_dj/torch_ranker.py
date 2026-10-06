"""PyTorch RankNet backend for learned transition ordering."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .ml import FEATURE_NAMES, MusicFeatures, RankedCandidate, TransitionRanker, transition_features
from .training import _evaluate, _ordered, load_labeled_rows


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch ranking requires: pip install -e '.[ml]'") from exc
    return torch


def _network(torch):
    return torch.nn.Sequential(
        torch.nn.Linear(len(FEATURE_NAMES), 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)


def _context_split(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    contexts = sorted({row["context_id"] for row in rows})
    if len(contexts) < 3:
        raise ValueError("At least three labeled source contexts are required for a leakage-safe train/test split")
    rng = np.random.default_rng(seed)
    rng.shuffle(contexts)
    test_count = max(1, round(len(contexts) * 0.25))
    test_contexts = set(contexts[:test_count])
    train_rows = _ordered([row for row in rows if row["context_id"] not in test_contexts])
    test_rows = _ordered([row for row in rows if row["context_id"] in test_contexts])
    return train_rows, test_rows


def _pair_indices(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["context_id"]].append(index)
    preferred, other = [], []
    for indices in grouped.values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                left_label, right_label = rows[left]["relevance"], rows[right]["relevance"]
                if left_label == right_label:
                    continue
                winner, loser = (left, right) if left_label > right_label else (right, left)
                preferred.append(winner)
                other.append(loser)
    if not preferred:
        raise ValueError("No unequal relevance pairs were found in the training contexts")
    return np.asarray(preferred, dtype=np.int64), np.asarray(other, dtype=np.int64)


def train_torch_ranker(
    labels_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    epochs: int = 24,
    batch_size: int = 512,
) -> dict:
    """Train RankNet using pairwise logistic loss and persist TorchScript."""
    torch = _require_torch()
    from sklearn.metrics import ndcg_score

    if epochs < 1:
        raise ValueError("epochs must be positive")
    rows = load_labeled_rows(labels_path)
    train_rows, test_rows = _context_split(rows, seed)
    if any(len({row["relevance"] for row in subset}) < 2 for subset in (train_rows, test_rows)):
        raise ValueError("Both train and test splits need at least two relevance levels")

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x = _matrix(train_rows)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (train_x - mean) / scale
    preferred, other = _pair_indices(train_rows)
    left = torch.from_numpy(normalized[preferred])
    right = torch.from_numpy(normalized[other])
    dataset = torch.utils.data.TensorDataset(left, right)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model = _network(torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0005)
    loss_function = torch.nn.BCEWithLogitsLoss()
    final_loss = 0.0
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        for preferred_x, other_x in loader:
            optimizer.zero_grad()
            difference = model(preferred_x).squeeze(1) - model(other_x).squeeze(1)
            loss = loss_function(difference, torch.ones_like(difference))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(preferred_x)
        final_loss = total_loss / len(dataset)

    model.eval()
    test_x = (_matrix(test_rows) - mean) / scale
    with torch.no_grad():
        predictions = model(torch.from_numpy(test_x)).squeeze(1).numpy()
    metrics = _evaluate(test_rows, predictions, ndcg_score)
    metrics["ranknet_pairwise_loss"] = round(final_loss, 6)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.cpu())
    scripted.save(str(output / "ranknet.pt"))
    metadata = {
        "model_type": "pytorch-ranknet-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "normalization_mean": mean.astype(float).tolist(),
        "normalization_scale": scale.astype(float).tolist(),
        "architecture": [len(FEATURE_NAMES), 32, 16, 1],
        "seed": seed,
        "epochs": epochs,
        "pairwise_examples": len(dataset),
        "training_rows": len(train_rows),
        "test_rows": len(test_rows),
        "training_contexts": len({row["context_id"] for row in train_rows}),
        "test_contexts": len({row["context_id"] for row in test_rows}),
        "metrics": metrics,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


class TorchTransitionRanker:
    def __init__(self, model, metadata: dict, torch):
        self.model = model
        self.metadata = metadata
        self.torch = torch
        self.mean = np.asarray(metadata["normalization_mean"], dtype=np.float32)
        self.scale = np.asarray(metadata["normalization_scale"], dtype=np.float32)

    @property
    def model_name(self) -> str:
        return str(self.metadata["model_type"])

    @classmethod
    def load(cls, model_dir: str | Path) -> "TorchTransitionRanker":
        torch = _require_torch()
        directory = Path(model_dir)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("PyTorch model feature schema does not match this application version")
        model = torch.jit.load(str(directory / "ranknet.pt"), map_location="cpu")
        model.eval()
        return cls(model, metadata, torch)

    def rank(
        self,
        source: MusicFeatures,
        candidates: Sequence[MusicFeatures],
        target_energy: float,
        direction: str = "maintain",
    ) -> list[RankedCandidate]:
        if not candidates:
            return []
        rows = [transition_features(source, item, target_energy, direction) for item in candidates]
        matrix = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
        tensor = self.torch.from_numpy((matrix - self.mean) / self.scale).requires_grad_(True)
        scores = self.model(tensor).squeeze(1)
        scores.sum().backward()
        contributions = (tensor.grad * tensor).detach().numpy()
        score_values = scores.detach().numpy()
        ranked = []
        for candidate, score, row, contribution in zip(candidates, score_values, rows, contributions):
            explanation = sorted(
                ({"feature": name, "contribution": round(float(value), 6)} for name, value in zip(FEATURE_NAMES, contribution)),
                key=lambda item: abs(item["contribution"]),
                reverse=True,
            )[:3]
            ranked.append(RankedCandidate(candidate.track_id, round(float(score), 6), row, explanation))
        return sorted(ranked, key=lambda item: item.score, reverse=True)


def load_preferred_ranker(model_dir: str | Path):
    """Prefer a neural artifact and retain the existing XGBoost/baseline path."""
    directory = Path(model_dir)
    if (directory / "ranknet.pt").exists() and (directory / "metadata.json").exists():
        return TorchTransitionRanker.load(directory)
    return TransitionRanker.load(directory)
