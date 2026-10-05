"""Leakage-safe training and evaluation for the transition ranker."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .ml import FEATURE_NAMES


def _require_training_dependencies():
    try:
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.metrics import ndcg_score
        from xgboost import XGBRanker
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Training requires the ML extra and a working XGBoost/OpenMP runtime") from exc
    return GroupShuffleSplit, ndcg_score, XGBRanker


def load_labeled_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("relevance", "").strip()]
    if not rows:
        raise ValueError("No labeled rows found; set relevance to 0 (reject), 1 (acceptable), or 2 (preferred)")
    for row in rows:
        relevance = int(row["relevance"])
        if relevance not in (0, 1, 2):
            raise ValueError(f"Invalid relevance {relevance!r} for {row.get('candidate_id')}; expected 0, 1, or 2")
        row["relevance"] = relevance
        for name in FEATURE_NAMES:
            row[name] = float(row[name])
    return rows


def _ordered(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row["context_id"], row.get("candidate_id", "")))


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)


def _groups(rows: list[dict]) -> list[int]:
    sizes: list[int] = []
    previous = None
    for row in rows:
        if row["context_id"] != previous:
            sizes.append(0)
            previous = row["context_id"]
        sizes[-1] += 1
    return sizes


def _evaluate(rows: list[dict], predictions: np.ndarray, ndcg_score) -> dict:
    ndcgs, correct, pairs = [], 0, 0
    start = 0
    for size in _groups(rows):
        labels = np.asarray([row["relevance"] for row in rows[start : start + size]], dtype=float)
        scores = predictions[start : start + size]
        if size > 1 and np.max(labels) > 0:
            ndcgs.append(float(ndcg_score(labels.reshape(1, -1), scores.reshape(1, -1), k=min(5, size))))
        for left in range(size):
            for right in range(left + 1, size):
                if labels[left] == labels[right]:
                    continue
                pairs += 1
                correct += int((scores[left] - scores[right]) * (labels[left] - labels[right]) > 0)
        start += size
    return {
        "ndcg_at_5": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 6),
        "pairwise_accuracy": round(correct / pairs if pairs else 0.0, 6),
        "evaluated_pairs": pairs,
    }


def train_ranker(labels_path: str | Path, output_dir: str | Path, seed: int = 42) -> dict:
    GroupShuffleSplit, ndcg_score, XGBRanker = _require_training_dependencies()
    rows = load_labeled_rows(labels_path)
    contexts = np.asarray([row["context_id"] for row in rows])
    unique_contexts = np.unique(contexts)
    if len(unique_contexts) < 3:
        raise ValueError("At least three labeled source contexts are required for a leakage-safe train/test split")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_indices, test_indices = next(splitter.split(np.zeros(len(rows)), groups=contexts))
    train_rows = _ordered([rows[index] for index in train_indices])
    test_rows = _ordered([rows[index] for index in test_indices])
    if any(len(set(row["relevance"] for row in subset)) < 2 for subset in (train_rows, test_rows)):
        raise ValueError("Both train and test splits need at least two relevance levels; label more contexts")

    model = XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg@5",
        n_estimators=180,
        max_depth=4,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=1,
    )
    train_x = _matrix(train_rows)
    train_y = np.asarray([row["relevance"] for row in train_rows])
    model.fit(train_x, train_y, group=_groups(train_rows), verbose=False)
    predictions = model.predict(_matrix(test_rows))
    metrics = _evaluate(test_rows, predictions, ndcg_score)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(output / "ranker.json")
    importance = dict(zip(FEATURE_NAMES, model.feature_importances_.astype(float)))
    metadata = {
        "model_type": "xgboost-rank-pairwise-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "seed": seed,
        "training_rows": len(train_rows),
        "test_rows": len(test_rows),
        "training_contexts": len({row["context_id"] for row in train_rows}),
        "test_contexts": len({row["context_id"] for row in test_rows}),
        "metrics": metrics,
        "feature_importance": {key: round(value, 8) for key, value in importance.items()},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
