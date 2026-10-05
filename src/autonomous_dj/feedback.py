"""Durable feedback events that can be fed directly back into training."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from .ml import FEATURE_NAMES, MusicFeatures, transition_features


FEEDBACK_FIELDS = [
    "context_id", "source_id", "candidate_id", "target_energy", "direction", "relevance", *FEATURE_NAMES
]


def append_feedback(
    path: str | Path,
    context_id: str,
    source: MusicFeatures,
    choices: Sequence[tuple[MusicFeatures, int]],
    target_energy: float,
    direction: str,
) -> int:
    """Append accepted/rejected or pairwise choices in training-compatible form."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEEDBACK_FIELDS)
        if needs_header:
            writer.writeheader()
        for candidate, relevance in choices:
            if relevance not in (0, 1, 2):
                raise ValueError("Feedback relevance must be 0, 1, or 2")
            writer.writerow({
                "context_id": context_id,
                "source_id": source.track_id,
                "candidate_id": candidate.track_id,
                "target_energy": target_energy,
                "direction": direction,
                "relevance": relevance,
                **transition_features(source, candidate, target_energy, direction),
            })
    return len(choices)
