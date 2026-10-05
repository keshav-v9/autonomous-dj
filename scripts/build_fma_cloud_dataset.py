#!/usr/bin/env python3
"""Build a Phase catalog and weakly supervised ranking set from FMA metadata.

This consumes only FMA's metadata and precomputed features. It never downloads
or redistributes the audio files themselves.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from autonomous_dj.ml import (
    FEATURE_NAMES,
    HEURISTIC_WEIGHTS,
    KEY_NAMES,
    MAJOR_CAMELOT,
    MINOR_CAMELOT,
    MusicFeatures,
    transition_features,
    write_labeling_csv,
)

FMA_PROJECT_URL = "https://github.com/mdeff/fma"


def _column(frame: pd.DataFrame, key: tuple[str, ...]) -> pd.Series:
    if key not in frame.columns:
        raise ValueError(f"FMA column is missing: {key}")
    return pd.to_numeric(frame[key], errors="coerce")


def _key_from_chroma(values: np.ndarray) -> tuple[str, str]:
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major = [np.corrcoef(values, np.roll(major_profile, root))[0, 1] for root in range(12)]
    minor = [np.corrcoef(values, np.roll(minor_profile, root))[0, 1] for root in range(12)]
    is_major = max(major) >= max(minor)
    root = int(np.argmax(major if is_major else minor))
    return f"{KEY_NAMES[root]} {'major' if is_major else 'minor'}", (MAJOR_CAMELOT if is_major else MINOR_CAMELOT)[root]


def _normalize(values: pd.Series) -> pd.Series:
    low, high = values.quantile(0.02), values.quantile(0.98)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return pd.Series(0.5, index=values.index)
    return ((values - low) / (high - low)).clip(0.0, 1.0)


def build_catalog(metadata_dir: Path, limit: int, genre: str, seed: int) -> tuple[list[MusicFeatures], list[dict]]:
    tracks = pd.read_csv(metadata_dir / "tracks.csv", index_col=0, header=[0, 1], low_memory=False)
    echonest = pd.read_csv(metadata_dir / "echonest.csv", index_col=0, header=[0, 1, 2], low_memory=False)
    features = pd.read_csv(metadata_dir / "features.csv", index_col=0, header=[0, 1, 2], low_memory=False)
    for frame in (tracks, echonest, features):
        frame.index = pd.to_numeric(frame.index, errors="coerce")

    eligible = tracks.index[tracks[("track", "genre_top")].fillna("").str.casefold() == genre.casefold()]
    eligible = eligible.intersection(echonest.index).intersection(features.index)
    tempo = _column(echonest, ("echonest", "audio_features", "tempo"))
    energy = _column(echonest, ("echonest", "audio_features", "energy"))
    instrumentalness = _column(echonest, ("echonest", "audio_features", "instrumentalness"))
    danceability = _column(echonest, ("echonest", "audio_features", "danceability"))
    rms = _column(features, ("rmse", "mean", "01"))
    centroid = _column(features, ("spectral_centroid", "mean", "01"))
    required = pd.concat([tempo, energy, instrumentalness, danceability, rms, centroid], axis=1).dropna().index
    eligible = eligible.intersection(required)
    if len(eligible) < limit:
        raise ValueError(f"Only {len(eligible)} complete {genre!r} tracks are available; requested {limit}")

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(eligible.to_numpy(dtype=int), size=limit, replace=False))
    normalized_centroid = _normalize(centroid.loc[chosen])
    catalog: list[MusicFeatures] = []
    manifest: list[dict] = []
    for track_id in chosen:
        chroma = np.array([features.loc[track_id, ("chroma_cens", "mean", f"{number:02d}")] for number in range(1, 13)], dtype=float)
        key, camelot = _key_from_chroma(chroma)
        bpm = float(tempo.loc[track_id])
        rms_value = max(float(rms.loc[track_id]), 1e-9)
        feature = MusicFeatures(
            track_id=f"fma-{track_id:06d}",
            path=f"fma://{track_id}",
            duration_seconds=float(pd.to_numeric(tracks.loc[track_id, ("track", "duration")], errors="coerce") or 30.0),
            bpm=round(bpm, 3),
            key=key,
            camelot=camelot,
            energy=round(float(np.clip(energy.loc[track_id], 0.0, 1.0)), 6),
            loudness_db=round(float(20.0 * np.log10(rms_value)), 4),
            vocal_activity=round(float(np.clip(1.0 - instrumentalness.loc[track_id], 0.0, 1.0)), 6),
            spectral_centroid=round(float(centroid.loc[track_id]), 3),
            phrase_seconds=round(16.0 * 60.0 / max(bpm, 1.0), 4),
            phrase_confidence=round(float(np.clip(0.55 + 0.35 * danceability.loc[track_id] + 0.1 * normalized_centroid.loc[track_id], 0.0, 1.0)), 6),
        )
        catalog.append(feature)
        manifest.append({
            **feature.to_dict(),
            "title": str(tracks.loc[track_id, ("track", "title")]),
            "artist": str(tracks.loc[track_id, ("artist", "name")]),
            "genre": str(tracks.loc[track_id, ("track", "genre_top")]),
            "license": str(tracks.loc[track_id, ("track", "license")]),
            "dataset": "Free Music Archive",
            "dataset_url": FMA_PROJECT_URL,
        })
    return catalog, manifest


def build_training_rows(catalog: list[MusicFeatures], candidates_per_context: int, seed: int) -> list[dict]:
    if candidates_per_context < 6 or candidates_per_context >= len(catalog):
        raise ValueError("candidates-per-context must be at least 6 and smaller than the catalog")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    directions = ("maintain", "energy_up", "energy_down")
    offsets = {"maintain": 0.0, "energy_up": 0.15, "energy_down": -0.15}
    for index, source in enumerate(catalog):
        direction = directions[index % len(directions)]
        target = float(np.clip(source.energy + offsets[direction], 0.05, 0.95))
        pool = [candidate for candidate in catalog if candidate.track_id != source.track_id]
        selected = rng.choice(len(pool), size=candidates_per_context, replace=False)
        context: list[dict] = []
        for candidate_index in selected:
            candidate = pool[int(candidate_index)]
            vector = transition_features(source, candidate, target, direction)
            score = sum(vector[name] * HEURISTIC_WEIGHTS[name] for name in FEATURE_NAMES)
            context.append({
                "context_id": f"weak-{source.track_id}-{direction}",
                "source_id": source.track_id,
                "candidate_id": candidate.track_id,
                "target_energy": round(target, 6),
                "direction": direction,
                "_score": score,
                **{name: round(vector[name], 8) for name in FEATURE_NAMES},
            })
        context.sort(key=lambda row: row["_score"], reverse=True)
        preferred = max(2, candidates_per_context // 5)
        acceptable = max(2, candidates_per_context // 3)
        for rank, row in enumerate(context):
            row["relevance"] = 2 if rank < preferred else (1 if rank < preferred + acceptable else 0)
            row.pop("_score")
            rows.append(row)
    return rows


def write_catalog(manifest: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_fields = [item.name for item in fields(MusicFeatures)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*feature_fields, "title", "artist", "genre", "license", "dataset", "dataset_url"])
        writer.writeheader()
        writer.writerows(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--genre", default="Electronic")
    parser.add_argument("--candidates-per-context", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog, manifest = build_catalog(args.metadata_dir, args.limit, args.genre, args.seed)
    labels = build_training_rows(catalog, args.candidates_per_context, args.seed)
    write_catalog(manifest, args.output_dir / "fma_catalog.csv")
    write_labeling_csv(labels, args.output_dir / "weak_transition_labels.csv")
    report = {
        "tracks": len(catalog),
        "genre": args.genre,
        "seed": args.seed,
        "contexts": len(catalog),
        "candidates_per_context": args.candidates_per_context,
        "training_rows": len(labels),
        "label_source": "Phase heuristic weak supervision; replace or augment with human DJ feedback",
        "audio_downloaded": False,
        "source": FMA_PROJECT_URL,
    }
    (args.output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
