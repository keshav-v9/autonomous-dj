"""Feature extraction, candidate construction, and learned transition ranking.

The heavyweight ML libraries are imported lazily so the deterministic audio
engine can still run with the base install. Install ``autonomous-dj[ml]`` to
use the dataset and training commands.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FEATURE_NAMES = (
    "tempo_distance",
    "harmonic_distance",
    "energy_delta",
    "energy_target_error",
    "loudness_distance",
    "vocal_overlap",
    "spectral_distance",
    "phrase_confidence",
    "direction_fit",
)


@dataclass(slots=True)
class MusicFeatures:
    track_id: str
    path: str
    duration_seconds: float
    bpm: float
    key: str
    camelot: str
    energy: float
    loudness_db: float
    vocal_activity: float
    spectral_centroid: float
    phrase_seconds: float
    phrase_confidence: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "MusicFeatures":
        return cls(**{item.name: value[item.name] for item in fields(cls)})


@dataclass(slots=True)
class RankedCandidate:
    track_id: str
    score: float
    features: dict[str, float]
    explanation: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


KEY_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_CAMELOT = ("8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B")
MINOR_CAMELOT = ("5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A")


def _require_librosa():
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Music analysis requires the ML extra: pip install -e '.[ml]'") from exc
    return librosa


def analyze_music(path: str | Path, track_id: str | None = None) -> MusicFeatures:
    """Extract stable, normalized musical features from WAV/MP3/FLAC audio."""
    librosa = _require_librosa()
    source = Path(path)
    y, sample_rate = librosa.load(source, sr=22_050, mono=True)
    if not len(y):
        raise ValueError(f"No audio samples in {source}")

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate, units="frames")
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    chroma = librosa.feature.chroma_cqt(y=y, sr=sample_rate)
    chroma_mean = np.mean(chroma, axis=1)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_scores = [np.corrcoef(chroma_mean, np.roll(major_profile, root))[0, 1] for root in range(12)]
    minor_scores = [np.corrcoef(chroma_mean, np.roll(minor_profile, root))[0, 1] for root in range(12)]
    is_major = max(major_scores) >= max(minor_scores)
    root = int(np.argmax(major_scores if is_major else minor_scores))
    key = f"{KEY_NAMES[root]} {'major' if is_major else 'minor'}"
    camelot = (MAJOR_CAMELOT if is_major else MINOR_CAMELOT)[root]

    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    loudness_db = float(librosa.amplitude_to_db(np.array([max(rms_mean, 1e-9)]), ref=1.0)[0])
    onset = librosa.onset.onset_strength(y=y, sr=sample_rate)
    energy = float(np.clip((loudness_db + 45.0) / 35.0, 0.0, 1.0))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sample_rate)))

    # A reproducible vocal-presence proxy: energy in the 300–3400 Hz band
    # relative to total spectral energy. It is intentionally named activity,
    # not source separation confidence.
    spectrum = np.abs(librosa.stft(y)) ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate)
    vocal_band = spectrum[(frequencies >= 300) & (frequencies <= 3400)].sum()
    vocal_activity = float(np.clip(vocal_band / max(float(spectrum.sum()), 1e-12), 0.0, 1.0))

    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    beat_period = 60.0 / max(bpm, 1.0)
    phrase_seconds = beat_period * 16.0  # four 4/4 bars
    if len(beat_times) >= 32:
        phrase_onsets = np.interp(beat_times, librosa.times_like(onset, sr=sample_rate), onset)
        accents = phrase_onsets[::16]
        phrase_confidence = float(np.clip(np.std(accents) / max(np.mean(onset), 1e-8), 0.0, 1.0))
    else:
        phrase_confidence = 0.0

    return MusicFeatures(
        track_id=track_id or source.stem,
        path=str(source.resolve()),
        duration_seconds=round(len(y) / sample_rate, 3),
        bpm=round(bpm, 3),
        key=key,
        camelot=camelot,
        energy=round(energy, 6),
        loudness_db=round(loudness_db, 4),
        vocal_activity=round(vocal_activity, 6),
        spectral_centroid=round(centroid, 3),
        phrase_seconds=round(phrase_seconds, 4),
        phrase_confidence=round(phrase_confidence, 6),
    )


def camelot_distance(left: str, right: str) -> float:
    """Return a normalized harmonic-wheel distance in the range [0, 1]."""
    try:
        left_number, left_mode = int(left[:-1]), left[-1]
        right_number, right_mode = int(right[:-1]), right[-1]
    except (ValueError, IndexError):
        return 1.0
    circle = min((left_number - right_number) % 12, (right_number - left_number) % 12)
    mode_cost = 0.0 if left_mode == right_mode else (0.15 if left_number == right_number else 0.35)
    return float(np.clip(circle / 6.0 + mode_cost, 0.0, 1.0))


def transition_features(
    source: MusicFeatures,
    destination: MusicFeatures,
    target_energy: float,
    direction: str = "maintain",
) -> dict[str, float]:
    delta = destination.energy - source.energy
    expected_sign = {"energy_up": 1.0, "raise": 1.0, "energy_down": -1.0, "lower": -1.0}.get(direction, 0.0)
    if expected_sign == 0.0:
        direction_fit = 1.0 - min(abs(delta) / 0.35, 1.0)
    else:
        direction_fit = float(np.clip(0.5 + expected_sign * delta / 0.5, 0.0, 1.0))
    target = target_energy / 10.0 if target_energy > 1.0 else target_energy
    return {
        "tempo_distance": min(abs(math.log2(max(destination.bpm, 1.0) / max(source.bpm, 1.0))), 1.0),
        "harmonic_distance": camelot_distance(source.camelot, destination.camelot),
        "energy_delta": delta,
        "energy_target_error": min(abs(destination.energy - target), 1.0),
        "loudness_distance": min(abs(destination.loudness_db - source.loudness_db) / 18.0, 1.0),
        "vocal_overlap": source.vocal_activity * destination.vocal_activity,
        "spectral_distance": min(abs(destination.spectral_centroid - source.spectral_centroid) / 4000.0, 1.0),
        "phrase_confidence": (source.phrase_confidence + destination.phrase_confidence) / 2.0,
        "direction_fit": direction_fit,
    }


HEURISTIC_WEIGHTS = {
    "tempo_distance": -2.2,
    "harmonic_distance": -1.8,
    "energy_delta": 0.15,
    "energy_target_error": -1.7,
    "loudness_distance": -0.8,
    "vocal_overlap": -0.9,
    "spectral_distance": -0.45,
    "phrase_confidence": 0.7,
    "direction_fit": 1.4,
}


class TransitionRanker:
    """Serve an XGBoost ranker when available, otherwise a visible baseline."""

    def __init__(self, model=None, metadata: dict | None = None):
        self.model = model
        self.metadata = metadata or {"model_type": "heuristic-v1", "feature_names": list(FEATURE_NAMES)}

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_type", "xgboost-ranker"))

    @classmethod
    def load(cls, model_dir: str | Path) -> "TransitionRanker":
        directory = Path(model_dir)
        metadata_path = directory / "metadata.json"
        model_path = directory / "ranker.json"
        if not metadata_path.exists() or not model_path.exists():
            return cls()
        try:
            from xgboost import XGBRanker
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Could not load XGBoost; install the ML extra and its OpenMP runtime") from exc
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Model feature schema does not match this application version")
        model = XGBRanker()
        model.load_model(model_path)
        return cls(model, metadata)

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
        if self.model is None:
            scores = matrix @ np.asarray([HEURISTIC_WEIGHTS[name] for name in FEATURE_NAMES])
            contributions = matrix * np.asarray([HEURISTIC_WEIGHTS[name] for name in FEATURE_NAMES])
        else:
            scores = self.model.predict(matrix)
            contributions = self._contributions(matrix)
        result = []
        for candidate, score, row, contribution in zip(candidates, scores, rows, contributions):
            explanation = sorted(
                ({"feature": name, "contribution": round(float(value), 6)} for name, value in zip(FEATURE_NAMES, contribution)),
                key=lambda item: abs(item["contribution"]),
                reverse=True,
            )[:3]
            result.append(RankedCandidate(candidate.track_id, round(float(score), 6), row, explanation))
        return sorted(result, key=lambda item: item.score, reverse=True)

    def _contributions(self, matrix: np.ndarray) -> np.ndarray:
        try:
            from xgboost import DMatrix

            values = self.model.get_booster().predict(DMatrix(matrix, feature_names=list(FEATURE_NAMES)), pred_contribs=True)
            return values[:, :-1]
        except (ImportError, ValueError):  # Older XGBoost wrappers can omit feature names.
            return np.zeros_like(matrix)


def write_feature_catalog(items: Iterable[MusicFeatures], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [item.to_dict() for item in items]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[item.name for item in fields(MusicFeatures)])
        writer.writeheader()
        writer.writerows(rows)


def read_feature_catalog(path: str | Path) -> list[MusicFeatures]:
    numeric = {item.name for item in fields(MusicFeatures)} - {"track_id", "path", "key", "camelot"}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [MusicFeatures.from_dict({key: float(value) if key in numeric else value for key, value in row.items()}) for row in rows]


def build_labeling_rows(
    catalog: Sequence[MusicFeatures], target_energy: float = 0.7, direction: str = "maintain"
) -> list[dict]:
    rows: list[dict] = []
    for source in catalog:
        for destination in catalog:
            if source.track_id == destination.track_id:
                continue
            features = transition_features(source, destination, target_energy, direction)
            rows.append({
                "context_id": source.track_id,
                "source_id": source.track_id,
                "candidate_id": destination.track_id,
                "target_energy": target_energy,
                "direction": direction,
                "relevance": "",
                **{name: round(features[name], 8) for name in FEATURE_NAMES},
            })
    return rows


def write_labeling_csv(rows: Sequence[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["context_id", "source_id", "candidate_id", "target_energy", "direction", "relevance", *FEATURE_NAMES]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
