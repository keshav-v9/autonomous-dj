from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from autonomous_dj.feedback import append_feedback
from autonomous_dj.ml import (
    MusicFeatures,
    TransitionRanker,
    build_labeling_rows,
    camelot_distance,
    read_feature_catalog,
    transition_features,
    write_feature_catalog,
    write_labeling_csv,
)
from autonomous_dj.training import train_ranker

try:
    import xgboost  # noqa: F401
except Exception:
    HAS_XGBOOST = False
else:
    HAS_XGBOOST = True


def track(track_id: str, bpm: float, camelot: str, energy: float) -> MusicFeatures:
    return MusicFeatures(
        track_id=track_id,
        path=f"/{track_id}.wav",
        duration_seconds=240.0,
        bpm=bpm,
        key=camelot,
        camelot=camelot,
        energy=energy,
        loudness_db=-12.0,
        vocal_activity=0.25,
        spectral_centroid=2000.0,
        phrase_seconds=8.0,
        phrase_confidence=0.9,
    )


class MLFeatureTests(unittest.TestCase):
    def test_camelot_distance_recognizes_exact_and_adjacent_keys(self) -> None:
        self.assertEqual(camelot_distance("8A", "8A"), 0.0)
        self.assertLess(camelot_distance("8A", "9A"), camelot_distance("8A", "2B"))

    def test_direction_changes_candidate_features(self) -> None:
        source = track("source", 122, "8A", 0.5)
        higher = track("higher", 124, "8A", 0.8)
        up = transition_features(source, higher, 8.0, "energy_up")
        down = transition_features(source, higher, 4.0, "energy_down")
        self.assertGreater(up["direction_fit"], down["direction_fit"])

    def test_heuristic_fallback_ranks_compatible_track_first(self) -> None:
        source = track("source", 122, "8A", 0.6)
        compatible = track("compatible", 123, "8A", 0.7)
        poor = track("poor", 82, "2B", 0.2)
        ranked = TransitionRanker().rank(source, [poor, compatible], 7.0)
        self.assertEqual(ranked[0].track_id, "compatible")
        self.assertTrue(ranked[0].explanation)
        self.assertEqual(TransitionRanker().rank(source, [], 7.0), [])

    def test_catalog_and_labeling_round_trip(self) -> None:
        items = [track("a", 120, "8A", 0.5), track("b", 122, "9A", 0.6)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "catalog.csv"
            write_feature_catalog(items, path)
            loaded = read_feature_catalog(path)
            self.assertEqual([item.track_id for item in loaded], ["a", "b"])
            rows = build_labeling_rows(loaded)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["relevance"], "")

    def test_feedback_is_training_compatible(self) -> None:
        source = track("source", 122, "8A", 0.5)
        winner = track("winner", 123, "8A", 0.7)
        loser = track("loser", 98, "2B", 0.2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "feedback.csv"
            count = append_feedback(path, "set-1", source, [(winner, 2), (loser, 0)], 7.0, "maintain")
            self.assertEqual(count, 2)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["relevance"] for row in rows], ["2", "0"])
            self.assertIn("harmonic_distance", rows[0])

    def test_runtime_queue_uses_ranker_contract(self) -> None:
        from autonomous_dj.server import ranked_queue

        queue = ranked_queue()
        self.assertGreaterEqual(len(queue), 2)
        self.assertIn("score", queue[0])
        self.assertIn("explanation", queue[0])

    @unittest.skipUnless(HAS_XGBOOST, "XGBoost/OpenMP not installed")
    def test_pairwise_training_persists_a_reloadable_model(self) -> None:
        items = [track(str(index), 120 + index, f"{index + 1}A", 0.4 + index * 0.08) for index in range(6)]
        rows = build_labeling_rows(items)
        for row in rows:
            source = int(row["source_id"])
            candidate = int(row["candidate_id"])
            row["relevance"] = 2 if candidate == (source + 1) % 6 else (1 if abs(candidate - source) == 1 else 0)
        with tempfile.TemporaryDirectory() as folder:
            labels = Path(folder) / "labels.csv"
            model_dir = Path(folder) / "model"
            write_labeling_csv(rows, labels)
            metadata = train_ranker(labels, model_dir)
            loaded = TransitionRanker.load(model_dir)
            ranked = loaded.rank(items[0], items[1:], 7.0)
            self.assertEqual(metadata["model_type"], loaded.model_name)
            self.assertTrue(ranked)
            self.assertGreater(metadata["metrics"]["evaluated_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
