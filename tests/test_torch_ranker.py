from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autonomous_dj.ml import MusicFeatures, write_labeling_csv
from autonomous_dj.torch_ranker import TorchTransitionRanker, train_torch_ranker
from scripts.build_fma_cloud_dataset import build_training_rows

try:
    import torch  # noqa: F401
except Exception:
    HAS_TORCH = False
else:
    HAS_TORCH = True


def track(index: int) -> MusicFeatures:
    return MusicFeatures(
        track_id=f"track-{index}",
        path="",
        duration_seconds=180.0,
        bpm=116.0 + index,
        key=f"{index + 1}A",
        camelot=f"{index + 1}A",
        energy=0.3 + index * 0.07,
        loudness_db=-18.0 + index,
        vocal_activity=0.15 + index * 0.05,
        spectral_centroid=1500.0 + index * 180,
        phrase_seconds=8.0,
        phrase_confidence=0.65 + index * 0.03,
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TorchRankerTests(unittest.TestCase):
    def test_ranknet_trains_exports_and_reloads(self) -> None:
        catalog = [track(index) for index in range(8)]
        rows = build_training_rows(catalog, candidates_per_context=6, seed=42)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = root / "labels.csv"
            model_dir = root / "model"
            write_labeling_csv(rows, labels)
            metadata = train_torch_ranker(labels, model_dir, seed=42, epochs=2, batch_size=32)
            ranker = TorchTransitionRanker.load(model_dir)
            ranked = ranker.rank(catalog[0], catalog[1:], 0.7, "energy_up")
            self.assertEqual(metadata["model_type"], "pytorch-ranknet-v1")
            self.assertTrue((model_dir / "ranknet.pt").exists())
            self.assertEqual(len(ranked), 7)
            self.assertTrue(ranked[0].explanation)


if __name__ == "__main__":
    unittest.main()
