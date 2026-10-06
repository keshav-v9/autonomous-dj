from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_fma_cloud_dataset import build_catalog, build_training_rows


class FMACloudDatasetTests(unittest.TestCase):
    def _metadata(self, root: Path, count: int = 10) -> None:
        index = np.arange(1, count + 1)
        tracks = pd.DataFrame(
            {
                ("track", "genre_top"): ["Electronic"] * count,
                ("track", "duration"): [30 + item for item in index],
                ("track", "title"): [f"Track {item}" for item in index],
                ("track", "license"): ["CC BY 4.0"] * count,
                ("artist", "name"): [f"Artist {item}" for item in index],
            },
            index=index,
        )
        audio_columns = pd.MultiIndex.from_tuples([
            ("echonest", "audio_features", "tempo"),
            ("echonest", "audio_features", "energy"),
            ("echonest", "audio_features", "instrumentalness"),
            ("echonest", "audio_features", "danceability"),
        ])
        echonest = pd.DataFrame(
            np.column_stack([
                115 + index,
                np.linspace(0.35, 0.9, count),
                np.linspace(0.1, 0.8, count),
                np.linspace(0.4, 0.85, count),
            ]),
            index=index,
            columns=audio_columns,
        )
        feature_columns = pd.MultiIndex.from_tuples([
            *[("chroma_cens", "mean", f"{item:02d}") for item in range(1, 13)],
            ("rmse", "mean", "01"),
            ("spectral_centroid", "mean", "01"),
        ])
        rows = []
        profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        for item in index:
            rows.append([*np.roll(profile, item % 12), 0.1 + item / 100, 1600 + item * 90])
        features = pd.DataFrame(rows, index=index, columns=feature_columns)
        tracks.to_csv(root / "tracks.csv")
        echonest.to_csv(root / "echonest.csv")
        features.to_csv(root / "features.csv")

    def test_catalog_and_weak_preferences_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._metadata(root)
            catalog, manifest = build_catalog(root, limit=8, genre="Electronic", seed=42)
            labels = build_training_rows(catalog, candidates_per_context=6, seed=42)
            repeated, _ = build_catalog(root, limit=8, genre="Electronic", seed=42)
            self.assertEqual([item.track_id for item in catalog], [item.track_id for item in repeated])
            self.assertEqual(len(catalog), 8)
            self.assertEqual(len(manifest), 8)
            self.assertEqual(len(labels), 48)
            self.assertEqual({row["relevance"] for row in labels}, {0, 1, 2})
            self.assertTrue(all(item.path.startswith("fma://") for item in catalog))


if __name__ == "__main__":
    unittest.main()
