from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from autonomous_dj.audio import STANDARD_SAMPLE_RATE, analyze_track, render_transition, synthesize_demo, waveform_peaks
from autonomous_dj.models import TransitionPlan


class AudioEngineTests(unittest.TestCase):
    def test_waveform_peaks_are_normalized_for_the_frontend(self) -> None:
        peaks = waveform_peaks(np.array([0.0, -0.5, 1.0, -0.25], dtype=np.float32), bins=2)
        self.assertEqual(peaks, [0.5, 1.0])

    def test_demo_analysis_finds_dance_tempo(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = synthesize_demo(Path(folder) / "track.wav", 120.0, 110.0, seconds=12)
            result = analyze_track(path)
            self.assertEqual(result.sample_rate, STANDARD_SAMPLE_RATE)
            self.assertGreater(result.duration_seconds, 11.9)
            self.assertLess(abs(result.bpm - 120.0), 5.0)
            self.assertGreater(len(result.beat_times), 15)

    def test_transition_contract_has_duration_and_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a = synthesize_demo(root / "a.wav", 120.0, 110.0, seconds=20)
            b = synthesize_demo(root / "b.wav", 124.0, 130.81, seconds=20)
            output = root / "transition.wav"
            plan = TransitionPlan("a", "b", 0.0, 0.0, transition_bars=4, target_bpm=122.0)
            metrics = render_transition(a, b, output, plan, source_bpm=120.0, destination_bpm=124.0)
            self.assertTrue(output.exists())
            self.assertEqual(metrics.clipped_samples, 0)
            self.assertLessEqual(metrics.peak_dbfs, -0.5)
            self.assertAlmostEqual(metrics.duration_seconds, plan.duration_seconds, places=2)
            with wave.open(str(output), "rb") as handle:
                self.assertEqual(handle.getframerate(), STANDARD_SAMPLE_RATE)
                self.assertEqual(handle.getnchannels(), 1)


if __name__ == "__main__":
    unittest.main()
