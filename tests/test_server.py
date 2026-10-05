from __future__ import annotations

import unittest

from autonomous_dj.server import render_selected_mix


class MixAPITests(unittest.TestCase):
    def test_selected_tracks_render_with_visualization_data(self) -> None:
        result = render_selected_mix("soft-focus", "glass-horizon", "bass_swap", 4)
        self.assertTrue(result["ok"])
        self.assertEqual(result["source"]["id"], "soft-focus")
        self.assertEqual(result["destination"]["id"], "glass-horizon")
        self.assertEqual(len(result["waveform"]), 160)
        self.assertTrue(result["url"].endswith("soft-focus-to-glass-horizon.wav"))

    def test_same_track_cannot_be_mixed_into_itself(self) -> None:
        with self.assertRaisesRegex(ValueError, "different songs"):
            render_selected_mix("afterimage", "afterimage", "eq_blend", 8)


if __name__ == "__main__":
    unittest.main()
