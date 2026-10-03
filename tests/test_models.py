import unittest

from autonomous_dj.models import TransitionPlan


class TransitionPlanTests(unittest.TestCase):
    def test_sixteen_bar_duration(self) -> None:
        plan = TransitionPlan("a", "b", 0, 0, transition_bars=16, target_bpm=120)
        self.assertEqual(plan.duration_seconds, 32.0)


if __name__ == "__main__":
    unittest.main()

