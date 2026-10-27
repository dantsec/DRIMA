"""Unit tests for DRIMA movement pattern normalization schemes (Section 3.2)."""

import unittest

from src.drima.normalization import (
    normalize_equal_displacement,
    normalize_pattern,
    normalize_proportional,
)


class TestNormalization(unittest.TestCase):
    """Tests for `normalize_equal_displacement` and `normalize_proportional`."""

    def test_equal_displacement_single_pass_excess(self) -> None:
        """When all positive fields exceed the equal displacement, a single pass suffices."""
        # 8 directions with 0.125, East increased by 0.08 -> sum = 1.08
        # Equal displacement across 8 positive fields = 0.08 / 8 = 0.01
        raw = [0.205, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0]
        norm = normalize_equal_displacement(raw, mobility="F")
        self.assertAlmostEqual(sum(norm), 1.0, places=12)
        self.assertAlmostEqual(norm[0], 0.195, places=9)
        for idx in range(1, 8):
            self.assertAlmostEqual(norm[idx], 0.115, places=9)
        self.assertEqual(norm[8], 0.0)

    def test_equal_displacement_multi_iteration_clamping(self) -> None:
        """When small positive fields clamp to 0, normalization iterates until sum == 1.0."""
        # Suppose sum = 1.20 across 4 positive fields: [0.90, 0.25, 0.03, 0.02, 0, 0, 0, 0, 0]
        # Pass 1: excess = 0.20, k = 4 -> subtract 0.05 -> [0.85, 0.20, -0.02->0, -0.03->0, ...]
        # Sum after clamping = 1.05, remaining positive k = 2
        # Pass 2: excess = 0.05, k = 2 -> subtract 0.025 -> [0.825, 0.175, 0, ...]
        raw = [0.90, 0.25, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
        norm = normalize_equal_displacement(raw, mobility="F")
        self.assertAlmostEqual(sum(norm), 1.0, places=12)
        self.assertAlmostEqual(norm[0], 0.825, places=9)
        self.assertAlmostEqual(norm[1], 0.175, places=9)
        for idx in range(2, 9):
            self.assertEqual(norm[idx], 0.0)

    def test_equal_displacement_deficit_below_one(self) -> None:
        """When total probability < 1.0 (e.g., two-way decrease), deficit is added to >0 fields."""
        raw = [0.40, 0.20, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # sum = 0.70, k = 3
        # Deficit = -0.30, divided by 3 positive fields -> +0.10 each -> [0.50, 0.30, 0.20, 0...]
        norm = normalize_equal_displacement(raw, mobility="F")
        self.assertAlmostEqual(sum(norm), 1.0, places=12)
        self.assertAlmostEqual(norm[0], 0.50, places=9)
        self.assertAlmostEqual(norm[1], 0.30, places=9)
        self.assertAlmostEqual(norm[2], 0.20, places=9)
        for idx in range(3, 9):
            self.assertEqual(norm[idx], 0.0)

    def test_proportional_and_none_methods(self) -> None:
        """Verify proportional normalization preserves ratios and 'none' clamps to [0, 1]."""
        raw = [0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        prop = normalize_proportional(raw, mobility="F")
        self.assertAlmostEqual(prop[0], 0.5, places=9)
        self.assertAlmostEqual(prop[1], 0.5, places=9)

        unnorm = normalize_pattern(raw, mobility="F", method="none")
        self.assertAlmostEqual(unnorm[0], 0.6, places=9)
        self.assertAlmostEqual(unnorm[1], 0.6, places=9)


if __name__ == "__main__":
    unittest.main()
