"""Unit tests for DRIMA direct binary interactions (Section 3.1 & Section 3.2)."""

import unittest

from src.drima.interactions import (
    compute_delta_p,
    determine_modifiability,
    interact_pair,
    resolve_interaction_types,
)
from src.drima.models import Agent, SimulationConfig


class TestInteractions(unittest.TestCase):
    """Tests for interaction types, delta_p equations, clamping, and R-agent convergence."""

    def test_conflict_resolution_and_modifiability(self) -> None:
        """DD prevails over dd; RR prevails over rr; dR/Rd modifies R but not D."""
        resolved = resolve_interaction_types(["dd", "DD", "rr", "RR", "dR"])
        self.assertIn("DD", resolved)
        self.assertNotIn("dd", resolved)
        self.assertIn("RR", resolved)
        self.assertNotIn("rr", resolved)

        d1 = Agent.create_d_agent("D01", direction="E", position=(1, 1))
        d2 = Agent.create_d_agent("D02", direction="W", position=(1, 2))
        r1 = Agent.create_r_agent("R01", position=(1, 1))
        r2 = Agent.create_r_agent("R02", position=(1, 2))

        default_active = resolve_interaction_types(["dd", "RR", "dR", "Rd"])
        self.assertEqual(determine_modifiability(d1, d2, default_active), (False, False))
        self.assertEqual(determine_modifiability(r1, r2, default_active), (True, True))
        self.assertEqual(determine_modifiability(d1, r1, default_active), (False, True))
        self.assertEqual(determine_modifiability(r1, d1, default_active), (True, False))

        # Empty interaction_types disables all modifications
        self.assertEqual(determine_modifiability(r1, r2, set()), (False, False))

    def test_compute_delta_p_fixed_and_variable(self) -> None:
        """Verify Equations (1) and (2) from Section 3.2."""
        # Fixed mode: F_p / 9
        self.assertAlmostEqual(
            compute_delta_p(0.125, 1.0, fp=1.0, delta_mode="fixed"),
            1.0 / 9.0,
            places=12,
        )
        # Variable mode: (F_p / 9) * |P_i - P_j|
        # When |P_i - P_j| = 8/9 and F_p = 1, delta_p = 8/81
        self.assertAlmostEqual(
            compute_delta_p(1.0 / 9.0, 1.0, fp=1.0, delta_mode="variable"),
            (1.0 / 9.0) * (8.0 / 9.0),
            places=12,
        )

    def test_clamping_smaller_probability_cannot_exceed_larger(self) -> None:
        """Section 3.2: smaller probability cannot grow larger than the pre-interaction higher value."""
        # Give R01 0.20 in East and R02 0.22 in East, with a high F_p=5 (delta_p = 5/9 = 0.555)
        r01 = Agent.create_r_agent(
            "R01",
            position=(1, 1),
            mobility="F",
            movement_pattern=[0.20, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.0],
        )
        r02 = Agent.create_r_agent(
            "R02",
            position=(1, 1),
            mobility="F",
            movement_pattern=[0.22, 0.18, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.0],
        )
        config = SimulationConfig(
            fp=5.0,
            delta_mode="fixed",
            normalization_method="none",
            world_mobility="F",
        )
        interact_pair(r01, r02, config)
        # Without clamping, r01[0] would jump to 0.20 + 0.555 = 0.755; with clamping, it caps at 0.22
        self.assertAlmostEqual(r01.movement_pattern[0], 0.22, places=9)
        # And in NE (index 1), r02 had 0.18 < 0.20, so r02[1] caps at 0.20!
        self.assertAlmostEqual(r02.movement_pattern[1], 0.20, places=9)

    def test_r_agent_converges_to_d_agent_over_repeated_interactions(self) -> None:
        """Section 2.1 & 3.2: Repeated interactions between an R-agent and a D-agent converge R to D."""
        d01 = Agent.create_d_agent("D01", direction="E", position=(2, 2), mobility="F")
        r01 = Agent.create_r_agent("R01", position=(2, 2), mobility="F")
        config = SimulationConfig(
            fp=1.0,
            delta_mode="fixed",
            world_mobility="F",
            normalization_method="equal_displacement",
        )

        for _ in range(100):
            interact_pair(d01, r01, config)

        # D01 remains completely unchanged
        self.assertEqual(d01.movement_pattern, [1.0, 0, 0, 0, 0, 0, 0, 0, 0])
        # R01 has converged to moving deterministically East (probability 1.0)
        self.assertAlmostEqual(r01.movement_pattern[0], 1.0, places=6)
        for idx in range(1, 9):
            self.assertAlmostEqual(r01.movement_pattern[idx], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
