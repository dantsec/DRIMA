"""Unit tests for the 2D periodic grid, Moore neighborhood distance, and agent movement."""

import random
import unittest

from src.drima.grid import (
    get_neighbors,
    moore_distance,
    move_agent,
    wrap_position,
)
from src.drima.models import Agent


class TestGridAndNeighborhoods(unittest.TestCase):
    """Tests for `wrap_position`, `moore_distance`, `get_neighbors`, and `move_agent`."""

    def test_periodic_boundary_wrapping(self) -> None:
        """Verify 1-indexed toroidal wrapping across all 4 edges and corners."""
        # Standard interior position
        self.assertEqual(wrap_position(3, 4, x_max=5, y_max=10), (3, 4))
        # Wrap East -> West and West -> East
        self.assertEqual(wrap_position(6, 4, x_max=5, y_max=10), (1, 4))
        self.assertEqual(wrap_position(0, 4, x_max=5, y_max=10), (5, 4))
        # Wrap North -> South and South -> North
        self.assertEqual(wrap_position(3, 11, x_max=5, y_max=10), (3, 1))
        self.assertEqual(wrap_position(3, 0, x_max=5, y_max=10), (3, 10))
        # Corner wrap-around
        self.assertEqual(wrap_position(0, 0, x_max=5, y_max=10), (5, 10))
        self.assertEqual(wrap_position(6, 11, x_max=5, y_max=10), (1, 1))

    def test_moore_distance_isotropic_and_periodic(self) -> None:
        """Moore distance should be 0 on same cell, 1 on contiguous/wrapped cells."""
        self.assertEqual(moore_distance((2, 2), (2, 2), x_max=8, y_max=5), 0)
        self.assertEqual(moore_distance((2, 2), (3, 3), x_max=8, y_max=5), 1)
        # Opposite corners (1, 1) and (8, 5) on an 8x5 periodic grid are diagonal neighbors (dist=1)
        self.assertEqual(moore_distance((1, 1), (8, 5), x_max=8, y_max=5), 1)
        self.assertEqual(moore_distance((1, 3), (8, 3), x_max=8, y_max=5), 1)
        self.assertEqual(moore_distance((1, 1), (4, 3), x_max=8, y_max=5), 3)

    def test_get_neighbors_range_zero_and_range_one(self) -> None:
        """Verify R=0 only matches same cell, while R=1 includes contiguous and wrapped cells."""
        a1 = Agent.create_r_agent("R01", position=(1, 1), mobility="F")
        a_same = Agent.create_r_agent("R02", position=(1, 1), mobility="F")
        a_wrap = Agent.create_d_agent("D01", direction="E", position=(8, 5), mobility="F")
        a_far = Agent.create_r_agent("R03", position=(4, 3), mobility="F")
        all_agents = [a1, a_same, a_wrap, a_far]

        # Range R = 0: only a_same is at (1, 1)
        neighbors_r0 = get_neighbors(a1, all_agents, x_max=8, y_max=5, interaction_range=0)
        self.assertEqual([n.name for n in neighbors_r0], ["R02"])

        # Range R = 1: both a_same (dist 0) and a_wrap at (8, 5) (dist 1) are neighbors
        neighbors_r1 = get_neighbors(a1, all_agents, x_max=8, y_max=5, interaction_range=1)
        self.assertEqual({n.name for n in neighbors_r1}, {"R02", "D01"})

    def test_move_agent_shared_and_exclusive_cells(self) -> None:
        """Test deterministic movement across boundary and exclusive cell collision handling."""
        rng = random.Random(42)
        d01 = Agent.create_d_agent("D01", direction="NW", position=(1, 5), mobility="F")
        # From (1, 5) on a 5x5 grid, NW is (-1, +1) -> (0, 6) -> wraps to (5, 1)
        rec = move_agent(d01, x_max=5, y_max=5, rng=rng, allow_shared_cells=True)
        self.assertEqual(d01.position, (5, 1))
        self.assertEqual(rec.direction, "NW")
        self.assertFalse(rec.blocked_by_occupancy)

        # Exclusive occupancy: if target cell (5, 1) is already occupied, move is blocked
        d02 = Agent.create_d_agent("D02", direction="NW", position=(1, 5), mobility="F")
        occupied = {(1, 5), (5, 1)}
        rec_blocked = move_agent(
            d02,
            x_max=5,
            y_max=5,
            rng=rng,
            occupied_positions=occupied,
            allow_shared_cells=False,
        )
        self.assertEqual(d02.position, (1, 5))
        self.assertTrue(rec_blocked.blocked_by_occupancy)


if __name__ == "__main__":
    unittest.main()
