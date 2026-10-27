"""Unit tests for DRIMA core domain models (`Agent`, `World`, `AgentProperties`, `SimulationConfig`).

Covers Section 3.1 of the DRIMA specification, including Full (`F`) and Partial (`P`) world
mobility, canonical nested-list representations, and edge-case validations.
"""

import unittest

from src.drima.models import (
    Agent,
    AgentProperties,
    SimulationConfig,
    World,
)


class TestAgentAndWorldModels(unittest.TestCase):
    """Tests for `AgentProperties`, `Agent`, `World`, and `SimulationConfig`."""

    def test_create_d_agent_full_and_partial_mobility(self) -> None:
        """D-agent should have 1.0 in its active direction and 1000 in its transition potential."""
        d_full = Agent.create_d_agent(
            name="D01", direction="NW", position=(2, 2), mobility="F"
        )
        self.assertEqual(d_full.agent_type, "D")
        self.assertEqual(d_full.dominant_direction, "NW")
        self.assertEqual(d_full.movement_pattern, [0, 0, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(
            d_full.properties.transition_potential_pattern,
            [0, 0, 0, 1000, 0, 0, 0, 0, 0],
        )
        # Verify canonical representation from Section 3.1 (lines 147-148)
        self.assertEqual(
            d_full.to_drima_list(),
            [["D01", [0, 0, 0, 1, 0, 0, 0, 0, 0], [[0, 0, 0, 1000, 0, 0, 0, 0, 0]]], [2, 2]],
        )

        # Partial mobility allows X (no-move, index 8) to be 1.0 (lines 155-156)
        d_partial = Agent.create_d_agent(
            name="D01", direction="X", position=(2, 2), mobility="P"
        )
        self.assertEqual(d_partial.movement_pattern, [0, 0, 0, 0, 0, 0, 0, 0, 1])

        # Full mobility must reject direction X for D-agent
        with self.assertRaises(ValueError):
            Agent.create_d_agent(name="D02", direction="X", position=(1, 1), mobility="F")

    def test_create_r_agent_full_and_partial_mobility(self) -> None:
        """R-agent initializes uniformly: 0.125 in 8 dirs for F, 1/9 in 9 dirs for P."""
        r_full = Agent.create_r_agent(name="R01", position=(4, 6), mobility="F")
        self.assertEqual(r_full.agent_type, "R")
        self.assertEqual(r_full.movement_pattern, [0.125] * 8 + [0.0])
        self.assertAlmostEqual(sum(r_full.movement_pattern), 1.0, places=9)

        r_partial = Agent.create_r_agent(name="R01", position=(4, 6), mobility="P")
        for prob in r_partial.movement_pattern:
            self.assertAlmostEqual(prob, 1.0 / 9.0, places=9)
        self.assertAlmostEqual(sum(r_partial.movement_pattern), 1.0, places=9)

    def test_agent_validation_edge_cases(self) -> None:
        """Invalid names, negative probabilities, bad sums, or 0-indexed coords raise ValueError."""
        with self.assertRaises(ValueError):
            Agent(name="X01", movement_pattern=[0.125] * 8 + [0.0], position=(1, 1))

        with self.assertRaises(ValueError):
            Agent(name="R01", movement_pattern=[0.5, 0.5], position=(1, 1))

        with self.assertRaises(ValueError):
            Agent(
                name="R01",
                movement_pattern=[-0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0],
                position=(1, 1),
            )

        with self.assertRaises(ValueError):
            Agent(
                name="R01",
                movement_pattern=[0.5] * 8 + [0.0],
                position=(1, 1),
            )

        with self.assertRaises(ValueError):
            Agent(
                name="R01",
                movement_pattern=[0.125] * 8 + [0.0],
                position=(0, 1),
            )

    def test_world_representation_and_deserialization(self) -> None:
        """Verify exact Section 3.1 World Representation example (lines 163-169 of paper)."""
        paper_world_list = [
            [
                [
                    [
                        "D01",
                        [0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [[0, 0, 0, 1000, 0, 0, 0, 0, 0]],
                    ],
                    [2, 2],
                ],
                [
                    [
                        "R01",
                        [0, 0.13, 0.14, 0.11, 0.25, 0.15, 0.10, 0.12, 0],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    ],
                    [5, 3],
                ],
                [
                    [
                        "R02",
                        [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    ],
                    [4, 6],
                ],
            ],
            [5, 10],
            ["dd", "RR", "dR", "Rd"],
            "F",
        ]
        world = World.from_dict(paper_world_list)
        self.assertEqual(world.grid_size, (5, 10))
        self.assertEqual(world.world_mobility, "F")
        self.assertEqual(world.interaction_types, ["dd", "RR", "dR", "Rd"])
        self.assertEqual(len(world.agents), 3)
        self.assertEqual(world.get_agent("R01").dominant_direction, "W")

        # Cloning must produce an independent deep copy
        cloned = world.clone()
        cloned.agents[1].movement_pattern[0] = 0.5
        self.assertEqual(world.agents[1].movement_pattern[0], 0.0)

    def test_world_validation_edge_cases(self) -> None:
        """World rejects out-of-bounds positions, duplicate names, and non-zero X in F mode."""
        r01 = Agent.create_r_agent("R01", position=(6, 1), mobility="F")
        with self.assertRaises(ValueError):
            World(agents=[r01], grid_size=(5, 5), world_mobility="F")

        r01_a = Agent.create_r_agent("R01", position=(2, 2), mobility="F")
        r01_b = Agent.create_r_agent("R01", position=(3, 3), mobility="F")
        with self.assertRaises(ValueError):
            World(agents=[r01_a, r01_b], grid_size=(5, 5), world_mobility="F")

        r_partial = Agent.create_r_agent("R02", position=(2, 2), mobility="P")
        with self.assertRaises(ValueError):
            World(agents=[r_partial], grid_size=(5, 5), world_mobility="F")

    def test_simulation_config_validation(self) -> None:
        """SimulationConfig validates grid dimensions, exclusive cell capacity, and modes."""
        with self.assertRaises(ValueError):
            SimulationConfig(
                grid_width=2,
                grid_height=2,
                num_d_agents=3,
                num_r_agents=3,
                allow_shared_cells=False,
            )
        with self.assertRaises(ValueError):
            SimulationConfig(interaction_range=-1)
        with self.assertRaises(ValueError):
            SimulationConfig(delta_mode="invalid")


if __name__ == "__main__":
    unittest.main()
