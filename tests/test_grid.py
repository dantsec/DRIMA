import unittest
from src.main import World, Agent

class TestGrid(unittest.TestCase):
    def test_grid_initialization(self):
        """Test that the world initializes correctly with periodic boundaries."""
        world = World(grid_size=(10, 10), interaction_range=1, world_mobility='F')
        self.assertEqual(world.grid_size, (10, 10))
        self.assertEqual(world.interaction_range, 1)
        self.assertEqual(world.world_mobility, 'F')

    def test_agent_placement(self):
        """Test that agents are added correctly to the world."""
        world = World(grid_size=(10, 10), interaction_range=1, world_mobility='F')
        agent = Agent(agent_id="D1", position=(5, 5), movement_pattern={"E": 1, "X": 0})
        world.add_agent(agent)
        self.assertIn(agent, world.agents)

if __name__ == "__main__":
    unittest.main()
