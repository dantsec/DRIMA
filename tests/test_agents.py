import unittest
from src.main import Agent

class TestAgents(unittest.TestCase):
    def test_agent_initialization(self):
        """Test that agents initialize correctly with agent_id, position, and movement_pattern."""
        agent = Agent(agent_id="D1", position=(0, 0), movement_pattern={"E": 1, "X": 0})
        self.assertEqual(agent.agent_id, "D1")
        self.assertEqual(agent.position, (0, 0))

    def test_agent_movement(self):
        """Test that agents move according to their movement pattern."""
        agent = Agent(agent_id="R1", position=(0, 0), movement_pattern={"E": 0.5, "X": 0.5})
        agent.move(grid_size=(10, 10), world_mobility='F')
        self.assertIn(agent.position, [(1, 0), (0, 0)])

if __name__ == "__main__":
    unittest.main()
