import unittest
from src.main import Interaction, Agent

class TestInteractions(unittest.TestCase):
    def test_interaction_range(self):
        """Test that interactions occur within the defined range."""
        interaction = Interaction(interaction_type='RR', intensity=1.0, mode='fixed')
        agent1 = Agent(agent_id="R1", position=(0, 0), movement_pattern={"E": 0.5, "X": 0.5})
        agent2 = Agent(agent_id="R2", position=(0, 1), movement_pattern={"E": 0.5, "X": 0.5})
        self.assertTrue(interaction._calculate_delta(0.5, 0.5) >= 0)

    def test_interaction_effect(self):
        """Test that interactions modify agent movement patterns correctly."""
        interaction = Interaction(interaction_type='RR', intensity=1.0, mode='fixed')
        agent1 = Agent(agent_id="R1", position=(0, 0), movement_pattern={"E": 0.5, "X": 0.5})
        agent2 = Agent(agent_id="R2", position=(0, 1), movement_pattern={"E": 0.2, "X": 0.8})
        interaction.process(agent1, agent2)
        self.assertAlmostEqual(agent1.movement_pattern["E"], 0.5, places=2)

if __name__ == "__main__":
    unittest.main()
