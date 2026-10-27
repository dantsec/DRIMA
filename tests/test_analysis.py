"""Unit tests for DRIMA temporal series extraction (`src/drima/analysis.py`) and `analysis.ipynb`."""

import json
from pathlib import Path
import unittest

from src.drima.analysis import (
    extract_agent_probability_series,
    extract_convergence_series,
    extract_interaction_series,
    extract_position_series,
    load_or_validate_history,
)
from src.drima.models import DIRECTIONS, SimulationConfig
from src.drima.simulation import run_simulation


class TestAnalysisAndNotebook(unittest.TestCase):
    """Tests for temporal series extraction functions and notebook validity."""

    def setUp(self) -> None:
        """Create a deterministic 15-step simulation result for testing."""
        self.config = SimulationConfig(
            grid_width=6,
            grid_height=5,
            num_d_agents=2,
            num_r_agents=4,
            d_directions=["E", "SE"],
            iterations=15,
            seed=42,
            log_file=None,
        )
        self.result = run_simulation(self.config, write_output=False)
        self.payload = self.result.payload

    def test_extract_agent_probability_series(self) -> None:
        """Probability series has length iterations+1 for all 9 directions per agent."""
        prob_series = extract_agent_probability_series(self.payload)
        self.assertIn("D01", prob_series)
        self.assertIn("R01", prob_series)

        for name, agent_series in prob_series.items():
            self.assertEqual(len(agent_series["step"]), 16)
            for d_name in DIRECTIONS:
                self.assertEqual(len(agent_series[d_name]), 16)
            # At every step t, probabilities for each agent must sum to 1.0
            for t_idx in range(16):
                total_t = sum(agent_series[d][t_idx] for d in DIRECTIONS)
                self.assertAlmostEqual(
                    total_t,
                    1.0,
                    places=6,
                    msg=f"Agent {name} probabilities at step {t_idx} did not sum to 1.0",
                )

    def test_extract_convergence_and_interaction_series(self) -> None:
        """Convergence and interaction series match step count and cumulative invariants."""
        conv = extract_convergence_series(self.payload)
        self.assertEqual(conv["steps"], list(range(16)))
        self.assertEqual(len(conv["r_mean_max_prob"]), 16)
        self.assertAlmostEqual(conv["r_mean_max_prob"][0], 0.125, places=6)

        inter = extract_interaction_series(self.payload)
        self.assertEqual(len(inter["steps"]), 16)
        # Step 0 (initial world) has 0 interactions
        self.assertEqual(inter["total_interactions_per_step"][0], 0)
        self.assertEqual(
            inter["cumulative_total_interactions"][-1],
            sum(inter["total_interactions_per_step"]),
        )
        self.assertEqual(
            inter["cumulative_effective_interactions"][-1],
            sum(inter["effective_interactions_per_step"]),
        )

    def test_extract_position_series_and_invalid_inputs(self) -> None:
        """Position trajectories remain within 1-indexed grid bounds; invalid inputs raise errors."""
        pos = extract_position_series(self.payload)
        for name, coords in pos.items():
            self.assertEqual(len(coords), 16)
            for (x, y) in coords:
                self.assertTrue(1 <= x <= 6 and 1 <= y <= 5, f"Out of bounds for {name}: {(x, y)}")

        with self.assertRaises(ValueError):
            load_or_validate_history({"history": []})
        with self.assertRaises(TypeError):
            load_or_validate_history(12345)

    def test_analysis_notebook_valid_json_structure(self) -> None:
        """Verify `analysis.ipynb` is valid JSON conforming to Jupyter Notebook v4 schema."""
        nb_path = Path("analysis.ipynb")
        self.assertTrue(nb_path.exists())
        nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb_data.get("nbformat"), 4)
        cells = nb_data.get("cells", [])
        self.assertGreaterEqual(len(cells), 10)
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 5)

        # Ensure all code cells in analysis.ipynb compile without syntax errors
        for idx, cell in enumerate(code_cells):
            source_code = "".join(cell.get("source", []))
            compile(source_code, f"analysis.ipynb#cell-{idx}", "exec")


if __name__ == "__main__":
    unittest.main()
