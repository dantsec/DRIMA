"""Unit tests for world updates (`sync` and `async`), CLI parsing, config merging, and JSON history output."""

from pathlib import Path
import tempfile
import unittest

from src.drima.cli import main as cli_main
from src.drima.config import build_simulation_config
from src.drima.io_handler import load_history_json
from src.drima.models import SimulationConfig
from src.drima.simulation import run_simulation
from src.drima.visualizer import display_history, display_world


class TestSimulationAndCLI(unittest.TestCase):
    """Tests for `sync_world_update`, `async_world_update`, CLI, and history JSON export."""

    def test_sync_update_at_most_one_interaction_per_agent_per_step(self) -> None:
        """Section 2.2.4: In sync mode, no agent participates in more than one interaction per step."""
        config = SimulationConfig(
            grid_width=3,
            grid_height=3,
            num_d_agents=2,
            num_r_agents=6,
            interaction_range=2,
            update_mode="sync",
            sync_only_unprocessed_neighbors=True,
            iterations=20,
            seed=123,
            log_file=None,
        )
        result = run_simulation(config, write_output=False)
        self.assertEqual(len(result.history), 21)

        for step_log in result.step_logs:
            participating = []
            for inter in step_log.interactions:
                participating.append(inter.agent_i_name)
                participating.append(inter.agent_j_name)
            # Every participating agent in this step is unique
            self.assertEqual(len(participating), len(set(participating)))

    def test_async_update_mode(self) -> None:
        """Section 2.2.4: Async mode selects at most one interacting pair per step."""
        config = SimulationConfig(
            grid_width=4,
            grid_height=4,
            num_d_agents=1,
            num_r_agents=4,
            interaction_range=2,
            update_mode="async",
            iterations=15,
            seed=99,
            log_file=None,
        )
        result = run_simulation(config, write_output=False)
        self.assertEqual(len(result.history), 16)
        for step_log in result.step_logs:
            self.assertLessEqual(len(step_log.interactions), 1)

    def test_config_file_loading_and_cli_override_precedence(self) -> None:
        """CLI flags must override values loaded from a JSON configuration file."""
        config = build_simulation_config(
            config_file="examples/figure2_experiment.json",
            cli_overrides={"iterations": 12, "fp": 2.5, "world_mobility": "P"},
        )
        self.assertEqual(config.grid_width, 8)
        self.assertEqual(config.grid_height, 5)
        self.assertEqual(config.iterations, 12)
        self.assertEqual(config.fp, 2.5)
        self.assertEqual(config.world_mobility, "P")

    def test_cli_end_to_end_creates_history_and_log_files(self) -> None:
        """End-to-end test of CLI execution, log file creation, and JSON history structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            history_path = Path(tmp_dir) / "test_history.json"
            log_path = Path(tmp_dir) / "test_drima.log"

            exit_code = cli_main(
                [
                    "--config",
                    "examples/section3_world.json",
                    "--iterations",
                    "10",
                    "--seed",
                    "42",
                    "--output",
                    str(history_path),
                    "--log-file",
                    str(log_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(history_path.exists())
            self.assertTrue(log_path.exists())

            loaded = load_history_json(str(history_path))
            self.assertEqual(loaded["metadata"]["total_steps"], 10)
            self.assertEqual(loaded["metadata"]["total_snapshots"], 11)
            self.assertEqual(len(loaded["history"]), 11)
            self.assertEqual(len(loaded["drima_history_representation"]), 11)

            log_content = log_path.read_text(encoding="utf-8")
            self.assertIn("DRIMA Simulation Initialized", log_content)
            self.assertIn("Simulation Completed", log_content)

    def test_visualizers_display_world_and_history(self) -> None:
        """Verify `display_world` and `display_history` produce formatted ASCII output."""
        config = SimulationConfig(
            grid_width=5,
            grid_height=4,
            num_d_agents=1,
            num_r_agents=2,
            d_directions=["E"],
            iterations=5,
            seed=11,
            log_file=None,
        )
        result = run_simulation(config, write_output=False)
        world_str = display_world(result.final_world)
        history_str = display_history(result.history)

        self.assertIn("D01", world_str)
        self.assertIn("R01", world_str)
        self.assertIn("DRIMA History Evolution", history_str)


if __name__ == "__main__":
    unittest.main()
