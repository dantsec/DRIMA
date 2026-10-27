"""High-level simulation runner orchestrating DRIMA world histories.

Project Overview
----------------
As described in Section 3 of the DRIMA specification, running an experiment consists of:
1. Initializing a world at `t = 0` with all agents, positions, movement patterns, and properties.
2. Iteratively applying `SyncWorldUpdate` or `AsyncWorldUpdate` for `iterations` steps (`Fold` /
   `Nest` in the original *Mathematica* implementation).
3. Recording every world snapshot into a `history` list, logging all events, and writing out
   the final JSON history file.

Example Usage
-------------
```python
from src.drima.models import SimulationConfig
from src.drima.simulation import run_simulation

config = SimulationConfig(
    grid_width=6,
    grid_height=8,
    num_d_agents=4,
    num_r_agents=9,
    iterations=25,
    seed=101,
    output_file="history.json",
    log_file=None,
)
result = run_simulation(config, write_output=False)
assert len(result.history) == 26  # Initial world + 25 update steps
```
"""

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

from src.drima.io_handler import build_history_payload, save_history_json
from src.drima.logger import log_initial_state, log_step, setup_logger
from src.drima.models import SimulationConfig, StepLog, World
from src.drima.world import (
    async_world_update,
    compute_convergence_metrics,
    create_world,
    sync_world_update,
)


@dataclass
class SimulationResult:
    """Container for all artifacts produced by a DRIMA simulation run.

    Attributes:
        config: The `SimulationConfig` used for the run.
        history: List of all `World` snapshots from `t = 0` to `t = iterations`.
        step_logs: List of `StepLog` instances for steps `1..iterations`.
        payload: Full JSON-serializable dictionary of the history and metadata.
        output_path: Resolved `Path` where the JSON history file was saved (if written).
    """

    config: SimulationConfig
    history: List[World]
    step_logs: List[StepLog]
    payload: Dict[str, Any]
    output_path: Optional[Path] = None

    @property
    def initial_world(self) -> World:
        """Return the initial `World` snapshot (`t = 0`)."""
        return self.history[0]

    @property
    def final_world(self) -> World:
        """Return the final `World` snapshot (`t = iterations`)."""
        return self.history[-1]


def run_simulation(
    config: SimulationConfig,
    write_output: bool = True,
    console_log: bool = False,
) -> SimulationResult:
    """Execute a complete DRIMA simulation according to `config`.

    Args:
        config: Validated `SimulationConfig` instance.
        write_output: If `True` and `config.output_file` is set, writes the JSON history file.
        console_log: Whether to also stream log messages to `stderr`.

    Returns:
        `SimulationResult` containing the full sequence of `World` snapshots, step logs, and
        JSON payload.
    """
    logger = setup_logger(
        log_file=config.log_file,
        log_level=config.log_level,
        console_output=console_log,
    )

    rng = random.Random(config.seed)
    initial_world = create_world(config, rng)

    log_initial_state(logger, config, initial_world)

    worlds: List[World] = [initial_world]
    step_logs: List[StepLog] = []
    current_world = initial_world

    update_fn = (
        sync_world_update
        if config.update_mode.lower() == "sync"
        else async_world_update
    )

    for step_idx in range(1, config.iterations + 1):
        current_world, step_log = update_fn(
            world=current_world,
            config=config,
            rng=rng,
            step_index=step_idx,
        )
        worlds.append(current_world)
        step_logs.append(step_log)
        log_step(logger, step_log, current_world)

    final_metrics = compute_convergence_metrics(current_world)
    logger.info("=" * 72)
    logger.info(
        "Simulation Completed | Total Steps: %d | Mean R Max-Prob: %.4f | "
        "R Converged (>=0.95): %d/%d",
        config.iterations,
        final_metrics["r_mean_max_prob"],
        final_metrics["r_converged_95_count"],
        final_metrics["num_r_agents"],
    )
    for agent in current_world.agents:
        probs_str = ", ".join(f"{p:.3f}" for p in agent.movement_pattern)
        logger.info(
            "Final Agent   %-4s (%s) at pos=%s | dom=%-2s | max_p=%.3f | pattern=[%s]",
            agent.name,
            agent.agent_type,
            agent.position,
            agent.dominant_direction,
            agent.max_probability,
            probs_str,
        )

    payload = build_history_payload(config=config, worlds=worlds, step_logs=step_logs)
    output_path: Optional[Path] = None
    if write_output and config.output_file:
        output_path = save_history_json(payload, config.output_file)
        logger.info("Saved full history JSON to: %s", output_path)
    logger.info("=" * 72)

    # Flush and close file handlers cleanly to avoid open file descriptor leaks
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    return SimulationResult(
        config=config,
        history=worlds,
        step_logs=step_logs,
        payload=payload,
        output_path=output_path,
    )
