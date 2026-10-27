"""Structured logging utilities for DRIMA simulations.

Project Overview
----------------
To satisfy the requirement that all simulation information be logged clearly and reproducibly,
this module configures a dedicated logger (`"drima"`) that writes structured trace events to
both a log file (if specified) and the console.

Debugging with DRIMA Logs
-------------------------
When diagnosing unexpected multi-agent dynamics (such as why an R-agent oscillates between two
directions or never changes its movement pattern):
1. Check the **Initialization Log** to confirm `world_mobility` (`F` vs `P`), `interaction_types`,
   and initial agent coordinates/directions.
2. Set `--log-level DEBUG` to inspect every binary interaction's pre- and post-normalization
   movement patterns and verify whether agents were within Moore range `R`.
3. Inspect the **Step Summary** and **Final Convergence Summary** to see each R-agent's dominant
   direction and maximum probability.

Example Usage
-------------
```python
from src.drima.logger import setup_logger

logger = setup_logger(log_file="drima_run.log", log_level="INFO")
logger.info("DRIMA simulation initialized.")
```
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.drima.models import SimulationConfig, StepLog, World


def setup_logger(
    log_file: Optional[str] = "drima.log",
    log_level: str = "INFO",
    console_output: bool = True,
) -> logging.Logger:
    """Configure and return the `'drima'` logger instance.

    Clears any previously attached handlers on `'drima'` so repeated programmatic calls do not
    duplicate log lines or leak file descriptors.

    Args:
        log_file: Optional path to the log file on disk. Pass `None` or `""` to disable file
            logging.
        log_level: Logging level name (`'DEBUG'`, `'INFO'`, `'WARNING'`, `'ERROR'`).
        console_output: Whether to also emit logs to `sys.stderr`.

    Returns:
        Configured `logging.Logger` instance.
    """
    logger = logging.getLogger("drima")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Close and remove existing handlers to prevent resource leaks across multiple runs
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        log_path = Path(log_file)
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console_output:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(numeric_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def log_initial_state(
    logger: logging.Logger,
    config: SimulationConfig,
    world: World,
) -> None:
    """Log the simulation configuration and initial world state (`t = 0`)."""
    logger.info("=" * 72)
    logger.info("DRIMA Simulation Initialized")
    logger.info(
        "Grid Size: %dx%d | Mobility: %s | Interaction Types: %s | Range R: %d",
        world.grid_size[0],
        world.grid_size[1],
        world.world_mobility,
        world.interaction_types,
        config.interaction_range,
    )
    logger.info(
        "Update Mode: %s | Direction Mode: %s | Delta Mode: %s | F_p: %.4f | Norm: %s",
        config.update_mode,
        config.interaction_direction,
        config.delta_mode,
        config.fp,
        config.normalization_method,
    )
    logger.info(
        "Iterations: %d | Seed: %s | Shared Cells Allowed: %s",
        config.iterations,
        str(config.seed),
        config.allow_shared_cells,
    )
    logger.info("-" * 72)
    for agent in world.agents:
        formatted_probs = ", ".join(f"{p:.3f}" for p in agent.movement_pattern)
        logger.info(
            "Initial Agent %-4s (%s) at pos=%s | dom=%-2s | pattern=[%s]",
            agent.name,
            agent.agent_type,
            agent.position,
            agent.dominant_direction,
            formatted_probs,
        )
    logger.info("=" * 72)


def log_step(
    logger: logging.Logger,
    step_log: StepLog,
    world: World,
) -> None:
    """Log interactions, movements, and state summary for a single update step."""
    modified_count = sum(
        1
        for inter in step_log.interactions
        if inter.agent_i_modified or inter.agent_j_modified
    )
    logger.info(
        "Step %04d [%s] | Interactions: %d (effective: %d) | Movements: %d",
        step_log.step,
        step_log.update_mode,
        len(step_log.interactions),
        modified_count,
        len(step_log.movements),
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("  Step %04d permutation order: %s", step_log.step, step_log.agent_order)
        for inter in step_log.interactions:
            logger.debug(
                "  Interaction: %s %s <-> %s %s | mod=(%s, %s)",
                inter.agent_i_name,
                inter.agent_i_pos,
                inter.agent_j_name,
                inter.agent_j_pos,
                inter.agent_i_modified,
                inter.agent_j_modified,
            )
            if inter.agent_i_modified:
                logger.debug(
                    "    %s after: [%s]",
                    inter.agent_i_name,
                    ", ".join(f"{p:.4f}" for p in inter.agent_i_pattern_after),
                )
            if inter.agent_j_modified:
                logger.debug(
                    "    %s after: [%s]",
                    inter.agent_j_name,
                    ", ".join(f"{p:.4f}" for p in inter.agent_j_pattern_after),
                )
        for mov in step_log.movements:
            logger.debug(
                "  Move: %-4s dir=%-2s %s -> %s%s",
                mov.agent_name,
                mov.direction,
                mov.position_before,
                mov.position_after,
                " (BLOCKED)" if mov.blocked_by_occupancy else "",
            )
        for agent in world.agents:
            logger.debug(
                "  State %-4s pos=%s dom=%-2s max_p=%.4f",
                agent.name,
                agent.position,
                agent.dominant_direction,
                agent.max_probability,
            )
