"""JSON history and configuration I/O handlers for DRIMA.

Project Overview
----------------
A **history** in DRIMA (Section 1 & Section 3) is the sequence of worlds generated through a
pathway of agent interactions, starting from an initial world configuration (`t = 0`) through
`t = iterations`.

This module serializes the full history, step logs, configuration parameters, and convergence
metrics into a structured JSON file so experiments can be inspected, audited, or re-loaded.

Example Usage
-------------
```python
from src.drima.models import SimulationConfig
from src.drima.simulation import run_simulation
from src.drima.io_handler import save_history_json, load_history_json

config = SimulationConfig(iterations=10, output_file="sample_history.json", log_file=None)
result = run_simulation(config)
save_history_json(result, "sample_history.json")
loaded = load_history_json("sample_history.json")
assert len(loaded["history"]) == 11  # t=0 plus 10 steps
```
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any, Dict, List

from src.drima.models import SimulationConfig, StepLog, World
from src.drima.world import compute_convergence_metrics


def build_history_payload(
    config: SimulationConfig,
    worlds: List[World],
    step_logs: List[StepLog],
) -> Dict[str, Any]:
    """Assemble the complete JSON-serializable history dictionary.

    Includes:
    - `metadata`: Timestamp, Python version, total iterations, total interactions, configuration,
      and initial/final convergence metrics.
    - `initial_world`: World state at `t = 0`.
    - `final_world`: World state at `t = N`.
    - `drima_history_representation`: List of canonical DRIMA world lists
      `{list_of_agents, grid_size, types_of_interaction, world_mobility}` for every step.
    - `history`: Detailed step-by-step snapshots combining the world state with the step's
      interactions and movements.

    Args:
        config: `SimulationConfig` used for the simulation.
        worlds: List of `World` snapshots from `t = 0` to `t = iterations`.
        step_logs: List of `StepLog` records from `step = 1` to `step = iterations`.

    Returns:
        Dictionary ready for JSON serialization.
    """
    total_interactions = sum(len(s.interactions) for s in step_logs)
    effective_interactions = sum(
        sum(1 for i in s.interactions if i.agent_i_modified or i.agent_j_modified)
        for s in step_logs
    )

    history_entries: List[Dict[str, Any]] = []
    for idx, world_snapshot in enumerate(worlds):
        entry: Dict[str, Any] = {
            "step": idx,
            "world": world_snapshot.to_dict(),
            "convergence": compute_convergence_metrics(world_snapshot),
        }
        if idx == 0:
            entry["events"] = {
                "update_mode": "initial",
                "agent_order": [a.name for a in world_snapshot.agents],
                "interactions": [],
                "movements": [],
            }
        else:
            entry["events"] = step_logs[idx - 1].to_dict()
        history_entries.append(entry)

    return {
        "metadata": {
            "system": "DRIMA: Dynamics of Randomly Interacting Moving Agents",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "total_steps": len(step_logs),
            "total_snapshots": len(worlds),
            "total_interactions": total_interactions,
            "effective_interactions": effective_interactions,
            "config": config.to_dict(),
            "initial_convergence": compute_convergence_metrics(worlds[0]),
            "final_convergence": compute_convergence_metrics(worlds[-1]),
        },
        "initial_world": worlds[0].to_dict(),
        "final_world": worlds[-1].to_dict(),
        "drima_history_representation": [w.to_drima_list() for w in worlds],
        "history": history_entries,
    }


def save_history_json(
    payload: Dict[str, Any],
    output_file: str,
) -> Path:
    """Write the simulation history dictionary to a formatted JSON file.

    Args:
        payload: Dictionary returned by `build_history_payload`.
        output_file: Destination file path.

    Returns:
        Resolved `Path` to the written JSON file.
    """
    out_path = Path(output_file)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    return out_path.resolve()


def load_history_json(file_path: str) -> Dict[str, Any]:
    """Load and parse a previously generated DRIMA JSON history file.

    Args:
        file_path: Path to the JSON history file.

    Returns:
        Parsed dictionary containing `metadata`, `initial_world`, `final_world`, and `history`.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"History file not found: '{file_path}'.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
