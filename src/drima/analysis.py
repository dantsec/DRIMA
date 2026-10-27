"""Temporal series extraction and analysis utilities for DRIMA `history.json` files.

Project Overview
----------------
After a DRIMA simulation finishes, it exports a comprehensive `history.json` file containing
every world snapshot from `t = 0` to `t = N`, along with step-level interaction and movement
events.

This module provides pure-Python and `pandas`/`matplotlib`-compatible functions to transform
`history.json` into temporal series for plotting and quantitative analysis:
1. **Agent Movement Pattern Series (`extract_agent_probability_series`):**
   Extracts the 9 directional probability trajectories `P_d(t)` over time `t = 0..N` for every
   agent—directly corresponding to `DisplayHistory[history]` (Figure 2 in de Oliveira, 2010).
2. **Convergence & Determinism Series (`extract_convergence_series`):**
   Extracts per-step aggregate R-agent convergence (`r_mean_max_prob`, `r_converged_95_count`)
   and per-agent `max_probability` and `determinism_score` (`1 - normalized_entropy`).
3. **Interaction & Event Series (`extract_interaction_series`):**
   Extracts per-step total vs. effective interactions, cumulative interaction counts, and a
   pairwise interaction matrix across agent names.
4. **Spatial Trajectory Series (`extract_position_series`):**
   Extracts `(x_t, y_t)` coordinates and dominant directions over time `t = 0..N`.

Common Pitfalls & What NOT To Do
--------------------------------
- **Do NOT confuse total interactions with effective interactions:**
  In the default interaction setup `["dd", "RR", "dR", "Rd"]`, when two D-agents meet (`dd`) or
  two R-agents with identical movement patterns meet, an interaction event occurs, but neither
  agent's movement pattern changes (`effective = False`). Always distinguish total encounters
  from pattern-changing interactions in temporal series plots.
- **Do NOT plot raw Euclidean line segments across toroidal wrap-arounds without accounting for
  periodic boundaries:** When an agent steps East from `x = X_max` to `x = 1`, its coordinate
  jumps across the grid even though it moved only 1 cell.

Example Usage
-------------
```python
from src.drima.io_handler import load_history_json
from src.drima.analysis import (
    extract_agent_probability_series,
    extract_convergence_series,
    extract_interaction_series,
)

history_data = load_history_json("history.json")
prob_series = extract_agent_probability_series(history_data)
conv_series = extract_convergence_series(history_data)
inter_series = extract_interaction_series(history_data)

print("Steps:", len(conv_series["steps"]))
print("Final mean R max-prob:", conv_series["r_mean_max_prob"][-1])
```
"""

from typing import Any, Dict, List, Tuple

from src.drima.io_handler import load_history_json
from src.drima.models import DIRECTIONS


def load_or_validate_history(history_source: Any) -> Dict[str, Any]:
    """Accept either a file path (`str`/`Path`) or an already-loaded history dictionary.

    Args:
        history_source: Path to `history.json` or a dictionary returned by `load_history_json`.

    Returns:
        Validated history dictionary containing `'history'` and `'metadata'`.
    """
    if isinstance(history_source, str):
        data = load_history_json(history_source)
    elif isinstance(history_source, dict):
        data = history_source
    else:
        raise TypeError(
            f"history_source must be a file path string or dict, got {type(history_source).__name__}."
        )

    if "history" not in data or not isinstance(data["history"], list) or not data["history"]:
        raise ValueError("Invalid history data: missing or empty 'history' list.")
    return data


def extract_agent_probability_series(
    history_source: Any,
) -> Dict[str, Dict[str, List[float]]]:
    """Extract temporal series of directional movement probabilities for every agent.

    Args:
        history_source: Path to `history.json` or loaded history dictionary.

    Returns:
        Nested dictionary `series[agent_name][direction] -> List[float]` of length `T`
        (from `step = 0` to `step = N`), plus `series[agent_name]["step"] -> List[int]`.
    """
    data = load_or_validate_history(history_source)
    snapshots = data["history"]

    series: Dict[str, Dict[str, List[float]]] = {}
    first_agents = snapshots[0]["world"]["agents"]
    for agent_info in first_agents:
        name = agent_info["name"]
        series[name] = {d: [] for d in DIRECTIONS}
        series[name]["step"] = []

    for snap in snapshots:
        step = int(snap["step"])
        for agent_info in snap["world"]["agents"]:
            name = agent_info["name"]
            if name not in series:
                series[name] = {d: [] for d in DIRECTIONS}
                series[name]["step"] = []
            series[name]["step"].append(step)
            pattern = agent_info["movement_pattern"]
            for idx, d_name in enumerate(DIRECTIONS):
                series[name][d_name].append(float(pattern[idx]))

    return series


def extract_convergence_series(history_source: Any) -> Dict[str, Any]:
    """Extract temporal series of convergence metrics across all steps `t = 0..N`.

    Args:
        history_source: Path to `history.json` or loaded history dictionary.

    Returns:
        Dictionary containing:
        - `'steps'`: `List[int]`
        - `'r_mean_max_prob'`: `List[float]`
        - `'r_converged_95_count'`: `List[int]`
        - `'r_all_converged_95'`: `List[bool]`
        - `'per_agent_max_prob'`: `Dict[str, List[float]]`
        - `'per_agent_determinism'`: `Dict[str, List[float]]`
        - `'per_agent_dominant_dir'`: `Dict[str, List[str]]`
    """
    data = load_or_validate_history(history_source)
    snapshots = data["history"]

    steps: List[int] = []
    r_mean_max_prob: List[float] = []
    r_converged_95_count: List[int] = []
    r_all_converged_95: List[bool] = []

    first_agents = snapshots[0]["world"]["agents"]
    per_agent_max_prob: Dict[str, List[float]] = {a["name"]: [] for a in first_agents}
    per_agent_determinism: Dict[str, List[float]] = {a["name"]: [] for a in first_agents}
    per_agent_dominant_dir: Dict[str, List[str]] = {a["name"]: [] for a in first_agents}

    for snap in snapshots:
        steps.append(int(snap["step"]))
        conv = snap["convergence"]
        r_mean_max_prob.append(float(conv["r_mean_max_prob"]))
        r_converged_95_count.append(int(conv["r_converged_95_count"]))
        r_all_converged_95.append(bool(conv["r_all_converged_95"]))

        for name, agent_conv in conv["agents"].items():
            per_agent_max_prob.setdefault(name, []).append(
                float(agent_conv["max_probability"])
            )
            per_agent_determinism.setdefault(name, []).append(
                float(agent_conv["determinism_score"])
            )
            per_agent_dominant_dir.setdefault(name, []).append(
                str(agent_conv["dominant_direction"])
            )

    return {
        "steps": steps,
        "r_mean_max_prob": r_mean_max_prob,
        "r_converged_95_count": r_converged_95_count,
        "r_all_converged_95": r_all_converged_95,
        "per_agent_max_prob": per_agent_max_prob,
        "per_agent_determinism": per_agent_determinism,
        "per_agent_dominant_dir": per_agent_dominant_dir,
    }


def extract_interaction_series(history_source: Any) -> Dict[str, Any]:
    """Extract temporal series of interaction counts and pairwise interaction matrices.

    Args:
        history_source: Path to `history.json` or loaded history dictionary.

    Returns:
        Dictionary containing:
        - `'steps'`: `List[int]` (`0..N`)
        - `'total_interactions_per_step'`: `List[int]`
        - `'effective_interactions_per_step'`: `List[int]`
        - `'cumulative_total_interactions'`: `List[int]`
        - `'cumulative_effective_interactions'`: `List[int]`
        - `'agent_names'`: `List[str]`
        - `'pairwise_total_matrix'`: `Dict[Tuple[str, str], int]`
        - `'pairwise_effective_matrix'`: `Dict[Tuple[str, str], int]`
        - `'per_agent_modifications'`: `Dict[str, int]`
    """
    data = load_or_validate_history(history_source)
    snapshots = data["history"]

    agent_names = [a["name"] for a in snapshots[0]["world"]["agents"]]
    steps: List[int] = []
    total_per_step: List[int] = []
    effective_per_step: List[int] = []
    cumulative_total: List[int] = []
    cumulative_effective: List[int] = []

    pairwise_total: Dict[Tuple[str, str], int] = {
        (a, b): 0 for a in agent_names for b in agent_names
    }
    pairwise_effective: Dict[Tuple[str, str], int] = {
        (a, b): 0 for a in agent_names for b in agent_names
    }
    per_agent_modifications: Dict[str, int] = {a: 0 for a in agent_names}

    running_total = 0
    running_effective = 0

    for snap in snapshots:
        step = int(snap["step"])
        steps.append(step)
        interactions = snap.get("events", {}).get("interactions", [])

        step_tot = len(interactions)
        step_eff = 0
        for inter in interactions:
            ai = inter["agent_i"]
            aj = inter["agent_j"]
            mod_i = bool(inter["agent_i_modified"])
            mod_j = bool(inter["agent_j_modified"])

            pairwise_total[(ai, aj)] = pairwise_total.get((ai, aj), 0) + 1
            pairwise_total[(aj, ai)] = pairwise_total.get((aj, ai), 0) + 1

            if mod_i or mod_j:
                step_eff += 1
                pairwise_effective[(ai, aj)] = pairwise_effective.get((ai, aj), 0) + 1
                pairwise_effective[(aj, ai)] = pairwise_effective.get((aj, ai), 0) + 1
            if mod_i:
                per_agent_modifications[ai] = per_agent_modifications.get(ai, 0) + 1
            if mod_j:
                per_agent_modifications[aj] = per_agent_modifications.get(aj, 0) + 1

        running_total += step_tot
        running_effective += step_eff

        total_per_step.append(step_tot)
        effective_per_step.append(step_eff)
        cumulative_total.append(running_total)
        cumulative_effective.append(running_effective)

    return {
        "steps": steps,
        "total_interactions_per_step": total_per_step,
        "effective_interactions_per_step": effective_per_step,
        "cumulative_total_interactions": cumulative_total,
        "cumulative_effective_interactions": cumulative_effective,
        "agent_names": agent_names,
        "pairwise_total_matrix": pairwise_total,
        "pairwise_effective_matrix": pairwise_effective,
        "per_agent_modifications": per_agent_modifications,
    }


def extract_position_series(
    history_source: Any,
) -> Dict[str, List[Tuple[int, int]]]:
    """Extract `(x, y)` grid position trajectories over `t = 0..N` for every agent.

    Args:
        history_source: Path to `history.json` or loaded history dictionary.

    Returns:
        Dictionary mapping `agent_name -> List[(x, y)]`.
    """
    data = load_or_validate_history(history_source)
    snapshots = data["history"]

    positions: Dict[str, List[Tuple[int, int]]] = {
        a["name"]: [] for a in snapshots[0]["world"]["agents"]
    }
    for snap in snapshots:
        for agent_info in snap["world"]["agents"]:
            pos = agent_info["position"]
            positions.setdefault(agent_info["name"], []).append(
                (int(pos[0]), int(pos[1]))
            )
    return positions
