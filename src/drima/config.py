"""Configuration loading and merging for DRIMA (JSON config files + CLI overrides).

Project Overview
----------------
To give the user complete control over every variable in DRIMA—from initial agent placement and
movement patterns to world mobility, interaction types, neighborhood range, and normalization
methods—this module supports:
1. Loading a JSON configuration file (`load_config_file`).
2. Parsing inline JSON or shorthand agent specifications.
3. Merging CLI flags on top of JSON configuration values (`build_simulation_config`), where any
   explicitly provided CLI flag takes precedence over the JSON file.

Example Usage
-------------
```python
from src.drima.config import build_simulation_config

# Build config from a dictionary and override iterations and fp via CLI overrides
base_dict = {"grid_width": 8, "grid_height": 5, "num_d_agents": 2, "num_r_agents": 6}
cli_overrides = {"iterations": 200, "fp": 1.5}
config = build_simulation_config(config_dict=base_dict, cli_overrides=cli_overrides)
assert config.iterations == 200
assert config.fp == 1.5
```
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.drima.models import DEFAULT_INTERACTION_TYPES, SimulationConfig


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Read and parse a JSON configuration file from disk.

    Also supports loading a canonical DRIMA world list directly if the JSON root is a 4-element
    list `[list_of_agents, [X_max, Y_max], interaction_types, world_mobility]`.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Dictionary of configuration parameters.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: '{config_path}'.")

    with path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    if isinstance(raw_data, list):
        # Canonical 4-element DRIMA world list:
        # [list_of_agents, [x_max, y_max], interaction_types, world_mobility]
        if len(raw_data) != 4:
            raise ValueError(
                f"Expected 4-element DRIMA world list in '{config_path}', got {len(raw_data)}."
            )
        agents_raw, grid_raw, types_raw, mobility_raw = raw_data
        if (
            len(agents_raw) == 1
            and isinstance(agents_raw[0], list)
            and agents_raw[0]
            and isinstance(agents_raw[0][0], list)
            and len(agents_raw[0][0]) == 2
            and isinstance(agents_raw[0][0][0], list)
        ):
            agents_raw = agents_raw[0]
        return {
            "agents": agents_raw,
            "grid_width": int(grid_raw[0]),
            "grid_height": int(grid_raw[1]),
            "interaction_types": [str(t) for t in types_raw],
            "world_mobility": str(mobility_raw).upper(),
        }

    if not isinstance(raw_data, dict):
        raise ValueError(
            f"JSON config root must be an object or 4-element DRIMA world list, "
            f"got {type(raw_data).__name__}."
        )

    # Normalize optional `grid_size: [width, height]` shorthand in JSON config
    normalized = dict(raw_data)
    if "grid_size" in normalized and isinstance(normalized["grid_size"], list):
        grid_size = normalized.pop("grid_size")
        if len(grid_size) != 2:
            raise ValueError(f"grid_size must have 2 elements [X_max, Y_max], got {grid_size}.")
        normalized.setdefault("grid_width", int(grid_size[0]))
        normalized.setdefault("grid_height", int(grid_size[1]))

    return normalized


def parse_interaction_types_arg(raw_value: Optional[str]) -> Optional[List[str]]:
    """Parse a comma- or space-separated string of interaction types from the CLI.

    Passing `'none'` or `''` yields an empty list `[]` (no interactions allowed in the world,
    as described in Section 3.1).

    Args:
        raw_value: String such as `'dd,RR,dR,Rd'` or `'none'`.

    Returns:
        List of interaction type strings, or `None` if `raw_value` is `None`.
    """
    if raw_value is None:
        return None
    stripped = raw_value.strip()
    if not stripped or stripped.lower() in ("none", "{}", "[]"):
        return []
    tokens = [
        tok.strip()
        for part in stripped.split(",")
        for tok in part.split()
        if tok.strip()
    ]
    return tokens


def parse_d_directions_arg(raw_value: Optional[str]) -> Optional[List[str]]:
    """Parse a comma- or space-separated list of initial directions for D-agents.

    Args:
        raw_value: String such as `'E,SE'` or `'NW'`.

    Returns:
        List of uppercase direction strings, or `None` if `raw_value` is `None`.
    """
    if raw_value is None:
        return None
    tokens = [
        tok.strip().upper()
        for part in raw_value.split(",")
        for tok in part.split()
        if tok.strip()
    ]
    return tokens if tokens else None


def build_simulation_config(
    config_file: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> SimulationConfig:
    """Construct a `SimulationConfig` by merging defaults, JSON config, and CLI overrides.

    Precedence order (highest to lowest):
    1. `cli_overrides` (any key whose value is not `None`)
    2. `config_dict` / `config_file` values
    3. `SimulationConfig` defaults

    Args:
        config_file: Optional path to a JSON configuration file.
        config_dict: Optional pre-loaded configuration dictionary.
        cli_overrides: Optional dictionary of explicit CLI flag overrides.

    Returns:
        Validated `SimulationConfig` instance.
    """
    merged: Dict[str, Any] = {}

    if config_file:
        merged.update(load_config_file(config_file))
    if config_dict:
        merged.update(config_dict)
    if cli_overrides:
        for key, val in cli_overrides.items():
            if val is not None:
                # If CLI explicitly overrides num_d_agents or num_r_agents, clear static agents list
                # unless `agents` itself was also passed on the CLI
                if key in ("num_d_agents", "num_r_agents", "d_directions") and (
                    cli_overrides.get("agents") is None
                ):
                    merged.pop("agents", None)
                merged[key] = val

    valid_fields = {
        "grid_width",
        "grid_height",
        "world_mobility",
        "interaction_types",
        "allow_shared_cells",
        "num_d_agents",
        "num_r_agents",
        "d_directions",
        "d_transition_potential",
        "agents",
        "interaction_range",
        "update_mode",
        "interaction_direction",
        "delta_mode",
        "fp",
        "normalization_method",
        "sync_only_unprocessed_neighbors",
        "async_move_all",
        "iterations",
        "seed",
        "output_file",
        "log_file",
        "log_level",
        "display_world",
        "display_history",
    }

    unknown_keys = set(merged.keys()) - valid_fields
    if unknown_keys:
        raise ValueError(
            f"Unknown configuration parameter(s): {sorted(unknown_keys)}. "
            f"Valid parameters are: {sorted(valid_fields)}."
        )

    if "interaction_types" not in merged:
        merged["interaction_types"] = list(DEFAULT_INTERACTION_TYPES)

    return SimulationConfig(**merged)
