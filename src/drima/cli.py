"""Command-Line Interface (CLI) for DRIMA.

Project Overview
----------------
Provides a full-featured CLI allowing users to configure and run DRIMA simulations from either
command-line flags, JSON configuration files, or a combination of both (with CLI flags taking
precedence).

Example CLI Invocations
-----------------------
1. **Run a default 8x5 simulation (2 D-agents, 6 R-agents) for 100 steps and display results:**
   ```bash
   python3 main.py --iterations 100 --display-world --display-history
   ```

2. **Reproduce Figure 2 setup (8x5 world, 2 D-agents moving E and SE, 6 R-agents, F_p=1):**
   ```bash
   python3 main.py --grid-width 8 --grid-height 5 --num-d-agents 2 --num-r-agents 6 \
       --d-directions E,SE --mobility F --delta-mode fixed --fp 1.0 --iterations 200 \
       --seed 42 --output history_fig2.json --log-file drima_fig2.log
   ```

3. **Load a JSON config file and override the number of iterations and increment factor `F_p`:**
   ```bash
   python3 main.py --config examples/figure2_experiment.json --iterations 500 --fp 5.0
   ```
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from src.drima.config import (
    build_simulation_config,
    parse_d_directions_arg,
    parse_interaction_types_arg,
)
from src.drima.simulation import run_simulation
from src.drima.visualizer import display_history, display_world


def build_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the `argparse.ArgumentParser` for DRIMA."""
    parser = argparse.ArgumentParser(
        prog="drima",
        description=(
            "DRIMA: Dynamics of Randomly Interacting Moving Agents — "
            "A Minimal System for Probing the Dynamics of Change in a Reactive Multi-Agent Setting."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 main.py --grid-width 8 --grid-height 5 --num-d-agents 2 --num-r-agents 6 \\\n"
            "      --d-directions E,SE --iterations 100 --display-world --display-history\n\n"
            "  python3 main.py --config examples/figure2_experiment.json --iterations 400 --seed 42\n"
        ),
    )

    # Configuration file input
    config_group = parser.add_argument_group("Configuration Input")
    config_group.add_argument(
        "-c",
        "--config",
        dest="config_file",
        type=str,
        default=None,
        help="Path to a JSON configuration file (CLI flags override file values).",
    )

    # World & Grid parameters
    world_group = parser.add_argument_group("World & Grid Parameters")
    world_group.add_argument(
        "--grid-width",
        "-X",
        type=int,
        default=None,
        help="Grid width X_max (1-indexed periodic horizontal dimension, default: 8).",
    )
    world_group.add_argument(
        "--grid-height",
        "-Y",
        type=int,
        default=None,
        help="Grid height Y_max (1-indexed periodic vertical dimension, default: 5).",
    )
    world_group.add_argument(
        "--mobility",
        choices=["F", "P", "f", "p"],
        default=None,
        help="World mobility: 'F' (Full: no-move X is always 0) or 'P' (Partial: X >= 0).",
    )
    world_group.add_argument(
        "--interaction-types",
        type=str,
        default=None,
        help=(
            "Comma-separated allowed interaction types from {dd,DD,rr,RR,dR,Rd,Dr,rD} "
            "or 'none' for no interactions (default: 'dd,RR,dR,Rd')."
        ),
    )

    occupancy_group = world_group.add_mutually_exclusive_group()
    occupancy_group.add_argument(
        "--allow-shared-cells",
        dest="allow_shared_cells",
        action="store_true",
        default=None,
        help="Allow multiple agents to occupy the same grid cell (default).",
    )
    occupancy_group.add_argument(
        "--exclusive-cells",
        dest="allow_shared_cells",
        action="store_false",
        help="Enforce at most one agent per grid cell.",
    )

    # Agent generation & specification
    agent_group = parser.add_argument_group("Initial Agents Configuration")
    agent_group.add_argument(
        "--num-d-agents",
        type=int,
        default=None,
        help="Number of deterministic D-agents to generate (default: 2).",
    )
    agent_group.add_argument(
        "--num-r-agents",
        type=int,
        default=None,
        help="Number of random R-agents to generate (default: 6).",
    )
    agent_group.add_argument(
        "--d-directions",
        type=str,
        default=None,
        help="Comma-separated initial directions for generated D-agents (e.g., 'E,SE').",
    )
    agent_group.add_argument(
        "--d-transition-potential",
        type=float,
        default=None,
        help="Transition potential value assigned to D-agents' active direction (default: 1000).",
    )
    agent_group.add_argument(
        "--agents-json",
        type=str,
        default=None,
        help="Inline JSON string specifying explicit initial agents list.",
    )

    # Dimensions of the Interactions (Section 2.2 & 3.2)
    inter_group = parser.add_argument_group("Dimensions of the Interactions")
    inter_group.add_argument(
        "--range",
        "-R",
        dest="interaction_range",
        type=int,
        default=None,
        help="Moore neighborhood interaction range R >= 0 (default: 1; 0 means same-cell only).",
    )
    inter_group.add_argument(
        "--update-mode",
        choices=["sync", "async"],
        default=None,
        help="World update mode: 'sync' (SyncWorldUpdate) or 'async' (AsyncWorldUpdate).",
    )
    inter_group.add_argument(
        "--interaction-direction",
        choices=["one-way", "two-way"],
        default=None,
        help="Direction of interaction effect: 'one-way' (default) or 'two-way'.",
    )
    inter_group.add_argument(
        "--delta-mode",
        choices=["fixed", "variable"],
        default=None,
        help="Probability increment mode: 'fixed' (F_p/9) or 'variable' ((F_p/9)*|P_i - P_j|).",
    )
    inter_group.add_argument(
        "--fp",
        type=float,
        default=None,
        help="Probability increment factor F_p >= 0 (default: 1.0).",
    )
    inter_group.add_argument(
        "--normalization",
        dest="normalization_method",
        choices=["equal_displacement", "proportional", "none"],
        default=None,
        help="Movement pattern normalization scheme (default: 'equal_displacement').",
    )
    inter_group.add_argument(
        "--sync-only-unprocessed-neighbors",
        action="store_true",
        default=None,
        help="In sync mode, restrict neighbor candidates to agents not yet processed in the step.",
    )
    async_move_group = inter_group.add_mutually_exclusive_group()
    async_move_group.add_argument(
        "--async-move-all",
        dest="async_move_all",
        action="store_true",
        default=None,
        help="In async mode, move all agents at the end of each update step (default).",
    )
    async_move_group.add_argument(
        "--async-move-pair-only",
        dest="async_move_all",
        action="store_false",
        help="In async mode, move only the selected interacting agents.",
    )

    # Simulation execution, logging, and output
    exec_group = parser.add_argument_group("Execution, Logging & Visualization")
    exec_group.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=None,
        help="Number of world update iterations (history length, default: 100).",
    )
    exec_group.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducible simulation runs.",
    )
    exec_group.add_argument(
        "--output",
        "-o",
        dest="output_file",
        type=str,
        default=None,
        help="Path to the output JSON history file (default: 'history.json').",
    )
    exec_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to the simulation log file (default: 'drima.log'; pass '' to disable).",
    )
    exec_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging verbosity level (default: 'INFO').",
    )
    exec_group.add_argument(
        "--console-log",
        action="store_true",
        default=False,
        help="Stream step log messages to stderr during execution.",
    )
    exec_group.add_argument(
        "--display-world",
        action="store_true",
        default=None,
        help="Display initial and final DRIMA worlds in ASCII table/grid format.",
    )
    exec_group.add_argument(
        "--display-history",
        action="store_true",
        default=None,
        help="Display movement pattern evolution across the history (DisplayHistory).",
    )

    return parser


def extract_cli_overrides(parsed_args: argparse.Namespace) -> Dict[str, Any]:
    """Convert parsed `argparse.Namespace` into a dictionary of explicit CLI overrides."""
    agents_list: Optional[List[Any]] = None
    if parsed_args.agents_json is not None:
        agents_list = json.loads(parsed_args.agents_json)
        if not isinstance(agents_list, list):
            raise ValueError("--agents-json must parse to a JSON list of agents.")

    return {
        "grid_width": parsed_args.grid_width,
        "grid_height": parsed_args.grid_height,
        "world_mobility": (
            parsed_args.mobility.upper() if parsed_args.mobility is not None else None
        ),
        "interaction_types": parse_interaction_types_arg(parsed_args.interaction_types),
        "allow_shared_cells": parsed_args.allow_shared_cells,
        "num_d_agents": parsed_args.num_d_agents,
        "num_r_agents": parsed_args.num_r_agents,
        "d_directions": parse_d_directions_arg(parsed_args.d_directions),
        "d_transition_potential": parsed_args.d_transition_potential,
        "agents": agents_list,
        "interaction_range": parsed_args.interaction_range,
        "update_mode": parsed_args.update_mode,
        "interaction_direction": parsed_args.interaction_direction,
        "delta_mode": parsed_args.delta_mode,
        "fp": parsed_args.fp,
        "normalization_method": parsed_args.normalization_method,
        "sync_only_unprocessed_neighbors": parsed_args.sync_only_unprocessed_neighbors,
        "async_move_all": parsed_args.async_move_all,
        "iterations": parsed_args.iterations,
        "seed": parsed_args.seed,
        "output_file": parsed_args.output_file,
        "log_file": parsed_args.log_file,
        "log_level": parsed_args.log_level,
        "display_world": parsed_args.display_world,
        "display_history": parsed_args.display_history,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for the DRIMA CLI tool.

    Args:
        argv: Optional list of CLI argument strings (defaults to `sys.argv[1:]`).

    Returns:
        Exit code (`0` on success, `1` on configuration/runtime error).
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        overrides = extract_cli_overrides(args)
        config = build_simulation_config(
            config_file=args.config_file,
            cli_overrides=overrides,
        )
        result = run_simulation(
            config=config,
            write_output=True,
            console_log=args.console_log,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if config.display_world:
        print(display_world(result.initial_world, title="Initial World (t = 0)"))
        print()
        print(
            display_world(
                result.final_world,
                title=f"Final World (t = {config.iterations})",
            )
        )
        print()

    if config.display_history:
        print(display_history(result.history))
        print()

    final_conv = result.payload["metadata"]["final_convergence"]
    print(
        f"DRIMA run finished: {config.iterations} iterations | "
        f"Interactions: {result.payload['metadata']['total_interactions']} "
        f"(effective: {result.payload['metadata']['effective_interactions']}) | "
        f"Mean R max-prob: {final_conv['r_mean_max_prob']:.4f} | "
        f"History saved to: {result.output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
