"""Terminal/ASCII visualization functions (`DisplayWorld` and `DisplayHistory` equivalents).

Project Overview
----------------
In the original *Mathematica* implementation of DRIMA (Section 3, Figures 1 & 2), two core
visualization functions are provided:
- `DisplayWorld[world]`: Visualizes the 2D cellular grid, showing the positions of all D-agents
  and R-agents and their tags.
- `DisplayHistory[history]`: Visualizes the evolution of movement pattern probabilities for each
  agent across the sequence of worlds in a history.

This module implements pure-Python terminal/ASCII equivalents (`display_world` and
`display_history`) using only the Python standard library.

Example Usage
-------------
```python
from src.drima.models import Agent, World
from src.drima.visualizer import display_world, display_history

d01 = Agent.create_d_agent("D01", direction="E", position=(2, 2), mobility="F")
r01 = Agent.create_r_agent("R01", position=(4, 3), mobility="F")
world = World(agents=[d01, r01], grid_size=(5, 4), world_mobility="F")

print(display_world(world))
print(display_history([world]))
```
"""

from typing import Dict, List, Optional, Sequence, Tuple

from src.drima.models import DIRECTIONS, World


DIRECTION_ARROWS: Dict[str, str] = {
    "E": "→",
    "NE": "↗",
    "N": "↑",
    "NW": "↖",
    "W": "←",
    "SW": "↙",
    "S": "↓",
    "SE": "↘",
    "X": "•",
}

SPARKLINE_BLOCKS: str = " ·▂▃▄▅▆▇█"


def display_world(world: World, title: Optional[str] = None) -> str:
    """Render an ASCII grid and agent table representing a DRIMA `World` (`DisplayWorld`).

    Rows are displayed from `y = Y_max` (North, top) down to `y = 1` (South, bottom), and
    columns from `x = 1` (West, left) to `x = X_max` (East, right).

    Args:
        world: The `World` snapshot to render.
        title: Optional header title.

    Returns:
        Multi-line formatted string depicting the grid and all agent movement patterns.
    """
    x_max, y_max = world.grid_size
    cell_map: Dict[Tuple[int, int], List[str]] = {}
    for agent in world.agents:
        arrow = DIRECTION_ARROWS.get(agent.dominant_direction, "")
        tag = f"{agent.name}{arrow}"
        cell_map.setdefault(agent.position, []).append(tag)

    cell_width = max(
        6,
        max((len(",".join(tags)) for tags in cell_map.values()), default=4) + 2,
    )

    lines: List[str] = []
    header = title or (
        f"DRIMA World ({x_max}x{y_max} grid, mobility={world.world_mobility}, "
        f"interactions={world.interaction_types})"
    )
    lines.append(header)
    lines.append("=" * len(header))

    border_segment = "+" + "+".join("-" * cell_width for _ in range(x_max)) + "+"
    for y in range(y_max, 0, -1):
        lines.append(f"    {border_segment}")
        row_cells: List[str] = []
        for x in range(1, x_max + 1):
            tags = cell_map.get((x, y), [])
            cell_text = ",".join(tags) if tags else "."
            row_cells.append(cell_text.center(cell_width))
        lines.append(f"y={y:<2d}|" + "|".join(row_cells) + "|")
    lines.append(f"    {border_segment}")

    x_labels = [f"x={x}".center(cell_width) for x in range(1, x_max + 1)]
    lines.append("     " + " ".join(x_labels))
    lines.append("")

    # Agent Movement Pattern Table
    dir_headers = " ".join(f"{d:>6s}" for d in DIRECTIONS)
    table_header = f"{'Agent':<6s} {'Pos':<8s} {'Dom':<4s} | {dir_headers}"
    lines.append(table_header)
    lines.append("-" * len(table_header))
    for agent in world.agents:
        pos_str = f"({agent.position[0]},{agent.position[1]})"
        probs_str = " ".join(f"{p:6.3f}" for p in agent.movement_pattern)
        lines.append(
            f"{agent.name:<6s} {pos_str:<8s} {agent.dominant_direction:<4s} | {probs_str}"
        )

    return "\n".join(lines)


def _render_sparkline(values: Sequence[float], width: int = 32) -> str:
    """Downsample or map a sequence of probabilities in `[0, 1]` to a compact sparkline."""
    if not values:
        return ""
    if len(values) <= width:
        sampled = list(values)
    else:
        sampled = [
            values[int(round(i * (len(values) - 1) / (width - 1)))]
            for i in range(width)
        ]

    chars: List[str] = []
    max_idx = len(SPARKLINE_BLOCKS) - 1
    for val in sampled:
        clamped = max(0.0, min(1.0, val))
        idx = int(round(clamped * max_idx))
        chars.append(SPARKLINE_BLOCKS[idx])
    return "".join(chars)


def display_history(
    history: Sequence[World],
    agent_names: Optional[Sequence[str]] = None,
    sparkline_width: int = 36,
) -> str:
    """Render the evolution of movement pattern probabilities across a history (`DisplayHistory`).

    Mirrors Figure 2 of the DRIMA paper by plotting the probability trajectory of each direction
    for every selected agent from `t = 0` to `t = len(history) - 1`.

    Args:
        history: Sequence of `World` snapshots in chronological order.
        agent_names: Optional subset of agent names to display. Defaults to all agents.
        sparkline_width: Number of character columns in each direction's probability sparkline.

    Returns:
        Multi-line string summarizing the history evolution.
    """
    if not history:
        return "Empty DRIMA history."

    initial_world = history[0]
    final_world = history[-1]
    selected_names = (
        list(agent_names)
        if agent_names is not None
        else [a.name for a in initial_world.agents]
    )

    lines: List[str] = []
    title = f"DRIMA History Evolution ({len(history) - 1} steps, {len(history)} snapshots)"
    lines.append(title)
    lines.append("=" * len(title))

    for name in selected_names:
        agent_init = initial_world.get_agent(name)
        agent_final = final_world.get_agent(name)
        if agent_init is None or agent_final is None:
            continue

        lines.append(
            f"\nAgent {name} (type={agent_final.agent_type}) | "
            f"pos: {agent_init.position} -> {agent_final.position} | "
            f"dominant: {agent_init.dominant_direction} ({agent_init.max_probability:.3f}) -> "
            f"{agent_final.dominant_direction} ({agent_final.max_probability:.3f})"
        )
        lines.append("-" * 68)

        for d_idx, d_name in enumerate(DIRECTIONS):
            series = [
                w.get_agent(name).movement_pattern[d_idx]  # type: ignore[union-attr]
                for w in history
            ]
            max_series = max(series)
            # Always display directions that had non-zero probability at some point
            if max_series <= 1e-9 and d_name != "X":
                continue
            spark = _render_sparkline(series, width=sparkline_width)
            lines.append(
                f"  {d_name:<2s} | start={series[0]:.3f} -> end={series[-1]:.3f} "
                f"(max={max_series:.3f}) | [{spark}]"
            )

    return "\n".join(lines)
