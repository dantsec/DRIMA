"""Two-dimensional periodic grid, Moore neighborhood discovery, and agent movement.

Project Overview & Conceptual Foundation
----------------------------------------
In DRIMA (Section 2.1 & Section 2.2.1), agents inhabit a discrete two-dimensional grid of size
`{X_max, Y_max}` with **periodic boundary conditions** (a torus): moving past the East edge
(`x = X_max`) wraps around to the West edge (`x = 1`), and moving past the North edge
(`y = Y_max`) wraps around to the South edge (`y = 1`).

Starting Simple and Building Up
-------------------------------
1. **Coordinate Wrapping (`wrap_position`):**
   Because DRIMA uses 1-indexed coordinates (`1..X_max`, `1..Y_max`), wrapping a raw coordinate
   `x` is computed as `((x - 1) % X_max) + 1`. For example, on a `5 x 10` grid, stepping West
   (`dx = -1`) from `x = 1` yields `((0 - 1) % 5) + 1 = 5`.
2. **Isotropic Moore Neighborhood (`moore_distance` and `get_neighbors`):**
   The range of action `R` is defined by a Moore neighborhood (Chebyshev distance on a torus).
   Along each axis, the shortest toroidal distance is `min(|a - b|, size - |a - b|)`. The Moore
   distance is `max(dx_torus, dy_torus)`.
   - When `R = 0`, `moore_distance == 0`, meaning two agents can interact *only* when they share
     the exact same grid cell `(x, y)` (Section 2.2.1).
   - When `R = 1` (the standard DRIMA setting), an agent interacts with any other agent on its
     own cell or any of the 8 contiguous cells around it.
3. **Probabilistic Movement (`move_agent`):**
   Each agent samples one of the 9 movement options (`E, NE, N, NW, W, SW, S, SE, X`) weighted
   by its `movement_pattern`. In Full (`F`) mobility, `X` has probability `0.0`, so the agent
   always steps to one of the 8 neighboring cells.

Common Pitfalls & What NOT To Do
--------------------------------
- **Do NOT use Euclidean distance or forget toroidal wrap-around in neighborhood checks:**
  On an `8 x 5` periodic grid, `(1, 1)` and `(8, 5)` are diagonal neighbors at Moore distance `1`,
  not `7`. Forgetting toroidal distance creates artificial boundary walls that distort the
  emergent multi-agent dynamics.
- **Do NOT include an agent as its own neighbor:** When scanning `all_agents` for neighbors
  within range `R`, always exclude the querying agent itself (`candidate.name != agent.name`),
  even when multiple agents share the same cell.

Example Usage
-------------
```python
import random
from src.drima.models import Agent
from src.drima.grid import wrap_position, moore_distance, get_neighbors, move_agent

# 1. Wrapping across periodic boundaries on a 5x10 grid
assert wrap_position(0, 11, x_max=5, y_max=10) == (5, 1)

# 2. Toroidal Moore distance between opposite corners (1, 1) and (5, 10) is 1
assert moore_distance((1, 1), (5, 10), x_max=5, y_max=10) == 1

# 3. Moving a deterministic agent East from (5, 2) wraps to (1, 2)
d01 = Agent.create_d_agent("D01", direction="E", position=(5, 2), mobility="F")
rec = move_agent(d01, x_max=5, y_max=10, rng=random.Random(1))
assert d01.position == (1, 2)
```
"""

import random
from typing import List, Optional, Sequence, Set, Tuple

from src.drima.models import (
    DIRECTIONS,
    DIRECTION_DELTAS,
    Agent,
    MovementRecord,
)


def wrap_position(x: int, y: int, x_max: int, y_max: int) -> Tuple[int, int]:
    """Wrap 1-indexed `(x, y)` coordinates onto the periodic grid `[1..x_max] x [1..y_max]`.

    Args:
        x: Raw horizontal coordinate (1-indexed).
        y: Raw vertical coordinate (1-indexed).
        x_max: Grid width (`X_max >= 1`).
        y_max: Grid height (`Y_max >= 1`).

    Returns:
        Wrapped `(x_wrapped, y_wrapped)` tuple with `1 <= x_wrapped <= x_max` and
        `1 <= y_wrapped <= y_max`.
    """
    if x_max < 1 or y_max < 1:
        raise ValueError(f"Grid dimensions must be >= 1, got ({x_max}, {y_max}).")
    wrapped_x = ((x - 1) % x_max) + 1
    wrapped_y = ((y - 1) % y_max) + 1
    return (wrapped_x, wrapped_y)


def moore_distance(
    pos1: Tuple[int, int],
    pos2: Tuple[int, int],
    x_max: int,
    y_max: int,
) -> int:
    """Compute the isotropic Moore (Chebyshev) distance on a periodic 2D grid.

    As defined in Section 2.2.1 of the DRIMA specification, the range of action is a Moore
    neighborhood of radius `R` on a periodic grid.

    Args:
        pos1: First 1-indexed position `(x1, y1)`.
        pos2: Second 1-indexed position `(x2, y2)`.
        x_max: Grid width (`X_max`).
        y_max: Grid height (`Y_max`).

    Returns:
        Integer Moore distance `>= 0` accounting for toroidal wrap-around.
    """
    dx_raw = abs(pos1[0] - pos2[0])
    dy_raw = abs(pos1[1] - pos2[1])
    dx = min(dx_raw, x_max - dx_raw)
    dy = min(dy_raw, y_max - dy_raw)
    return max(dx, dy)


def get_neighbors(
    agent: Agent,
    all_agents: Sequence[Agent],
    x_max: int,
    y_max: int,
    interaction_range: int = 1,
    excluded_names: Optional[Set[str]] = None,
) -> List[Agent]:
    """Return all other agents within Moore neighborhood radius `interaction_range`.

    Args:
        agent: The focal `Agent` looking for interaction partners.
        all_agents: Sequence of all agents in the world.
        x_max: Grid width (`X_max`).
        y_max: Grid height (`Y_max`).
        interaction_range: Moore neighborhood radius `R >= 0`. If `R == 0`, only agents sharing
            the exact same cell are returned.
        excluded_names: Optional set of agent names to exclude (e.g., already-processed agents
            when `sync_only_unprocessed_neighbors` is enabled).

    Returns:
        List of neighboring `Agent` instances within distance `<= interaction_range`.
    """
    if interaction_range < 0:
        raise ValueError(f"interaction_range must be >= 0, got {interaction_range}.")

    neighbors: List[Agent] = []
    for candidate in all_agents:
        if candidate.name == agent.name:
            continue
        if excluded_names and candidate.name in excluded_names:
            continue
        dist = moore_distance(agent.position, candidate.position, x_max, y_max)
        if dist <= interaction_range:
            neighbors.append(candidate)
    return neighbors


def sample_direction(
    movement_pattern: Sequence[float],
    rng: random.Random,
) -> str:
    """Sample a compass direction (`E`..`X`) according to the probability distribution.

    Args:
        movement_pattern: 9-element sequence of non-negative probabilities summing to 1.0.
        rng: `random.Random` instance for reproducible sampling.

    Returns:
        Chosen direction string from `DIRECTIONS`.
    """
    threshold = rng.random()
    cumulative = 0.0
    last_positive_idx = 0
    for idx, prob in enumerate(movement_pattern):
        if prob > 0.0:
            last_positive_idx = idx
            cumulative += prob
            if threshold <= cumulative:
                return DIRECTIONS[idx]
    return DIRECTIONS[last_positive_idx]


def move_agent(
    agent: Agent,
    x_max: int,
    y_max: int,
    rng: random.Random,
    occupied_positions: Optional[Set[Tuple[int, int]]] = None,
    allow_shared_cells: bool = True,
) -> MovementRecord:
    """Advance `agent` on the periodic grid according to its `movement_pattern`.

    If `allow_shared_cells` is `True` (default in DRIMA), any number of agents may occupy the
    same cell. If `allow_shared_cells` is `False` and `occupied_positions` is provided, moving
    into an already occupied cell is prevented (the agent first attempts alternative directions
    with positive probability, or remains in its current cell if all reachable target cells are
    occupied).

    Args:
        agent: The `Agent` to move (mutates `agent.position` in place).
        x_max: Grid width (`X_max`).
        y_max: Grid height (`Y_max`).
        rng: `random.Random` instance.
        occupied_positions: Set of currently occupied `(x, y)` cells when
            `allow_shared_cells=False`. Updated in place if provided.
        allow_shared_cells: Whether multiple agents can occupy the same grid cell.

    Returns:
        `MovementRecord` describing the chosen direction and pre/post positions.
    """
    pos_before = agent.position
    chosen_dir = sample_direction(agent.movement_pattern, rng)
    dx, dy = DIRECTION_DELTAS[chosen_dir]
    candidate_pos = wrap_position(pos_before[0] + dx, pos_before[1] + dy, x_max, y_max)

    if allow_shared_cells or occupied_positions is None:
        agent.position = candidate_pos
        return MovementRecord(
            agent_name=agent.name,
            direction=chosen_dir,
            position_before=pos_before,
            position_after=candidate_pos,
            blocked_by_occupancy=False,
        )

    # Single-agent-per-cell occupancy mode
    if candidate_pos == pos_before or candidate_pos not in occupied_positions:
        occupied_positions.discard(pos_before)
        occupied_positions.add(candidate_pos)
        agent.position = candidate_pos
        return MovementRecord(
            agent_name=agent.name,
            direction=chosen_dir,
            position_before=pos_before,
            position_after=candidate_pos,
            blocked_by_occupancy=False,
        )

    # Target cell is occupied: stay in current cell and mark blocked_by_occupancy
    return MovementRecord(
        agent_name=agent.name,
        direction="X",
        position_before=pos_before,
        position_after=pos_before,
        blocked_by_occupancy=True,
    )
