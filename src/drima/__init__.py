"""DRIMA: A Minimal System for Probing the Dynamics of Change in a Reactive Multi-Agent Setting.

Project Overview
----------------
DRIMA (Dynamics of Randomly Interacting Moving Agents) is a cellular multi-agent simulation
system originally conceived by Pedro P. B. de Oliveira (*The Mathematica Journal*, 12, 2010).
It serves as a minimal computational metaphor for how reactive agents "attract" others toward
their behavioral or conceptual orientation—such as a consumer choosing between competing
products, an apprentice adopting habits, or random-walking physical particles aligning under
local forces.

In DRIMA, all behavioral state is distilled into a single observable action: how an agent
moves across a two-dimensional periodic (toroidal) grid. Two types of agents inhabit the grid:
- **D-agents (Deterministic):** Move in a single fixed direction with probability 1.0 (and 0.0
  in all other directions). They act as stable attractors or behavioral anchors.
- **R-agents (Random):** Move probabilistically according to a 9-component movement pattern
  covering the 8 compass directions (`E, NE, N, NW, W, SW, S, SE`) plus no-movement (`X`).

At each step of a world history, agents follow an **interact-first-then-move** cycle:
1. **Interact:** Agents within a Moore neighborhood of radius `R` interact pairwise. In a
   direct one-way interaction, for each direction independently, the smaller probability is
   increased toward the larger probability by `delta_p`, followed by iterative
   equal-displacement normalization so probabilities continue to sum to 1.0.
2. **Move:** Agents advance on the periodic grid by sampling a direction from their updated
   movement pattern.

Example Usage
-------------
```python
from src.drima.models import SimulationConfig
from src.drima.simulation import run_simulation
from src.drima.visualizer import display_world

# Configure an 8x5 world with 2 D-agents (moving East and Southeast) and 6 R-agents
config = SimulationConfig(
    grid_width=8,
    grid_height=5,
    num_d_agents=2,
    num_r_agents=6,
    d_directions=["E", "SE"],
    world_mobility="F",
    interaction_types=["dd", "RR", "dR", "Rd"],
    update_mode="sync",
    delta_mode="fixed",
    fp=1.0,
    interaction_range=1,
    iterations=50,
    seed=42,
)

result = run_simulation(config)
print(display_world(result.final_world))
```
"""

from src.drima.models import (
    DIRECTIONS,
    DIRECTION_DELTAS,
    DEFAULT_INTERACTION_TYPES,
    VALID_INTERACTION_TYPES,
    Agent,
    AgentProperties,
    InteractionRecord,
    MovementRecord,
    SimulationConfig,
    StepLog,
    World,
)
from src.drima.grid import (
    get_neighbors,
    moore_distance,
    move_agent,
    wrap_position,
)
from src.drima.normalization import (
    normalize_equal_displacement,
    normalize_pattern,
    normalize_proportional,
)
from src.drima.interactions import (
    compute_delta_p,
    determine_modifiability,
    interact_pair,
    resolve_interaction_types,
)
from src.drima.world import (
    async_world_update,
    compute_convergence_metrics,
    create_world,
    sync_world_update,
)
from src.drima.simulation import SimulationResult, run_simulation
from src.drima.visualizer import display_history, display_world
from src.drima.analysis import (
    extract_agent_probability_series,
    extract_convergence_series,
    extract_interaction_series,
    extract_position_series,
    load_or_validate_history,
)

__all__ = [
    "DIRECTIONS",
    "DIRECTION_DELTAS",
    "DEFAULT_INTERACTION_TYPES",
    "VALID_INTERACTION_TYPES",
    "Agent",
    "AgentProperties",
    "InteractionRecord",
    "MovementRecord",
    "SimulationConfig",
    "SimulationResult",
    "StepLog",
    "World",
    "async_world_update",
    "compute_convergence_metrics",
    "compute_delta_p",
    "create_world",
    "determine_modifiability",
    "display_history",
    "display_world",
    "extract_agent_probability_series",
    "extract_convergence_series",
    "extract_interaction_series",
    "extract_position_series",
    "get_neighbors",
    "interact_pair",
    "load_or_validate_history",
    "moore_distance",
    "move_agent",
    "normalize_equal_displacement",
    "normalize_pattern",
    "normalize_proportional",
    "resolve_interaction_types",
    "run_simulation",
    "sync_world_update",
    "wrap_position",
]

__version__ = "1.0.0"
