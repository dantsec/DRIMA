"""World creation, synchronous/asynchronous update cycles, and convergence analysis.

Project Overview & How World Updates Work
-----------------------------------------
In DRIMA (Section 2.1, Section 2.2.4, & Section 3), the dynamics of the system unfold over a
sequence of discrete time steps ("worlds") that together form a **history**.

Every update step follows the **interact-first-then-move** cycle (Section 2.1):
1. **Interaction Phase:**
   - **Synchronous Mode (`sync_world_update` / `SyncWorldUpdate`):**
     At the start of the iteration, a **random permutation** of all agents is generated to avoid
     ordering artifacts. The system iterates through this permuted sequence:
     - If an agent has *already* been processed in the current iteration (i.e., it was previously
       selected as a neighboring partner in an earlier interaction during this same step), it is
       skipped.
     - Otherwise, all of its neighbors within Moore distance `<= R` are identified. One neighbor
       is chosen at random, the binary interaction (`interact_pair`) is carried out, and both
       agents are marked as processed for the current step.
   - **Asynchronous Mode (`async_world_update` / `AsyncWorldUpdate`):**
     A single agent is picked uniformly at random from the world, one of its neighbors within
     range `R` is randomly selected (if any exist), and the binary interaction happens between
     those two agents.
2. **Movement Phase:**
   After the interaction phase concludes, agents step on the periodic grid according to their
   updated movement patterns.

Common Pitfalls & What NOT To Do
--------------------------------
- **Do NOT move agents before interacting:** Section 2.1 emphasizes that DRIMA uses an
  *interact-first-then-move* cycle because the initial world starts with predefined uniform
  movement patterns.
- **Do NOT iterate over agents in a fixed index order (`D01, D02, R01, ...`) in synchronous
  mode:** Always take a random permutation at every step (Section 2.2.4) so earlier agents in
  the list do not systematically dominate neighborhood pairings.
- **Do NOT allow an agent that was already picked as a neighbor in synchronous mode to initiate
  a second interaction in the same step:** Section 2.2.4 states: *"every agent in the world is
  sequentially considered for interaction, but only those that have not yet been processed in
  the current iteration, that is, those that have not yet been selected as a neighboring agent
  of a previous interaction."*

Example Usage
-------------
```python
import random
from src.drima.models import SimulationConfig
from src.drima.world import create_world, sync_world_update, compute_convergence_metrics

config = SimulationConfig(grid_width=8, grid_height=5, num_d_agents=1, num_r_agents=4, seed=7)
rng = random.Random(config.seed)
world = create_world(config, rng)

next_world, step_log = sync_world_update(world, config, rng, step_index=1)
metrics = compute_convergence_metrics(next_world)
print(f"Step 1 interactions: {len(step_log.interactions)}, mean R max-prob: {metrics['r_mean_max_prob']:.3f}")
```
"""

import math
import random
from typing import Any, Dict, List, Set, Tuple

from src.drima.grid import get_neighbors, move_agent
from src.drima.interactions import interact_pair
from src.drima.models import (
    DIRECTIONS,
    Agent,
    InteractionRecord,
    MovementRecord,
    SimulationConfig,
    StepLog,
    World,
)


def _sample_initial_position(
    x_max: int,
    y_max: int,
    rng: random.Random,
    occupied: Set[Tuple[int, int]],
    allow_shared_cells: bool,
) -> Tuple[int, int]:
    """Sample a 1-indexed `(x, y)` grid position, respecting `allow_shared_cells`."""
    if allow_shared_cells:
        return (rng.randint(1, x_max), rng.randint(1, y_max))

    available = [
        (x, y)
        for x in range(1, x_max + 1)
        for y in range(1, y_max + 1)
        if (x, y) not in occupied
    ]
    if not available:
        raise ValueError(
            f"No unoccupied cells remaining on {x_max}x{y_max} grid "
            f"(allow_shared_cells=False)."
        )
    chosen = rng.choice(available)
    occupied.add(chosen)
    return chosen


def create_world(config: SimulationConfig, rng: random.Random) -> World:
    """Create and initialize a DRIMA `World` from `SimulationConfig`.

    If `config.agents` is provided, agents are constructed directly from those definitions
    (supporting both JSON dictionaries and canonical DRIMA lists). Otherwise, `num_d_agents`
    (`D01`, `D02`, ...) and `num_r_agents` (`R01`, `R02`, ...) are generated automatically.

    Args:
        config: `SimulationConfig` specifying grid dimensions, mobility, and agent setup.
        rng: `random.Random` instance for reproducible initialization.

    Returns:
        Initialized `World` snapshot at `t = 0`.
    """
    x_max = config.grid_width
    y_max = config.grid_height
    mobility = config.world_mobility.upper()
    occupied: Set[Tuple[int, int]] = set()
    agents: List[Agent] = []

    if config.agents is not None:
        for agent_spec in config.agents:
            if isinstance(agent_spec, dict) and "position" not in agent_spec:
                spec_copy = dict(agent_spec)
                spec_copy["position"] = list(
                    _sample_initial_position(
                        x_max, y_max, rng, occupied, config.allow_shared_cells
                    )
                )
                agent = Agent.from_dict(spec_copy, default_mobility=mobility)
            else:
                agent = Agent.from_dict(agent_spec, default_mobility=mobility)
                if not config.allow_shared_cells and agent.position in occupied:
                    raise ValueError(
                        f"Agent '{agent.name}' shares position {agent.position} with another "
                        f"agent, but allow_shared_cells=False."
                    )
                occupied.add(agent.position)
            agents.append(agent)
    else:
        allowed_d_dirs = (
            list(DIRECTIONS[:8]) if mobility == "F" else list(DIRECTIONS)
        )
        pad_d = max(2, len(str(max(1, config.num_d_agents))))
        for idx in range(1, config.num_d_agents + 1):
            name = f"D{idx:0{pad_d}d}"
            if config.d_directions and (idx - 1) < len(config.d_directions):
                direction = config.d_directions[idx - 1].upper()
            elif config.d_directions:
                direction = config.d_directions[(idx - 1) % len(config.d_directions)].upper()
            else:
                direction = rng.choice(allowed_d_dirs)

            pos = _sample_initial_position(
                x_max, y_max, rng, occupied, config.allow_shared_cells
            )
            agents.append(
                Agent.create_d_agent(
                    name=name,
                    direction=direction,
                    position=pos,
                    mobility=mobility,
                    transition_potential=config.d_transition_potential,
                )
            )

        pad_r = max(2, len(str(max(1, config.num_r_agents))))
        for idx in range(1, config.num_r_agents + 1):
            name = f"R{idx:0{pad_r}d}"
            pos = _sample_initial_position(
                x_max, y_max, rng, occupied, config.allow_shared_cells
            )
            agents.append(
                Agent.create_r_agent(
                    name=name,
                    position=pos,
                    mobility=mobility,
                )
            )

    return World(
        agents=agents,
        grid_size=(x_max, y_max),
        interaction_types=list(config.interaction_types),
        world_mobility=mobility,
    )


def sync_world_update(
    world: World,
    config: SimulationConfig,
    rng: random.Random,
    step_index: int,
) -> Tuple[World, StepLog]:
    """Perform a single synchronous world update (`SyncWorldUpdate` in Section 3).

    Follows the interact-first-then-move cycle (Section 2.1 & Section 2.2.4):
    1. Clone the world to preserve the previous snapshot in history.
    2. Permute all agents randomly.
    3. Sequentially consider each agent not yet processed in this iteration. Pick one of its
       neighbors within range `R` at random, perform the binary interaction, and mark both
       as processed.
    4. Move all agents on the periodic grid according to their updated movement patterns.

    Args:
        world: Current `World` snapshot.
        config: `SimulationConfig` governing interaction and movement parameters.
        rng: `random.Random` instance.
        step_index: 1-based step number for logging.

    Returns:
        Tuple `(next_world, step_log)`.
    """
    next_world = world.clone()
    x_max, y_max = next_world.grid_size

    permuted_agents = list(next_world.agents)
    rng.shuffle(permuted_agents)
    agent_order = [a.name for a in permuted_agents]

    processed_names: Set[str] = set()
    interactions: List[InteractionRecord] = []

    # 1. Interaction Phase
    for agent in permuted_agents:
        if agent.name in processed_names:
            continue

        excluded = processed_names if config.sync_only_unprocessed_neighbors else None
        neighbors = get_neighbors(
            agent=agent,
            all_agents=next_world.agents,
            x_max=x_max,
            y_max=y_max,
            interaction_range=config.interaction_range,
            excluded_names=excluded,
        )
        processed_names.add(agent.name)

        if not neighbors:
            continue

        partner = rng.choice(neighbors)
        processed_names.add(partner.name)

        record = interact_pair(agent, partner, config)
        interactions.append(record)

    # 2. Movement Phase
    occupied_positions = (
        {a.position for a in next_world.agents}
        if not config.allow_shared_cells
        else None
    )
    movements: List[MovementRecord] = []
    for agent in next_world.agents:
        move_rec = move_agent(
            agent=agent,
            x_max=x_max,
            y_max=y_max,
            rng=rng,
            occupied_positions=occupied_positions,
            allow_shared_cells=config.allow_shared_cells,
        )
        movements.append(move_rec)

    step_log = StepLog(
        step=step_index,
        update_mode="sync",
        agent_order=agent_order,
        interactions=interactions,
        movements=movements,
    )
    return (next_world, step_log)


def async_world_update(
    world: World,
    config: SimulationConfig,
    rng: random.Random,
    step_index: int,
) -> Tuple[World, StepLog]:
    """Perform a single asynchronous world update (`AsyncWorldUpdate` in Section 3).

    As described in Section 2.2.4:
    1. Clone the world to preserve the previous snapshot in history.
    2. Randomly pick one agent from the world, and randomly pick one of its neighbors within
       range `R` (if any exist).
    3. Execute the binary interaction between the pair.
    4. Move the agents (all agents if `config.async_move_all` is `True`, or only the selected
       interacting pair if `False`).

    Args:
        world: Current `World` snapshot.
        config: `SimulationConfig` governing interaction and movement parameters.
        rng: `random.Random` instance.
        step_index: 1-based step number for logging.

    Returns:
        Tuple `(next_world, step_log)`.
    """
    next_world = world.clone()
    x_max, y_max = next_world.grid_size

    focal_agent = rng.choice(next_world.agents)
    agent_order = [focal_agent.name]
    interactions: List[InteractionRecord] = []
    involved_agents = [focal_agent]

    neighbors = get_neighbors(
        agent=focal_agent,
        all_agents=next_world.agents,
        x_max=x_max,
        y_max=y_max,
        interaction_range=config.interaction_range,
    )
    if neighbors:
        partner = rng.choice(neighbors)
        agent_order.append(partner.name)
        involved_agents.append(partner)
        record = interact_pair(focal_agent, partner, config)
        interactions.append(record)

    occupied_positions = (
        {a.position for a in next_world.agents}
        if not config.allow_shared_cells
        else None
    )
    agents_to_move = next_world.agents if config.async_move_all else involved_agents
    movements: List[MovementRecord] = []
    for agent in agents_to_move:
        move_rec = move_agent(
            agent=agent,
            x_max=x_max,
            y_max=y_max,
            rng=rng,
            occupied_positions=occupied_positions,
            allow_shared_cells=config.allow_shared_cells,
        )
        movements.append(move_rec)

    step_log = StepLog(
        step=step_index,
        update_mode="async",
        agent_order=agent_order,
        interactions=interactions,
        movements=movements,
    )
    return (next_world, step_log)


def compute_convergence_metrics(world: World) -> Dict[str, Any]:
    """Compute convergence and determinism statistics for R-agents in `world`.

    In DRIMA (Section 2.1 & Section 4), a central analytical question is how rapidly R-agents
    converge from uniform random movement (`max_prob = 0.125` in Full mobility or `0.111` in
    Partial mobility) toward deterministic movement (`max_prob -> 1.0`).

    Args:
        world: `World` snapshot to analyze.

    Returns:
        Dictionary with per-agent dominant directions, maximum probabilities, normalized Shannon
        entropies, and aggregate R-agent convergence metrics.
    """
    r_agents = [a for a in world.agents if a.agent_type == "R"]
    d_agents = [a for a in world.agents if a.agent_type == "D"]

    max_entropy = math.log(8.0) if world.world_mobility == "F" else math.log(9.0)
    per_agent: Dict[str, Dict[str, Any]] = {}

    for agent in world.agents:
        entropy = 0.0
        for prob in agent.movement_pattern:
            if prob > 1e-15:
                entropy -= prob * math.log(prob)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        per_agent[agent.name] = {
            "type": agent.agent_type,
            "position": list(agent.position),
            "dominant_direction": agent.dominant_direction,
            "max_probability": round(agent.max_probability, 6),
            "determinism_score": round(max(0.0, 1.0 - norm_entropy), 6),
            "is_converged_95": agent.max_probability >= 0.95,
        }

    if r_agents:
        r_max_probs = [a.max_probability for a in r_agents]
        r_mean_max_prob = sum(r_max_probs) / len(r_max_probs)
        r_converged_count = sum(1 for p in r_max_probs if p >= 0.95)
    else:
        r_mean_max_prob = 1.0
        r_converged_count = 0

    return {
        "num_d_agents": len(d_agents),
        "num_r_agents": len(r_agents),
        "r_mean_max_prob": round(r_mean_max_prob, 6),
        "r_converged_95_count": r_converged_count,
        "r_all_converged_95": len(r_agents) > 0 and r_converged_count == len(r_agents),
        "agents": per_agent,
    }
