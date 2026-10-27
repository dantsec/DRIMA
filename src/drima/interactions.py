"""Direct binary interactions between agents in DRIMA.

Project Overview & How DRIMA Interactions Work
----------------------------------------------
In DRIMA (Section 2.2 & Section 3.2), interactions are **binary** (two agents at a time) and
**direct** (immediately modifying the interacting agents' movement patterns).

Starting Simple and Building Up
-------------------------------
1. **Who Can Be Modified? (`resolve_interaction_types` & `determine_modifiability`):**
   - World `interaction_types` is a list of two-letter codes such as `["dd", "RR", "dR", "Rd"]`.
   - **Uppercase** means the corresponding agent type is allowed to change; **lowercase** means
     it comes out of the interaction unchanged.
   - Conflict rule (Section 3.1): If both `"DD"` and `"dd"` are specified, `"DD"` prevails; if
     both `"RR"` and `"rr"` are specified, `"RR"` prevails.
   - Between a D-agent and an R-agent, specifying either `"dR"` or `"Rd"` allows the R-agent to
     be modified while the D-agent remains unchanged.

2. **How Much Does a Direction Probability Change? (`compute_delta_p`):**
   For each direction field `d` independently, given probabilities `P_d^i` and `P_d^j`:
   - **Fixed mode (Equation 1):**
     `delta_p = F_p / 9.0`
   - **Variable mode (Equation 2):**
     `delta_p = (F_p / 9.0) * abs(P_d^i - P_d^j)`

3. **Local Pairwise Operation & Clamping (`interact_pair`):**
   - In **one-way** interaction (Section 3.2): at each direction field `d`, only the *smaller*
     probability of the pair `(P_d^i, P_d^j)` is increased by `delta_p` (if that agent is
     modifiable).
   - Crucial subtlety from Section 3.2: Because this comparison happens *per direction field*,
     in an `RR` interaction between `R01` and `R02`, `R01` might have the smaller probability in
     East (and thus increase in East), while `R02` might have the smaller probability in North
     (and thus increase in North) during the *same* binary interaction!
   - **Clamping Rule (Section 3.2):** The increased smaller probability is never allowed to
     exceed the larger pre-interaction probability: `new_small = min(large, small + delta_p)`.
   - After all direction fields are updated, any modified agent's movement pattern is normalized
     via `normalize_pattern` (defaulting to iterative equal-displacement normalization).

Common Pitfalls & What NOT To Do
--------------------------------
- **Do NOT normalize after each individual direction field `d`:** Normalization happens *once* per
  agent after all 9 direction fields have undergone their pairwise local operation.
- **Do NOT mutate `agent_i.movement_pattern` while iterating over directions before `agent_j`
  reads the pre-interaction values:** Both agents must compare against each other's
  *pre-interaction* probabilities `P_d^i` and `P_d^j`.
- **Do NOT let `small + delta_p` overshoot `large`:** In fixed mode with `F_p = 1`
  (`delta_p = 1/9 ≈ 0.111`), if `P_d^i = 0.10` and `P_d^j = 0.15`, `P_d^i + delta_p = 0.211`
  would exceed `P_d^j = 0.15` without clamping. Clamping to `0.15` preserves the principle of
  attraction toward the other agent.

Example Usage
-------------
```python
from src.drima.models import Agent, SimulationConfig
from src.drima.interactions import interact_pair

d01 = Agent.create_d_agent("D01", direction="E", position=(2, 2), mobility="F")
r01 = Agent.create_r_agent("R01", position=(2, 3), mobility="F")
config = SimulationConfig(fp=1.0, delta_mode="fixed", world_mobility="F")

record = interact_pair(d01, r01, config)
# D01 is unchanged; R01's probability of moving East (index 0) has increased!
assert not record.agent_i_modified
assert record.agent_j_modified
assert r01.movement_pattern[0] > 0.125
```
"""

from typing import Iterable, Set, Tuple

from src.drima.models import (
    PROBABILITY_TOLERANCE,
    VALID_INTERACTION_TYPES,
    Agent,
    InteractionRecord,
    SimulationConfig,
)
from src.drima.normalization import normalize_pattern


def resolve_interaction_types(interaction_types: Iterable[str]) -> Set[str]:
    """Validate interaction type codes and resolve `DD`/`dd` and `RR`/`rr` conflicts.

    According to Section 3.1 of the DRIMA specification:
    - If both `'DD'` and `'dd'` are present, `'DD'` prevails.
    - If both `'RR'` and `'rr'` are present, `'RR'` prevails.

    Args:
        interaction_types: Iterable of interaction codes (subset of `VALID_INTERACTION_TYPES`).

    Returns:
        Set of active interaction codes with conflicts resolved.
    """
    active: Set[str] = set()
    for code in interaction_types:
        if code not in VALID_INTERACTION_TYPES:
            raise ValueError(
                f"Invalid interaction type '{code}'. Allowed: {VALID_INTERACTION_TYPES}."
            )
        active.add(code)

    if "DD" in active and "dd" in active:
        active.discard("dd")
    if "RR" in active and "rr" in active:
        active.discard("rr")

    return active


def determine_modifiability(
    agent_i: Agent,
    agent_j: Agent,
    active_types: Set[str],
) -> Tuple[bool, bool]:
    """Determine whether `agent_i` and `agent_j` can be modified in a binary interaction.

    Implements the rules in Section 3.1:
    - If `active_types` is empty (`{}`), neither agent can be modified.
    - Both D-agents (`D` and `D`): both modifiable iff `'DD'` in `active_types`.
    - Both R-agents (`R` and `R`): both modifiable iff `'RR'` in `active_types`.
    - Mixed (`D` and `R` or `R` and `D`):
      - As stated in Section 3.1, specifying `'dR'` or `'Rd'` suffices for an interaction
        between D and R to occur in which the R-agent is modified (`R` uppercase) and the
        D-agent is unchanged (`d` lowercase).
      - If `'Dr'` or `'rD'` is explicitly present, the D-agent is also allowed to be modified.

    Args:
        agent_i: First interacting `Agent`.
        agent_j: Second interacting `Agent`.
        active_types: Conflict-resolved set of active interaction codes.

    Returns:
        Tuple `(can_modify_i, can_modify_j)` of booleans.
    """
    if not active_types:
        return (False, False)

    type_i = agent_i.agent_type
    type_j = agent_j.agent_type

    if type_i == "D" and type_j == "D":
        can_mod = "DD" in active_types
        return (can_mod, can_mod)

    if type_i == "R" and type_j == "R":
        can_mod = "RR" in active_types
        return (can_mod, can_mod)

    # One agent is D and the other is R
    r_modifiable = ("dR" in active_types) or ("Rd" in active_types)
    d_modifiable = ("Dr" in active_types) or ("rD" in active_types)

    if type_i == "D" and type_j == "R":
        return (d_modifiable, r_modifiable)
    return (r_modifiable, d_modifiable)


def compute_delta_p(
    p_i: float,
    p_j: float,
    fp: float = 1.0,
    delta_mode: str = "fixed",
) -> float:
    """Compute the probability increment `delta_p` for a single direction field.

    Implements Equations (1) and (2) from Section 3.2:
    - Fixed mode: `delta_p = F_p / 9`
    - Variable mode: `delta_p = (F_p / 9) * |P_d^i - P_d^j|`

    Args:
        p_i: Probability of `agent_i` in direction `d`.
        p_j: Probability of `agent_j` in direction `d`.
        fp: Probability increment factor `F_p >= 0` (default `1.0`).
        delta_mode: `'fixed'` or `'variable'`.

    Returns:
        Non-negative float `delta_p`.
    """
    mode = delta_mode.lower()
    if mode == "fixed":
        return float(fp) / 9.0
    if mode == "variable":
        return (float(fp) / 9.0) * abs(p_i - p_j)
    raise ValueError(f"delta_mode must be 'fixed' or 'variable', got '{delta_mode}'.")


def interact_pair(
    agent_i: Agent,
    agent_j: Agent,
    config: SimulationConfig,
) -> InteractionRecord:
    """Execute a direct binary interaction between `agent_i` and `agent_j`.

    Updates `agent_i.movement_pattern` and `agent_j.movement_pattern` in place (if modifiable)
    and normalizes any modified pattern using `config.normalization_method`.

    Args:
        agent_i: First interacting `Agent`.
        agent_j: Second interacting `Agent`.
        config: `SimulationConfig` governing `interaction_types`, `world_mobility`, `fp`,
            `delta_mode`, `interaction_direction`, and `normalization_method`.

    Returns:
        `InteractionRecord` documenting the pre- and post-interaction movement patterns.
    """
    pattern_i_before = list(agent_i.movement_pattern)
    pattern_j_before = list(agent_j.movement_pattern)

    active_types = resolve_interaction_types(config.interaction_types)
    can_mod_i, can_mod_j = determine_modifiability(agent_i, agent_j, active_types)

    if not can_mod_i and not can_mod_j:
        return InteractionRecord(
            agent_i_name=agent_i.name,
            agent_j_name=agent_j.name,
            agent_i_pos=agent_i.position,
            agent_j_pos=agent_j.position,
            agent_i_modified=False,
            agent_j_modified=False,
            agent_i_pattern_before=pattern_i_before,
            agent_j_pattern_before=pattern_j_before,
            agent_i_pattern_after=list(pattern_i_before),
            agent_j_pattern_after=list(pattern_j_before),
        )

    new_pattern_i = list(pattern_i_before)
    new_pattern_j = list(pattern_j_before)

    # Under Full ('F') mobility, index 8 (no-move X) is kept at 0.0
    num_active_dirs = 8 if config.world_mobility.upper() == "F" else 9
    two_way = config.interaction_direction.lower() == "two-way"

    # Scale factor for optional agent susceptibility & partner influence potential
    scale_on_i = agent_i.properties.susceptibility * agent_j.properties.influence_potential
    scale_on_j = agent_j.properties.susceptibility * agent_i.properties.influence_potential

    changed_i = False
    changed_j = False

    for d_idx in range(num_active_dirs):
        p_i = pattern_i_before[d_idx]
        p_j = pattern_j_before[d_idx]

        if abs(p_i - p_j) <= 1e-15:
            continue

        base_delta = compute_delta_p(
            p_i=p_i,
            p_j=p_j,
            fp=config.fp,
            delta_mode=config.delta_mode,
        )
        if base_delta <= 0.0:
            continue

        if p_i < p_j:
            # agent_i has the smaller probability in direction d_idx
            if can_mod_i:
                delta_i = base_delta * scale_on_i
                updated_i = min(p_j, p_i + delta_i)
                if abs(updated_i - p_i) > 1e-15:
                    new_pattern_i[d_idx] = updated_i
                    changed_i = True
            if two_way and can_mod_j:
                delta_j = base_delta * scale_on_j
                updated_j = max(p_i, p_j - delta_j)
                if abs(updated_j - p_j) > 1e-15:
                    new_pattern_j[d_idx] = updated_j
                    changed_j = True
        else:
            # p_j < p_i: agent_j has the smaller probability in direction d_idx
            if can_mod_j:
                delta_j = base_delta * scale_on_j
                updated_j = min(p_i, p_j + delta_j)
                if abs(updated_j - p_j) > 1e-15:
                    new_pattern_j[d_idx] = updated_j
                    changed_j = True
            if two_way and can_mod_i:
                delta_i = base_delta * scale_on_i
                updated_i = max(p_j, p_i - delta_i)
                if abs(updated_i - p_i) > 1e-15:
                    new_pattern_i[d_idx] = updated_i
                    changed_i = True

    if changed_i:
        new_pattern_i = normalize_pattern(
            new_pattern_i,
            mobility=config.world_mobility,
            method=config.normalization_method,
        )
        agent_i.movement_pattern = new_pattern_i

    if changed_j:
        new_pattern_j = normalize_pattern(
            new_pattern_j,
            mobility=config.world_mobility,
            method=config.normalization_method,
        )
        agent_j.movement_pattern = new_pattern_j

    modified_i = any(
        abs(a - b) > PROBABILITY_TOLERANCE
        for a, b in zip(pattern_i_before, agent_i.movement_pattern)
    )
    modified_j = any(
        abs(a - b) > PROBABILITY_TOLERANCE
        for a, b in zip(pattern_j_before, agent_j.movement_pattern)
    )

    return InteractionRecord(
        agent_i_name=agent_i.name,
        agent_j_name=agent_j.name,
        agent_i_pos=agent_i.position,
        agent_j_pos=agent_j.position,
        agent_i_modified=modified_i,
        agent_j_modified=modified_j,
        agent_i_pattern_before=pattern_i_before,
        agent_j_pattern_before=pattern_j_before,
        agent_i_pattern_after=list(agent_i.movement_pattern),
        agent_j_pattern_after=list(agent_j.movement_pattern),
    )
