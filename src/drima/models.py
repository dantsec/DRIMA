"""Core data models and domain representations for DRIMA.

Project Overview & Conceptual Foundation
----------------------------------------
In DRIMA (Section 3.1 of de Oliveira, 2010), the state of the simulation at any instant is
captured by two hierarchical abstractions:
1. **Agent (`{identity, position}`):**
   - `identity` is `{name, movement_pattern, list_of_properties}`:
     - `name`: Agent type (`"D"` or `"R"`) concatenated with a unique numeric identifier
       (e.g., `"D01"`, `"R01"`).
     - `movement_pattern`: A 9-element list of probabilities
       `[P_E, P_NE, P_N, P_NW, P_W, P_SW, P_S, P_SE, X]`, where `X` (index 8) is the
       probability of staying in the same grid cell (no-move).
     - `list_of_properties`: Additional agent features, starting with the 9-element
       `transition_potential_pattern` (set to `1000` in a D-agent's active direction and `0`
       elsewhere) and optional modifiers like `susceptibility` or `influence_potential`.
   - `position`: 1-indexed grid coordinates `[x, y]` where `1 <= x <= X_max` and
     `1 <= y <= Y_max`.

2. **World (`{list_of_agents, grid_size, types_of_interaction, world_mobility}`):**
   - `list_of_agents`: All agents currently in the world.
   - `grid_size`: `[X_max, Y_max]`.
   - `types_of_interaction`: Subset of `{"dd", "DD", "rr", "RR", "dR", "Rd"}` (plus optional
     `"Dr"`, `"rD"`). Uppercase letters indicate that the corresponding agent type is modified
     by the interaction; lowercase letters indicate it remains unchanged.
   - `world_mobility`: `"F"` (Full mobility, where no-move probability `X` is locked to `0.0`)
     or `"P"` (Partial mobility, where `X` is treated like any other direction).

Common Pitfalls & What NOT To Do
--------------------------------
- **Do NOT use 0-indexed coordinates for DRIMA world snapshots:** The paper specifies 1-indexed
  grid coordinates (`1..X_max`, `1..Y_max`). Mixing 0-indexed and 1-indexed coordinates causes
  off-by-one boundary wrapping errors.
- **Do NOT allow non-zero `X` (index 8) under Full (`"F"`) mobility:** In Full mobility, agents
  must move at every step, so `movement_pattern[8]` must always equal `0.0`.
- **Do NOT mutate shared agent lists across history snapshots:** Because a DRIMA history is a
  sequence of world snapshots over time, mutating an agent in place without cloning corrupts
  earlier snapshots in the history. Always use `World.clone()` or `Agent.clone()` when
  advancing a step.

Example Usage
-------------
```python
from src.drima.models import Agent, World

# Create a deterministic agent D01 moving Northwest (index 3) at position (2, 2)
d01 = Agent.create_d_agent(name="D01", direction="NW", position=(2, 2), mobility="F")

# Create a random agent R01 at position (4, 6) under Full mobility (0.125 in 8 directions, 0 in X)
r01 = Agent.create_r_agent(name="R01", position=(4, 6), mobility="F")

world = World(
    agents=[d01, r01],
    grid_size=(5, 10),
    interaction_types=["dd", "RR", "dR", "Rd"],
    world_mobility="F",
)
print(world.to_drima_list())
```
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Eight compass directions plus X (no movement) in the exact order defined in Section 3.1:
# {P_E, P_NE, P_N, P_NW, P_W, P_SW, P_S, P_SE, X}
DIRECTIONS: Tuple[str, ...] = ("E", "NE", "N", "NW", "W", "SW", "S", "SE", "X")

DIRECTION_TO_INDEX: Dict[str, int] = {direction: idx for idx, direction in enumerate(DIRECTIONS)}

# Coordinate displacements (dx, dy) on the 2D grid for each direction
DIRECTION_DELTAS: Dict[str, Tuple[int, int]] = {
    "E": (1, 0),
    "NE": (1, 1),
    "N": (0, 1),
    "NW": (-1, 1),
    "W": (-1, 0),
    "SW": (-1, -1),
    "S": (0, -1),
    "SE": (1, -1),
    "X": (0, 0),
}

# Standard interaction types defined in Section 3.1, plus optional Dr/rD for total configurability
STANDARD_INTERACTION_TYPES: Tuple[str, ...] = ("dd", "DD", "rr", "RR", "dR", "Rd")
VALID_INTERACTION_TYPES: Tuple[str, ...] = ("dd", "DD", "rr", "RR", "dR", "Rd", "Dr", "rD")
DEFAULT_INTERACTION_TYPES: List[str] = ["dd", "RR", "dR", "Rd"]

PROBABILITY_TOLERANCE: float = 1e-6


@dataclass
class AgentProperties:
    """Encapsulates the `list of properties` within an agent's identity.

    In DRIMA (Section 3.1), the first property in `list of properties` is the
    `transition_potential_pattern` (9 numerical values corresponding to the 9 movement fields).
    Additional optional properties mentioned in Section 3.1 and the Acknowledgments include
    `susceptibility` (multiplier on how strongly an agent is affected during an interaction)
    and `influence_potential` (multiplier on how strongly an agent pulls another agent).

    Attributes:
        transition_potential_pattern: List of 9 floats representing the transition potential
            for each direction `[E, NE, N, NW, W, SW, S, SE, X]`.
        susceptibility: Positive float scaling the intensity of change experienced by this
            agent (default `1.0`).
        influence_potential: Positive float scaling the intensity of change exerted by this
            agent on its interaction partner (default `1.0`).
        extra: Dictionary for any additional user-defined properties.
    """

    transition_potential_pattern: List[float] = field(
        default_factory=lambda: [0.0] * 9
    )
    susceptibility: float = 1.0
    influence_potential: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate property invariants."""
        if len(self.transition_potential_pattern) != 9:
            raise ValueError(
                f"transition_potential_pattern must have exactly 9 elements, "
                f"got {len(self.transition_potential_pattern)}."
            )
        self.transition_potential_pattern = [
            float(val) for val in self.transition_potential_pattern
        ]
        if self.susceptibility < 0:
            raise ValueError(
                f"susceptibility must be non-negative, got {self.susceptibility}."
            )
        if self.influence_potential < 0:
            raise ValueError(
                f"influence_potential must be non-negative, got {self.influence_potential}."
            )

    def clone(self) -> "AgentProperties":
        """Return an independent deep copy of these properties."""
        return AgentProperties(
            transition_potential_pattern=list(self.transition_potential_pattern),
            susceptibility=self.susceptibility,
            influence_potential=self.influence_potential,
            extra=dict(self.extra),
        )

    def to_drima_list(self) -> List[Any]:
        """Return the canonical DRIMA `list of properties` representation.

        In Section 3.1, `list of properties` is represented as `{{t_E, ..., t_X}}`.
        If custom susceptibility, influence_potential, or extra fields are set to non-default
        values, they are appended as a dictionary element after the transition potential list.
        """
        potentials = [
            int(v) if float(v).is_integer() else round(v, 6)
            for v in self.transition_potential_pattern
        ]
        result: List[Any] = [potentials]
        if (
            self.susceptibility != 1.0
            or self.influence_potential != 1.0
            or self.extra
        ):
            result.append(
                {
                    "susceptibility": self.susceptibility,
                    "influence_potential": self.influence_potential,
                    **self.extra,
                }
            )
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize properties to a JSON-friendly dictionary."""
        data: Dict[str, Any] = {
            "transition_potential_pattern": list(self.transition_potential_pattern),
            "susceptibility": self.susceptibility,
            "influence_potential": self.influence_potential,
        }
        if self.extra:
            data["extra"] = dict(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "AgentProperties":
        """Construct `AgentProperties` from a dictionary or canonical DRIMA list."""
        if isinstance(data, list):
            if not data:
                return cls()
            first = data[0]
            if isinstance(first, list) and len(first) == 9:
                extra_meta = data[1] if len(data) > 1 and isinstance(data[1], dict) else {}
                susceptibility = float(extra_meta.get("susceptibility", 1.0))
                influence = float(extra_meta.get("influence_potential", 1.0))
                extra = {
                    k: v
                    for k, v in extra_meta.items()
                    if k not in ("susceptibility", "influence_potential")
                }
                return cls(
                    transition_potential_pattern=[float(x) for x in first],
                    susceptibility=susceptibility,
                    influence_potential=influence,
                    extra=extra,
                )
            if len(data) == 9 and all(isinstance(x, (int, float)) for x in data):
                return cls(transition_potential_pattern=[float(x) for x in data])
            raise ValueError(f"Invalid list format for AgentProperties: {data}")

        if isinstance(data, dict):
            pattern = data.get("transition_potential_pattern", [0.0] * 9)
            susceptibility = float(data.get("susceptibility", 1.0))
            influence = float(data.get("influence_potential", 1.0))
            extra = dict(data.get("extra", {}))
            return cls(
                transition_potential_pattern=[float(x) for x in pattern],
                susceptibility=susceptibility,
                influence_potential=influence,
                extra=extra,
            )

        raise TypeError(f"Unsupported type for AgentProperties: {type(data).__name__}")


@dataclass
class Agent:
    """Represents a reactive agent in DRIMA (`{identity, position}`).

    Attributes:
        name: Unique string identifier starting with `'D'` or `'R'` followed by digits
            (e.g., `'D01'`, `'R04'`).
        movement_pattern: List of 9 floats `[P_E, P_NE, P_N, P_NW, P_W, P_SW, P_S, P_SE, X]`
            representing movement probabilities that sum to `1.0`.
        position: Tuple `(x, y)` of 1-indexed grid coordinates.
        properties: `AgentProperties` instance holding transition potentials and optional
            modifiers.
    """

    name: str
    movement_pattern: List[float]
    position: Tuple[int, int]
    properties: AgentProperties = field(default_factory=AgentProperties)
    allow_unnormalized: bool = False

    def __post_init__(self) -> None:
        """Validate agent name, movement pattern, and position invariants."""
        if not self.name or self.name[0] not in ("D", "R"):
            raise ValueError(
                f"Agent name must start with 'D' or 'R', got '{self.name}'."
            )
        if len(self.movement_pattern) != 9:
            raise ValueError(
                f"Agent '{self.name}' movement_pattern must have 9 elements "
                f"(8 directions + X), got {len(self.movement_pattern)}."
            )
        self.movement_pattern = [float(p) for p in self.movement_pattern]
        for idx, prob in enumerate(self.movement_pattern):
            if prob < -PROBABILITY_TOLERANCE:
                raise ValueError(
                    f"Agent '{self.name}' has negative probability {prob} at "
                    f"direction {DIRECTIONS[idx]} (index {idx})."
                )
            if prob < 0.0:
                self.movement_pattern[idx] = 0.0

        total_prob = sum(self.movement_pattern)
        if not self.allow_unnormalized and abs(total_prob - 1.0) > 1e-2:
            raise ValueError(
                f"Agent '{self.name}' movement_pattern probabilities must sum to 1.0 "
                f"(within tolerance), got {total_prob:.6f}."
            )

        if len(self.position) != 2:
            raise ValueError(
                f"Agent '{self.name}' position must be a 2-element (x, y) coordinate, "
                f"got {self.position}."
            )
        self.position = (int(self.position[0]), int(self.position[1]))
        if self.position[0] < 1 or self.position[1] < 1:
            raise ValueError(
                f"Agent '{self.name}' position must use 1-indexed coordinates >= 1, "
                f"got {self.position}."
            )

        if not isinstance(self.properties, AgentProperties):
            self.properties = AgentProperties.from_dict(self.properties)

    @property
    def agent_type(self) -> str:
        """Return `'D'` for deterministic agents or `'R'` for random agents."""
        return self.name[0]

    @property
    def dominant_direction(self) -> str:
        """Return the direction name (`E`..`X`) with the highest probability."""
        max_idx = max(range(9), key=lambda idx: self.movement_pattern[idx])
        return DIRECTIONS[max_idx]

    @property
    def max_probability(self) -> float:
        """Return the maximum single-direction probability in `movement_pattern`."""
        return max(self.movement_pattern)

    def clone(self) -> "Agent":
        """Return an independent deep copy of this agent."""
        return Agent(
            name=self.name,
            movement_pattern=list(self.movement_pattern),
            position=(self.position[0], self.position[1]),
            properties=self.properties.clone(),
            allow_unnormalized=True,
        )

    @classmethod
    def create_d_agent(
        cls,
        name: str,
        direction: str,
        position: Tuple[int, int],
        mobility: str = "F",
        transition_potential: float = 1000.0,
        susceptibility: float = 1.0,
        influence_potential: float = 1.0,
    ) -> "Agent":
        """Create a deterministic D-agent moving with probability 1.0 in `direction`.

        Args:
            name: Unique identifier starting with `'D'` (e.g., `'D01'`).
            direction: One of `DIRECTIONS` (`'E'`, `'NE'`, `'N'`, `'NW'`, `'W'`, `'SW'`,
                `'S'`, `'SE'`, or `'X'`). Note: `'X'` is only permitted when `mobility == 'P'`.
            position: 1-indexed `(x, y)` coordinates.
            mobility: `'F'` (Full) or `'P'` (Partial).
            transition_potential: Value placed in `transition_potential_pattern` at the active
                direction (default `1000.0`, as shown in Section 3.1 of the paper).
            susceptibility: Optional susceptibility multiplier.
            influence_potential: Optional influence potential multiplier.

        Returns:
            Initialized `Agent` instance.
        """
        if not name.startswith("D"):
            raise ValueError(f"D-agent name must start with 'D', got '{name}'.")
        direction_upper = direction.upper()
        if direction_upper not in DIRECTION_TO_INDEX:
            raise ValueError(
                f"Invalid direction '{direction}'. Must be one of {DIRECTIONS}."
            )
        if mobility.upper() == "F" and direction_upper == "X":
            raise ValueError(
                "D-agent cannot have direction 'X' (no-move) when world_mobility is 'F' (Full)."
            )

        dir_idx = DIRECTION_TO_INDEX[direction_upper]
        pattern = [0.0] * 9
        pattern[dir_idx] = 1.0

        potentials = [0.0] * 9
        potentials[dir_idx] = float(transition_potential)

        return cls(
            name=name,
            movement_pattern=pattern,
            position=position,
            properties=AgentProperties(
                transition_potential_pattern=potentials,
                susceptibility=susceptibility,
                influence_potential=influence_potential,
            ),
        )

    @classmethod
    def create_r_agent(
        cls,
        name: str,
        position: Tuple[int, int],
        mobility: str = "F",
        movement_pattern: Optional[Sequence[float]] = None,
        susceptibility: float = 1.0,
        influence_potential: float = 1.0,
    ) -> "Agent":
        """Create a random R-agent initialized uniformly (or with a custom pattern).

        Under Full mobility (`'F'`), probabilities are initialized to `1/8 = 0.125` for the 8
        compass directions and `0.0` for `'X'` (Section 3.1). Under Partial mobility (`'P'`),
        all 9 fields are initialized to `1/9` (`~0.111111...`).

        Args:
            name: Unique identifier starting with `'R'` (e.g., `'R01'`).
            position: 1-indexed `(x, y)` coordinates.
            mobility: `'F'` (Full) or `'P'` (Partial).
            movement_pattern: Optional explicit 9-element probability list.
            susceptibility: Optional susceptibility multiplier.
            influence_potential: Optional influence potential multiplier.

        Returns:
            Initialized `Agent` instance.
        """
        if not name.startswith("R"):
            raise ValueError(f"R-agent name must start with 'R', got '{name}'.")
        mobility_upper = mobility.upper()
        if mobility_upper not in ("F", "P"):
            raise ValueError(f"mobility must be 'F' or 'P', got '{mobility}'.")

        if movement_pattern is not None:
            pattern = [float(p) for p in movement_pattern]
            if mobility_upper == "F" and pattern[8] > PROBABILITY_TOLERANCE:
                raise ValueError(
                    f"Agent '{name}' cannot have non-zero no-move probability X ({pattern[8]}) "
                    f"under Full ('F') mobility."
                )
        else:
            if mobility_upper == "F":
                pattern = [0.125] * 8 + [0.0]
            else:
                pattern = [1.0 / 9.0] * 9

        return cls(
            name=name,
            movement_pattern=pattern,
            position=position,
            properties=AgentProperties(
                transition_potential_pattern=[0.0] * 9,
                susceptibility=susceptibility,
                influence_potential=influence_potential,
            ),
        )

    def to_drima_list(self) -> List[Any]:
        """Return the canonical nested-list structure `{identity, position}` from Section 3.1.

        Example output:
            `[["D01", [0, 0, 0, 1, 0, 0, 0, 0, 0], [[0, 0, 0, 1000, 0, 0, 0, 0, 0]]], [2, 2]]`
        """
        formatted_pattern = [
            int(p) if p in (0.0, 1.0) else round(p, 6)
            for p in self.movement_pattern
        ]
        identity = [
            self.name,
            formatted_pattern,
            self.properties.to_drima_list(),
        ]
        return [identity, [self.position[0], self.position[1]]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the agent into a descriptive JSON dictionary."""
        return {
            "name": self.name,
            "type": self.agent_type,
            "position": [self.position[0], self.position[1]],
            "movement_pattern": [round(p, 8) for p in self.movement_pattern],
            "dominant_direction": self.dominant_direction,
            "properties": self.properties.to_dict(),
            "drima_representation": self.to_drima_list(),
        }

    @classmethod
    def from_dict(cls, data: Any, default_mobility: str = "F") -> "Agent":
        """Deserialize an `Agent` from either a JSON dictionary or a canonical DRIMA list."""
        if isinstance(data, list):
            # Canonical DRIMA format: {{name, movement_pattern, list_of_properties}, {x, y}}
            if len(data) != 2 or not isinstance(data[0], list) or len(data[0]) < 2:
                raise ValueError(f"Invalid DRIMA list format for Agent: {data}")
            identity, pos = data[0], data[1]
            name = str(identity[0])
            pattern = [float(x) for x in identity[1]]
            props = (
                AgentProperties.from_dict(identity[2])
                if len(identity) > 2
                else AgentProperties()
            )
            return cls(
                name=name,
                movement_pattern=pattern,
                position=(int(pos[0]), int(pos[1])),
                properties=props,
            )

        if isinstance(data, dict):
            name = str(data["name"])
            pos_raw = data.get("position", [1, 1])
            position = (int(pos_raw[0]), int(pos_raw[1]))
            mobility = str(data.get("mobility", default_mobility)).upper()

            if "movement_pattern" in data and data["movement_pattern"] is not None:
                pattern = [float(p) for p in data["movement_pattern"]]
                if "properties" in data:
                    props = AgentProperties.from_dict(data["properties"])
                elif name.startswith("D"):
                    potentials = [
                        1000.0 if p == 1.0 else 0.0 for p in pattern
                    ]
                    props = AgentProperties(transition_potential_pattern=potentials)
                else:
                    props = AgentProperties()
                return cls(
                    name=name,
                    movement_pattern=pattern,
                    position=position,
                    properties=props,
                )

            if name.startswith("D"):
                direction = str(data.get("direction", "E"))
                potential = float(data.get("transition_potential", 1000.0))
                susceptibility = float(data.get("susceptibility", 1.0))
                influence = float(data.get("influence_potential", 1.0))
                return cls.create_d_agent(
                    name=name,
                    direction=direction,
                    position=position,
                    mobility=mobility,
                    transition_potential=potential,
                    susceptibility=susceptibility,
                    influence_potential=influence,
                )

            susceptibility = float(data.get("susceptibility", 1.0))
            influence = float(data.get("influence_potential", 1.0))
            return cls.create_r_agent(
                name=name,
                position=position,
                mobility=mobility,
                susceptibility=susceptibility,
                influence_potential=influence,
            )

        raise TypeError(f"Unsupported agent specification type: {type(data).__name__}")


@dataclass
class World:
    """Represents a snapshot of a DRIMA world (`{list_of_agents, grid_size, types_of_interaction, world_mobility}`).

    Attributes:
        agents: List of `Agent` objects currently in the world.
        grid_size: Tuple `(x_max, y_max)` defining the 2D periodic grid dimensions.
        interaction_types: List of allowed interaction type codes (e.g., `["dd", "RR", "dR", "Rd"]`).
        world_mobility: `'F'` (Full mobility: `X = 0` always) or `'P'` (Partial mobility: `X >= 0`).
    """

    agents: List[Agent]
    grid_size: Tuple[int, int]
    interaction_types: List[str] = field(
        default_factory=lambda: list(DEFAULT_INTERACTION_TYPES)
    )
    world_mobility: str = "F"

    def __post_init__(self) -> None:
        """Validate world configuration and ensure all agents respect grid and mobility rules."""
        if len(self.grid_size) != 2:
            raise ValueError(f"grid_size must be a 2-tuple (X_max, Y_max), got {self.grid_size}.")
        x_max, y_max = int(self.grid_size[0]), int(self.grid_size[1])
        if x_max < 1 or y_max < 1:
            raise ValueError(
                f"grid_size dimensions must be >= 1, got ({x_max}, {y_max})."
            )
        self.grid_size = (x_max, y_max)

        self.world_mobility = self.world_mobility.upper()
        if self.world_mobility not in ("F", "P"):
            raise ValueError(
                f"world_mobility must be 'F' (Full) or 'P' (Partial), got '{self.world_mobility}'."
            )

        for itype in self.interaction_types:
            if itype not in VALID_INTERACTION_TYPES:
                raise ValueError(
                    f"Invalid interaction type '{itype}'. Allowed values: {VALID_INTERACTION_TYPES}."
                )

        seen_names = set()
        for agent in self.agents:
            if agent.name in seen_names:
                raise ValueError(f"Duplicate agent name '{agent.name}' in world.")
            seen_names.add(agent.name)

            ax, ay = agent.position
            if not (1 <= ax <= x_max and 1 <= ay <= y_max):
                raise ValueError(
                    f"Agent '{agent.name}' position {agent.position} is outside "
                    f"grid bounds (1..{x_max}, 1..{y_max})."
                )
            if (
                self.world_mobility == "F"
                and agent.movement_pattern[8] > PROBABILITY_TOLERANCE
            ):
                raise ValueError(
                    f"Agent '{agent.name}' has non-zero no-move probability "
                    f"X={agent.movement_pattern[8]} in a Full ('F') mobility world."
                )

    def clone(self) -> "World":
        """Return a deep copy of this world snapshot."""
        return World(
            agents=[agent.clone() for agent in self.agents],
            grid_size=(self.grid_size[0], self.grid_size[1]),
            interaction_types=list(self.interaction_types),
            world_mobility=self.world_mobility,
        )

    def get_agent(self, name: str) -> Optional[Agent]:
        """Look up an agent by name, returning `None` if not present."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def to_drima_list(self) -> List[Any]:
        """Return the canonical DRIMA list `{list_of_agents, grid_size, interaction_types, world_mobility}`."""
        return [
            [agent.to_drima_list() for agent in self.agents],
            [self.grid_size[0], self.grid_size[1]],
            list(self.interaction_types),
            self.world_mobility,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the world state into a JSON-compatible dictionary."""
        return {
            "grid_size": [self.grid_size[0], self.grid_size[1]],
            "world_mobility": self.world_mobility,
            "interaction_types": list(self.interaction_types),
            "agents": [agent.to_dict() for agent in self.agents],
            "drima_representation": self.to_drima_list(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "World":
        """Deserialize a `World` from a dictionary or canonical DRIMA list."""
        if isinstance(data, list):
            if len(data) != 4:
                raise ValueError(f"Expected 4-element DRIMA world list, got {len(data)}.")
            agents_raw, grid_raw, types_raw, mobility_raw = data
            # Handle optional extra nesting around agents list as in line 163 of the paper
            if (
                len(agents_raw) == 1
                and isinstance(agents_raw[0], list)
                and agents_raw[0]
                and isinstance(agents_raw[0][0], list)
                and len(agents_raw[0][0]) == 2
                and isinstance(agents_raw[0][0][0], list)
            ):
                agents_raw = agents_raw[0]
            mobility = str(mobility_raw).upper()
            agents = [
                Agent.from_dict(a, default_mobility=mobility) for a in agents_raw
            ]
            return cls(
                agents=agents,
                grid_size=(int(grid_raw[0]), int(grid_raw[1])),
                interaction_types=[str(t) for t in types_raw],
                world_mobility=mobility,
            )

        if isinstance(data, dict):
            grid_raw = data.get("grid_size", [8, 5])
            mobility = str(data.get("world_mobility", "F")).upper()
            interaction_types = [
                str(t)
                for t in data.get("interaction_types", list(DEFAULT_INTERACTION_TYPES))
            ]
            agents = [
                Agent.from_dict(a, default_mobility=mobility)
                for a in data.get("agents", [])
            ]
            return cls(
                agents=agents,
                grid_size=(int(grid_raw[0]), int(grid_raw[1])),
                interaction_types=interaction_types,
                world_mobility=mobility,
            )

        raise TypeError(f"Unsupported world data type: {type(data).__name__}")


@dataclass
class InteractionRecord:
    """Records the details of a single binary interaction between two agents."""

    agent_i_name: str
    agent_j_name: str
    agent_i_pos: Tuple[int, int]
    agent_j_pos: Tuple[int, int]
    agent_i_modified: bool
    agent_j_modified: bool
    agent_i_pattern_before: List[float]
    agent_j_pattern_before: List[float]
    agent_i_pattern_after: List[float]
    agent_j_pattern_after: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize interaction record to a JSON-compatible dictionary."""
        return {
            "agent_i": self.agent_i_name,
            "agent_j": self.agent_j_name,
            "agent_i_position": list(self.agent_i_pos),
            "agent_j_position": list(self.agent_j_pos),
            "agent_i_modified": self.agent_i_modified,
            "agent_j_modified": self.agent_j_modified,
            "agent_i_pattern_before": [round(p, 8) for p in self.agent_i_pattern_before],
            "agent_j_pattern_before": [round(p, 8) for p in self.agent_j_pattern_before],
            "agent_i_pattern_after": [round(p, 8) for p in self.agent_i_pattern_after],
            "agent_j_pattern_after": [round(p, 8) for p in self.agent_j_pattern_after],
        }


@dataclass
class MovementRecord:
    """Records the movement of a single agent during a world update step."""

    agent_name: str
    direction: str
    position_before: Tuple[int, int]
    position_after: Tuple[int, int]
    blocked_by_occupancy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize movement record to a JSON-compatible dictionary."""
        return {
            "agent": self.agent_name,
            "direction": self.direction,
            "position_before": list(self.position_before),
            "position_after": list(self.position_after),
            "blocked_by_occupancy": self.blocked_by_occupancy,
        }


@dataclass
class StepLog:
    """Captures all events (permutation, interactions, movements) for one world update step."""

    step: int
    update_mode: str
    agent_order: List[str]
    interactions: List[InteractionRecord] = field(default_factory=list)
    movements: List[MovementRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize step log to a JSON-compatible dictionary."""
        return {
            "step": self.step,
            "update_mode": self.update_mode,
            "agent_order": list(self.agent_order),
            "interactions": [inter.to_dict() for inter in self.interactions],
            "movements": [mov.to_dict() for mov in self.movements],
        }


@dataclass
class SimulationConfig:
    """Complete configuration for a DRIMA world and simulation run.

    Allows the user to configure every initial world parameter and interaction dimension
    described in Section 2.2 and Section 3 of the DRIMA specification.
    """

    # Grid & World settings
    grid_width: int = 8
    grid_height: int = 5
    world_mobility: str = "F"
    interaction_types: List[str] = field(
        default_factory=lambda: list(DEFAULT_INTERACTION_TYPES)
    )
    allow_shared_cells: bool = True

    # Agent generation settings (used when explicit `agents` list is not provided)
    num_d_agents: int = 2
    num_r_agents: int = 6
    d_directions: Optional[List[str]] = None
    d_transition_potential: float = 1000.0
    agents: Optional[List[Dict[str, Any]]] = None

    # Interaction dimensions (Section 2.2 & Section 3.2)
    interaction_range: int = 1
    update_mode: str = "sync"
    interaction_direction: str = "one-way"
    delta_mode: str = "fixed"
    fp: float = 1.0
    normalization_method: str = "equal_displacement"
    sync_only_unprocessed_neighbors: bool = False
    async_move_all: bool = True

    # Simulation execution & output settings
    iterations: int = 100
    seed: Optional[int] = None
    output_file: str = "history.json"
    log_file: Optional[str] = "drima.log"
    log_level: str = "INFO"
    display_world: bool = False
    display_history: bool = False

    def __post_init__(self) -> None:
        """Validate all configuration parameters."""
        if self.grid_width < 1 or self.grid_height < 1:
            raise ValueError(
                f"Grid dimensions must be >= 1, got ({self.grid_width}, {self.grid_height})."
            )
        self.world_mobility = self.world_mobility.upper()
        if self.world_mobility not in ("F", "P"):
            raise ValueError(
                f"world_mobility must be 'F' or 'P', got '{self.world_mobility}'."
            )
        for itype in self.interaction_types:
            if itype not in VALID_INTERACTION_TYPES:
                raise ValueError(
                    f"Invalid interaction type '{itype}'. Must be in {VALID_INTERACTION_TYPES}."
                )
        if self.num_d_agents < 0 or self.num_r_agents < 0:
            raise ValueError("num_d_agents and num_r_agents must be >= 0.")
        if self.agents is None and (self.num_d_agents + self.num_r_agents) == 0:
            raise ValueError("World must contain at least one agent.")
        if (
            not self.allow_shared_cells
            and self.agents is None
            and (self.num_d_agents + self.num_r_agents) > (self.grid_width * self.grid_height)
        ):
            raise ValueError(
                f"Cannot place {self.num_d_agents + self.num_r_agents} agents on a "
                f"{self.grid_width}x{self.grid_height} grid when allow_shared_cells=False."
            )
        if self.interaction_range < 0:
            raise ValueError(
                f"interaction_range (R) must be >= 0, got {self.interaction_range}."
            )
        self.update_mode = self.update_mode.lower()
        if self.update_mode not in ("sync", "async"):
            raise ValueError(
                f"update_mode must be 'sync' or 'async', got '{self.update_mode}'."
            )
        self.interaction_direction = self.interaction_direction.lower()
        if self.interaction_direction not in ("one-way", "two-way"):
            raise ValueError(
                f"interaction_direction must be 'one-way' or 'two-way', "
                f"got '{self.interaction_direction}'."
            )
        self.delta_mode = self.delta_mode.lower()
        if self.delta_mode not in ("fixed", "variable"):
            raise ValueError(
                f"delta_mode must be 'fixed' or 'variable', got '{self.delta_mode}'."
            )
        if self.fp < 0:
            raise ValueError(f"fp (F_p) must be >= 0, got {self.fp}.")
        self.normalization_method = self.normalization_method.lower()
        if self.normalization_method not in (
            "equal_displacement",
            "proportional",
            "none",
        ):
            raise ValueError(
                f"normalization_method must be 'equal_displacement', 'proportional', "
                f"or 'none', got '{self.normalization_method}'."
            )
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        return {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "world_mobility": self.world_mobility,
            "interaction_types": list(self.interaction_types),
            "allow_shared_cells": self.allow_shared_cells,
            "num_d_agents": self.num_d_agents,
            "num_r_agents": self.num_r_agents,
            "d_directions": list(self.d_directions) if self.d_directions else None,
            "d_transition_potential": self.d_transition_potential,
            "agents": self.agents,
            "interaction_range": self.interaction_range,
            "update_mode": self.update_mode,
            "interaction_direction": self.interaction_direction,
            "delta_mode": self.delta_mode,
            "fp": self.fp,
            "normalization_method": self.normalization_method,
            "sync_only_unprocessed_neighbors": self.sync_only_unprocessed_neighbors,
            "async_move_all": self.async_move_all,
            "iterations": self.iterations,
            "seed": self.seed,
            "output_file": self.output_file,
            "log_file": self.log_file,
            "log_level": self.log_level,
            "display_world": self.display_world,
            "display_history": self.display_history,
        }
