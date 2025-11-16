import random
import argparse

class Agent:
    def __init__(self, agent_id: str, position: tuple[int, int], movement_pattern: dict[str, float]) -> None:
        """
        Initialize an agent.

        :param agent_id: Unique identifier for the agent.
        :type agent_id: str
        :param position: Tuple (x, y) representing the agent's position.
        :type position: tuple[int, int]
        :param movement_pattern: Dictionary of movement probabilities for each direction.
        :type movement_pattern: dict[str, float]
        """
        self.agent_id = agent_id
        self.position = position
        self.movement_pattern = movement_pattern


    def __str__(self) -> str:
        """
        Return a string representation of the agent.

        :return: String representation of the agent.
        :rtype: str
        """
        movement_pattern_str = ', '.join([f"{direction}: {prob:.2f}" for direction, prob in self.movement_pattern.items()])

        return f"Agent(id={self.agent_id}, position={self.position}, movement_pattern={{ {movement_pattern_str} }})"


    def move(self, grid_size: tuple[int, int], world_mobility: str) -> None:
        """
        Move the agent based on its movement pattern.

        :param grid_size: Tuple (width, height) representing the grid dimensions.
        :type grid_size: tuple[int, int]
        :param world_mobility: 'F' for full mobility, 'P' for partial mobility.
        :type world_mobility: str
        """
        if world_mobility == 'F':
            self.movement_pattern['X'] = 0.0

        # Normalize movement pattern.
        total_prob = sum(self.movement_pattern.values())
        normalized_pattern = {k: v / total_prob for k, v in self.movement_pattern.items()}

        # Choose a direction based on probabilities.
        directions = list(normalized_pattern.keys())
        probabilities = list(normalized_pattern.values())
        chosen_direction = random.choices(directions, probabilities)[0]

        # Update position based on chosen direction.
        movement_offsets = {
            "E": (1, 0), "NE": (1, -1), "N": (0, -1), "NW": (-1, -1),
            "W": (-1, 0), "SW": (-1, 1), "S": (0, 1), "SE": (1, 1), "X": (0, 0)
        }

        offset = movement_offsets[chosen_direction]

        self.position = (
            (self.position[0] + offset[0]) % grid_size[0],
            (self.position[1] + offset[1]) % grid_size[1]
        )


    def __str__(self) -> str:
        """
        Return a string representation of the agent.

        :return: String representation of the agent.
        :rtype: str
        """
        movement_pattern_str = ', '.join([f"{direction}: {prob:.2f}" for direction, prob in self.movement_pattern.items()])
        return f"Agent(id={self.agent_id}, position={self.position}, movement_pattern={{ {movement_pattern_str} }})"


class World:
    def __init__(self, grid_size: tuple[int, int], interaction_range: int, world_mobility: str) -> None:
        """
        Initialize the world.

        :param grid_size: Tuple (width, height) representing the grid dimensions.
        :type grid_size: tuple[int, int]
        :param interaction_range: Interaction range for agents.
        :type interaction_range: int
        :param world_mobility: 'F' for full mobility, 'P' for partial mobility.
        :type world_mobility: str
        """
        self.grid_size = grid_size
        self.interaction_range = interaction_range
        self.world_mobility = world_mobility
        self.agents: list[Agent] = []


    def __str__(self) -> str:
        """
        Return a string representation of the world.

        :return: String representation of the world.
        :rtype: str
        """
        agent_positions = ', '.join([f"{agent.agent_id}: {agent.position}" for agent in self.agents])

        return f"World(grid_size={self.grid_size}, interaction_range={self.interaction_range}, world_mobility={self.world_mobility}, agents=[{agent_positions}])"


    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the world.

        :param agent: The agent to be added.
        :type agent: Agent
        """
        self.agents.append(agent)


    def update(self, interaction_config: dict[str, str|float], update_mode: str = "Synchronous") -> None:
        """
        Update the world by moving agents and processing interactions.

        :param interaction_config: Configuration for interactions including type, intensity, and mode.
        :type interaction_config: dict[str, str|float]
        :param update_mode: Update mode, either "Synchronous" or "Asynchronous".
        :type update_mode: str
        """
        if update_mode == "Synchronous":
            random.shuffle(self.agents)

            for agent in self.agents:
                agent.move(self.grid_size, self.world_mobility)

            for i, agent1 in enumerate(self.agents):
                for agent2 in self.agents[i + 1:]:
                    if self._within_interaction_range(agent1.position, agent2.position):
                        interaction = Interaction(interaction_type=interaction_config["interaction_type"], intensity=interaction_config["intensity"], mode=interaction_config["mode"])
                        interaction.process(agent1, agent2)

        if update_mode == "Asynchronous":
            for _ in range(len(self.agents)):
                agent1 = random.choice(self.agents)
                agent2 = random.choice(self.agents)

                if agent1 != agent2 and self._within_interaction_range(agent1.position, agent2.position):
                    interaction = Interaction(interaction_type=interaction_config["interaction_type"], intensity=interaction_config["intensity"], mode=interaction_config["mode"])
                    interaction.process(agent1, agent2)


    def _within_interaction_range(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> bool:
        """
        Check if two positions are within the interaction range.

        :param pos1: Position of the first agent.
        :type pos1: tuple[int, int]
        :param pos2: Position of the second agent.
        :type pos2: tuple[int, int]
        :return: True if the positions are within the interaction range, False otherwise.
        :rtype: bool
        """
        dx = abs(pos1[0] - pos2[0]) % self.grid_size[0]
        dy = abs(pos1[1] - pos2[1]) % self.grid_size[1]

        return dx <= self.interaction_range and dy <= self.interaction_range


class Interaction:
    def __init__(self, interaction_type: str, intensity: float, mode: str = "fixed") -> None:
        """
        Initialize an interaction.

        DRIMA defines the interaction type as a specific category or mode of interaction that
        the system can analyze or process. In its current implementation, DRIMA is limited to
        accepting only one interaction type at a time, meaning it can handle and analyze a single
        predefined interaction type during a given operation.

        :param interaction_type: Type of interaction (e.g., 'dd', 'RR', etc.). Implemented types are: 'dd', 'RR', 'dR' and 'Rd'.
        :type interaction_type: str
        :param intensity: Intensity of the interaction.
        :type intensity: float
        :param mode: Interaction mode ('fixed' or 'variable').
        :type mode: str
        """
        self.interaction_type = interaction_type
        self.intensity = intensity
        self.mode = mode


    def __str__(self) -> str:
        """
        Return a string representation of the interaction.

        :return: String representation of the interaction.
        :rtype: str
        """
        return f"Interaction(type={self.interaction_type}, intensity={self.intensity}, mode={self.mode})"


    def _calculate_delta(self, p1: float, p2: float) -> float:
        """
        Calculate the change in probability based on the interaction mode.

        :param p1: Probability of the first agent in a specific direction.
        :type p1: float
        :param p2: Probability of the second agent in a specific direction.
        :type p2: float
        :return: Calculated delta probability.
        :rtype: float
        """
        if self.mode == "fixed":
            return self.intensity / 9
        elif self.mode == "variable":
            return (p2 - p1) * (self.intensity / 9)
        return 0.0


    def _normalize(self, agent: Agent) -> None:
        """
        Normalize the movement pattern of an agent.

        :param agent: The agent whose movement pattern will be normalized.
        :type agent: Agent
        """
        total_prob = sum(agent.movement_pattern.values())

        for direction in agent.movement_pattern:
            agent.movement_pattern[direction] /= total_prob


    def _update_movement_pattern(self, target_agent: Agent, source_agent: Agent) -> None:
        """
        Update the movement pattern of the target agent based on the source agent.

        :param target_agent: The agent whose movement pattern will be updated.
        :type target_agent: Agent
        :param source_agent: The agent influencing the target agent.
        :type source_agent: Agent
        """
        if target_agent.agent_id[0].lower() == 'd':
            return

        # Update probabilities for each direction.
        for direction in target_agent.movement_pattern:
            delta_p = self._calculate_delta(
                target_agent.movement_pattern[direction],
                source_agent.movement_pattern[direction]
            )

            target_agent.movement_pattern[direction] += delta_p

        self._normalize(target_agent)


    def process(self, agent1: Agent, agent2: Agent) -> None:
        """
        Process the interaction between two agents.
        Upper case letter in interaction_type means that the corresponding agent's movement pattern is modified.
        After processing, the movement patterns are normalized (probabilities will be equal 1 again).

        :param agent1: First agent involved in the interaction.
        :type agent1: Agent
        :param agent2: Second agent involved in the interaction.
        :type agent2: Agent
        """
        modify_agent1 = self.interaction_type[0].isupper()
        modify_agent2 = self.interaction_type[1].isupper()

        agent1_type = agent1.agent_id[0].lower()
        agent2_type = agent2.agent_id[0].lower()

        # Validate the interaction type for the given agents based on defined interaction type.
        valid_interaction = f"{agent1_type}{agent2_type}" in [
            self.interaction_type.lower(),
            self.interaction_type.lower()[::-1]
        ]

        if not valid_interaction:
            return

        if modify_agent1:
            self._update_movement_pattern(agent1, agent2)

        if modify_agent2:
            self._update_movement_pattern(agent2, agent1)


if __name__ == "__main__":
    # Argument parser setup.
    parser = argparse.ArgumentParser(description="Run the DRIMA simulation.")
    agents_group = parser.add_argument_group("Agents")
    agents_group.add_argument("--agents_file", type=str, required=True, default=None, help="Path to a file containing agent definitions. Default: None.")
    world_group = parser.add_argument_group("World")
    world_group.add_argument("--grid_size", type=str, default="10x10", help="Grid size in the format 'widthxheight'. Default: '10x10'.")
    world_group.add_argument("--interaction_range", type=int, default=1, help="Interaction range for agents. Default: 1.")
    world_group.add_argument("--world_mobility", type=str, default="F", help="World mobility ('F' for full, 'P' for partial). Default: 'F'.")
    world_group.add_argument("--iterations", type=int, default=100, help="Number of iterations to run the simulation. Default: 100.")
    world_group.add_argument("--update_mode", type=str, default="Synchronous", help="Update mode ('Synchronous' or 'Asynchronous'). Default: 'Synchronous'.")
    interaction_group = parser.add_argument_group("Interaction")
    interaction_group.add_argument("--interaction_type", type=str, default="dR", help="Type of interaction (e.g., 'dd', 'RR', 'dR', 'Rd'). Default: 'dR'.")
    interaction_group.add_argument("--intensity", type=float, default=1.0, help="Intensity of the interaction. Default: 1.0.")
    interaction_group.add_argument("--mode", type=str, default="fixed", help="Interaction mode ('fixed' or 'variable'). Default: 'fixed'.")
    args = parser.parse_args()

    # Create a world.
    world = World(
        grid_size=tuple(map(int, args.grid_size.split('x'))),
        interaction_range=args.interaction_range,
        world_mobility=args.world_mobility
    )

    # TODO: coiso pra implementar (1)
    # Create agents.
    agents = [
        Agent(
            agent_id="D1",
            position=(0, 0),
            movement_pattern={"E": 1.0, "X": 0.0}
        ),

        Agent(
            agent_id="R1",
            position=(1, 1),
            movement_pattern={"E": 0.2, "X": 0.8}
        )
    ]

    # Add agents to the world.
    for agent in agents:
        world.add_agent(agent)

    # Main loop.
    ITERATIONS = args.iterations

    for iteration in range(ITERATIONS):
        print("----------------------------- START OF ITERATION -----------------------------")
        print(f"Iteration [{iteration + 1}] ; Percentage [{(iteration + 1) / ITERATIONS * 100:.2f}%]")

        print(">>> WORLD STATE <<<")
        print(world)
        print()

        print(">>> AGENT STATES <<<")
        for agent in agents:
            print(agent)
        print()

        print(">>> UPDATING WORLD <<<")
        world.update(
            interaction_config={
                "interaction_type": args.interaction_type,
                "intensity": args.intensity,
                "mode": args.mode
            },
            update_mode=args.update_mode
        )
        print("----------------------------- FINAL OF ITERATION -----------------------------\n")
