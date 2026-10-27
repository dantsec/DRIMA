# DRIMA Implementation Walkthrough

## Summary of Changes

We implemented **DRIMA** (*Dynamics of Randomly Interacting Moving Agents*) in **Python 3.9.6** following the specification in [DRIMA: A Minimal System for Probing the Dynamics of Change in a Reactive Multi-Agent Setting.tex](file:///Users/cauesantana/Desktop/DRIMA/docs/DRIMA:%20A%20Minimal%20System%20for%20Probing%20the%20Dynamics%20of%20Change%20in%20a%20Reactive%20Multi-Agent%20Setting.tex) and the coding standards in [INSTRUCTIONS.md](file:///Users/cauesantana/Desktop/DRIMA/INSTRUCTIONS.md).

### Core Modules Created
- **Package Entry & Overview:** [\_\_init\_\_.py](file:///Users/cauesantana/Desktop/DRIMA/src/drima/__init__.py) and [main.py](file:///Users/cauesantana/Desktop/DRIMA/main.py)
- **Domain Models (`Agent`, `World`, `AgentProperties`, `SimulationConfig`):** [models.py](file:///Users/cauesantana/Desktop/DRIMA/src/drima/models.py)
  - Supports both structured JSON dictionaries and canonical nested DRIMA lists (`{{name, movement_pattern, list_of_properties}, {x, y}}` and `{list_of_agents, grid_size, types_of_interaction, world_mobility}`).
  - Enforces 1-indexed periodic coordinates, Full (`F`) vs. Partial (`P`) world mobility invariants, and D-agent / R-agent movement pattern rules.
- **2D Periodic Grid & Moore Neighborhoods:** [grid.py](file:///Users/cauesantana/Desktop/DRIMA/src/drima/grid.py)
  - Implements [`wrap_position`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/grid.py#L64-L81), [`moore_distance`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/grid.py#L84-L109), [`get_neighbors`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/grid.py#L112-L143), and [`move_agent`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/grid.py#L168-L226) (supporting both shared cells and exclusive single-agent-per-cell occupancy).
- **Iterative Equal-Displacement Normalization:** [normalization.py](file:///Users/cauesantana/Desktop/DRIMA/src/drima/normalization.py)
  - Implements [`normalize_equal_displacement`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/normalization.py#L60-L127) (distributing probability excess/deficit equally across strictly positive fields, clamping negatives to `0.0`, and iterating until convergence), plus [`normalize_proportional`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/normalization.py#L130-L157) and `"none"` for comparative experiments (Figures 2, 4, and 5).
- **Direct Binary Interactions:** [interactions.py](file:///Users/cauesantana/Desktop/DRIMA/src/drima/interactions.py)
  - Implements [`resolve_interaction_types`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/interactions.py#L67-L89) (`DD` over `dd`, `RR` over `rr`), [`determine_modifiability`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/interactions.py#L92-L137), [`compute_delta_p`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/interactions.py#L140-L165) (fixed $\Delta p = F_p / 9$ and variable $\Delta p = (F_p / 9)|P_d^i - P_d^j|$), and [`interact_pair`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/interactions.py#L168-L287) with upper-bound clamping.
- **World Updates & Simulation Runner:** [world.py](file:///Users/cauesantana/Desktop/DRIMA/src/drima/world.py) and [simulation.py](file:///Users/cauesantana/Desktop/DRIMA/src/drima/simulation.py)
  - Implements [`create_world`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/world.py#L84-L163), [`sync_world_update`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/world.py#L166-L238) (`SyncWorldUpdate`), [`async_world_update`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/world.py#L241-L304) (`AsyncWorldUpdate`), and [`compute_convergence_metrics`](file:///Users/cauesantana/Desktop/DRIMA/src/drima/world.py#L307-L356).
- **CLI, Config Loader, Logging, JSON History I/O & Visualizers:**
  - [config.py](file:///Users/cauesantana/Desktop/DRIMA/src/drima/config.py), [cli.py](file:///Users/cauesantana/Desktop/DRIMA/src/drima/cli.py), [logger.py](file:///Users/cauesantana/Desktop/DRIMA/src/drima/logger.py), [io_handler.py](file:///Users/cauesantana/Desktop/DRIMA/src/drima/io_handler.py), and [visualizer.py](file:///Users/cauesantana/Desktop/DRIMA/src/drima/visualizer.py) (`display_world` and `display_history`).
- **Example Configurations:**
  - [figure1_world.json](file:///Users/cauesantana/Desktop/DRIMA/examples/figure1_world.json), [figure2_experiment.json](file:///Users/cauesantana/Desktop/DRIMA/examples/figure2_experiment.json), and [section3_world.json](file:///Users/cauesantana/Desktop/DRIMA/examples/section3_world.json).

---

## Verification & Test Results

### 1. Automated Unit Tests
All 23 unit tests across 5 test modules ([test_models.py](file:///Users/cauesantana/Desktop/DRIMA/tests/test_models.py), [test_grid.py](file:///Users/cauesantana/Desktop/DRIMA/tests/test_grid.py), [test_normalization.py](file:///Users/cauesantana/Desktop/DRIMA/tests/test_normalization.py), [test_interactions.py](file:///Users/cauesantana/Desktop/DRIMA/tests/test_interactions.py), and [test_simulation_and_cli.py](file:///Users/cauesantana/Desktop/DRIMA/tests/test_simulation_and_cli.py)) pass:

```text
$ python3 -m unittest discover -s tests -v
Ran 23 tests in 0.019s
OK
```

### 2. End-to-End Simulation Verification (Figure 2 Experiment)
Running the $8 \times 5$ world with 2 D-agents (`D01` moving `SE` and `D02` moving `E`) and 6 R-agents (`R01`–`R06`) in fixed mode ($F_p = 1.0$) with iterative equal-displacement normalization reproduces the paper's emergent convergence behavior, where non-attractor directions decay to `0.000` and R-agents resolve toward the D-agent directions (`E` and `SE`), generating both the detailed `.log` trace and the complete `.json` history file.
