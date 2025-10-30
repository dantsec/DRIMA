# DRIMA: A Minimal System for Probing the Dynamics of Change in a Reactive Multi-Agent Setting

## Project Overview

- DRIMA (**D**ynamics of **R**andomly **I**nteracting **M**oving **A**gents) is a minimal multi-agent system designed to study how reactive agents influence and adapt to one another’s behaviors through interactions;
- Implemented in Mathematica, it models agents moving on a 2D grid with periodic boundaries;
- Each agent occupies one cell and follows either deterministic (D-agents) or random (R-agents) movement patterns;
- When agents come within an interaction range, their movement probabilities are modified — mimicking attraction or influence between individuals;
- Over time, these interactions can lead to emergent patterns such as convergence of random agents toward deterministic ones;
DRIMA serves as a metaphorical model for social, behavioral, or conceptual adaptation in groups, emphasizing simplicity and emergent complexity.

### Technologies

- Python 3.12.2.

### Coding Standards

- Follow PEP 8 guidelines for Python code style;
- Use meaningful variable and function names;
- Write docstrings for all public modules, functions, and classes;
- Include at least one example usage in comments. Provide an overview of the project you're working on, including its purpose, goals, and any relevant background information.

## Business Rules

### Environment Setup

- The world is a 2D grid with periodic boundaries (edges wrap around);
- Each cell can contain one agent;
- Simulation progresses through discrete time steps (world updates).

### Agent Types

- D-agent (Deterministic): Moves in one fixed direction with probability 1;
- R-agent (Random): Moves in all directions with varying probabilities that sum to 1.

### Movement Rules

- Movement directions: E, NE, N, NW, W, SW, S, SE, and X (no movement);
- The movement pattern is a probability vector for these 9 directions;
- Agents can have full mobility (always move) or partial mobility (can stay still).

### Interaction Rules

- Interaction Range: Agents interact if within a radius R (usually 1);
- Interaction Scope: Only binary interactions (two agents at a time);
- Interaction Type:
    - Same-type or different-type (e.g., D–D, R–R, D–R);
    - Case (uppercase/lowercase) defines whether an agent changes (e.g., DR → D changes R).
- Interaction Mode:
    - Synchronous: All agents processed sequentially per update;
    - Asynchronous: Random pairs interact one at a time.
- Interaction Direction:
    - One-way (only one agent changes) — implemented;
    - Two-way (both agents change) — conceptually supported but not implemented.
- Interaction Effect:
    - Direct: Immediate change in movement pattern (implemented);
    - Indirect: Change mediated by an internal state (conceptual).
- Interaction Intensity (Δp):
    - Fixed or variable increment to probability values;
    - Typical `Δp = 1/9` (moderate convergence).
- Determinism:
    - Movement is stochastic; interactions can be deterministic or probabilistic.
- Content:
    - Interactions are reactive (no memory, no internal model).

### Normalization Rule

- After each interaction, the sum of all movement probabilities for each agent must equal 1;
If any probabilities exceed 1 or become negative, they are adjusted and re-normalized.

### Outcome Rule

- Over time, R-agents may converge to deterministic movement patterns of nearby D-agents;
- Depending on parameters (Δp, grid size, density), dynamics may show stability, fluctuation, or chaotic patterns.

## Coding and Explanation Instructions

### Explaining Technical Concepts

- Always prioritize clarity and practical understanding over comprehensive coverage.

#### Start Simple, Build Up

- Introduce technical terms gradually after concepts are clear;
- Build each new idea on what was already explained;
- Use concrete examples before abstract theory.

#### Make It Practical

- Include working code examples that demonstrate the concept;
- Connect theory to problems developers actually face;
- Provide step-by-step implementation when relevant.

#### Address Common Confusion

- Explain what NOT to do and why;
- Address edge cases that often cause problems;
- Show debugging approaches when things go wrong.

### Reviewing Code

#### Performance Red Flags

- Spot inefficient loops and algorithmic issues;
- Check for memory leaks and resource cleanup;
- Review caching opportunities for expensive operations.

#### Code Quality Essentials

- Functions should be focused and appropriately sized;
- Use clear, descriptive naming conventions;
- Ensure proper error handling throughout.

#### Review Style

- Be specific and actionable in feedback;
- Explain the "why" behind recommendations;
- Acknowledge good patterns when you see them;
- Ask clarifying questions when code intent is unclear.
