# DRIMA: A Minimal System for Probing the Dynamics of Change in a Reactive Multi-Agent Setting

A Python 3.9.6 implementation and command-line interface (CLI) for **DRIMA** (*Dynamics of Randomly Interacting Moving Agents*), based on the model by Pedro P. B. de Oliveira ([*The Mathematica Journal*, 12, 2010](./docs/DRIMA:%20A%20Minimal%20System%20for%20Probing%20the%20Dynamics%20of%20Change%20in%20a%20Reactive%20Multi-Agent%20Setting.tex)).

---

## Overview & Motivation

DRIMA models how reactive agents have their behavior changed by the behavior of others as the outcome of local pairwise interactions. All behavioral and conceptual orientation is represented through a single observable action: **how agents move across a two-dimensional periodic (toroidal) grid**.

- **D-agents (Deterministic):** Move in a single direction with probability `1.0` (and `0.0` in all other directions). They act as fixed behavioral attractors.
- **R-agents (Random):** Move probabilistically across the 8 compass directions (`E, NE, N, NW, W, SW, S, SE`) plus no-movement (`X`).
- **Interact-First-Then-Move Cycle:** At each world update step (`sync` or `async`), agents within Moore neighborhood radius $R$ interact pairwise in a direct manner, normalize their updated movement patterns using iterative equal-displacement normalization, and step on the periodic grid.

---

## Quick Start

### 1. Run via CLI Flags
```bash
python3 main.py \
    --grid-width 8 \
    --grid-height 5 \
    --num-d-agents 2 \
    --num-r-agents 6 \
    --d-directions SE,E \
    --mobility F \
    --interaction-types dd,RR,dR,Rd \
    --range 1 \
    --update-mode sync \
    --delta-mode fixed \
    --fp 1.0 \
    --iterations 100 \
    --seed 42 \
    --display-world \
    --display-history \
    --output history.json \
    --log-file drima.log
```

### 2. Run from a JSON Configuration File (with Optional CLI Overrides)
```bash
# Run the Figure 2 experiment from JSON and override iterations and seed via CLI
python3 main.py --config examples/figure2_experiment.json --iterations 200 --seed 42 --display-history
```

### 3. Run the Exact Section 3.1 World Example
```bash
python3 main.py --config examples/section3_world.json --display-world
```

### 4. View All Configurable Parameters
```bash
python3 main.py --help
```

---

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```
