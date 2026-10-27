#!/usr/bin/env python3
"""Top-level command-line entry point for DRIMA.

Project Overview
----------------
DRIMA (Dynamics of Randomly Interacting Moving Agents) is a reactive multi-agent simulation
system modeled on a 2D periodic cellular grid, based on Pedro P. B. de Oliveira's article
in *The Mathematica Journal* (12, 2010).

Run `python3 main.py --help` to view all configurable initial world parameters and interaction
settings.

Example Usage
-------------
```bash
# Run an 8x5 world with 2 D-agents (moving East and Southeast) and 6 R-agents
python3 main.py --grid-width 8 --grid-height 5 --num-d-agents 2 --num-r-agents 6 \
    --d-directions E,SE --iterations 100 --seed 42 --display-world --display-history \
    --output history.json --log-file drima.log
```
"""

import sys

from src.drima.cli import main


if __name__ == "__main__":
    sys.exit(main())
