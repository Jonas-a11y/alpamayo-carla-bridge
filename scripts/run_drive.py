#!/usr/bin/env python3
"""Run the importable CARLA bridge from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.driver import main

if __name__ == "__main__":
    main()
