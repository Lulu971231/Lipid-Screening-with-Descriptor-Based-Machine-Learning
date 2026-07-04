#!/usr/bin/env python
"""
Entry point: run ensemble-based virtual screening on the full
combinatorial lipid library.

Prerequisites:
    - output/basic_descriptors.csv must exist (run run_model_selection.py first)
    - output/all_des.csv must exist

Usage:
    cd <project_root>
    python scripts/run_virtual_screening.py
"""

import sys
from pathlib import Path

# Ensure the package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lnp_screen.virtual_screening import run_virtual_screening


def main():
    print("=" * 60)
    print("Virtual Screening: Ensemble XGBoost")
    print("=" * 60)
    run_virtual_screening()


if __name__ == "__main__":
    main()
