#!/usr/bin/env python
"""
Entry point: generate molecular descriptors, build feature matrix,
and run nested cross-validation for model selection.

Usage:
    cd <project_root>
    python scripts/run_model_selection.py
"""

import sys
from pathlib import Path

# Ensure the package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lnp_screen.descriptors import build_feature_matrix, generate_descriptors
from lnp_screen.model_selection import run_model_selection


def main():
    print("=" * 60)
    print("Step 1: Generating PaDEL molecular descriptors")
    print("=" * 60)
    generate_descriptors()

    print("\n" + "=" * 60)
    print("Step 2: Building combined feature matrix")
    print("=" * 60)
    build_feature_matrix()

    print("\n" + "=" * 60)
    print("Step 3: Nested cross-validation with Bayesian optimization")
    print("=" * 60)
    run_model_selection()


if __name__ == "__main__":
    main()
