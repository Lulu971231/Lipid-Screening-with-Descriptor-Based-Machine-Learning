"""
Central configuration for paths, thresholds, and hyperparameters.
"""

import os
from pathlib import Path

# ===========================================================
# Project paths
# ===========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Input files
SMILES_FILE = DATA_DIR / "basic_smiles.csv"
RESULTS_FILE = DATA_DIR / "autodevice_results.csv"

# Output files
DESCRIPTORS_FILE = OUTPUT_DIR / "basic_descriptors.csv"
FEATURE_MATRIX_FILE = OUTPUT_DIR / "all_des.csv"
SCREENING_RESULTS_FILE = OUTPUT_DIR / "screening_results.csv"

# ===========================================================
# Classification thresholds
# ===========================================================
LABEL_THRESHOLD = 50000  # average >= this value => positive class

# ===========================================================
# Model selection configuration
# ===========================================================
RANDOM_STATE = 42
N_SPLITS_OUTER = 5       # Outer cross-validation folds
N_SPLITS_INNER = 3       # Inner cross-validation folds (hyperparameter search)
N_ITER_BAYES = 20        # Bayesian optimization iterations per model

# ===========================================================
# Virtual screening configuration
# ===========================================================
N_ENSEMBLE_RUNS = 1000   # Number of XGBoost models in ensemble
PREDICTION_THRESHOLD = 0.6  # Probability threshold for hit classification
TOP_N_COMPONENTS = 10    # Number of top components to report

# ===========================================================
# Component index ranges in basic_smiles.csv
# These correspond to the row order in basic_smiles.csv:
#   - Tails:      rows 0-35  (36 entries)
#   - Amines:     rows 36-67 (32 entries)
#   - Phosphates: rows 68-80 (13 entries)
# ===========================================================
TAIL_START = 0
TAIL_END = 36
AMINE_START = 36
AMINE_END = 68
PHOSPHATE_START = 68
PHOSPHATE_END = 81

# ===========================================================
# Java environment for PaDEL
# ===========================================================
os.environ["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8"
