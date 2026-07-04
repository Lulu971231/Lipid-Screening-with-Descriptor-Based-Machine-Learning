"""
Virtual lipid library construction and ensemble-based screening.
Enumerates all possible three-component combinations, predicts their
performance using an ensemble of XGBoost models, and reports the
most frequently predicted hit components.
"""

from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from lnp_screen.config import (
    AMINE_END,
    AMINE_START,
    DESCRIPTORS_FILE,
    FEATURE_MATRIX_FILE,
    LABEL_THRESHOLD,
    N_ENSEMBLE_RUNS,
    OUTPUT_DIR,
    PHOSPHATE_END,
    PHOSPHATE_START,
    PREDICTION_THRESHOLD,
    RANDOM_STATE,
    RESULTS_FILE,
    SCREENING_RESULTS_FILE,
    TAIL_END,
    TAIL_START,
    TOP_N_COMPONENTS,
)


def _build_virtual_library(desc_df):
    """
    Construct the virtual library by enumerating all possible combinations
    of tail, phosphate, and amine components.

    Returns:
        combinations: List of (tail, phosphate, amine) name tuples.
        features: np.ndarray of concatenated descriptor vectors.
    """
    # Extract component groups by their position in the descriptor file
    component_name_tails = desc_df["Name"].iloc[TAIL_START:TAIL_END].values
    component_tails = desc_df.iloc[TAIL_START:TAIL_END, 1:].values

    component_name_amines = desc_df["Name"].iloc[AMINE_START:AMINE_END].values
    component_amines = desc_df.iloc[AMINE_START:AMINE_END, 1:].values

    component_name_phosphates = desc_df["Name"].iloc[PHOSPHATE_START:PHOSPHATE_END].values
    component_phosphates = desc_df.iloc[PHOSPHATE_START:PHOSPHATE_END, 1:].values

    print(f"  Tails: {len(component_name_tails)} components")
    print(f"  Amines: {len(component_name_amines)} components")
    print(f"  Phosphates: {len(component_name_phosphates)} components")

    # Enumerate all combinations
    combinations = list(
        product(component_name_tails, component_name_phosphates, component_name_amines)
    )
    num_combinations = len(combinations)
    n_features = (component_tails.shape[1] + component_phosphates.shape[1]
                  + component_amines.shape[1])

    print(f"  Total combinations: {num_combinations}")

    features = np.zeros((num_combinations, n_features))

    for i, (tail, phos, amine) in enumerate(combinations):
        tail_idx = np.where(component_name_tails == tail)[0][0]
        phos_idx = np.where(component_name_phosphates == phos)[0][0]
        amine_idx = np.where(component_name_amines == amine)[0][0]
        features[i] = np.hstack((
            component_tails[tail_idx],
            component_phosphates[phos_idx],
            component_amines[amine_idx],
        ))

    return combinations, features


def run_virtual_screening():
    """
    Perform ensemble-based virtual screening:
    1. Train multiple XGBoost models with different random seeds.
    2. Predict hit probability for all virtual library entries.
    3. Collect entries exceeding the prediction threshold.
    4. Analyze component frequency among predicted hits.

    Returns:
        pd.DataFrame: Screening results sorted by hit frequency.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data
    X = pd.read_csv(FEATURE_MATRIX_FILE, header=None).values
    results_df = pd.read_csv(RESULTS_FILE)
    y = results_df["average"].apply(
        lambda x: 1 if x >= LABEL_THRESHOLD else 0
    ).values

    # Train/test split and SMOTE oversampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"Training data after SMOTE: {X_res.shape[0]} samples "
          f"(pos={int(y_res.sum())}, neg={len(y_res) - int(y_res.sum())})")

    # Build virtual library
    print("\nBuilding virtual library...")
    desc_df = pd.read_csv(DESCRIPTORS_FILE)
    combinations, virtual_features = _build_virtual_library(desc_df)

    # Ensemble screening
    print(f"\nRunning ensemble screening ({N_ENSEMBLE_RUNS} models, "
          f"threshold={PREDICTION_THRESHOLD})...")

    top_lipids_indices = []
    rng = np.random.RandomState(RANDOM_STATE)
    seeds = rng.randint(0, 10000, size=N_ENSEMBLE_RUNS)

    for run_idx, seed in enumerate(seeds):
        if (run_idx + 1) % 100 == 0:
            print(f"  Progress: {run_idx + 1}/{N_ENSEMBLE_RUNS}")

        model = XGBClassifier(
            random_state=int(seed), eval_metric="logloss", verbosity=0
        )
        model.fit(X_res, y_res)
        predictions = model.predict_proba(virtual_features)[:, 1]

        hit_indices = [idx for idx in range(len(predictions))
                       if predictions[idx] > PREDICTION_THRESHOLD]
        top_lipids_indices.extend(hit_indices)

    print(f"\nTotal hit events collected: {len(top_lipids_indices)}")

    if len(top_lipids_indices) == 0:
        print("No hits found. Consider lowering PREDICTION_THRESHOLD in config.py.")
        return pd.DataFrame()

    # Frequency analysis by component
    top_combinations = np.array(combinations, dtype=object)[top_lipids_indices]

    tail_counter = Counter(top_combinations[:, 0])
    phosphate_counter = Counter(top_combinations[:, 1])
    amine_counter = Counter(top_combinations[:, 2])

    top_tails = tail_counter.most_common(TOP_N_COMPONENTS)
    top_phosphates = phosphate_counter.most_common(TOP_N_COMPONENTS)
    top_amines = amine_counter.most_common(TOP_N_COMPONENTS)

    print(f"\nTop {TOP_N_COMPONENTS} tail components:")
    for component, freq in top_tails:
        print(f"  {component}: {freq}")

    print(f"\nTop {TOP_N_COMPONENTS} phosphate linkers:")
    for component, freq in top_phosphates:
        print(f"  {component}: {freq}")

    print(f"\nTop {TOP_N_COMPONENTS} amine heads:")
    for component, freq in top_amines:
        print(f"  {component}: {freq}")

    # Save detailed results
    hit_counter = Counter(top_lipids_indices)
    results_rows = []
    for idx, count in hit_counter.most_common():
        combo = combinations[idx]
        results_rows.append({
            "tail": combo[0],
            "phosphate": combo[1],
            "amine": combo[2],
            "frequency": count,
        })

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(SCREENING_RESULTS_FILE, index=False)
    print(f"\nScreening results saved to {SCREENING_RESULTS_FILE}")

    return results_df
