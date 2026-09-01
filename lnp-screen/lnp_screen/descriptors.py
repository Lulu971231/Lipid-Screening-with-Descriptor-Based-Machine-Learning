"""
Molecular descriptor generation using PaDEL and feature matrix construction
for multi-component lipid formulations.
"""

import numpy as np
import pandas as pd
from padelpy import from_smiles

from lnp_screen.config import (
    DESCRIPTORS_FILE,
    FEATURE_MATRIX_FILE,
    OUTPUT_DIR,
    RESULTS_FILE,
    SMILES_FILE,
)


def generate_descriptors():
    """
    Compute PaDEL molecular descriptors from SMILES in basic_smiles.csv.
    Saves results to OUTPUT_DIR/basic_descriptors.csv with the component
    identifier (flag) as the Name column.

    Returns:
        pd.DataFrame: Descriptor dataframe with Name column = flag.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    smiles_df = pd.read_csv(SMILES_FILE)
    smiles_list = smiles_df["smiles"].dropna().astype(str).tolist()
    flags = smiles_df["flag"].dropna().astype(str).tolist()

    print(f"Computing PaDEL descriptors for {len(smiles_list)} molecules...")
    _ = from_smiles(smiles_list, output_csv=str(DESCRIPTORS_FILE))

    # Replace auto-generated Name column with component identifiers
    desc_df = pd.read_csv(DESCRIPTORS_FILE)
    desc_df["Name"] = flags[:len(desc_df)]
    desc_df.to_csv(DESCRIPTORS_FILE, index=False)

    print(f"Descriptors saved to {DESCRIPTORS_FILE}")
    print(f"  Shape: {desc_df.shape[0]} molecules x {desc_df.shape[1]-1} descriptors")

    return desc_df


def build_feature_matrix():
    """
    Build a combined feature matrix for all experimentally tested
    formulations. Each formulation is represented by the concatenation
    of descriptors of its three components (tail + phosphate + amine).

    Requires basic_descriptors.csv to exist (run generate_descriptors first).

    Returns:
        np.ndarray: Feature matrix of shape (n_formulations, 3 * n_descriptors).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load experimental formulation data
    results_df = pd.read_csv(RESULTS_FILE)
    comp_tails = results_df["coma-l"].astype(str).tolist()
    comp_phosphates = results_df["comP1-P8"].astype(str).tolist()
    comp_amines = results_df["com1-20"].astype(str).tolist()
    n_formulations = len(comp_tails)

    # Load descriptor lookup table
    desc_df = pd.read_csv(DESCRIPTORS_FILE)
    desc_names = desc_df["Name"].astype(str).tolist()
    desc_matrix = desc_df.iloc[:, 1:].values

    print(f"Building feature matrix for {n_formulations} formulations...")

    # For each formulation, find and concatenate the three component descriptors
    feature_rows = []
    for i in range(n_formulations):
        row_features = []

        # Tail component
        tail_found = False
        for j in range(len(desc_names)):
            if comp_tails[i] == desc_names[j]:
                row_features.extend(desc_matrix[j])
                tail_found = True
                break
        if not tail_found:
            raise ValueError(
                f"Tail component '{comp_tails[i]}' not found in descriptors "
                f"(formulation index {i})"
            )

        # Phosphate linker component
        phos_found = False
        for j in range(len(desc_names)):
            if comp_phosphates[i] == desc_names[j]:
                row_features.extend(desc_matrix[j])
                phos_found = True
                break
        if not phos_found:
            raise ValueError(
                f"Phosphate component '{comp_phosphates[i]}' not found in "
                f"descriptors (formulation index {i})"
            )

        # Amine head component
        amine_found = False
        for j in range(len(desc_names)):
            if comp_amines[i] == desc_names[j]:
                row_features.extend(desc_matrix[j])
                amine_found = True
                break
        if not amine_found:
            raise ValueError(
                f"Amine component '{comp_amines[i]}' not found in descriptors "
                f"(formulation index {i})"
            )

        feature_rows.append(row_features)

    feature_matrix = np.array(feature_rows)
    np.savetxt(FEATURE_MATRIX_FILE, feature_matrix, delimiter=",", fmt="%.8f")

    print(f"Feature matrix saved to {FEATURE_MATRIX_FILE}")
    print(f"  Shape: {feature_matrix.shape}")

    return feature_matrix
