# Data Dictionary

## basic_smiles.csv

Molecular building blocks for lipid nanoparticle formulations.

| Column   | Type   | Description                                 |
|----------|--------|---------------------------------------------|
| flag     | string | Unique identifier for the component         |
| smiles   | string | SMILES notation of the molecule             |

### Component categories (determined by row order)

- Rows 0–35 (36 entries): Aldehyde tail components (hydrophobic tails)
- Rows 36–67 (32 entries): Amine head components (ionizable headgroups)
- Rows 68–80 (13 entries): Phosphate linker components

---

## autodevice_results.csv

Experimental transfection efficiency data for combinatorial LNP formulations.

| Column   | Type    | Description                                                      |
|----------|---------|------------------------------------------------------------------|
| num      | integer | Row index / formulation number                                   |
| coma-l   | string  | Aldehyde tail component identifier (references basic_smiles.csv) |
| comP1-P8 | string  | Phosphate linker identifier (references basic_smiles.csv)        |
| com1-20  | string  | Amine head component identifier (references basic_smiles.csv)    |
| result1  | integer | Transfection efficiency, replicate 1 (RLU)                       |
| result2  | integer | Transfection efficiency, replicate 2 (RLU)                       |
| result3  | integer | Transfection efficiency, replicate 3 (RLU)                       |
| average  | float   | Mean of result1, result2, result3                                |

### Binary label derivation

- Label = 1 if average >= 50,000
- Label = 0 if average < 50,000
