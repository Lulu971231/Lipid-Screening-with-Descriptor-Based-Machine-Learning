# Lipid-Screening-with-Descriptor-Based-Machine-Learning
README: Lipid Screening with Descriptor-Based Machine Learning
==============================================================

Project Overview
----------------
This script implements a machine learning-based virtual screening workflow to identify promising lipid-like compounds 
from combinatorial components. The workflow includes:

1. Molecular descriptor generation from SMILES.
2. Feature construction for real and virtual lipid candidates.
3. Classification model training using experimental data and SMOTE for class balancing.
4. Prediction and ranking of virtual candidates using ensemble-based screening.

The main classifier used is XGBoost.

Required Python Packages
-------------------------
Make sure the following Python packages are installed:
- pandas
- numpy
- imblearn (imbalanced-learn)
- scikit-learn
- xgboost
- padelpy
- matplotlib (optional, for visualization)

You can install them using pip:

    pip install pandas numpy imbalanced-learn scikit-learn xgboost padelpy matplotlib

Input Files
-----------
The script requires the following input files:

1. basic_smiles.xlsx  
   Excel file containing a column named `smiles` with SMILES strings of basic molecular components.

2. 结构的基元组分.xlsx  
   Excel file containing columns:
   - 产物编号: Identifier for each final product.
   - 基元1, 基元2, 基元3: Names of three basic building blocks (head group, linker, tail).
   - average: Experimental activity or performance value used for labeling (binary classification, threshold = 150000).

Optional:
- basic_descriptors0702.csv  
  CSV file containing pre-calculated descriptors for components used in virtual library screening.

Output Files
------------
- basic_descriptors.csv  
  Descriptors generated from SMILES by PaDEL (automatically generated).

- all_des.csv  
  Feature matrix of real products constructed from component descriptors (can be avoided if working purely in-memory).

- xgboost_classification_150000_model_0328.json  
  Trained XGBoost model saved for reuse.

How to Run
----------
1. Ensure that basic_smiles.xlsx and 结构的基元组分.xlsx are present in your working directory.
2. Run the Python script. It will:
   - Generate descriptors using PaDEL.
   - Construct feature vectors for known compounds.
   - Train a classification model with SMOTE-balanced data.
   - Generate all virtual combinations of 3-part components.
   - Predict their probabilities and rank high-performing candidates.
   - Output top components and combinations by occurrence frequency.

Model Details
-------------
- Binary classification: Label is 1 if `average >= 150000`, else 0.
- SMOTE is applied to balance the training dataset.
- XGBoost classifier is used with GPU acceleration (tree_method = gpu_hist).
- Ensemble prediction: The virtual library is screened with 1000 models using different random seeds.
- Candidates with predicted probability > 0.6 are selected as promising hits.

Output Interpretation
---------------------
At the end of execution, the script prints:
- Top 10 most frequent head groups (component1)
- Top 10 most frequent linkers (component2)
- Top 10 most frequent tail groups (component3)
- Top 50 full lipid combinations with their frequency of being selected in ensemble runs

Notes
-----
- PaDEL (used via padelpy) requires Java and must be properly set up in your environment.
- Large virtual libraries and multiple ensemble runs may consume significant memory and GPU resources.
- You can modify the activity threshold (150000) and the ensemble vote threshold (0.6) in the code to suit your data.

License
-------
This code is intended for academic and research use only.
