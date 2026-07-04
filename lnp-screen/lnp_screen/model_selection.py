"""
Nested cross-validation with Bayesian hyperparameter optimization
for model selection. Compares XGBoost, Random Forest, and Logistic
Regression with SMOTE oversampling for class imbalance.
"""

import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, make_scorer, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBClassifier

from lnp_screen.config import (
    FEATURE_MATRIX_FILE,
    LABEL_THRESHOLD,
    N_ITER_BAYES,
    N_SPLITS_INNER,
    N_SPLITS_OUTER,
    OUTPUT_DIR,
    RANDOM_STATE,
    RESULTS_FILE,
)


def _pr_auc_score(y_true, y_probs):
    """Compute area under the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)


PR_SCORER = make_scorer(_pr_auc_score, needs_proba=True)


def _get_search_spaces():
    """Define Bayesian search spaces for each algorithm."""
    xgb_space = {
        "max_depth": Integer(3, 10),
        "learning_rate": Real(0.01, 0.3, "log-uniform"),
        "n_estimators": Integer(50, 500),
        "subsample": Real(0.5, 1.0),
        "colsample_bytree": Real(0.5, 1.0),
        "gamma": Real(0, 5),
        "min_child_weight": Integer(1, 10),
    }

    rf_space = {
        "n_estimators": Integer(50, 500),
        "max_depth": Integer(3, 20),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 5),
    }

    lr_space = {
        "C": Real(0.01, 100, "log-uniform"),
        "max_iter": Integer(500, 2000),
    }

    return xgb_space, rf_space, lr_space


def _resample_training_data(X_train, y_train):
    """Apply SMOTE or random oversampling depending on minority class size."""
    n_pos = int(np.sum(y_train == 1))

    if n_pos >= 2:
        k_neighbors = min(5, n_pos - 1)
        sampler = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
    else:
        sampler = RandomOverSampler(random_state=RANDOM_STATE)

    return sampler.fit_resample(X_train, y_train)


def run_model_selection():
    """
    Perform nested cross-validation with Bayesian hyperparameter
    optimization. Trains XGBoost, Random Forest, and Logistic Regression
    on each outer fold and saves the best model for each algorithm.

    Returns:
        dict: Best model info for each algorithm with keys
              'model', 'roc_auc', 'pr_auc'.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    X = pd.read_csv(FEATURE_MATRIX_FILE, header=None).values
    results_df = pd.read_csv(RESULTS_FILE)
    y = results_df["average"].apply(
        lambda x: 1 if x >= LABEL_THRESHOLD else 0
    ).values

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Positive (>={LABEL_THRESHOLD}): {y.sum()}, "
          f"Negative: {len(y) - y.sum()}")

    # Setup
    xgb_space, rf_space, lr_space = _get_search_spaces()
    outer_cv = StratifiedKFold(
        n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_STATE
    )

    best_models = {
        "xgb": {"model": None, "roc_auc": -1, "pr_auc": -1},
        "rf":  {"model": None, "roc_auc": -1, "pr_auc": -1},
        "lr":  {"model": None, "roc_auc": -1, "pr_auc": -1},
    }

    # Nested cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        print(f"\n--- Outer fold {fold_idx}/{N_SPLITS_OUTER} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        n_pos_train = int(np.sum(y_train == 1))
        n_pos_test = int(np.sum(y_test == 1))
        print(f"  Train pos/neg: {n_pos_train}/{len(y_train) - n_pos_train} | "
              f"Test pos/neg: {n_pos_test}/{len(y_test) - n_pos_test}")

        if n_pos_train == 0:
            print("  WARNING: no positive samples in training fold. Skipping.")
            continue

        # Resample training data
        X_res, y_res = _resample_training_data(X_train, y_train)
        print(f"  After resampling: pos={int(np.sum(y_res == 1))}, "
              f"neg={len(y_res) - int(np.sum(y_res == 1))}")

        # XGBoost
        search_xgb = BayesSearchCV(
            estimator=XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
                random_state=RANDOM_STATE, verbosity=0,
            ),
            search_spaces=xgb_space,
            n_iter=N_ITER_BAYES, scoring=PR_SCORER,
            cv=N_SPLITS_INNER, n_jobs=-1, verbose=0,
            random_state=RANDOM_STATE,
        )
        search_xgb.fit(X_res, y_res)

        # Random Forest
        search_rf = BayesSearchCV(
            estimator=RandomForestClassifier(random_state=RANDOM_STATE),
            search_spaces=rf_space,
            n_iter=N_ITER_BAYES, scoring=PR_SCORER,
            cv=N_SPLITS_INNER, n_jobs=-1, verbose=0,
            random_state=RANDOM_STATE,
        )
        search_rf.fit(X_res, y_res)

        # Logistic Regression
        search_lr = BayesSearchCV(
            estimator=LogisticRegression(
                random_state=RANDOM_STATE, solver="liblinear"
            ),
            search_spaces=lr_space,
            n_iter=N_ITER_BAYES, scoring=PR_SCORER,
            cv=N_SPLITS_INNER, n_jobs=-1, verbose=0,
            random_state=RANDOM_STATE,
        )
        search_lr.fit(X_res, y_res)

        # Evaluate on held-out test fold
        fold_models = {
            "xgb": search_xgb.best_estimator_,
            "rf": search_rf.best_estimator_,
            "lr": search_lr.best_estimator_,
        }

        for name, model in fold_models.items():
            probs = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, probs)
            precision, recall, _ = precision_recall_curve(y_test, probs)
            roc_auc_val = auc(fpr, tpr)
            pr_auc_val = auc(recall, precision)

            print(f"  {name.upper():3s}  ROC AUC = {roc_auc_val:.4f}, "
                  f"PR AUC = {pr_auc_val:.4f}")

            if pr_auc_val > best_models[name]["pr_auc"]:
                best_models[name]["model"] = model
                best_models[name]["roc_auc"] = roc_auc_val
                best_models[name]["pr_auc"] = pr_auc_val

    # Save best models
    print("\n--- Saving best models ---")
    for name, info in best_models.items():
        if info["model"] is None:
            print(f"  {name}: no valid model produced")
            continue
        save_path = OUTPUT_DIR / f"best_{name}_model.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(info["model"], f)
        print(f"  Saved {save_path.name} | "
              f"ROC AUC = {info['roc_auc']:.4f}, PR AUC = {info['pr_auc']:.4f}")

    print("\nModel selection complete.")
    return best_models
