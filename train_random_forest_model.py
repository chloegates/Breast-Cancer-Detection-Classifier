#!/usr/bin/env python3
"""Train Random Forest on unscaled breast cancer development data with 5-fold CV."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

PROJECT_DIR = Path(__file__).resolve().parent
DEV_PATH = PROJECT_DIR / "development_unscaled.csv"
TEST_PATH = PROJECT_DIR / "test_unscaled_FINAL_HOLDOUT.csv"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_PATH = PROJECT_DIR / "random_forest_cv5_results.json"
MODEL_PATH = MODEL_DIR / "random_forest_cv5_model.joblib"
FEATURE_IMPORTANCE_PATH = PROJECT_DIR / "random_forest_feature_importance.csv"

RANDOM_STATE = 42
N_SPLITS = 5

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}


def load_xy(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    ids = df["id"].copy()
    y = df["diagnosis"].astype(int)
    X = df.drop(columns=["id", "diagnosis"])
    return X, y, ids


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    X_dev, y_dev, _ = load_xy(DEV_PATH)
    X_test, y_test, _ = load_xy(TEST_PATH)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=1,
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    grid.fit(X_dev, y_dev)
    best_model: RandomForestClassifier = grid.best_estimator_

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv_scores = cross_validate(
        best_model,
        X_dev,
        y_dev,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_test_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
    }

    fold_metrics = {
        metric: {
            "mean": float(np.mean(cv_scores[f"test_{metric}"])),
            "std": float(np.std(cv_scores[f"test_{metric}"])),
            "fold_values": [float(v) for v in cv_scores[f"test_{metric}"]],
        }
        for metric in scoring
    }

    feature_importance = pd.DataFrame(
        {
            "feature": X_dev.columns,
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    results = {
        "model": "RandomForestClassifier",
        "data": {
            "development_file": str(DEV_PATH.name),
            "holdout_file": str(TEST_PATH.name),
            "development_samples": int(len(X_dev)),
            "holdout_samples": int(len(X_test)),
            "feature_count": int(X_dev.shape[1]),
            "scaling": "none (unscaled features)",
        },
        "validation": {
            "method": "5-fold stratified cross-validation",
            "tuning": "GridSearchCV on development set",
            "selection_metric": "accuracy",
        },
        "best_hyperparameters": grid.best_params_,
        "cv5_metrics": fold_metrics,
        "holdout_metrics": test_metrics,
        "confusion_matrix_holdout": confusion_matrix(y_test, y_test_pred).tolist(),
        "classification_report_holdout": classification_report(
            y_test, y_test_pred, output_dict=True, zero_division=0
        ),
        "top_features": feature_importance.head(10).to_dict(orient="records"),
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    joblib.dump(best_model, MODEL_PATH)

    print("Training complete.")
    print(f"Best params: {grid.best_params_}")
    print(f"5-fold CV accuracy: {fold_metrics['accuracy']['mean']:.4f} +/- {fold_metrics['accuracy']['std']:.4f}")
    print(f"Holdout accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
