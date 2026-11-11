"""
src/evaluation.py
-----------------
Evaluation utilities: test-set metrics, repeated CV, overfitting check,
confusion matrix data, ROC curves. All return DataFrames or dicts for saving.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate


def evaluate_on_test(model_name, fitted_pipeline, X_test, y_test):
    """
    Compute test-set metrics for a fitted pipeline.
    Returns a dict of metrics and the classification report string.
    """
    y_pred = fitted_pipeline.predict(X_test)
    y_prob = fitted_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "CM": confusion_matrix(y_test, y_pred).tolist(),
        "FN": confusion_matrix(y_test, y_pred)[1, 0],  # false negatives (miss disease)
        "FP": confusion_matrix(y_test, y_pred)[0, 1],  # false positives
    }
    report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])
    return metrics, report


def repeated_cv(model_name, fitted_pipeline_unfitted, X, y, n_splits=5, n_repeats=5):
    """
    Repeated StratifiedKFold CV to assess robustness.
    fitted_pipeline_unfitted: an unfitted pipeline (clone of best estimator).
    Returns a DataFrame with mean ± std of accuracy, precision, recall, F1, ROC_AUC.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv_results = cross_validate(
        fitted_pipeline_unfitted, X, y,
        cv=rskf, scoring=scoring, n_jobs=-1,
        return_train_score=True
    )

    summary = {}
    for metric in scoring:
        vals = cv_results[f"test_{metric}"]
        summary[f"{metric}_mean"] = np.mean(vals)
        summary[f"{metric}_std"] = np.std(vals)

    summary["Model"] = model_name
    return summary, cv_results


def overfitting_check(model_name, grid_search, X_test, y_test):
    """
    Compare train CV score, test CV score (best_score_), and final test score.
    Returns a dict row.
    """
    best_index = grid_search.best_index_
    cv_results = grid_search.cv_results_

    train_score = cv_results["mean_train_score"][best_index]
    cv_score = grid_search.best_score_
    y_pred = grid_search.best_estimator_.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)

    gap = train_score - cv_score
    status = "OK" if gap < 0.08 else "OVERFIT?"

    return {
        "Model": model_name,
        "Train_F1_CV": round(train_score, 4),
        "Val_F1_CV": round(cv_score, 4),
        "Test_F1": round(test_f1, 4),
        "Train_Val_Gap": round(gap, 4),
        "Status": status,
    }


def build_metrics_df(all_metrics):
    """Convert list of metric dicts to a clean DataFrame (drop CM, FN, FP cols)."""
    rows = []
    for m in all_metrics:
        row = {k: v for k, v in m.items() if k not in ("CM", "FN", "FP")}
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")

