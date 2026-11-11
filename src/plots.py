"""
src/plots.py
------------
All visualization functions. Each saves a PNG at 300 dpi to outputs/figures/.
Returns the saved filepath.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc
from sklearn.inspection import permutation_importance

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
DPI = 300
PALETTE = "Set2"


def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] Saved {path}")
    return path


# -----------------------------------------------------------------------
# EDA Plots
# -----------------------------------------------------------------------

def plot_target_distribution(df, target_col="target"):
    """Bar chart of binary target distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Binary
    counts = df[target_col].value_counts().sort_index()
    labels = ["No Disease (0)", "Disease (1)"]
    colors = sns.color_palette(PALETTE, 2)
    axes[0].bar(labels, counts.values, color=colors)
    axes[0].set_title("Binary Target Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

    # Original num distribution
    orig_counts = df["num"].value_counts().sort_index()
    axes[1].bar(orig_counts.index.astype(str), orig_counts.values,
                color=sns.color_palette("Blues_d", len(orig_counts)))
    axes[1].set_title("Original Target Distribution (num 0–4)")
    axes[1].set_xlabel("num")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(orig_counts.values):
        axes[1].text(i, v + 1, str(v), ha="center", fontweight="bold")

    fig.suptitle("Target Variable Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig_2_1_target_distribution.png")


def plot_numeric_histograms(df, numeric_cols):
    """Histogram grid for numeric features, coloured by target."""
    n = len(numeric_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        for t, label, color in [(0, "No Disease", "steelblue"), (1, "Disease", "tomato")]:
            axes[i].hist(df[df["target"] == t][col].dropna(),
                         bins=20, alpha=0.6, label=label, color=color)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Histograms of Numeric Features by Target", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig_2_2_numeric_histograms.png")


def plot_boxplots_by_target(df, numeric_cols):
    """Box plots of numeric features grouped by target."""
    n = len(numeric_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()
    palette = {0: "steelblue", 1: "tomato"}
    for i, col in enumerate(numeric_cols):
        sub = df[[col, "target"]].dropna()
        sub["Target"] = sub["target"].map({0: "No Disease", 1: "Disease"})
        sns.boxplot(data=sub, x="Target", y=col, palette={"No Disease": "steelblue", "Disease": "tomato"},
                    ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Box Plots of Numeric Features by Target", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig_2_3_boxplots_by_target.png")


def plot_categorical_countplots(df, cat_cols):
    """Count plots of categorical features by target."""
    n = len(cat_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()
    palette = {0: "steelblue", 1: "tomato"}
    for i, col in enumerate(cat_cols):
        sub = df[[col, "target"]].dropna()
        sub["Target"] = sub["target"].map({0: "No Disease", 1: "Disease"})
        sns.countplot(data=sub, x=col, hue="Target",
                      palette={"No Disease": "steelblue", "Disease": "tomato"},
                      ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].legend(title="Target", fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Categorical Feature Count Plots by Target", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig_2_4_categorical_countplots.png")


def plot_correlation_heatmap(df, feature_cols):
    """Pearson correlation heatmap including target."""
    cols = feature_cols + ["target"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Pearson Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig_2_5_correlation_heatmap.png")


def plot_outlier_check(df, cols):
    """IQR-based outlier count bar chart for specified columns."""
    outlier_counts = {}
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts[col] = ((df[col] < lower) | (df[col] > upper)).sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(list(outlier_counts.keys()), list(outlier_counts.values()),
                  color=sns.color_palette(PALETTE, len(outlier_counts)))
    ax.set_title("IQR Outlier Counts per Feature", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Outliers")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(int(bar.get_height())), ha="center", va="bottom")
    fig.tight_layout()
    return _save(fig, "fig_2_6_iqr_outlier_counts.png")


# -----------------------------------------------------------------------
# Feature Selection Plots
# -----------------------------------------------------------------------

def plot_mi_scores(mi_scores, feature_names):
    """Bar chart of Mutual Information scores."""
    idx = np.argsort(mi_scores)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_scores = [mi_scores[i] for i in idx]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(sorted_names[::-1], sorted_scores[::-1],
                   color=sns.color_palette("viridis", len(sorted_names)))
    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Feature Ranking by Mutual Information (Training Data Only)",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=0.05, color="red", linestyle="--", label="Cutoff = 0.05")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "fig_3_1_mutual_information_scores.png")


# -----------------------------------------------------------------------
# Evaluation Plots
# -----------------------------------------------------------------------

def plot_confusion_matrices(all_cms, model_names):
    """Grid of confusion matrices for all models."""
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, cm_data, name in zip(axes, all_cms, model_names):
        cm = np.array(cm_data)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["No Disease", "Disease"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10)
    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig_4_1_confusion_matrices_grid.png")


def plot_individual_confusion_matrix(cm_data, model_name, feature_set_label="A"):
    """Single confusion matrix for a model."""
    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name} — Set {feature_set_label}", fontsize=11)
    fig.tight_layout()
    fname = f"fig_4_cm_{model_name.lower().replace(' ', '_')}_set{feature_set_label}.png"
    return _save(fig, fname)


def plot_roc_curves(roc_data, feature_set_label="A"):
    """
    Overlaid ROC curves for all models.
    roc_data: list of (model_name, fpr, tpr, auc_score)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette(PALETTE, len(roc_data))
    for (name, fpr, tpr, auc_score), color in zip(roc_data, colors):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — Feature Set {feature_set_label}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"fig_4_2_roc_curves_set{feature_set_label}.png")


def plot_metrics_bar(metrics_df, feature_set_label="A"):
    """Grouped bar chart of test-set metrics for all models."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    df_plot = metrics_df[metric_cols].copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df_plot))
    width = 0.15
    colors = sns.color_palette(PALETTE, len(metric_cols))
    for i, (metric, color) in enumerate(zip(metric_cols, colors)):
        bars = ax.bar(x + i * width, df_plot[metric], width, label=metric, color=color)
    ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
    ax.set_xticklabels(df_plot.index, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Model Comparison — Feature Set {feature_set_label}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"fig_4_3_metrics_bar_set{feature_set_label}.png")


def plot_cv_boxplots(cv_results_all, model_names, metric="f1", feature_set_label="A"):
    """Box plots of CV scores per model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [cv_results_all[name][f"test_{metric}"] for name in model_names]
    bp = ax.boxplot(data, tick_labels=model_names, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    colors = sns.color_palette(PALETTE, len(model_names))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_title(f"CV {metric.upper()} Score Distribution — Set {feature_set_label}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel(f"{metric.upper()} Score")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"fig_4_4_cv_boxplots_{metric}_set{feature_set_label}.png")


def plot_rf_feature_importance(rf_pipeline, feature_names, feature_set_label="A"):
    """Random Forest feature importance bar chart."""
    rf = rf_pipeline.named_steps["clf"]
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp = importances[idx]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(sorted_names[::-1], sorted_imp[::-1],
            color=sns.color_palette("viridis", len(sorted_names)))
    ax.set_xlabel("Importance")
    ax.set_title(f"Random Forest Feature Importances — Set {feature_set_label}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"fig_4_5_rf_feature_importance_set{feature_set_label}.png")


def plot_permutation_importance(rf_pipeline, X_test, y_test, feature_names, feature_set_label="A"):
    """Permutation importance for RF (or any model)."""
    from sklearn.inspection import permutation_importance as pi
    result = pi(rf_pipeline, X_test, y_test, n_repeats=20, random_state=42, scoring="f1")
    perm_imp = result.importances_mean
    perm_std = result.importances_std
    idx = np.argsort(perm_imp)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp = perm_imp[idx]
    sorted_std = perm_std[idx]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(sorted_names[::-1], sorted_imp[::-1], xerr=sorted_std[::-1],
            color=sns.color_palette("magma", len(sorted_names)), capsize=3)
    ax.set_xlabel("Mean Decrease in F1 (Permutation Importance)")
    ax.set_title(f"Permutation Feature Importance — Set {feature_set_label}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"fig_4_6_permutation_importance_set{feature_set_label}.png")


def plot_lr_coefficients(lr_pipeline, feature_names, feature_set_label="A"):
    """Logistic Regression coefficients / odds ratios."""
    lr = lr_pipeline.named_steps["clf"]
    coefs = lr.coef_[0]
    odds_ratios = np.exp(coefs)
    idx = np.argsort(np.abs(coefs))[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_coefs = coefs[idx]
    sorted_or = odds_ratios[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["tomato" if c > 0 else "steelblue" for c in sorted_coefs]
    axes[0].barh(sorted_names[::-1], sorted_coefs[::-1], color=colors[::-1])
    axes[0].set_xlabel("Coefficient Value")
    axes[0].set_title("LR Coefficients (standardised features)")
    axes[0].axvline(0, color="black", lw=0.8)

    axes[1].barh(sorted_names[::-1], sorted_or[::-1],
                 color=sns.color_palette("RdYlGn", len(sorted_names)))
    axes[1].set_xlabel("Odds Ratio")
    axes[1].set_title("LR Odds Ratios")
    axes[1].axvline(1, color="black", lw=0.8, linestyle="--")

    fig.suptitle(f"Logistic Regression Interpretability — Set {feature_set_label}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"fig_4_7_lr_coefficients_set{feature_set_label}.png")


def plot_decision_tree(dt_pipeline, feature_names, feature_set_label="A"):
    """Shallow Decision Tree visualization."""
    from sklearn.tree import plot_tree
    dt = dt_pipeline.named_steps["clf"]
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        dt, feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        filled=True, rounded=True, impurity=True,
        max_depth=3, fontsize=8, ax=ax
    )
    ax.set_title(f"Decision Tree (max_depth shown=3) — Set {feature_set_label}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"fig_4_8_decision_tree_set{feature_set_label}.png")


def plot_set_comparison(results_A, results_B, results_C, metric="F1"):
    """Grouped bar chart comparing models across feature sets A, B, C."""
    models = results_A.index.tolist()
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, results_A[metric], width, label="Set A (Primary)", color="steelblue")
    ax.bar(x, results_B[metric], width, label="Set B (All 13)", color="darkorange")
    ax.bar(x + width, results_C[metric], width, label="Set C (Auto-selected)", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{metric} Score: Feature Set A vs B vs C", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"fig_4_9_set_comparison_{metric.lower()}.png")


def plot_fn_fp_analysis(all_metrics, feature_set_label="A"):
    """Stacked bar of FN vs FP per model."""
    models = [m["Model"] for m in all_metrics]
    fns = [m["FN"] for m in all_metrics]
    fps = [m["FP"] for m in all_metrics]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, fns, label="False Negatives (missed disease)", color="tomato")
    ax.bar(x, fps, bottom=fns, label="False Positives (healthy classified as disease)", color="gold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"False Negatives vs False Positives — Set {feature_set_label}",
                 fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return _save(fig, f"fig_4_10_fn_fp_analysis_set{feature_set_label}.png")
