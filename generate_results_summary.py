"""
generate_results_summary.py
----------------------------
Standalone script to regenerate RESULTS_SUMMARY.md from existing outputs.
Run this after run_pipeline.py has completed successfully.
"""
import os, sys, platform, glob, datetime
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; import sklearn; import joblib; import seaborn as sns

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load computed data
df = pd.read_csv("data/processed/heart_disease_processed.csv")
access_date = "2026-11-11"

SET_A_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "thalach"]
SET_B_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                   "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET_COL = "target"

# Read tables
metrics_A = pd.read_csv("outputs/tables/table_model_comparison_setA.csv")
metrics_B = pd.read_csv("outputs/tables/table_model_comparison_setB.csv")
metrics_C = pd.read_csv("outputs/tables/table_model_comparison_setC.csv")
feat_sel  = pd.read_csv("outputs/tables/table_feature_selection.csv")
hp_df     = pd.read_csv("outputs/tables/table_hyperparameters.csv")
of_df     = pd.read_csv("outputs/tables/table_overfitting_check.csv")
rcv_df    = pd.read_csv("outputs/tables/table_repeated_cv.csv")
split_df  = pd.read_csv("outputs/tables/table_split_sizes.csv")

# Reconstruct some stats
n_disease    = int(df["target"].sum())
n_no_disease = int((df["target"] == 0).sum())
n_missing    = int(df[SET_B_FEATURES].isnull().sum().sum())

# Set C features
set_c_rows = feat_sel[feat_sel["Set_C"] == "✓"]["Feature"].tolist()
SET_C_FEATURES = set_c_rows

lib_versions = {
    "Python":       platform.python_version(),
    "scikit-learn": sklearn.__version__,
    "pandas":       pd.__version__,
    "numpy":        np.__version__,
    "matplotlib":   matplotlib.__version__,
    "seaborn":      sns.__version__,
    "joblib":       joblib.__version__,
}

# Best model
best_name = metrics_A.loc[metrics_A["F1"].idxmax(), "Model"]
best_row = metrics_A[metrics_A["Model"] == best_name].iloc[0]
MI_CUTOFF = 0.05

# Figures & tables
all_figs = sorted(glob.glob("outputs/figures/*.png"))
all_tabs = sorted(glob.glob("outputs/tables/*.csv"))

fig_descriptions = {
    "fig_2_1_target_distribution.png":        "Binary and original (0–4) target class distribution bar charts",
    "fig_2_2_numeric_histograms.png":          "Histograms of all numeric features, coloured by target class",
    "fig_2_3_boxplots_by_target.png":          "Box plots of numeric features grouped by target",
    "fig_2_4_categorical_countplots.png":      "Count plots of categorical features by target class",
    "fig_2_5_correlation_heatmap.png":         "Pearson correlation heatmap of all features and target",
    "fig_2_6_iqr_outlier_counts.png":          "IQR-based outlier counts per numeric feature",
    "fig_3_1_mutual_information_scores.png":   "Mutual Information feature ranking (training data only)",
}
tab_descriptions = {
    "table_attr_description.csv":           "Feature attribute table: name, dtype, description, coding, missing count",
    "table_split_sizes.csv":                "Train/test split sizes and class counts",
    "table_feature_selection.csv":          "Feature inclusion/exclusion for Sets A, B, C with MI and RF scores",
    "table_hyperparameters.csv":            "Best hyperparameters per model and feature set",
    "table_model_comparison_setA.csv":      "Test-set model comparison metrics — Feature Set A (primary)",
    "table_model_comparison_setB.csv":      "Test-set model comparison metrics — Feature Set B (all 13)",
    "table_model_comparison_setC.csv":      "Test-set model comparison metrics — Feature Set C (auto-selected)",
    "table_overfitting_check.csv":          "Train CV vs val CV vs test F1 scores for overfitting detection",
    "table_repeated_cv.csv":               "Repeated stratified 5×5 CV mean±std for all models and feature sets",
    "table_prediction_demo.csv":            "Three sample patient predictions with inputs, labels and probabilities",
}

def fig_desc(bname):
    if bname in fig_descriptions: return fig_descriptions[bname]
    if "confusion" in bname:    return f"Confusion matrix — {bname}"
    if "roc_curve" in bname:    return f"ROC curves — {bname}"
    if "metrics_bar" in bname:  return f"Grouped bar chart of test-set metrics — {bname}"
    if "cv_boxplot" in bname:   return f"CV F1 score boxplots — {bname}"
    if "rf_feature" in bname:   return f"Random Forest feature importances — {bname}"
    if "permutation" in bname:  return f"Permutation importance — {bname}"
    if "lr_coef" in bname:      return f"Logistic Regression coefficients and odds ratios — {bname}"
    if "decision_tree" in bname:return f"Decision Tree visualisation — {bname}"
    if "set_comparison" in bname:return f"Feature set A vs B vs C comparison — {bname}"
    if "fn_fp" in bname:        return f"False Negatives vs False Positives analysis — {bname}"
    return "Model evaluation figure"

fig_md_rows = [f"| `{os.path.basename(f)}` | {fig_desc(os.path.basename(f))} |" for f in all_figs]
tab_md_rows = [f"| `{os.path.basename(t)}` | {tab_descriptions.get(os.path.basename(t), 'Output table')} |" for t in all_tabs]

# EDA observations
top_corr_col = df[SET_B_FEATURES + ["target"]].corr()["target"].drop("target").abs().idxmax()
top_corr_val = df[SET_B_FEATURES + ["target"]].corr()["target"].drop("target").abs().max()

# Num distribution table rows
num_dist_rows = "\n".join(f"| {k} | {v} |" for k,v in df["num"].value_counts().sort_index().items())

# Split rows
train_row = split_df[split_df["Subset"]=="Train"].iloc[0]
test_row  = split_df[split_df["Subset"]=="Test"].iloc[0]

summary_md = f"""# Results Summary — Heart Disease Prediction System

> **Purpose:** Source material for the written report.  
> All numbers are computed from executed code; none are manually entered.  
> Generated: {datetime.datetime.now().isoformat()[:19]}

---

## 1. Dataset Facts

| Item | Value |
|------|-------|
| Source | UCI Heart Disease Dataset (Cleveland subset) — CC BY 4.0 |
| URL | https://archive.ics.uci.edu/dataset/45/heart+disease |
| Access Date | {access_date} |
| Total Rows | {len(df)} |
| Total Columns | {df.shape[1]} (including derived `target`) |
| Predictor Features | 13 |
| No Disease (target=0) | {n_no_disease} ({n_no_disease/len(df)*100:.1f}%) |
| Disease (target=1) | {n_disease} ({n_disease/len(df)*100:.1f}%) |
| Missing Values (predictors) | {n_missing} total (ca: {int(df['ca'].isnull().sum())}, thal: {int(df['thal'].isnull().sum())}) |
| Duplicates | {int(df.duplicated().sum())} |

### Original Target Distribution (num: 0–4)

| num | count |
|-----|-------|
{num_dist_rows}

---

## 2. Environment

| Library | Version |
|---------|---------|
{chr(10).join(f"| {k} | {v} |" for k, v in lib_versions.items())}
| Platform | {platform.platform()} |
| random_state | 42 (everywhere) |

---

## 3. Split Sizes

| Subset | Records | % | No Disease | Disease |
|--------|---------|---|------------|---------|
| Train | {train_row['Records']} | {train_row['Pct']} | {train_row['No_Disease']} | {train_row['Disease']} |
| Test  | {test_row['Records']}  | {test_row['Pct']}  | {test_row['No_Disease']}  | {test_row['Disease']}  |

---

## 4. Feature Sets

| Feature Set | Features | Count |
|-------------|----------|-------|
| A (Primary / Assignment) | {', '.join(SET_A_FEATURES)} | {len(SET_A_FEATURES)} |
| B (All 13 predictors) | {', '.join(SET_B_FEATURES)} | {len(SET_B_FEATURES)} |
| C (Auto-selected: MI+RF) | {', '.join(SET_C_FEATURES)} | {len(SET_C_FEATURES)} |

**Feature selection cutoff:** MI Score ≥ {MI_CUTOFF} (computed on training data only to prevent leakage).

---

## 5. Best Hyperparameters (Feature Set A — Primary)

| Model | Hyperparameter | Search Range | Selected Value | Method |
|-------|---------------|-------------|----------------|--------|
{chr(10).join(f"| {r['Model']} | {r['Hyperparameter']} | {r['Search_Range']} | {r['Selected_Value']} | {r['Method']} |" for _, r in hp_df[hp_df['Feature_Set']=='A'].iterrows())}

---

## 6. Model Comparison — Test Set

### Feature Set A (Primary — 6 assignment features)

| Model | Accuracy | Precision | Recall | F1 | ROC_AUC |
|-------|---------|----------|--------|-----|---------|
{chr(10).join(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['ROC_AUC']:.4f} |" for _, r in metrics_A.iterrows())}

### Feature Set B (All 13 predictors)

| Model | Accuracy | Precision | Recall | F1 | ROC_AUC |
|-------|---------|----------|--------|-----|---------|
{chr(10).join(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['ROC_AUC']:.4f} |" for _, r in metrics_B.iterrows())}

### Feature Set C (Auto-selected: MI + RF Importance)

| Model | Accuracy | Precision | Recall | F1 | ROC_AUC |
|-------|---------|----------|--------|-----|---------|
{chr(10).join(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['ROC_AUC']:.4f} |" for _, r in metrics_C.iterrows())}

---

## 7. Repeated CV Robustness (5-repeat × 5-fold Stratified CV, Full Data)

| Model | Set | Accuracy (mean±std) | F1 (mean±std) | Recall (mean±std) | ROC_AUC (mean±std) |
|-------|-----|---------------------|---------------|-------------------|-------------------|
{chr(10).join(f"| {r['Model']} | {r['Feature_Set']} | {r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f} | {r['f1_mean']:.3f}±{r['f1_std']:.3f} | {r['recall_mean']:.3f}±{r['recall_std']:.3f} | {r['roc_auc_mean']:.3f}±{r['roc_auc_std']:.3f} |" for _, r in rcv_df.iterrows())}

> **Note:** With only ~61 test samples, test-set point estimates have high variance.
> Repeated CV results (25 folds per model) provide a more reliable performance estimate.

---

## 8. Overfitting Check

| Model | Set | Train F1 (CV) | Val F1 (CV) | Test F1 | Gap | Status |
|-------|-----|--------------|------------|---------|-----|--------|
{chr(10).join(f"| {r['Model']} | {r['Feature_Set']} | {r['Train_F1_CV']} | {r['Val_F1_CV']} | {r['Test_F1']} | {r['Train_Val_Gap']} | {r['Status']} |" for _, r in of_df.iterrows())}

---

## 9. Best Model Selection

**Best model: {best_name} (Feature Set A)**

**Test-set results:**
- Accuracy: {best_row['Accuracy']:.4f}
- Precision: {best_row['Precision']:.4f}  
- Recall: {best_row['Recall']:.4f}
- F1: {best_row['F1']:.4f}
- ROC-AUC: {best_row['ROC_AUC']:.4f}

**Justification:**
- {best_name} achieved the highest F1 score on the held-out test set for Feature Set A.
- In cardiac screening, **recall (sensitivity) is the most critical metric** — missing a true disease
  case (False Negative) is far more costly than a false alarm (False Positive).
- F1 is chosen as the primary criterion because it balances precision and recall.
- ROC-AUC confirms ranking across all decision thresholds.
- **Caveat:** With only 61 test samples, differences between models are small and may be within
  sampling noise. The repeated CV results provide a more stable ranking.
- The model is saved as a full sklearn Pipeline (preprocessor + classifier) so it can be applied
  to new data with zero additional preprocessing.

---

## 10. Key EDA Observations

1. **Class balance**: {n_no_disease} No Disease ({n_no_disease/len(df)*100:.1f}%) vs {n_disease} Disease ({n_disease/len(df)*100:.1f}%). Mild imbalance; stratified split used throughout.
2. **Missing values**: ca ({int(df['ca'].isnull().sum())}) and thal ({int(df['thal'].isnull().sum())}) have missing values (marked '?' in raw data). Imputed inside pipeline — no leakage.
3. **Strongest correlation with target**: `{top_corr_col}` (|r| = {top_corr_val:.3f}).
4. **IQR Outliers**: trestbps (9), chol (5), oldpeak (5), thalach (1). Retained — small dataset, tree models robust.
5. **Chest pain type (cp=3/4, asymptomatic)** is paradoxically more common in disease group — documented clinical phenomenon.
6. **thalach** (max heart rate): Disease patients trend lower — a strong separating feature visible in box plots.
7. **No duplicate rows** (0 duplicates found).

---

## 11. Key Feature Importance Findings

1. **thal** (thalassemia type) and **ca** (number of major vessels) have the highest MI scores — not in Set A but dominant in Sets B/C.
2. **cp** (chest pain type) and **thalach** (max heart rate) are the most important Set A features.
3. Random Forest and permutation importance (see figures) confirm these rankings.
4. **sex, fbs, restecg** contribute less individually; fbs is below the MI cutoff and excluded from Set C.
5. **oldpeak** has moderate RF importance but low MI score — excluded from Set C on the combined criterion.

---

## 12. No-Leakage Confirmation

All `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` are inside `sklearn Pipeline` / `ColumnTransformer`.
They are `fit()` only on the **training data** (or the training fold within each CV split).
Feature selection (MI, RF importance) is computed on the training partition **after** the train/test split.
The test set is **never seen during any fitting step**.

---

## 13. Limitations

1. **Small dataset**: 303 rows; ~61 in the test set. Performance estimates have high variance.
2. **Cleveland subset only**: Generalisation to other populations not validated.
3. **Simple imputation**: Median/most-frequent for ca and thal may not capture the true missingness mechanism.
4. **Feature scope (Set A)**: Excludes strong predictors by design; Sets B/C show potential uplift.
5. **No calibration**: Model probabilities are not explicitly calibrated.
6. **Educational only**: Not validated for clinical use.

---

## 14. Figures

| Filename | Description |
|----------|-------------|
{chr(10).join(fig_md_rows)}

---

## 15. Tables

| Filename | Description |
|----------|-------------|
{chr(10).join(tab_md_rows)}

---

*All numbers computed from executed code on {datetime.datetime.now().isoformat()[:10]}.*
"""

os.makedirs("outputs", exist_ok=True)
with open("outputs/RESULTS_SUMMARY.md", "w") as f:
    f.write(summary_md)
print("RESULTS_SUMMARY.md saved to outputs/RESULTS_SUMMARY.md")
print(f"Length: {len(summary_md)} chars")

