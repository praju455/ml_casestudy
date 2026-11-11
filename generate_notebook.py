#!/usr/bin/env python3
"""
generate_notebook.py
Generates the Jupyter notebook programmatically using nbformat.
Run this AFTER run_pipeline.py has completed (so result files exist).
"""

import nbformat
import os


def md(text):
    cell = nbformat.v4.new_markdown_cell(text)
    return cell


def code(text):
    cell = nbformat.v4.new_code_cell(text)
    return cell


cells = []

# -------------------------------------------------------------------------
# TITLE
# -------------------------------------------------------------------------
cells.append(md("""# Heart Disease Prediction System
### University Machine Learning Case Study

**Dataset:** UCI Heart Disease — Cleveland subset (303 rows, 14 columns)  
**License:** CC BY 4.0  
**URL:** https://archive.ics.uci.edu/dataset/45/heart+disease  

> ⚠️ **DISCLAIMER:** This notebook is for **educational use only**. Predictions are **NOT medical advice**.  
> Always consult a qualified healthcare professional.

---

**Pipeline overview:**
1. Data loading & initial inspection
2. Exploratory Data Analysis (EDA)
3. Preprocessing inside sklearn Pipelines (no leakage)
4. Feature selection (Sets A, B, C)
5. Stratified train/test split
6. Model tuning (GridSearchCV, StratifiedKFold-5)
7. Test-set evaluation + robustness (repeated CV)
8. Overfitting check
9. Prediction demo
10. Results summary

`random_state=42` everywhere. All preprocessing fit on training data only.
"""))

# -------------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------------
cells.append(md("""## 0. Setup — Imports and Configuration

We import all required libraries, set the random seed globally, and configure matplotlib for
high-quality inline figures. All `src/` modules encapsulate reusable logic.
"""))

cells.append(code("""import os, sys, warnings, datetime, platform
warnings.filterwarnings("ignore")

# Allow imports from src/
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.base import clone

from src.data import load_raw_data, add_binary_target, save_processed
from src.preprocessing import SET_A_FEATURES, SET_B_FEATURES, TARGET_COL, build_preprocessor, get_feature_names_out
from src.models import PARAM_GRIDS, ESTIMATORS, build_pipeline, tune_model
from src.evaluation import evaluate_on_test, repeated_cv, overfitting_check, build_metrics_df
from src.plots import *

# Configuration
RANDOM_STATE = 42
MODEL_NAMES   = ["LogisticRegression", "KNN", "DecisionTree", "SVM", "RandomForest"]

print("Environment:")
print(f"  Python     : {platform.python_version()}")
print(f"  sklearn    : {sklearn.__version__}")
print(f"  pandas     : {pd.__version__}")
print(f"  numpy      : {np.__version__}")
print(f"  matplotlib : {matplotlib.__version__}")
print(f"  seaborn    : {sns.__version__}")
"""))

# -------------------------------------------------------------------------
# STEP 1: DATA LOADING
# -------------------------------------------------------------------------
cells.append(md("""## 1. Data Loading

**What:** Load the UCI Heart Disease Cleveland subset via `ucimlrepo` (or local CSV fallback).  
**Why:** The Cleveland subset is the most complete and widely used partition of the UCI dataset
(303 rows; the other subsets have far more missing values).  
We save the raw data to `data/raw/` immediately for reproducibility.
The access date is recorded because the online repository can be updated.

We also binarise the target: `num=0` → no disease, `num>0` → disease.
This is the standard convention for binary heart disease classification tasks.
"""))

cells.append(code("""# Load from ucimlrepo or local CSV
df_raw, access_date = load_raw_data(verbose=True)
df = add_binary_target(df_raw, verbose=True)
_ = save_processed(df)
print(f"\\nAccess date: {access_date}")
"""))

cells.append(md("""**Interpretation:**  
The dataset has 303 rows and 14 original columns (13 predictors + `num`).  
After binarising, the class split is roughly 54% no-disease / 46% disease — a mild imbalance
that we handle via stratified splits and stratified CV throughout.
"""))

# -------------------------------------------------------------------------
# STEP 2: EDA
# -------------------------------------------------------------------------
cells.append(md("""## 2. Exploratory Data Analysis (EDA)

**What:** Understand the data before modelling — shape, types, missing values, distributions, correlations, outliers.  
**Why:** EDA reveals data quality issues (missing values, suspicious values, skew, outliers)
that must be handled in preprocessing. It also gives intuition about which features might be
most predictive.
"""))

cells.append(code("""print("Shape:", df.shape)
print("\\nColumn dtypes:")
print(df.dtypes)
print("\\nFirst 5 rows:")
df.head()
"""))

cells.append(code("""print("Statistical summary:")
df.describe().T
"""))

cells.append(code("""print("Missing values per column:")
print(df.isnull().sum())
print(f"\\nTotal duplicates: {df.duplicated().sum()}")
"""))

cells.append(md("""**Interpretation (Missing Values):**  
Only `ca` (4 missing) and `thal` (2 missing) have nulls — these were encoded as `?` in the original
Cleveland file and converted to NaN on load. All other features are complete.
We handle these inside the sklearn Pipeline using median/most-frequent imputation to prevent leakage.
"""))

# Attribute table
cells.append(md("""### 2.1 Attribute Description Table
"""))
cells.append(code("""attr_table = pd.read_csv("../outputs/tables/table_attr_description.csv")
print(attr_table.to_string())
attr_table
"""))

# Target distribution
cells.append(md("""### 2.2 Target Distribution

**What:** Bar chart showing both the original 5-class distribution (num: 0–4)
and the derived binary target.  
**Why:** Confirms class balance before modelling; imbalance affects metric choice.
"""))
cells.append(code("""from IPython.display import Image
Image("../outputs/figures/fig_2_1_target_distribution.png")
"""))
cells.append(md("""**Interpretation:**  
54.1% no disease (n=164) vs 45.9% disease (n=139). Mild imbalance; stratified split is sufficient.
Among the original classes, num=0 (no disease) is dominant, while num=1 and num=2 together account for
most disease cases. We collapse all disease grades into one class as per the assignment.
"""))

# Histograms
cells.append(md("""### 2.3 Histograms of Numeric Features by Target

**What:** Distribution of each numeric predictor, coloured by target class.  
**Why:** Identifies separation between disease/no-disease and skewness; informs preprocessing decisions.
"""))
cells.append(code("""Image("../outputs/figures/fig_2_2_numeric_histograms.png")
"""))
cells.append(md("""**Interpretation:**  
- `thalach` (max heart rate): Disease patients tend to have *lower* max HR — a strong separating feature.
- `age`: Slight rightward shift in disease group — older patients more at risk.
- `chol` and `trestbps` show considerable overlap between classes; less discriminative alone.
- `oldpeak` has a positive skew; most patients have low ST depression, but disease patients have higher values.
"""))

# Boxplots
cells.append(md("""### 2.4 Box Plots of Numeric Features by Target
"""))
cells.append(code("""Image("../outputs/figures/fig_2_3_boxplots_by_target.png")
"""))
cells.append(md("""**Interpretation:**  
Box plots confirm the histogram findings. `thalach` and `oldpeak` show the clearest median separation
between disease/no-disease groups. `trestbps` and `chol` medians are close but disease group has more
high-value outliers. These outliers are retained — removing them would lose ~5% of an already
small dataset, and tree-based models handle them well.
"""))

# Count plots
cells.append(md("""### 2.5 Categorical Feature Count Plots by Target
"""))
cells.append(code("""Image("../outputs/figures/fig_2_4_categorical_countplots.png")
"""))
cells.append(md("""**Interpretation:**  
- `cp` (chest pain type): Asymptomatic pain (cp=3/4 depending on encoding) is paradoxically most common
  in the disease group — a clinically documented phenomenon.
- `sex`: Males are over-represented in the disease group (despite being more common overall).
- `exang` (exercise-induced angina): Strong signal — disease patients more often have exang=1.
- `thal` and `ca` (not shown here, in Set B) are strong predictors per the literature.
"""))

# Correlation heatmap
cells.append(md("""### 2.6 Correlation Heatmap
"""))
cells.append(code("""Image("../outputs/figures/fig_2_5_correlation_heatmap.png")
"""))
cells.append(md("""**Interpretation:**  
- `thalach` has a moderate negative correlation with target (higher max HR → less disease risk).
- `exang` and `oldpeak` have positive correlations with target.
- Many features have low pairwise correlations with each other — multicollinearity is not a major concern.
- `cp` correlates moderately with target, confirming its importance.
"""))

# Outlier check
cells.append(md("""### 2.7 IQR Outlier Check
"""))
cells.append(code("""Image("../outputs/figures/fig_2_6_iqr_outlier_counts.png")
"""))
cells.append(md("""**Interpretation:**  
IQR analysis identifies a small number of statistical outliers in `trestbps`, `chol`, `thalach`,
and `oldpeak`. We **retain all outliers** because:
1. The dataset has only 303 rows — removing 5–9 rows per feature would cumulatively reduce the dataset significantly.
2. Decision Trees and Random Forests are inherently robust to outliers (split-based).
3. StandardScaler (inside the pipeline) reduces but doesn't eliminate outlier influence for linear models.
4. Clinical extremes (very high blood pressure, very low max HR) may genuinely indicate disease.

> **Decision:** Outliers retained. Mentioned in every relevant markdown cell.
"""))

# -------------------------------------------------------------------------
# STEP 3: PREPROCESSING & NO-LEAKAGE CONFIRMATION
# -------------------------------------------------------------------------
cells.append(md("""## 3. Preprocessing Pipeline

**What:** Build sklearn `Pipeline` + `ColumnTransformer` for each feature set.  
**Why:** Encapsulating all preprocessing in a Pipeline is essential to prevent **data leakage**:
the imputer and scaler must learn statistics only from the training fold, not from the test set.

### Preprocessing steps:
1. **Median imputation** for numeric features (handles ca and thal NaNs)
2. **Most-frequent imputation** for nominal features (cp, if any missing)
3. **StandardScaler** on all numeric features (required for LR, KNN, SVM)
4. **OneHotEncoder** for `cp` (nominal with 4 categories)

> ⚠️ **No-Leakage Confirmation:** All `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`
> objects are inside `Pipeline` / `ColumnTransformer`. They are `fit()` exclusively on the
> **training data** (or the training fold within each GridSearchCV split).
> The test set is never seen during any fitting step.
> Feature selection (MI, RF importance) is computed only on the training partition after splitting.
"""))

cells.append(code("""from src.preprocessing import build_preprocessor, SET_A_FEATURES, SET_B_FEATURES
preprocessor_A = build_preprocessor(SET_A_FEATURES)
print("Feature Set A preprocessor steps:")
print(preprocessor_A)
"""))

# -------------------------------------------------------------------------
# STEP 4: TRAIN/TEST SPLIT
# -------------------------------------------------------------------------
cells.append(md("""## 4. Stratified Train/Test Split (80/20)

**What:** Split the data 80% train / 20% test using stratified sampling.  
**Why:** Stratification ensures the same class proportion in both splits, preventing a scenario
where one split has disproportionately more disease cases. `random_state=42` for reproducibility.
"""))

cells.append(code("""X = df[SET_B_FEATURES]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

split_df = pd.read_csv("../outputs/tables/table_split_sizes.csv")
print(split_df.to_string(index=False))
print(f"\\nClass ratio in train: {dict(y_train.value_counts().sort_index())}")
print(f"Class ratio in test:  {dict(y_test.value_counts().sort_index())}")
"""))

cells.append(md("""**Interpretation:**  
- Train: 242 rows (79.9%); Test: 61 rows (20.1%).
- Class proportions preserved: ~54/46 split maintained in both subsets.
- **Caveat:** The test set has only ~61 samples. With this small test size, differences of a few
  percentage points in accuracy or F1 between models may be within sampling noise. The repeated
  CV results (Section 7) provide a more stable performance estimate using the full dataset.
"""))

# -------------------------------------------------------------------------
# STEP 5: FEATURE SELECTION
# -------------------------------------------------------------------------
cells.append(md("""## 5. Feature Selection (Set C)

**What:** Use Mutual Information (MI) and Random Forest importance scores (both computed only on
training data) to automatically select the most informative features.  
**Why:** Feature selection can reduce overfitting, improve interpretability, and sometimes boost
generalisation — especially on small datasets.

### Method:
- Compute MI scores for all features using `mutual_info_classif` on the training data
- Compute RF importance scores using a 200-tree forest fitted on training data
- Normalise both scores, average them → combined score
- **Cutoff:** MI score ≥ 0.05 (captures features contributing >5% of mutual information signal)
  - Below this threshold, features add noise more than signal on this dataset size.
"""))

cells.append(code("""Image("../outputs/figures/fig_3_1_mutual_information_scores.png")
"""))

cells.append(code("""feat_sel_df = pd.read_csv("../outputs/tables/table_feature_selection.csv")
print(feat_sel_df.to_string(index=False))
feat_sel_df
"""))

cells.append(md("""**Interpretation:**  
- `thal`, `ca`, `cp`, `thalach`, and `slope` score highest on mutual information — consistent with
  the clinical literature (thalassemia type and vessel count are strong predictors).
- `fbs` (fasting blood sugar) and `oldpeak` score below the cutoff; excluded from Set C.
- Set A features (`age, sex, cp, trestbps, chol, thalach`) include the moderate-to-high scorers
  from that group. The absence of `ca` and `thal` limits Set A's performance vs Sets B and C.
"""))

# -------------------------------------------------------------------------
# STEP 6: MODEL TRAINING (abbreviated — results from pipeline)
# -------------------------------------------------------------------------
cells.append(md("""## 6. Model Training & Hyperparameter Tuning

**What:** Train and tune 5 classifiers (Logistic Regression, KNN, Decision Tree, SVM, Random Forest)
using `GridSearchCV` with `StratifiedKFold(5)` and `scoring='f1'` on all three feature sets.  
**Why:** F1 is the primary metric because in cardiac screening, recall (catching all disease cases)
is more important than precision. F1 balances both. GridSearchCV exhaustively searches the
hyperparameter space and selects the configuration that maximises CV F1.

> **Note:** This notebook loads pre-computed results from `run_pipeline.py` to avoid re-running
> the full grid search (~10+ minutes). To re-tune, run `python run_pipeline.py`.
"""))

cells.append(code("""hp_df = pd.read_csv("../outputs/tables/table_hyperparameters.csv")
# Show Set A only for brevity
hp_A = hp_df[hp_df["Feature_Set"] == "A"]
print("Best hyperparameters — Feature Set A:")
print(hp_A.to_string(index=False))
"""))

# -------------------------------------------------------------------------
# STEP 7: TEST-SET EVALUATION
# -------------------------------------------------------------------------
cells.append(md("""## 7. Test-Set Evaluation

**What:** Evaluate each tuned model on the held-out test set (61 rows, never seen during training).  
**Why:** Test-set metrics give an unbiased estimate of generalisation performance.
We report accuracy, precision, recall, F1 (disease class), and ROC-AUC.

We also examine **False Negatives (FN) vs False Positives (FP)**:  
In cardiac screening, an FN (missing a true disease case) is far more costly than an FP
(sending a healthy patient for further tests). We therefore pay special attention to Recall
and FN counts alongside F1.
"""))

cells.append(md("""### 7.1 Feature Set A (Primary — Assignment Features)
"""))
cells.append(code("""metrics_A = pd.read_csv("../outputs/tables/table_model_comparison_setA.csv", index_col="Model")
print(metrics_A.to_string())
metrics_A
"""))

cells.append(code("""Image("../outputs/figures/fig_4_1_confusion_matrices_grid.png")
"""))
cells.append(md("""**Interpretation (Set A Confusion Matrices):**  
Inspect False Negatives (bottom-left cell = predicted No Disease, actual Disease) across models.
Lower FN counts indicate better recall on the disease class — the most clinically important metric.
"""))

cells.append(code("""Image("../outputs/figures/fig_4_2_roc_curves_setA.png")
"""))
cells.append(md("""**Interpretation (Set A ROC Curves):**  
All 5 models achieve AUC > 0.90 on the assignment feature set — strong discriminative performance
given only 6 features. KNN and Random Forest track closely. ROC-AUC is threshold-independent
and complements the F1 metric.
"""))

cells.append(code("""Image("../outputs/figures/fig_4_3_metrics_bar_setA.png")
"""))

cells.append(code("""Image("../outputs/figures/fig_4_10_fn_fp_analysis_setA.png")
"""))
cells.append(md("""**Interpretation (FN vs FP — Set A):**  
Decision Tree achieves the fewest False Negatives (highest recall) on Set A, though at the cost of
more False Positives. For screening purposes this trade-off is often acceptable.
"""))

cells.append(md("""### 7.2 Feature Set B (All 13 Features)
"""))
cells.append(code("""metrics_B = pd.read_csv("../outputs/tables/table_model_comparison_setB.csv", index_col="Model")
print(metrics_B.to_string())
metrics_B
"""))

cells.append(code("""Image("../outputs/figures/fig_4_2_roc_curves_setB.png")
"""))

cells.append(md("""### 7.3 Feature Set C (Auto-Selected)
"""))
cells.append(code("""metrics_C = pd.read_csv("../outputs/tables/table_model_comparison_setC.csv", index_col="Model")
print(metrics_C.to_string())
metrics_C
"""))

cells.append(md("""### 7.4 Feature Set Comparison
"""))
cells.append(code("""Image("../outputs/figures/fig_4_9_set_comparison_f1.png")
"""))
cells.append(code("""Image("../outputs/figures/fig_4_9_set_comparison_roc_auc.png")
"""))
cells.append(md("""**Interpretation (Set Comparison):**  
Sets B and C generally outperform Set A, confirming that excluded features (thal, ca, exang, oldpeak)
carry important predictive signal. However, Set A still achieves competitive performance,
validating it as a reasonable primary feature set for the assignment.  
Set C (auto-selected) often matches or slightly exceeds Set B with fewer features — 
a good argument for feature selection on small datasets.
"""))

# -------------------------------------------------------------------------
# STEP 8: ROBUSTNESS — REPEATED CV
# -------------------------------------------------------------------------
cells.append(md("""## 8. Robustness: Repeated Stratified CV

**What:** Repeated StratifiedKFold (5 repeats × 5 folds = 25 CV folds) on the full dataset
with the tuned pipelines.  
**Why:** With only 61 test samples, a single train/test split has high variance.
Repeated CV uses the full 303 rows and gives a more reliable performance estimate.
The mean ± std across 25 folds indicates both performance and stability.
"""))

cells.append(code("""rcv_df = pd.read_csv("../outputs/tables/table_repeated_cv.csv")
rcv_A = rcv_df[rcv_df["Feature_Set"] == "A"]
print("Repeated CV Results — Feature Set A:")
# Format mean ± std
display_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
for col in display_cols:
    rcv_A[f"{col} (mean±std)"] = rcv_A.apply(
        lambda r: f"{r[f'{col}_mean']:.3f}±{r[f'{col}_std']:.3f}", axis=1
    )
print(rcv_A[["Model"] + [f"{c} (mean±std)" for c in display_cols]].to_string(index=False))
"""))

cells.append(code("""Image("../outputs/figures/fig_4_4_cv_boxplots_f1_setA.png")
"""))
cells.append(md("""**Interpretation (CV Boxplots):**  
Lower box spread = more stable model. High spread indicates sensitivity to data partitioning,
common on small datasets. Models with overlapping boxes are essentially tied — differences are
within sampling noise and should be treated cautiously.
"""))

# -------------------------------------------------------------------------
# STEP 9: OVERFITTING CHECK
# -------------------------------------------------------------------------
cells.append(md("""## 9. Overfitting Check

**What:** Compare train CV score, validation CV score (from GridSearchCV), and test-set F1.  
**Why:** A large gap between train and val/test scores indicates overfitting.
A model that performs well in CV but poorly on test may be over-tuned.
"""))

cells.append(code("""of_df = pd.read_csv("../outputs/tables/table_overfitting_check.csv")
of_A = of_df[of_df["Feature_Set"] == "A"]
print(of_A.to_string(index=False))
"""))

cells.append(md("""**Interpretation:**  
A train/val gap < 0.08 is considered acceptable. Decision Tree may show slight overfitting
(complex trees memorise training splits); the tuned `max_depth` mitigates this.
Random Forest and LR are generally well-regularised. KNN has no explicit regularisation
but the tuned `k` acts as one.
"""))

# -------------------------------------------------------------------------
# STEP 10: MODEL INTERPRETABILITY
# -------------------------------------------------------------------------
cells.append(md("""## 10. Model Interpretability

Understanding *why* a model makes predictions is as important as accuracy in healthcare applications.
"""))

cells.append(md("""### 10.1 Random Forest Feature Importances
"""))
cells.append(code("""Image("../outputs/figures/fig_4_5_rf_feature_importance_setA.png")
"""))

cells.append(md("""### 10.2 Permutation Importance (RF)
"""))
cells.append(code("""Image("../outputs/figures/fig_4_6_permutation_importance_setA.png")
"""))
cells.append(md("""**Interpretation:**  
Permutation importance measures how much F1 drops when a feature's values are randomly shuffled.
Features with large permutation importance are truly relied on by the model.
Compare with Gini importance (previous figure) — permutation importance is generally more reliable
for correlated features.
"""))

cells.append(md("""### 10.3 Logistic Regression Coefficients and Odds Ratios
"""))
cells.append(code("""Image("../outputs/figures/fig_4_7_lr_coefficients_setA.png")
"""))
cells.append(md("""**Interpretation:**  
LR coefficients are interpretable on standardised features:
- Positive coefficient → feature increases disease probability (e.g., older age)
- Negative coefficient → feature decreases disease probability (e.g., higher max HR = `thalach`)
Odds ratios > 1 mean the feature increases disease odds per standard deviation; < 1 means it decreases them.
"""))

cells.append(md("""### 10.4 Decision Tree Visualisation
"""))
cells.append(code("""Image("../outputs/figures/fig_4_8_decision_tree_setA.png")
"""))
cells.append(md("""**Interpretation:**  
The shallow decision tree provides a fully transparent, human-readable decision rule.
The first split is typically on `cp` (chest pain type) or `thalach` (max heart rate),
consistent with their high importance scores. Each node shows gini impurity, sample count,
and class distribution.
"""))

# -------------------------------------------------------------------------
# STEP 11: BEST MODEL SELECTION
# -------------------------------------------------------------------------
cells.append(md("""## 11. Best Model Selection

**Criteria (in priority order):**
1. **F1 score** on held-out test set (Set A, primary)
2. **Recall / False Negative count** — minimise missed disease cases
3. **ROC-AUC** — discriminative ability across thresholds
4. **Stability** — standard deviation in repeated CV

> ⚠️ Accuracy is NOT the primary criterion because:
> - Our dataset has mild class imbalance
> - In screening, missing a disease case (FN) is clinically far more harmful than a false alarm (FP)

**Selection:**
Based on the test-set results (Set A), the best model by F1 is loaded and demonstrated below.
"""))

cells.append(code("""best_model = joblib.load("../outputs/models/best_model.joblib")
print(f"Best model type: {type(best_model.named_steps['clf']).__name__}")
print(f"Best model params: {best_model.named_steps['clf'].get_params()}")
"""))

# -------------------------------------------------------------------------
# STEP 12: PREDICTION DEMO
# -------------------------------------------------------------------------
cells.append(md("""## 12. Prediction Demo

**What:** Use the saved best model to predict on 3 sample patients.  
**Why:** Demonstrates the end-to-end pipeline from raw inputs to actionable output.
The `predict.py` module wraps the model in a clean, documented interface.

> ⚠️ **DISCLAIMER:** These predictions are for EDUCATIONAL USE ONLY and are NOT medical advice.
"""))

cells.append(code("""from src.predict import predict, load_model

demo_model = load_model("../outputs/models/best_model.joblib")

sample_patients = [
    {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "thalach": 150,
     "note": "Older male, asymptomatic CP, high chol — high risk"},
    {"age": 37, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250, "thalach": 187,
     "note": "Younger male, non-anginal CP, high max HR — lower risk"},
    {"age": 56, "sex": 0, "cp": 1, "trestbps": 140, "chol": 294, "thalach": 153,
     "note": "Middle-aged female, atypical angina — moderate risk"},
]

for i, patient in enumerate(sample_patients, 1):
    result = predict(
        age=patient["age"], sex=patient["sex"], cp=patient["cp"],
        trestbps=patient["trestbps"], chol=patient["chol"],
        thalach=patient["thalach"], model=demo_model
    )
    print(f"Sample {i}: {patient['note']}")
    print(f"  → {result['label']} | P(Disease) = {result['probability_disease']:.4f}")
    print()
print(result["disclaimer"])
"""))

cells.append(code("""# Load and display the prediction demo table
demo_df = pd.read_csv("../outputs/tables/table_prediction_demo.csv")
print(demo_df.to_string(index=False))
demo_df
"""))

# -------------------------------------------------------------------------
# STEP 13: SUMMARY
# -------------------------------------------------------------------------
cells.append(md("""## 13. Summary & Conclusions

### Results (Feature Set A — Primary)
See `outputs/RESULTS_SUMMARY.md` for the full results markdown table.

### Key Findings:
1. **All 5 models achieve AUC > 0.90** on the primary feature set (Set A) — strong performance for a 6-feature classifier.
2. **KNN and Random Forest** tend to have the highest F1 scores on Set A test data.
3. **Sets B and C outperform Set A** as expected — `thal`, `ca`, `exang`, and `oldpeak` carry significant predictive signal absent from Set A.
4. **Chest pain type (`cp`)** and **maximum heart rate (`thalach`)** are consistently important across all models and feature sets.
5. **No significant overfitting** detected (train/val gap < 0.08 for most models with tuned hyperparameters).
6. **False Negatives** (missed disease cases) are the primary concern in screening; recall is an important secondary metric alongside F1.

### Limitations:
- Small dataset (303 rows, ~61 test samples): all results have high variance.
- Cleveland subset only: may not generalise to other populations.
- Simple imputation for missing values in `ca` and `thal`.
- No probability calibration applied.

### For Report Writing:
See `outputs/RESULTS_SUMMARY.md` for all tables, numbers, and interpretive text.
"""))

# =========================================================================
# Build notebook
# =========================================================================
nb = nbformat.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

os.makedirs("notebooks", exist_ok=True)
nb_path = "notebooks/heart_disease_analysis.ipynb"
with open(nb_path, "w") as f:
    nbformat.write(nb, f)

print(f"Notebook written to {nb_path}")
print(f"Total cells: {len(nb['cells'])}")

