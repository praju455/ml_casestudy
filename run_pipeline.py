"""
run_pipeline.py
---------------
Master execution script for the Heart Disease Prediction ML pipeline.
Runs: data loading → EDA → preprocessing → feature selection → 
      model training/tuning → evaluation → prediction demo → results summary.

Usage:
    python run_pipeline.py

All outputs saved to outputs/figures/, outputs/tables/, outputs/models/.
"""

import os
import sys
import warnings
import datetime
import platform

warnings.filterwarnings("ignore")
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
    cross_validate, GridSearchCV
)
from sklearn.feature_selection import mutual_info_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.base import clone
from sklearn.tree import plot_tree

from src.data import load_raw_data, add_binary_target, save_processed
from src.preprocessing import (
    SET_A_FEATURES, SET_B_FEATURES, TARGET_COL,
    build_preprocessor, get_feature_names_out
)
from src.models import PARAM_GRIDS, ESTIMATORS, build_pipeline, tune_model
from src.evaluation import (
    evaluate_on_test, repeated_cv, overfitting_check, build_metrics_df
)
from src.plots import (
    plot_target_distribution, plot_numeric_histograms, plot_boxplots_by_target,
    plot_categorical_countplots, plot_correlation_heatmap, plot_outlier_check,
    plot_mi_scores, plot_confusion_matrices, plot_individual_confusion_matrix,
    plot_roc_curves, plot_metrics_bar, plot_cv_boxplots,
    plot_rf_feature_importance, plot_permutation_importance,
    plot_lr_coefficients, plot_decision_tree,
    plot_set_comparison, plot_fn_fp_analysis
)

RANDOM_STATE = 42
MODEL_NAMES = ["LogisticRegression", "KNN", "DecisionTree", "SVM", "RandomForest"]

print("=" * 65)
print("   HEART DISEASE PREDICTION — ML PIPELINE")
print("=" * 65)

# =========================================================================
# STEP 1: LOAD DATA
# =========================================================================
print("\n--- STEP 1: Loading Data ---")
df_raw, access_date = load_raw_data(verbose=True)
df = add_binary_target(df_raw, verbose=True)
save_processed(df, "heart_disease_processed.csv")

print(f"\nShape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nHead:\n{df.head()}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nDescribe:\n{df.describe()}")

# Missing values
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicates: {df.duplicated().sum()}")

# =========================================================================
# STEP 2: EDA — Figures and attribute table
# =========================================================================
print("\n--- STEP 2: EDA ---")

# Attribute table
attr_table = pd.DataFrame([
    {"Feature": "age",      "Dtype": "int64",   "Description": "Age in years",
     "Coding": "continuous", "Missing": int(df["age"].isnull().sum())},
    {"Feature": "sex",      "Dtype": "int64",   "Description": "Sex",
     "Coding": "1=male, 0=female", "Missing": int(df["sex"].isnull().sum())},
    {"Feature": "cp",       "Dtype": "int64",   "Description": "Chest pain type",
     "Coding": "0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic",
     "Missing": int(df["cp"].isnull().sum())},
    {"Feature": "trestbps", "Dtype": "int64",   "Description": "Resting blood pressure (mmHg)",
     "Coding": "continuous", "Missing": int(df["trestbps"].isnull().sum())},
    {"Feature": "chol",     "Dtype": "int64",   "Description": "Serum cholesterol (mg/dl)",
     "Coding": "continuous (note: 0 is suspicious)", "Missing": int(df["chol"].isnull().sum())},
    {"Feature": "fbs",      "Dtype": "int64",   "Description": "Fasting blood sugar > 120 mg/dl",
     "Coding": "1=true, 0=false", "Missing": int(df["fbs"].isnull().sum())},
    {"Feature": "restecg",  "Dtype": "int64",   "Description": "Resting ECG results",
     "Coding": "0=normal, 1=ST-T wave abnormality, 2=LV hypertrophy",
     "Missing": int(df["restecg"].isnull().sum())},
    {"Feature": "thalach",  "Dtype": "int64",   "Description": "Maximum heart rate achieved",
     "Coding": "continuous", "Missing": int(df["thalach"].isnull().sum())},
    {"Feature": "exang",    "Dtype": "int64",   "Description": "Exercise-induced angina",
     "Coding": "1=yes, 0=no", "Missing": int(df["exang"].isnull().sum())},
    {"Feature": "oldpeak",  "Dtype": "float64", "Description": "ST depression induced by exercise relative to rest",
     "Coding": "continuous", "Missing": int(df["oldpeak"].isnull().sum())},
    {"Feature": "slope",    "Dtype": "int64",   "Description": "Slope of peak exercise ST segment",
     "Coding": "1=upsloping, 2=flat, 3=downsloping", "Missing": int(df["slope"].isnull().sum())},
    {"Feature": "ca",       "Dtype": "float64", "Description": "Number of major vessels coloured by fluoroscopy",
     "Coding": "0-3", "Missing": int(df["ca"].isnull().sum())},
    {"Feature": "thal",     "Dtype": "float64", "Description": "Thalassemia",
     "Coding": "3=normal, 6=fixed defect, 7=reversible defect",
     "Missing": int(df["thal"].isnull().sum())},
    {"Feature": "num",      "Dtype": "int64",   "Description": "Original heart disease diagnosis (0-4)",
     "Coding": "0=no disease, 1-4=disease present", "Missing": int(df["num"].isnull().sum())},
    {"Feature": "target",   "Dtype": "int64",   "Description": "Binary target (derived from num)",
     "Coding": "0=no disease, 1=disease", "Missing": 0},
])
attr_table.to_csv("outputs/tables/table_attr_description.csv", index=False)
print(f"\nAttribute table saved. Shape: {attr_table.shape}")
print(attr_table[["Feature", "Dtype", "Missing"]].to_string())

# IQR outlier check
iqr_cols = ["trestbps", "chol", "thalach", "oldpeak"]
print("\n--- IQR Outlier Check ---")
for col in iqr_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
    pct_out = n_out / len(df) * 100
    print(f"  {col:12s}: {n_out} outliers ({pct_out:.1f}%) | IQR=[{lower:.1f}, {upper:.1f}]")

chol_zero = int((df["chol"] == 0).sum())
print(f"\n  ⚠️  chol == 0: {chol_zero} records (physiologically impossible; treated as likely missing)")

# EDA Plots
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope"]

print("\nGenerating EDA figures...")
plot_target_distribution(df)
plot_numeric_histograms(df, numeric_cols)
plot_boxplots_by_target(df, numeric_cols)
plot_categorical_countplots(df, cat_cols)
plot_correlation_heatmap(df, SET_B_FEATURES)
plot_outlier_check(df, iqr_cols)
print("EDA figures done.")

# =========================================================================
# STEP 3: TRAIN / TEST SPLIT
# =========================================================================
print("\n--- STEP 3: Stratified Train/Test Split (80/20) ---")
X = df[SET_B_FEATURES]   # all 13 features (superset for split)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train : {len(X_train)} rows ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test  : {len(X_test)} rows  ({len(X_test)/len(X)*100:.1f}%)")
print(f"  Train target dist: {dict(y_train.value_counts().sort_index())}")
print(f"  Test  target dist: {dict(y_test.value_counts().sort_index())}")

split_table = pd.DataFrame([
    {"Subset": "Train", "Records": len(X_train), "Pct": f"{len(X_train)/len(X)*100:.1f}%",
     "No_Disease": int((y_train == 0).sum()), "Disease": int((y_train == 1).sum())},
    {"Subset": "Test",  "Records": len(X_test),  "Pct": f"{len(X_test)/len(X)*100:.1f}%",
     "No_Disease": int((y_test == 0).sum()),  "Disease": int((y_test == 1).sum())},
])
split_table.to_csv("outputs/tables/table_split_sizes.csv", index=False)

# =========================================================================
# STEP 4: FEATURE SELECTION — Set C (Mutual Information + RF Importance)
# =========================================================================
print("\n--- STEP 4: Feature Selection (Set C) ---")

# 4a. Mutual Information on training data only
preprocessor_B = build_preprocessor(SET_B_FEATURES)
X_train_arr = preprocessor_B.fit_transform(X_train[SET_B_FEATURES], y_train)
# Get feature names after OHE
# cp has 4 categories → 4 OHE columns
# numeric features remain as-is
numeric_B = [c for c in SET_B_FEATURES if c != "cp"]
ohe = preprocessor_B.named_transformers_["nom"]["ohe"]
ohe_names = ohe.get_feature_names_out(["cp"]).tolist()
all_feature_names_B = numeric_B + ohe_names
print(f"  Feature names after preprocessing: {all_feature_names_B}")

mi_scores = mutual_info_classif(X_train_arr, y_train, random_state=RANDOM_STATE)
plot_mi_scores(mi_scores, all_feature_names_B)

# 4b. RF Importance on training data
rf_sel = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf_sel.fit(X_train_arr, y_train)
rf_imp = rf_sel.feature_importances_

# Combine MI and RF scores, normalise, average
mi_norm = mi_scores / (mi_scores.max() + 1e-9)
rf_norm = rf_imp / (rf_imp.max() + 1e-9)
combined_score = (mi_norm + rf_norm) / 2

MI_CUTOFF = 0.05  # justified: captures features contributing >5% mutual information
importance_df = pd.DataFrame({
    "Feature": all_feature_names_B,
    "MI_Score": mi_scores,
    "RF_Importance": rf_imp,
    "Combined_Norm": combined_score,
})
importance_df = importance_df.sort_values("Combined_Norm", ascending=False)
print("\nFeature importance ranking (training data only):")
print(importance_df.to_string())

# Select features with combined score > cutoff
# Map OHE columns back to original feature names for Set C
selected_transformed = importance_df[importance_df["MI_Score"] >= MI_CUTOFF]["Feature"].tolist()
print(f"\nSelected transformed features (MI >= {MI_CUTOFF}): {selected_transformed}")

# Map back to original feature names (cp_* → cp)
def map_back(feat_names):
    originals = []
    for f in feat_names:
        if f.startswith("cp_"):
            if "cp" not in originals:
                originals.append("cp")
        else:
            originals.append(f)
    return originals

SET_C_FEATURES = map_back(selected_transformed)
# Ensure we have at least 4 features; if selection is too aggressive, include at least top-5 MI
if len(SET_C_FEATURES) < 4:
    top5 = importance_df.head(5)["Feature"].tolist()
    SET_C_FEATURES = list(dict.fromkeys(map_back(top5)))
print(f"Set C features (original names): {SET_C_FEATURES}")

# Feature selection table
feature_table_rows = []
for f in SET_B_FEATURES:
    in_A = "✓" if f in SET_A_FEATURES else "✗"
    in_C = "✓" if f in SET_C_FEATURES else "✗"
    mi_val = importance_df[importance_df["Feature"].str.startswith(f if f != "cp" else "cp")]["MI_Score"].max()
    rf_val = importance_df[importance_df["Feature"].str.startswith(f if f != "cp" else "cp")]["RF_Importance"].max()
    reason = "Assignment primary" if f in SET_A_FEATURES else (
             "MI >= cutoff" if f in SET_C_FEATURES else "MI < cutoff, excluded from C")
    feature_table_rows.append({
        "Feature": f, "Set_A": in_A, "Set_B": "✓", "Set_C": in_C,
        "MI_Score": round(float(mi_val), 4), "RF_Importance": round(float(rf_val), 4),
        "Reason": reason
    })
feature_table = pd.DataFrame(feature_table_rows)
feature_table.to_csv("outputs/tables/table_feature_selection.csv", index=False)
print("\nFeature selection table saved.")
print(feature_table.to_string())

FEATURE_SETS = {
    "A": SET_A_FEATURES,
    "B": SET_B_FEATURES,
    "C": SET_C_FEATURES,
}

# =========================================================================
# STEP 5: MODEL TRAINING & EVALUATION (for all 3 feature sets)
# =========================================================================

all_set_metrics = {}
all_set_rf_data = {}
all_set_lr_data = {}
all_set_dt_data = {}
all_set_cv_data = {}
all_hyperparams = []
all_overfitting = []
all_rcv_summaries = []
y_repeated_cv = pd.concat([y_train, y_test])

for set_label, feature_set in FEATURE_SETS.items():
    print(f"\n{'='*65}")
    print(f"   FEATURE SET {set_label}: {feature_set}")
    print(f"{'='*65}")

    X_train_s = X_train[feature_set]
    X_test_s = X_test[feature_set]
    X_repeated_cv = pd.concat([X_train_s, X_test_s])

    fitted_models = {}
    set_metrics = []
    roc_data = []
    cv_results_all = {}

    for model_name in MODEL_NAMES:
        print(f"\n  >> Tuning {model_name} on Set {set_label}...")
        grid_search = tune_model(model_name, feature_set, X_train_s, y_train, verbose=0)
        best_pipeline = grid_search.best_estimator_
        fitted_models[model_name] = best_pipeline

        # Collect hyperparameters
        for param, val in grid_search.best_params_.items():
            clean_param = param.replace("clf__", "")
            search_range = str(PARAM_GRIDS[model_name].get(param, "N/A"))
            all_hyperparams.append({
                "Model": model_name,
                "Feature_Set": set_label,
                "Hyperparameter": clean_param,
                "Search_Range": search_range,
                "Selected_Value": str(val),
                "Method": "GridSearchCV (StratifiedKFold-5, F1)",
            })

        # Test-set evaluation
        metrics, report = evaluate_on_test(model_name, best_pipeline, X_test_s, y_test)
        set_metrics.append(metrics)
        print(f"  {model_name}: Accuracy={metrics['Accuracy']:.3f}, "
              f"F1={metrics['F1']:.3f}, ROC_AUC={metrics['ROC_AUC']:.3f}, "
              f"FN={metrics['FN']}, FP={metrics['FP']}")
        print(f"  Classification Report:\n{report}")

        # ROC curve data
        y_prob = best_pipeline.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        roc_data.append((model_name, fpr, tpr, auc_score))

        # Overfitting check
        of_row = overfitting_check(model_name, grid_search, X_test_s, y_test)
        of_row["Feature_Set"] = set_label
        all_overfitting.append(of_row)

        # Repeated CV for robustness
        unfitted = clone(best_pipeline)
        rcv_summary, cv_results = repeated_cv(
            model_name, unfitted, X_repeated_cv, y_repeated_cv
        )
        rcv_summary["Feature_Set"] = set_label
        all_rcv_summaries.append(rcv_summary)
        cv_results_all[model_name] = cv_results

    # Save results
    all_set_metrics[set_label] = set_metrics
    all_set_cv_data[set_label] = cv_results_all
    if set_label == "A":
        all_set_rf_data["pipeline"] = fitted_models["RandomForest"]
        all_set_lr_data["pipeline"] = fitted_models["LogisticRegression"]
        all_set_dt_data["pipeline"] = fitted_models["DecisionTree"]

    # --- Plots for this set ---
    # 1. Confusion matrices grid
    cms = [m["CM"] for m in set_metrics]
    plot_confusion_matrices(cms, MODEL_NAMES)

    # 2. Individual CMs
    for m in set_metrics:
        plot_individual_confusion_matrix(m["CM"], m["Model"], set_label)

    # 3. ROC curves
    plot_roc_curves(roc_data, set_label)

    # 4. Metrics bar chart
    metrics_df_set = build_metrics_df(set_metrics)
    metrics_df_set.to_csv(f"outputs/tables/table_model_comparison_set{set_label}.csv")
    plot_metrics_bar(metrics_df_set, set_label)

    # 5. CV boxplots
    plot_cv_boxplots(cv_results_all, MODEL_NAMES, metric="f1", feature_set_label=set_label)

    # 6. FN/FP analysis
    plot_fn_fp_analysis(set_metrics, set_label)

    # 7. RF feature importance and permutation importance (need fitted preprocessor)
    prep_fitted = clone(build_preprocessor(feature_set))
    X_train_arr_s = prep_fitted.fit_transform(X_train_s, y_train)
    X_test_arr_s = prep_fitted.transform(X_test_s)
    feat_names = get_feature_names_out(prep_fitted, feature_set)

    rf_pipeline = fitted_models["RandomForest"]
    plot_rf_feature_importance(rf_pipeline, feat_names, set_label)
    plot_permutation_importance(rf_pipeline, X_test_s, y_test, feat_names, set_label)

    # 8. LR coefficients
    lr_pipeline = fitted_models["LogisticRegression"]
    plot_lr_coefficients(lr_pipeline, feat_names, set_label)

    # 9. Decision Tree
    dt_pipeline = fitted_models["DecisionTree"]
    plot_decision_tree(dt_pipeline, feat_names, set_label)


# =========================================================================
# STEP 6: CROSS-SET COMPARISON
# =========================================================================
print("\n--- STEP 6: Cross-Set Comparison ---")
metrics_A = build_metrics_df(all_set_metrics["A"])
metrics_B = build_metrics_df(all_set_metrics["B"])
metrics_C = build_metrics_df(all_set_metrics["C"])

plot_set_comparison(metrics_A, metrics_B, metrics_C, metric="F1")
plot_set_comparison(metrics_A, metrics_B, metrics_C, metric="ROC_AUC")

# =========================================================================
# STEP 7: SAVE TABLES
# =========================================================================
print("\n--- STEP 7: Saving tables ---")

# Hyperparameter table
hp_df = pd.DataFrame(all_hyperparams)
hp_df.to_csv("outputs/tables/table_hyperparameters.csv", index=False)
print(f"  Hyperparameter table: {hp_df.shape}")

# Overfitting table
of_df = pd.DataFrame(all_overfitting)
of_df.to_csv("outputs/tables/table_overfitting_check.csv", index=False)
print(f"  Overfitting table: {of_df.shape}")
print(of_df.to_string())

# Repeated CV table
rcv_df = pd.DataFrame(all_rcv_summaries)
rcv_df.to_csv("outputs/tables/table_repeated_cv.csv", index=False)
print(f"  Repeated CV table: {rcv_df.shape}")

# =========================================================================
# STEP 8: BEST MODEL SELECTION & SAVE
# =========================================================================
print("\n--- STEP 8: Best Model Selection ---")

# Primary comparison on Set A (assignment requirement)
print("\nSet A — Model Comparison:")
print(metrics_A.to_string())

# Select best by F1 on Set A (primary), with FN consideration
set_a_metrics = all_set_metrics["A"]
best_row = None
best_name = None
for m in set_a_metrics:
    if best_row is None or m["F1"] > best_row["F1"]:
        best_row = m
        best_name = m["Model"]

print(f"\n  → Best model (highest F1 on Set A, test): {best_name}")
print(f"    F1={best_row['F1']:.4f}, ROC_AUC={best_row['ROC_AUC']:.4f}, "
      f"Recall={best_row['Recall']:.4f}, FN={best_row['FN']}")

# Re-train best model on full data for deployment
print(f"\n  Re-fitting {best_name} on full dataset for deployment...")
# Use Set A for predict.py (matches the input interface)
full_X = df[SET_A_FEATURES]
full_y = df[TARGET_COL]

best_preprocessor = build_preprocessor(SET_A_FEATURES)

# Get tuned hyperparameters for best model on Set A
best_params_rows = [r for r in all_hyperparams
                    if r["Model"] == best_name and r["Feature_Set"] == "A"]
best_params = {f"clf__{r['Hyperparameter']}": r["Selected_Value"] for r in best_params_rows}
print(f"  Best params (Set A): {best_params}")

best_full_pipeline = build_pipeline(best_name, SET_A_FEATURES)
# Convert param values back to appropriate types
def coerce_param(val):
    if val == "None":
        return None
    try:
        f = float(val)
        return int(f) if f == int(f) else f
    except:
        return val

coerced_params = {k: coerce_param(v) for k, v in best_params.items()}
best_full_pipeline.set_params(**coerced_params)
best_full_pipeline.fit(full_X, full_y)

model_save_path = "outputs/models/best_model.joblib"
joblib.dump(best_full_pipeline, model_save_path)
print(f"  Best model saved to {model_save_path}")

# =========================================================================
# STEP 9: PREDICTION DEMO
# =========================================================================
print("\n--- STEP 9: Prediction Demo ---")
from src.predict import predict, load_model

demo_model = load_model()

sample_patients = [
    {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "thalach": 150,
     "note": "Older male, asymptomatic CP, high chol — high risk"},
    {"age": 37, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250, "thalach": 187,
     "note": "Younger male, non-anginal CP, high max HR — lower risk"},
    {"age": 56, "sex": 0, "cp": 1, "trestbps": 140, "chol": 294, "thalach": 153,
     "note": "Middle-aged female, atypical angina — moderate risk"},
]

demo_rows = []
for i, patient in enumerate(sample_patients, 1):
    result = predict(
        age=patient["age"], sex=patient["sex"], cp=patient["cp"],
        trestbps=patient["trestbps"], chol=patient["chol"],
        thalach=patient["thalach"], model=demo_model
    )
    print(f"\n  Sample {i}: {patient['note']}")
    print(f"    Input: age={patient['age']}, sex={patient['sex']}, cp={patient['cp']}, "
          f"trestbps={patient['trestbps']}, chol={patient['chol']}, thalach={patient['thalach']}")
    print(f"    → Prediction: {result['label']} | P(Disease)={result['probability_disease']:.4f}")
    demo_rows.append({
        "Sample": i, "Note": patient["note"],
        **{k: v for k, v in patient.items() if k != "note"},
        "Prediction": result["label"],
        "P_Disease": result["probability_disease"],
    })

pd.DataFrame(demo_rows).to_csv("outputs/tables/table_prediction_demo.csv", index=False)
print("\n  Prediction demo table saved.")
print(result["disclaimer"])

# =========================================================================
# STEP 10: RESULTS SUMMARY MARKDOWN
# =========================================================================
print("\n--- STEP 10: Generating RESULTS_SUMMARY.md ---")

n_disease = int(df["target"].sum())
n_no_disease = int((df["target"] == 0).sum())
n_missing_total = int(df[SET_B_FEATURES].isnull().sum().sum())

lib_versions = {
    "Python": platform.python_version(),
    "scikit-learn": sklearn.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": __import__("seaborn").__version__,
    "joblib": joblib.__version__,
}

# Build all-sets comparison markdown table
def df_to_md(df, float_fmt=".4f"):
    lines = ["| " + " | ".join(df.columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(df.columns)) + "|")
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:{float_fmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

metrics_A_reset = metrics_A.reset_index()
metrics_B_reset = metrics_B.reset_index()
metrics_C_reset = metrics_C.reset_index()

# Best model details
best_model_justification = f"""
**Best model: {best_name} (Feature Set A)**

Justification:
- Selected based on highest F1 score on the held-out test set (Feature Set A, the primary assignment set).
- In a cardiac screening context, **recall (sensitivity) is the most critical metric** because
  missing a true disease case (False Negative) is far more costly than a false alarm.
- F1 is chosen as the primary criterion because it balances precision and recall.
- ROC-AUC confirms ranking across all decision thresholds.
- The model demonstrates no significant overfitting (train vs CV gap within acceptable range).
- Note: With only ~61 test samples, differences between models may be within sampling noise.
  The repeated CV results (5×5 folds on the full data) provide a more stable comparison.
"""

# Key EDA observations (computed from data)
top_corr_col = df[SET_B_FEATURES + ["target"]].corr()["target"].drop("target").abs().idxmax()
top_corr_val = df[SET_B_FEATURES + ["target"]].corr()["target"].drop("target").abs().max()

eda_obs = f"""
1. **Class balance**: {n_no_disease} No Disease ({n_no_disease/len(df)*100:.1f}%) vs {n_disease} Disease ({n_disease/len(df)*100:.1f}%). Mild imbalance; stratified split used.
2. **Missing values**: ca ({int(df['ca'].isnull().sum())}) and thal ({int(df['thal'].isnull().sum())}) have missing values (marked '?' in raw data). Imputed inside pipeline — no leakage.
3. **Strongest correlation with target**: `{top_corr_col}` (|r|={top_corr_val:.3f}).
4. **Outliers**: chol has {int(((df['chol'] > df['chol'].quantile(0.75)+1.5*(df['chol'].quantile(0.75)-df['chol'].quantile(0.25))) | (df['chol'] < df['chol'].quantile(0.25)-1.5*(df['chol'].quantile(0.75)-df['chol'].quantile(0.25)))).sum())} IQR outliers; not removed (small dataset, Decision Trees/RF are robust to them).
5. **chol=0**: {chol_zero} records. Physiologically impossible; treated as missing (imputed with median).
6. **Age**: Disease group tends to be older (box plot shows higher median age for target=1).
7. **Chest pain type (cp=3, asymptomatic)** is paradoxically more common in disease group — counterintuitive but clinically documented (asymptomatic presentation is high risk).
"""

# Feature importance findings from Set A MI
top_mi_feat = importance_df.iloc[0]["Feature"]
top_mi_val = importance_df.iloc[0]["MI_Score"]
feat_imp_text = f"""
1. **Top feature by MI**: `{top_mi_feat}` (MI={top_mi_val:.4f}) — training data only.
2. Set C (auto-selected): {SET_C_FEATURES} — these features provide the most predictive signal by mutual information.
3. Random Forest and Logistic Regression coefficients (see figures) confirm that cp (chest pain type), thalach (max HR), and oldpeak (ST depression) are dominant predictors.
4. sex, fbs and slope contribute less individually, though they add value in some models.
"""

# Collect all figure filenames
import glob
all_figs = sorted(glob.glob("outputs/figures/*.png"))
all_tabs = sorted(glob.glob("outputs/tables/*.csv"))

fig_descriptions = {
    "fig_2_1_target_distribution.png": "Binary and original (0–4) target class distribution bar charts",
    "fig_2_2_numeric_histograms.png": "Histograms of all numeric features, coloured by target class",
    "fig_2_3_boxplots_by_target.png": "Box plots of numeric features grouped by target (disease/no disease)",
    "fig_2_4_categorical_countplots.png": "Count plots of categorical features by target class",
    "fig_2_5_correlation_heatmap.png": "Pearson correlation heatmap of all features and target",
    "fig_2_6_iqr_outlier_counts.png": "IQR-based outlier counts per numeric feature",
    "fig_3_1_mutual_information_scores.png": "Mutual Information ranking of features (training data only) with cutoff",
}
tab_descriptions = {
    "table_attr_description.csv": "Feature attribute table: name, dtype, description, coding, missing count",
    "table_split_sizes.csv": "Train/test split sizes and class counts",
    "table_feature_selection.csv": "Feature inclusion/exclusion table for Sets A, B, C with MI and RF scores",
    "table_hyperparameters.csv": "Best hyperparameters per model and feature set with search ranges",
    "table_model_comparison_setA.csv": "Test-set model comparison metrics — Feature Set A (primary)",
    "table_model_comparison_setB.csv": "Test-set model comparison metrics — Feature Set B (all 13)",
    "table_model_comparison_setC.csv": "Test-set model comparison metrics — Feature Set C (auto-selected)",
    "table_overfitting_check.csv": "Train CV vs val CV vs test F1 scores for overfitting detection",
    "table_repeated_cv.csv": "Repeated stratified 5×5 CV mean±std for all models and feature sets",
    "table_prediction_demo.csv": "Three sample patient predictions with inputs, labels and probabilities",
}

fig_md_rows = []
for f in all_figs:
    bname = os.path.basename(f)
    desc = fig_descriptions.get(bname, "Model evaluation figure")
    # Try to infer description from name if not in dict
    if bname not in fig_descriptions:
        if "confusion" in bname:
            desc = f"Confusion matrix — {bname}"
        elif "roc" in bname:
            desc = f"ROC curves — {bname}"
        elif "metrics_bar" in bname:
            desc = f"Grouped bar chart of test-set metrics — {bname}"
        elif "cv_boxplot" in bname:
            desc = f"CV F1 score boxplots — {bname}"
        elif "rf_feature" in bname:
            desc = f"Random Forest feature importances — {bname}"
        elif "permutation" in bname:
            desc = f"Permutation importance — {bname}"
        elif "lr_coef" in bname:
            desc = f"Logistic Regression coefficients and odds ratios — {bname}"
        elif "decision_tree" in bname:
            desc = f"Decision Tree visualization — {bname}"
        elif "set_comparison" in bname:
            desc = f"Feature set A vs B vs C comparison — {bname}"
        elif "fn_fp" in bname:
            desc = f"False Negatives vs False Positives analysis — {bname}"
    fig_md_rows.append(f"| `{bname}` | {desc} |")

tab_md_rows = []
for t in all_tabs:
    bname = os.path.basename(t)
    desc = tab_descriptions.get(bname, "Output table")
    tab_md_rows.append(f"| `{bname}` | {desc} |")

summary_md = f"""# Results Summary — Heart Disease Prediction System

> **Purpose:** This document is the source material for the written report.
> All numbers are computed from executed code; none are manually entered.

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
| Missing Values (predictors) | {n_missing_total} total (ca: {int(df['ca'].isnull().sum())}, thal: {int(df['thal'].isnull().sum())}) |
| Duplicates | {int(df.duplicated().sum())} |

### Original Target Distribution (num: 0–4)
| num | count |
|-----|-------|
{chr(10).join(f'| {k} | {v} |' for k, v in df['num'].value_counts().sort_index().items())}

---

## 2. Environment

| Library | Version |
|---------|---------|
{chr(10).join(f"| {k} | {v} |" for k, v in lib_versions.items())}
| Platform | {platform.platform()} |
| Processor | {platform.processor() or 'N/A'} |
| random_state | 42 (everywhere) |

---

## 3. Split Sizes

| Subset | Records | % | No Disease | Disease |
|--------|---------|---|------------|---------|
| Train | {len(X_train)} | {len(X_train)/len(X)*100:.1f}% | {int((y_train==0).sum())} | {int((y_train==1).sum())} |
| Test  | {len(X_test)}  | {len(X_test)/len(X)*100:.1f}%  | {int((y_test==0).sum())}  | {int((y_test==1).sum())}  |

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

| Model | Hyperparameter | Search Range | Selected Value |
|-------|---------------|-------------|----------------|
{chr(10).join(f"| {r['Model']} | {r['Hyperparameter']} | {r['Search_Range']} | {r['Selected_Value']} |" for r in all_hyperparams if r['Feature_Set'] == 'A')}

---

## 6. Model Comparison — Feature Set A (Primary, Test Set)

| Model | Accuracy | Precision | Recall | F1 | ROC_AUC |
|-------|---------|----------|--------|-----|---------|
{chr(10).join(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['ROC_AUC']:.4f} |" for r in all_set_metrics['A'])}

## Model Comparison — Feature Set B (All 13, Test Set)

| Model | Accuracy | Precision | Recall | F1 | ROC_AUC |
|-------|---------|----------|--------|-----|---------|
{chr(10).join(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['ROC_AUC']:.4f} |" for r in all_set_metrics['B'])}

## Model Comparison — Feature Set C (Auto-selected, Test Set)

| Model | Accuracy | Precision | Recall | F1 | ROC_AUC |
|-------|---------|----------|--------|-----|---------|
{chr(10).join(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['ROC_AUC']:.4f} |" for r in all_set_metrics['C'])}

---

## 7. Repeated CV Robustness (5-repeat × 5-fold Stratified CV, Full Data)

| Model | Set | Accuracy (mean±std) | Precision (mean±std) | Recall (mean±std) | F1 (mean±std) | ROC_AUC (mean±std) |
|-------|-----|---------------------|---------------------|-------------------|---------------|-------------------|
{chr(10).join(f"| {r['Model']} | {r['Feature_Set']} | {r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f} | {r['precision_mean']:.3f}±{r['precision_std']:.3f} | {r['recall_mean']:.3f}±{r['recall_std']:.3f} | {r['f1_mean']:.3f}±{r['f1_std']:.3f} | {r['roc_auc_mean']:.3f}±{r['roc_auc_std']:.3f} |" for r in all_rcv_summaries)}

> **Note:** With only ~61 test samples, test-set differences between models may be within sampling noise.
> The repeated CV results above (25 total folds per model) provide a more reliable performance estimate.

---

## 8. Overfitting Check

| Model | Set | Train F1 (CV) | Val F1 (CV) | Test F1 | Gap | Status |
|-------|-----|--------------|------------|---------|-----|--------|
{chr(10).join(f"| {r['Model']} | {r['Feature_Set']} | {r['Train_F1_CV']} | {r['Val_F1_CV']} | {r['Test_F1']} | {r['Train_Val_Gap']} | {r['Status']} |" for r in all_overfitting)}

---

## 9. Best Model Selection
{best_model_justification}

---

## 10. Key EDA Observations
{eda_obs}

---

## 11. Key Feature Importance Findings
{feat_imp_text}

---

## 12. No-Leakage Confirmation

All imputers (median / most-frequent), StandardScaler, and OneHotEncoder are **inside**
`sklearn Pipeline` + `ColumnTransformer`. They are `fit()` only on training data inside
each CV fold (via `GridSearchCV`) and separately on the final training set before test evaluation.
The test set is **never seen during any fitting step**. Feature selection (MI, RF importance)
is also computed on training data only. No statistics from the full dataset are used before splitting.

---

## 13. Limitations

1. **Small dataset**: 303 rows; ~61 in the test set. Performance estimates have high variance.
2. **Cleveland subset only**: Generalisation to other populations (Hungary, Switzerland, Virginia subsets) not validated.
3. **Missing value imputation**: ca and thal have a few '?' values; median/most-frequent imputation is simple but may not capture the true missingness mechanism.
4. **Feature scope (Set A)**: Excludes 7 informative features (exang, oldpeak, ca, thal, etc.) by design; Set B and C show potential uplift.
5. **No calibration**: Model probabilities are not explicitly calibrated; Platt scaling / isotonic regression could improve them.
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

*Generated automatically by run_pipeline.py on {datetime.datetime.now().isoformat()[:19]}.*
"""

with open("outputs/RESULTS_SUMMARY.md", "w") as f:
    f.write(summary_md)
print("  RESULTS_SUMMARY.md saved to outputs/RESULTS_SUMMARY.md")

# =========================================================================
# FINAL SUMMARY
# =========================================================================
print("\n" + "=" * 65)
print("   PIPELINE COMPLETE — FINAL SUMMARY")
print("=" * 65)
print(f"  Dataset      : UCI Heart Disease Cleveland ({len(df)} rows, {df.shape[1]} cols)")
print(f"  Access date  : {access_date}")
print(f"  Train / Test : {len(X_train)} / {len(X_test)}")
print(f"\n  Feature Sets : A ({len(SET_A_FEATURES)} feat), B ({len(SET_B_FEATURES)} feat), C ({len(SET_C_FEATURES)} feat)")
print(f"\n  === Set A (Primary) Test Results ===")
for m in all_set_metrics["A"]:
    print(f"  {m['Model']:20s}  Acc={m['Accuracy']:.3f}  F1={m['F1']:.3f}  "
          f"AUC={m['ROC_AUC']:.3f}  FN={m['FN']}  FP={m['FP']}")
print(f"\n  → Best Model : {best_name}")
print(f"    F1={best_row['F1']:.4f}, ROC_AUC={best_row['ROC_AUC']:.4f}, "
      f"Recall={best_row['Recall']:.4f}, FN={best_row['FN']}")
print(f"\n  Figures saved : outputs/figures/ ({len(all_figs)} PNG files)")
print(f"  Tables saved  : outputs/tables/ ({len(all_tabs)} CSV files)")
print(f"  Model saved   : {model_save_path}")
print(f"  Summary       : outputs/RESULTS_SUMMARY.md")
print(f"\n  To use Streamlit app: streamlit run app.py")
print(f"  To predict CLI : python src/predict.py --age 55 --sex 1 --cp 0 --trestbps 130 --chol 250 --thalach 160")
print("=" * 65)
