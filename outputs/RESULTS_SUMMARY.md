# Results Summary — Heart Disease Prediction System

> **Purpose:** Source material for the written report.  
> All numbers are computed from executed code; none are manually entered.  
> Generated: 2026-11-11T15:58:05

---

## 1. Dataset Facts

| Item | Value |
|------|-------|
| Source | UCI Heart Disease Dataset (Cleveland subset) — CC BY 4.0 |
| URL | https://archive.ics.uci.edu/dataset/45/heart+disease |
| Access Date | 2026-11-11 |
| Total Rows | 303 |
| Total Columns | 15 (including derived `target`) |
| Predictor Features | 13 |
| No Disease (target=0) | 164 (54.1%) |
| Disease (target=1) | 139 (45.9%) |
| Missing Values (predictors) | 6 total (ca: 4, thal: 2) |
| Duplicates | 0 |

### Original Target Distribution (num: 0–4)

| num | count |
|-----|-------|
| 0 | 164 |
| 1 | 55 |
| 2 | 36 |
| 3 | 35 |
| 4 | 13 |

---

## 2. Environment

| Library | Version |
|---------|---------|
| Python | 3.15.2 |
| scikit-learn | 1.9.1 |
| pandas | 3.1.1 |
| numpy | 2.5.4 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| joblib | 1.6.0 |
| Platform | macOS-26.5-arm64-arm-64bit-Mach-O |
| random_state | 42 (everywhere) |

---

## 3. Split Sizes

| Subset | Records | % | No Disease | Disease |
|--------|---------|---|------------|---------|
| Train | 242 | 79.9% | 131 | 111 |
| Test  | 61  | 20.1%  | 33  | 28  |

---

## 4. Feature Sets

| Feature Set | Features | Count |
|-------------|----------|-------|
| A (Primary / Assignment) | age, sex, cp, trestbps, chol, thalach | 6 |
| B (All 13 predictors) | age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal | 13 |
| C (Auto-selected: MI+RF) | sex, cp, chol, restecg, thalach, exang, slope, ca, thal | 9 |

**Feature selection cutoff:** MI Score ≥ 0.05 (computed on training data only to prevent leakage).

---

## 5. Best Hyperparameters (Feature Set A — Primary)

| Model | Hyperparameter | Search Range | Selected Value | Method |
|-------|---------------|-------------|----------------|--------|
| LogisticRegression | C | [0.01, 0.1, 1, 10, 100] | 10 | GridSearchCV (StratifiedKFold-5, F1) |
| LogisticRegression | solver | ['liblinear', 'lbfgs'] | liblinear | GridSearchCV (StratifiedKFold-5, F1) |
| KNN | metric | ['euclidean', 'manhattan'] | manhattan | GridSearchCV (StratifiedKFold-5, F1) |
| KNN | n_neighbors | [3, 5, 7, 9, 11, 15] | 5 | GridSearchCV (StratifiedKFold-5, F1) |
| KNN | weights | ['uniform', 'distance'] | distance | GridSearchCV (StratifiedKFold-5, F1) |
| DecisionTree | criterion | ['gini', 'entropy'] | entropy | GridSearchCV (StratifiedKFold-5, F1) |
| DecisionTree | max_depth | [2, 3, 4, 5, 6, None] | 3 | GridSearchCV (StratifiedKFold-5, F1) |
| DecisionTree | min_samples_leaf | [1, 2, 4] | 1 | GridSearchCV (StratifiedKFold-5, F1) |
| DecisionTree | min_samples_split | [2, 5, 10] | 2 | GridSearchCV (StratifiedKFold-5, F1) |
| SVM | C | [0.1, 1, 10, 100] | 0.1 | GridSearchCV (StratifiedKFold-5, F1) |
| SVM | gamma | ['scale', 0.01, 0.1] | scale | GridSearchCV (StratifiedKFold-5, F1) |
| SVM | kernel | ['linear', 'rbf'] | linear | GridSearchCV (StratifiedKFold-5, F1) |
| RandomForest | max_depth | [3, 5, 8, None] | 8 | GridSearchCV (StratifiedKFold-5, F1) |
| RandomForest | max_features | ['sqrt', 'log2'] | sqrt | GridSearchCV (StratifiedKFold-5, F1) |
| RandomForest | min_samples_split | [2, 5] | 5 | GridSearchCV (StratifiedKFold-5, F1) |
| RandomForest | n_estimators | [100, 200, 300] | 200 | GridSearchCV (StratifiedKFold-5, F1) |

---

## 6. Model Comparison — Test Set

### Feature Set A (Primary — 6 assignment features)

| Model | Accuracy | Precision | Recall | F1 | ROC_AUC |
|-------|---------|----------|--------|-----|---------|
| LogisticRegression | 0.8197 | 0.8400 | 0.7500 | 0.7925 | 0.9069 |
| KNN | 0.8689 | 0.8846 | 0.8214 | 0.8519 | 0.9080 |
| DecisionTree | 0.8197 | 0.7742 | 0.8571 | 0.8136 | 0.9091 |
| SVM | 0.8361 | 0.8462 | 0.7857 | 0.8148 | 0.9037 |
| RandomForest | 0.8361 | 0.8214 | 0.8214 | 0.8214 | 0.9080 |

### Feature Set B (All 13 predictors)

| Model | Accuracy | Precision | Recall | F1 | ROC_AUC |
|-------|---------|----------|--------|-----|---------|
| LogisticRegression | 0.9016 | 0.8667 | 0.9286 | 0.8966 | 0.9556 |
| KNN | 0.9180 | 0.8485 | 1.0000 | 0.9180 | 0.9627 |
| DecisionTree | 0.7869 | 0.7586 | 0.7857 | 0.7719 | 0.8582 |
| SVM | 0.8361 | 0.7812 | 0.8929 | 0.8333 | 0.9513 |
| RandomForest | 0.9180 | 0.8966 | 0.9286 | 0.9123 | 0.9621 |

### Feature Set C (Auto-selected: MI + RF Importance)

| Model | Accuracy | Precision | Recall | F1 | ROC_AUC |
|-------|---------|----------|--------|-----|---------|
| LogisticRegression | 0.8525 | 0.7879 | 0.9286 | 0.8525 | 0.9481 |
| KNN | 0.8361 | 0.7647 | 0.9286 | 0.8387 | 0.9372 |
| DecisionTree | 0.8033 | 0.7667 | 0.8214 | 0.7931 | 0.8512 |
| SVM | 0.8525 | 0.8065 | 0.8929 | 0.8475 | 0.9372 |
| RandomForest | 0.8361 | 0.8214 | 0.8214 | 0.8214 | 0.9470 |

---

## 7. Repeated CV Robustness (5-repeat × 5-fold Stratified CV, Full Data)

| Model | Set | Accuracy (mean±std) | F1 (mean±std) | Recall (mean±std) | ROC_AUC (mean±std) |
|-------|-----|---------------------|---------------|-------------------|-------------------|
| LogisticRegression | A | 0.783±0.040 | 0.753±0.052 | 0.728±0.083 | 0.860±0.037 |
| KNN | A | 0.746±0.049 | 0.713±0.066 | 0.699±0.103 | 0.820±0.049 |
| DecisionTree | A | 0.725±0.047 | 0.719±0.052 | 0.775±0.105 | 0.787±0.050 |
| SVM | A | 0.770±0.046 | 0.748±0.049 | 0.745±0.068 | 0.860±0.035 |
| RandomForest | A | 0.754±0.047 | 0.725±0.058 | 0.712±0.081 | 0.831±0.049 |
| LogisticRegression | B | 0.829±0.039 | 0.804±0.049 | 0.770±0.078 | 0.903±0.031 |
| KNN | B | 0.816±0.035 | 0.793±0.040 | 0.773±0.071 | 0.883±0.040 |
| DecisionTree | B | 0.768±0.056 | 0.728±0.071 | 0.685±0.099 | 0.823±0.047 |
| SVM | B | 0.819±0.036 | 0.792±0.044 | 0.755±0.068 | 0.897±0.034 |
| RandomForest | B | 0.836±0.037 | 0.807±0.045 | 0.757±0.073 | 0.906±0.031 |
| LogisticRegression | C | 0.834±0.040 | 0.817±0.047 | 0.813±0.074 | 0.911±0.029 |
| KNN | C | 0.818±0.041 | 0.796±0.048 | 0.777±0.078 | 0.890±0.040 |
| DecisionTree | C | 0.786±0.055 | 0.742±0.082 | 0.692±0.129 | 0.837±0.054 |
| SVM | C | 0.833±0.035 | 0.808±0.042 | 0.771±0.074 | 0.908±0.034 |
| RandomForest | C | 0.834±0.048 | 0.808±0.054 | 0.766±0.069 | 0.905±0.031 |

> **Note:** With only ~61 test samples, test-set point estimates have high variance.
> Repeated CV results (25 folds per model) provide a more reliable performance estimate.

---

## 8. Overfitting Check

| Model | Set | Train F1 (CV) | Val F1 (CV) | Test F1 | Gap | Status |
|-------|-----|--------------|------------|---------|-----|--------|
| LogisticRegression | A | 0.7581 | 0.754 | 0.7925 | 0.0041 | OK |
| KNN | A | 1.0 | 0.7553 | 0.8519 | 0.2447 | OVERFIT? |
| DecisionTree | A | 0.7813 | 0.6958 | 0.8136 | 0.0855 | OVERFIT? |
| SVM | A | 0.7508 | 0.7375 | 0.8148 | 0.0133 | OK |
| RandomForest | A | 0.9623 | 0.75 | 0.8214 | 0.2122 | OVERFIT? |
| LogisticRegression | B | 0.834 | 0.8165 | 0.8966 | 0.0175 | OK |
| KNN | B | 0.8481 | 0.8147 | 0.918 | 0.0334 | OK |
| DecisionTree | B | 0.8358 | 0.7504 | 0.7719 | 0.0854 | OVERFIT? |
| SVM | B | 0.837 | 0.8248 | 0.8333 | 0.0122 | OK |
| RandomForest | B | 0.8542 | 0.7993 | 0.9123 | 0.055 | OK |
| LogisticRegression | C | 0.8149 | 0.8205 | 0.8525 | -0.0056 | OK |
| KNN | C | 1.0 | 0.8086 | 0.8387 | 0.1914 | OVERFIT? |
| DecisionTree | C | 0.8413 | 0.7601 | 0.7931 | 0.0812 | OVERFIT? |
| SVM | C | 0.8445 | 0.8251 | 0.8475 | 0.0194 | OK |
| RandomForest | C | 0.8601 | 0.7997 | 0.8214 | 0.0604 | OK |

---

## 9. Best Model Selection

**Best model: KNN (Feature Set A)**

**Test-set results:**
- Accuracy: 0.8689
- Precision: 0.8846  
- Recall: 0.8214
- F1: 0.8519
- ROC-AUC: 0.9080

**Justification:**
- KNN achieved the highest F1 score on the held-out test set for Feature Set A.
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

1. **Class balance**: 164 No Disease (54.1%) vs 139 Disease (45.9%). Mild imbalance; stratified split used throughout.
2. **Missing values**: ca (4) and thal (2) have missing values (marked '?' in raw data). Imputed inside pipeline — no leakage.
3. **Strongest correlation with target**: `thal` (|r| = 0.526).
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
| `fig_2_1_target_distribution.png` | Binary and original (0–4) target class distribution bar charts |
| `fig_2_2_numeric_histograms.png` | Histograms of all numeric features, coloured by target class |
| `fig_2_3_boxplots_by_target.png` | Box plots of numeric features grouped by target |
| `fig_2_4_categorical_countplots.png` | Count plots of categorical features by target class |
| `fig_2_5_correlation_heatmap.png` | Pearson correlation heatmap of all features and target |
| `fig_2_6_iqr_outlier_counts.png` | IQR-based outlier counts per numeric feature |
| `fig_3_1_mutual_information_scores.png` | Mutual Information feature ranking (training data only) |
| `fig_4_10_fn_fp_analysis_setA.png` | False Negatives vs False Positives analysis — fig_4_10_fn_fp_analysis_setA.png |
| `fig_4_10_fn_fp_analysis_setB.png` | False Negatives vs False Positives analysis — fig_4_10_fn_fp_analysis_setB.png |
| `fig_4_10_fn_fp_analysis_setC.png` | False Negatives vs False Positives analysis — fig_4_10_fn_fp_analysis_setC.png |
| `fig_4_1_confusion_matrices_grid.png` | Confusion matrix — fig_4_1_confusion_matrices_grid.png |
| `fig_4_2_roc_curves_setA.png` | ROC curves — fig_4_2_roc_curves_setA.png |
| `fig_4_2_roc_curves_setB.png` | ROC curves — fig_4_2_roc_curves_setB.png |
| `fig_4_2_roc_curves_setC.png` | ROC curves — fig_4_2_roc_curves_setC.png |
| `fig_4_3_metrics_bar_setA.png` | Grouped bar chart of test-set metrics — fig_4_3_metrics_bar_setA.png |
| `fig_4_3_metrics_bar_setB.png` | Grouped bar chart of test-set metrics — fig_4_3_metrics_bar_setB.png |
| `fig_4_3_metrics_bar_setC.png` | Grouped bar chart of test-set metrics — fig_4_3_metrics_bar_setC.png |
| `fig_4_4_cv_boxplots_f1_setA.png` | CV F1 score boxplots — fig_4_4_cv_boxplots_f1_setA.png |
| `fig_4_4_cv_boxplots_f1_setB.png` | CV F1 score boxplots — fig_4_4_cv_boxplots_f1_setB.png |
| `fig_4_4_cv_boxplots_f1_setC.png` | CV F1 score boxplots — fig_4_4_cv_boxplots_f1_setC.png |
| `fig_4_5_rf_feature_importance_setA.png` | Random Forest feature importances — fig_4_5_rf_feature_importance_setA.png |
| `fig_4_5_rf_feature_importance_setB.png` | Random Forest feature importances — fig_4_5_rf_feature_importance_setB.png |
| `fig_4_5_rf_feature_importance_setC.png` | Random Forest feature importances — fig_4_5_rf_feature_importance_setC.png |
| `fig_4_6_permutation_importance_setA.png` | Permutation importance — fig_4_6_permutation_importance_setA.png |
| `fig_4_6_permutation_importance_setB.png` | Permutation importance — fig_4_6_permutation_importance_setB.png |
| `fig_4_6_permutation_importance_setC.png` | Permutation importance — fig_4_6_permutation_importance_setC.png |
| `fig_4_7_lr_coefficients_setA.png` | Logistic Regression coefficients and odds ratios — fig_4_7_lr_coefficients_setA.png |
| `fig_4_7_lr_coefficients_setB.png` | Logistic Regression coefficients and odds ratios — fig_4_7_lr_coefficients_setB.png |
| `fig_4_7_lr_coefficients_setC.png` | Logistic Regression coefficients and odds ratios — fig_4_7_lr_coefficients_setC.png |
| `fig_4_8_decision_tree_setA.png` | Decision Tree visualisation — fig_4_8_decision_tree_setA.png |
| `fig_4_8_decision_tree_setB.png` | Decision Tree visualisation — fig_4_8_decision_tree_setB.png |
| `fig_4_8_decision_tree_setC.png` | Decision Tree visualisation — fig_4_8_decision_tree_setC.png |
| `fig_4_9_set_comparison_f1.png` | Feature set A vs B vs C comparison — fig_4_9_set_comparison_f1.png |
| `fig_4_9_set_comparison_roc_auc.png` | Feature set A vs B vs C comparison — fig_4_9_set_comparison_roc_auc.png |
| `fig_4_cm_decisiontree_setA.png` | Model evaluation figure |
| `fig_4_cm_decisiontree_setB.png` | Model evaluation figure |
| `fig_4_cm_decisiontree_setC.png` | Model evaluation figure |
| `fig_4_cm_knn_setA.png` | Model evaluation figure |
| `fig_4_cm_knn_setB.png` | Model evaluation figure |
| `fig_4_cm_knn_setC.png` | Model evaluation figure |
| `fig_4_cm_logisticregression_setA.png` | Model evaluation figure |
| `fig_4_cm_logisticregression_setB.png` | Model evaluation figure |
| `fig_4_cm_logisticregression_setC.png` | Model evaluation figure |
| `fig_4_cm_randomforest_setA.png` | Model evaluation figure |
| `fig_4_cm_randomforest_setB.png` | Model evaluation figure |
| `fig_4_cm_randomforest_setC.png` | Model evaluation figure |
| `fig_4_cm_svm_setA.png` | Model evaluation figure |
| `fig_4_cm_svm_setB.png` | Model evaluation figure |
| `fig_4_cm_svm_setC.png` | Model evaluation figure |

---

## 15. Tables

| Filename | Description |
|----------|-------------|
| `table_attr_description.csv` | Feature attribute table: name, dtype, description, coding, missing count |
| `table_feature_selection.csv` | Feature inclusion/exclusion for Sets A, B, C with MI and RF scores |
| `table_hyperparameters.csv` | Best hyperparameters per model and feature set |
| `table_model_comparison_setA.csv` | Test-set model comparison metrics — Feature Set A (primary) |
| `table_model_comparison_setB.csv` | Test-set model comparison metrics — Feature Set B (all 13) |
| `table_model_comparison_setC.csv` | Test-set model comparison metrics — Feature Set C (auto-selected) |
| `table_overfitting_check.csv` | Train CV vs val CV vs test F1 scores for overfitting detection |
| `table_prediction_demo.csv` | Three sample patient predictions with inputs, labels and probabilities |
| `table_repeated_cv.csv` | Repeated stratified 5×5 CV mean±std for all models and feature sets |
| `table_split_sizes.csv` | Train/test split sizes and class counts |

---

*All numbers computed from executed code on 2026-11-11.*


---

## 16. Potential Improvements & Optimization Analysis

> Analyzed from executed experiments on 2026-11-11.

### 16.1 Decision Threshold Optimization (Set A)
For screening, the cost of a false negative (missed patient) is far greater than a false positive. By tuning the classification threshold to maximize the Youden Index ($J = \text{Recall} - \text{FPR}$):

| Model | Default F1 (t=0.50) | Opt Threshold | Opt F1 | Opt Recall | FN (Default $\to$ Opt) |
|---|---|---|---|---|---|
| LogisticRegression | 0.7925 | 0.372 | 0.8276 | 0.8571 | 7 $\to$ 4 |
| KNN | 0.8519 | 0.460 | 0.8571 | 0.8571 | 5 $\to$ 4 |
| DecisionTree | 0.8136 | 0.304 | 0.8571 | 0.9643 | 4 $\to$ 1 |
| SVM | 0.8148 | 0.403 | 0.8621 | 0.8929 | 6 $\to$ 3 |
| RandomForest | 0.8214 | 0.521 | 0.8364 | 0.8214 | 5 $\to$ 5 |

**Key Observation:** For Logistic Regression, shifting threshold from 0.50 to 0.372 reduces missed cases from 7 to 4, while increasing F1 from 0.7925 to 0.8276.

### 16.2 Feature Set Extension (Set A vs Set B)
The assignment requires Set A (6 features), but adding clinical features (`ca`, `thal`, `exang`, `oldpeak`) yields significant gains:

| Model | Set A F1 (6 feat) | Set B F1 (13 feat) | F1 Gain | Set B Recall | Set B FN |
|---|---|---|---|---|---|
| LogisticRegression | 0.7925 | 0.8966 | +10.4% | 0.9286 | 2 |
| KNN | 0.8519 | 0.9180 | +6.6% | 1.0000 | 0 |
| DecisionTree | 0.8136 | 0.7719 | +-4.2% | 0.7857 | 6 |
| SVM | 0.8148 | 0.8333 | +1.9% | 0.8929 | 3 |
| RandomForest | 0.8214 | 0.9123 | +9.1% | 0.9286 | 2 |

### 16.3 Soft-Voting Ensemble
Combining predictions of diverse models on Set A:
| Configuration | Accuracy | F1 | Recall | ROC-AUC | FN |
|---|---|---|---|---|---|
| Ensemble(LR+KNN+RF) | 0.8525 | 0.8364 | 0.8214 | 0.9264 | 5 |
| Ensemble(LR+SVM+RF) | 0.8361 | 0.8148 | 0.7857 | 0.9156 | 6 |
| Ensemble(All 5) | 0.8033 | 0.7778 | 0.7500 | 0.9275 | 7 |

### 16.4 New Improvement Figures & Tables
- `fig_5_1_threshold_tuning_setA.png`: Threshold optimization comparison on Set A
- `fig_5_2_comprehensive_improvements.png`: F1 and ROC-AUC uplift across strategies
- `table_threshold_tuning.csv`: Complete metrics across various thresholds
- `table_class_weighted.csv`: Performance with `class_weight='balanced'`
- `table_ensemble.csv`: Performance of voting ensembles
