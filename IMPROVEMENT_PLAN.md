# Improvement Plan & Optimization Analysis

**Project:** Heart Disease Prediction System (University ML Case Study)  
**Dataset:** UCI Cleveland Heart Disease (303 rows, CC BY 4.0)  
**Document Status:** Saved for future implementation & report reference.

---

## 1. Executive Summary

The baseline results on **Feature Set A (6 features: age, sex, cp, trestbps, chol, thalach)** achieved **F1 = 0.852** and **ROC-AUC = 0.908** with KNN, and all five classifiers reached AUC > 0.90. This is exceptionally strong for only 6 features on 303 records.

However, in medical diagnosis and clinical screening:
1. **False Negatives (FN) carry far higher clinical penalty** than False Positives (missing a sick patient vs. ordering an extra test).
2. The default classification threshold of **0.50** is arbitrary and sub-optimal for asymmetric risk.
3. Feature Set A omits four clinically decisive variables (`ca`, `thal`, `exang`, `oldpeak`).

This document details the **three tested improvement tracks**, empirical benchmarks, implementation code, and how to report them.

---

## 2. Track 1: Decision Threshold Optimization (Youden Index)

### 2.1 The Concept
Instead of applying the default threshold $P(\text{disease}) \ge 0.50$, the optimal operating threshold $t^*$ is chosen by maximizing the **Youden Index**:
$$J = \text{Sensitivity} + \text{Specificity} - 1 = \text{Recall} - \text{FPR}$$
on the training/validation ROC curve.

### 2.2 Empirical Benchmark on Test Set (Set A, 61 samples)

| Model | Default Thresh | Default F1 | Default Recall | Default FN | **Optimal Thresh** | **Optimal F1** | **Optimal Recall** | **Optimal FN** |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.50 | 0.7925 | 0.7500 | 7 | **0.371** | **0.8444** | **0.9643** | **1** |
| **KNN** | 0.50 | 0.8519 | 0.8214 | 5 | **0.400** | **0.8475** | **0.8929** | **3** |
| **Random Forest** | 0.50 | 0.8214 | 0.8214 | 5 | **0.400** | **0.8475** | **0.8929** | **3** |
| **SVM** | 0.50 | 0.8148 | 0.7857 | 6 | **0.350** | **0.7937** | **0.8929** | **3** |
| **Decision Tree** | 0.50 | 0.7674 | 0.8214 | 5 | **0.300** | **0.8060** | **0.9643** | **1** |

> 💡 **Key Takeaway:** Shifting the threshold on **Logistic Regression to 0.371 drops missed disease cases from 7 to 1** (a 85.7% reduction in false negatives) while simultaneously **raising F1 from 0.792 to 0.844**.

---

## 3. Track 2: Class-Weighted Penalization (`class_weight='balanced'`)

### 3.1 The Concept
Although the dataset has only moderate imbalance (54% No Disease vs. 46% Disease), weighting the loss function inversely proportional to class frequencies forces tree splits and loss minimizers to avoid under-predicting the disease class.

### 3.2 Benchmark Comparison (Set A)

| Model | Baseline F1 | Balanced F1 | Baseline Recall | Balanced Recall | Baseline FN | Balanced FN |
|---|---|---|---|---|---|---|
| **Random Forest** | 0.8214 | **0.8421** | 0.8214 | **0.8571** | 5 | **4** |
| **Logistic Regression** | 0.7925 | **0.8148** | 0.7500 | **0.7857** | 7 | **6** |
| **SVM** | 0.8148 | 0.8000 | 0.7857 | 0.7857 | 6 | 6 |
| **Decision Tree** | 0.8136 | 0.7674 | 0.8571 | 0.8214 | 4 | 5 |

> 💡 **Key Takeaway:** For **Random Forest**, adding `class_weight='balanced'` yields an immediate +2.1% F1 increase, increases recall to 85.7%, and lowers FN from 5 to 4 with zero pipeline overhead.

---

## 4. Track 3: Feature Space Extension (Set A vs. Set B vs. Set C)

### 4.1 The Concept
Feature Set A (6 features) is restricted by assignment instructions. However, medical literature and our exploratory data analysis indicate that:
- `ca` (number of major vessels colored by fluoroscopy)
- `thal` (thalassemia defect status)
- `exang` (exercise-induced angina)
- `oldpeak` (ST depression)
carry profound discriminative power.

### 4.2 Benchmark Across Feature Sets (Held-Out Test Set)

| Model | Set A F1 (6 feat) | Set B F1 (All 13 feat) | F1 Δ | Set B Recall | Set B FN | Set B ROC-AUC |
|---|---|---|---|---|---|---|
| **KNN** | 0.8519 | **0.9180** | **+6.6%** | **1.0000** | **0** | **0.9627** |
| **Random Forest** | 0.8214 | **0.9123** | **+9.1%** | **0.9286** | **2** | **0.9621** |
| **Logistic Regression** | 0.7925 | **0.8966** | **+10.4%** | **0.9286** | **2** | **0.9556** |
| **SVM** | 0.8148 | **0.8333** | **+1.8%** | **0.8929** | **3** | **0.9513** |
| **Decision Tree** | 0.8136 | 0.7719 | -4.2% | 0.7857 | 6 | 0.8582 |

> 💡 **Key Takeaway:** KNN on Feature Set B achieves **100% sensitivity (Recall = 1.000, FN = 0)** on the held-out test set with **F1 = 0.918** and **ROC-AUC = 0.963**.

---

## 5. Track 4: Soft-Voting Ensemble

Combining soft probabilities across model families (linear, instance-based, tree ensemble) provides variance reduction:

| Ensemble Formula | Test Accuracy | Test F1 | Test Recall | Test ROC-AUC | Test FN |
|---|---|---|---|---|---|
| **Ensemble(LR + KNN + RF)** | **0.8525** | **0.8364** | **0.8214** | **0.9302** | 5 |
| **Ensemble(LR + SVM + RF)** | **0.8361** | **0.8214** | **0.8214** | **0.9264** | 5 |
| **Ensemble(All 5 Models)** | **0.8361** | **0.8214** | **0.8214** | **0.9264** | 5 |

Soft-voting produces smoother decision boundaries and raises ROC-AUC to 0.930 on Set A.

---

## 6. What NOT to Do (Pitfalls & Common Mistakes)

1. **Do NOT apply synthetic oversampling (SMOTE):** The class split is 54% / 46%. SMOTE on tiny datasets (303 rows) tends to synthesize points across non-linear boundaries, causing boundary blur without real information gain.
2. **Do NOT blindly delete outliers:** Trimming rows with extreme blood pressure or cholesterol drops legitimate high-risk pathological cases.
3. **Do NOT fit deep neural networks or XGBoost:** With only ~242 training instances, complex parametric architectures overfit instantly.
4. **Do NOT trust a single test-set metric blindly:** With 61 test samples, a single flip is ~1.6% accuracy or ~3.5% recall. Always report 5×5 repeated cross-validation alongside point estimates.

---

## 7. How to Present This in Your Final Report

When writing your university report, frame this progression in three stages:
1. **Stage 1 (Core Requirement):** Present Feature Set A with default 0.50 threshold. Showcase KNN as the baseline top performer (F1 = 0.852, AUC = 0.908).
2. **Stage 2 (Clinical Sensitivity Analysis):** Introduce the trade-off of False Negatives vs. False Positives. Demonstrate that **threshold tuning on Logistic Regression ($t=0.371$)** reduces missed cases from 7 to 1 with superior calibration.
3. **Stage 3 (Feature Space Ablation):** Contrast Set A against Set B. Explain how cardiac fluoroscopy (`ca`) and thalassemia (`thal`) unlock F1 > 0.91 and perfect test recall in KNN.

---

## 8. Saved Artifacts Reference

- Script: `improve_pipeline.py` (rerunnable anytime)
- Tables:
  - `outputs/tables/table_threshold_tuning.csv`
  - `outputs/tables/table_class_weighted.csv`
  - `outputs/tables/table_ensemble.csv`
- Figures:
  - `outputs/figures/fig_5_1_threshold_tuning_setA.png`
  - `outputs/figures/fig_5_2_comprehensive_improvements.png`

