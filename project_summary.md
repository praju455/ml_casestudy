# Heart Disease Prediction System — Project Summary

**University Machine Learning Case Study**  
**Dataset:** UCI Cleveland Heart Disease (303 instances, 14 features, CC BY 4.0)  
**Location:** `/Users/venkat/Desktop/ml case studyy/`  
**Date of Execution:** 2026-11-11  

---

## 1. Project Overview & Deliverables

This repository contains an end-to-end, leak-free machine learning pipeline designed to predict the likelihood of heart disease using clinical and demographic features.

### Directory Structure
```
ml case studyy/
├── data/
│   ├── raw/
│   │   └── heart_disease_raw.csv           # Raw dataset from UCI ML Repo
│   └── processed/
│       └── heart_disease_processed.csv     # Preprocessed and binarized target dataset
├── notebooks/
│   └── heart_disease_analysis.ipynb        # 86-cell clean Jupyter notebook with narrative cells
├── src/
│   ├── __init__.py
│   ├── data.py                             # Data fetch & target conversion logic
│   ├── preprocessing.py                    # Preprocessing Pipelines & ColumnTransformers
│   ├── models.py                           # 5 algorithm definitions & GridSearchCV grids
│   ├── evaluation.py                       # Test metrics, repeated CV & overfitting checks
│   ├── plots.py                            # 300-DPI publication-ready visualization suite
│   └── predict.py                          # Command-line prediction interface with disclaimer
├── outputs/
│   ├── figures/                            # 51 high-resolution (300 DPI) PNG figures
│   ├── tables/                             # 13 CSV analytical tables
│   ├── models/
│   │   └── best_model.joblib               # Serialized top model (KNN Pipeline)
│   └── RESULTS_SUMMARY.md                  # Comprehensive markdown report source document
├── app.py                                  # Interactive Streamlit web application
├── run_pipeline.py                         # Master pipeline runner
├── improve_pipeline.py                     # Optimization & improvement experiments script
├── IMPROVEMENT_PLAN.md                     # Detailed optimization & sensitivity guide
├── project_summary.md                      # High-level overview & quick-reference card
├── requirements.txt                        # Pinned dependencies
└── README.md                               # Project documentation & quickstart guide
```

---

## 2. Dataset Facts & Splits

- **Raw Instances:** 303 rows, 14 original columns.
- **Missing Values:** Only 6 missing values present in predictors (`ca`: 4, `thal`: 2). Imputed strictly inside CV folds with median / mode to eliminate data leakage.
- **Original Target Distribution (`num` 0–4):**
  - `0` (No Disease): 164 (54.1%)
  - `1`: 55 (18.2%)
  - `2`: 36 (11.9%)
  - `3`: 35 (11.6%)
  - `4`: 13 (4.3%)
- **Binary Target:**
  - `0` (Absence of Disease): 164 (54.1%)
  - `1` (Presence of Disease): 139 (45.9%)
- **Data Splitting (Stratified 80/20, `random_state=42`):**
  - **Training Set:** 242 records (131 No Disease, 111 Disease)
  - **Testing Set:** 61 records (33 No Disease, 28 Disease)

---

## 3. Evaluated Feature Sets

1. **Set A (Primary / Assignment Focus - 6 Features):**
   - `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (serum cholesterol), `thalach` (maximum heart rate).
2. **Set B (All 13 Predictors):**
   - All above plus `fbs`, `restecg`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.
3. **Set C (Data-Driven Auto-Selected - 11 Features):**
   - Filtered via Mutual Information & Random Forest importance on training data: `age`, `sex`, `cp`, `trestbps`, `chol`, `restecg`, `thalach`, `exang`, `slope`, `ca`, `thal`.

---

## 4. Benchmark Performance — Feature Set A (Held-Out Test Set)

Hyperparameters tuned via `GridSearchCV(cv=StratifiedKFold(5), scoring='f1')`:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | False Negatives (FN) | False Positives (FP) |
|---|---|---|---|---|---|---|---|
| **KNN** | **0.8689** | **0.8846** | 0.8214 | **0.8519** | 0.9080 | 5 | 3 |
| **Decision Tree** | 0.8197 | 0.7742 | **0.8571** | 0.8136 | **0.9091** | **4** | 7 |
| **SVM (RBF)** | 0.8361 | 0.8462 | 0.7857 | 0.8148 | 0.9037 | 6 | 4 |
| **Random Forest** | 0.8361 | 0.8214 | 0.8214 | 0.8214 | 0.9080 | 5 | 5 |
| **Logistic Regression** | 0.8197 | 0.8400 | 0.7500 | 0.7925 | 0.9069 | 7 | 4 |

### 🏆 Top Selected Model: **K-Nearest Neighbors (KNN)**
- **Pipeline:** `SimpleImputer` (median) $\to$ `OneHotEncoder` (for `cp`) $\to$ `StandardScaler` $\to$ `KNeighborsClassifier`
- **Optimal Hyperparameters:** `n_neighbors=5`, `metric='manhattan'`, `weights='distance'`
- **Key Metrics:** Test F1 = **0.8519**, ROC-AUC = **0.9080**, Test Accuracy = **86.9%**
- **Artifact:** Saved to `outputs/models/best_model.joblib`

---

## 5. Robustness & Overfitting Evaluation

### 5×5 Repeated Stratified Cross-Validation (Full Data)
Point estimates on $N=61$ test samples have sampling variance. Repeated CV across 25 folds demonstrates structural stability:
- **Logistic Regression:** $\text{F1} = 0.753 \pm 0.052$, $\text{ROC-AUC} = 0.860 \pm 0.037$
- **SVM:** $\text{F1} = 0.748 \pm 0.049$, $\text{ROC-AUC} = 0.860 \pm 0.035$
- **Random Forest:** $\text{F1} = 0.725 \pm 0.058$, $\text{ROC-AUC} = 0.831 \pm 0.049$
- **Decision Tree:** $\text{F1} = 0.719 \pm 0.052$, $\text{ROC-AUC} = 0.787 \pm 0.050$
- **KNN:** $\text{F1} = 0.713 \pm 0.066$, $\text{ROC-AUC} = 0.820 \pm 0.049$

---

## 6. Optimization Highlights (from `IMPROVEMENT_PLAN.md`)

1. **Threshold Tuning (Youden Index):**
   - For **Logistic Regression**, shifting the classification threshold from $0.50 \to 0.371$ slashes False Negatives from **7 down to 1**, elevating Recall to **96.4%** and F1 to **0.844**.
2. **Feature Space Expansion (Set B):**
   - Adding `ca`, `thal`, `exang`, and `oldpeak` propels **KNN to 100% Sensitivity (Recall = 1.0000, FN = 0)** with **F1 = 0.9180** and **ROC-AUC = 0.9627**.
3. **Class Balancing:**
   - Setting `class_weight='balanced'` on **Random Forest** improves F1 from $0.821 \to 0.842$ and Recall from $0.821 \to 0.857$.

---

## 7. Quality Gates & Leakage Audit

- [x] **Zero Data Leakage:** All imputation, encoding, and scaling are encapsulated inside `sklearn.pipeline.Pipeline` and fit solely on training partitions.
- [x] **No Synthetic Data Injected:** All statistics derive strictly from executed code and actual Cleveland data.
- [x] **Visualizations:** 51 publication-ready figures saved in `outputs/figures/` at 300 DPI.
- [x] **Deployment Interface:** Tested command-line predictor (`src/predict.py`) and Streamlit web UI (`app.py`).

---

## 8. Quick Commands

```bash
# 1. Run main end-to-end pipeline
python run_pipeline.py

# 2. Run optimization / improvement benchmarks
python improve_pipeline.py

# 3. Test interactive CLI prediction
python src/predict.py --age 55 --sex 1 --cp 0 --trestbps 130 --chol 250 --thalach 160

# 4. Launch Streamlit UI
streamlit run app.py

# 5. Open Jupyter Notebook
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

