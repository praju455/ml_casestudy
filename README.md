# Heart Disease Prediction System

A complete machine learning pipeline for predicting heart disease using the UCI Heart Disease (Cleveland) dataset.

> ⚠️ **DISCLAIMER:** This project is for **educational use only**. It is **not** medical advice and must not be used for clinical diagnosis or treatment decisions.

---

## Project Structure

```
ml case studyy/
├── data/
│   ├── raw/                    # Raw CSV from UCI / ucimlrepo
│   └── processed/              # Cleaned, binarised dataset
├── notebooks/
│   └── heart_disease_analysis.ipynb  # Main Jupyter notebook (runs top-to-bottom)
├── src/
│   ├── __init__.py
│   ├── data.py                 # Data loading and saving
│   ├── preprocessing.py        # Feature sets, Pipeline/ColumnTransformer builders
│   ├── models.py               # Model definitions, hyperparameter grids, GridSearchCV
│   ├── evaluation.py           # Metrics, repeated CV, overfitting check
│   ├── plots.py                # All visualisation functions (save 300 dpi PNGs)
│   └── predict.py              # Prediction interface (CLI + importable)
├── outputs/
│   ├── figures/                # PNG plots (300 dpi)
│   ├── tables/                 # CSV result tables
│   ├── models/                 # Saved best model (joblib)
│   └── RESULTS_SUMMARY.md      # Full results for report writing
├── run_pipeline.py             # Master script — runs everything end-to-end
├── app.py                      # Optional Streamlit web app
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

This will:
- Fetch the UCI Heart Disease dataset (Cleveland, 303 rows)
- Run full EDA and save figures
- Train and tune 5 classifiers (LR, KNN, Decision Tree, SVM, Random Forest) on 3 feature sets
- Evaluate on held-out test set + repeated CV
- Save all outputs, tables, figures, best model
- Generate `outputs/RESULTS_SUMMARY.md`

### 3. Run the Jupyter notebook

```bash
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

### 4. Make a prediction (CLI)

```bash
python src/predict.py --age 55 --sex 1 --cp 0 --trestbps 130 --chol 250 --thalach 160
```

### 5. Launch the Streamlit app (optional)

```bash
streamlit run app.py
```

---

## Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease Dataset (Cleveland subset)
- **URL:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **License:** CC BY 4.0
- **Rows:** 303 | **Features:** 13 predictors + 1 target
- **Target:** `num` (0–4) binarised → 0 = no disease, 1 = disease

---

## Algorithms

| Model | Library |
|-------|---------|
| Logistic Regression | sklearn |
| K-Nearest Neighbours | sklearn |
| Decision Tree | sklearn |
| Support Vector Machine | sklearn |
| Random Forest | sklearn |

Hyperparameters tuned with **GridSearchCV (StratifiedKFold-5, F1 scoring)**.

---

## Feature Sets

| Set | Features | Purpose |
|-----|----------|---------|
| A (Primary) | age, sex, cp, trestbps, chol, thalach | Assignment requirement |
| B | All 13 predictors | Full feature baseline |
| C | Auto-selected (Mutual Information + RF Importance) | Data-driven selection |

---

## Key Design Decisions

- **No data leakage:** All preprocessing (imputation, scaling, OHE) is inside `sklearn Pipeline` / `ColumnTransformer`, fit on training data only.
- **Outliers kept:** Small dataset (303 rows); removing outliers would lose too much information. Decision Trees and Random Forests are robust to outliers.
- **Stratified splits & CV:** Class imbalance handled via stratification throughout.
- **F1 as primary metric:** Heart disease screening — missing a true case (False Negative) is costlier than a false alarm.
- **random_state=42** everywhere for reproducibility.

---

## Outputs

- `outputs/figures/` — ~30+ PNG plots (300 dpi)
- `outputs/tables/` — CSV tables (metrics, hyperparameters, feature selection, etc.)
- `outputs/models/best_model.joblib` — Serialised best pipeline
- `outputs/RESULTS_SUMMARY.md` — Full results markdown for report writing

---

*For educational use only. Not medical advice.*
