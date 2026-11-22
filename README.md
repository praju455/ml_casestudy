# Heart Disease Prediction System

This machine learning case study compares five classifiers on 303 records from the Cleveland subset of the UCI Heart Disease dataset. It evaluates the six assignment predictors alongside the full and data-selected predictor sets. The models classify recorded disease status for an educational demonstration.

## Dataset and features

The Cleveland data are available as [raw](data/raw/heart_disease_raw.csv) and [processed](data/processed/heart_disease_processed.csv) CSVs under CC BY 4.0. The binary target maps `num = 0` to absence and `num > 0` to presence. There are 164 negative and 139 positive records. The only missing predictor values are four in `ca` and two in `thal`.

| Set | Predictors | Purpose |
| --- | --- | --- |
| A (6) | `age`, `sex`, `cp`, `trestbps`, `chol`, `thalach` | Primary assignment comparison |
| B (13) | All 13 predictors | Full-input comparison |
| C (9) | `sex`, `cp`, `chol`, `restecg`, `thalach`, `exang`, `slope`, `ca`, `thal` | Training-data feature selection |

Set C keeps predictors whose transformed columns meet the mutual information cutoff of 0.05. Random Forest importance contributes to the ranking. The [feature-selection table](outputs/tables/table_feature_selection.csv) shows the membership and scores for each predictor.

## Experiment design

- The stratified 80/20 split uses random state 42, with 242 training and 61 test records shared across all three feature sets.
- Each pipeline uses median imputation and standard scaling for the numeric branch, plus most-frequent imputation and one-hot encoding for chest pain type. Grid search fits these transformations inside the training folds.
- Logistic Regression, K-Nearest Neighbours (KNN), Decision Tree, Support Vector Machine and Random Forest are tuned with five stratified folds and mean validation F1. Set C membership is fixed using the full 242-record training pool before this grid search.
- Repeated cross-validation uses five repeats of five stratified folds over all 303 records with fixed grid settings. Its rankings describe variation across splits under this design, which reuses the selected settings and features.

## Results

| Feature set | Highest test F1 | Test F1 | Test ROC-AUC | Highest mean repeated-CV F1 |
| --- | --- | ---: | ---: | --- |
| A | KNN | 0.8519 | 0.9080 | Logistic Regression (0.753) |
| B | KNN | 0.9180 | 0.9627 | Random Forest (0.807) |
| C | Logistic Regression | 0.8525 | 0.9481 | Logistic Regression (0.817) |

On the 61-record test split, Set A KNN missed 5 of 28 positive cases and produced 3 false alarms. Set B KNN found all 28 positives and produced 5 false alarms. The saved six-feature KNN was selected using test F1 and then refitted on all 303 rows for the demo.

The follow-up experiments compare decision thresholds, class weights and soft voting on the same test split. The threshold sweeps choose and score cutoffs on those test rows, so their gains are exploratory. The small sample and single cohort limit generalisation. The model is intended for education.

See the [results summary](outputs/RESULTS_SUMMARY.md), [experiment tables](outputs/tables/), and [figures](outputs/figures/) for the full comparisons.

## Run the project

Run these commands from the repository root.

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
python improve_pipeline.py
```

The main pipeline first tries `ucimlrepo`. Its offline fallback expects the original `processed.cleveland.data` file at `data/raw/processed.cleveland.data`. The exploratory improvement script uses the processed CSV and hyperparameter table from the main pipeline.

Open the [analysis notebook](notebooks/heart_disease_analysis.ipynb) to review the workflow and generated outputs.

```bash
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

The CLI and Streamlit demo use the saved [Set A pipeline](outputs/models/best_model.joblib).

```bash
python src/predict.py --age 55 --sex 1 --cp 2 --trestbps 130 --chol 250 --thalach 160
streamlit run app.py
```

The demo accepts chest-pain codes 0–3, while the committed training data use 1–4. Interpret the demo’s category labels accordingly.

## Project files

| Location | Purpose |
| --- | --- |
| [`src/`](src/) | Data loading, preprocessing, model definitions, evaluation, plots and prediction |
| [`run_pipeline.py`](run_pipeline.py) | Main training and evaluation pipeline |
| [`improve_pipeline.py`](improve_pipeline.py) | Threshold, class-weight and soft-voting experiments |
| [`notebooks/`](notebooks/) | Reproducible analysis and visual results |
| [`outputs/`](outputs/) | Saved tables, plots, model and results summary |
