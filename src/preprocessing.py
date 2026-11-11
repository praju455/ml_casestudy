"""
src/preprocessing.py
--------------------
Feature sets, sklearn Pipeline/ColumnTransformer builders.
All imputation and scaling is done INSIDE the pipeline (no leakage).
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# -----------------------------------------------------------------------
# Feature definitions
# -----------------------------------------------------------------------

# Assignment primary features (Set A)
SET_A_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "thalach"]

# All 13 predictor features (Set B)
SET_B_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal"
]

TARGET_COL = "target"

# Nominal (multi-category) features that need OHE
NOMINAL_FEATURES = ["cp"]             # chest pain type (0-3)
# Binary/ordinal kept as numeric
# (sex=0/1, fbs=0/1, restecg=0/1/2, exang=0/1, slope=1/2/3, ca=0-3, thal=3/6/7)

# Numeric continuous features
NUMERIC_CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Numeric but discrete/ordinal — no OHE, keep as-is after scaling
NUMERIC_ORDINAL = ["sex", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def _get_column_groups(feature_set):
    """Return (numeric_cols, nominal_cols, all_cols) for a given feature_set."""
    nominal_cols = [c for c in NOMINAL_FEATURES if c in feature_set]
    numeric_cols = [c for c in feature_set if c not in nominal_cols]
    return numeric_cols, nominal_cols


def build_preprocessor(feature_set):
    """
    Build a ColumnTransformer that:
      - Imputes numeric features with median
      - Imputes nominal features with most frequent
      - Scales numeric features with StandardScaler
      - One-hot encodes nominal features (cp)

    All fitted on training data only when used inside a Pipeline.
    """
    numeric_cols, nominal_cols = _get_column_groups(feature_set)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformers = [
        ("num", numeric_transformer, numeric_cols),
    ]

    if nominal_cols:
        nominal_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("nom", nominal_transformer, nominal_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def get_feature_names_out(preprocessor, feature_set):
    """
    Return the final feature names after ColumnTransformer transform.
    Call only AFTER fitting the preprocessor.
    """
    numeric_cols, nominal_cols = _get_column_groups(feature_set)
    names = list(numeric_cols)
    if nominal_cols:
        try:
            ohe = preprocessor.named_transformers_["nom"]["ohe"]
            ohe_names = ohe.get_feature_names_out(nominal_cols).tolist()
            names += ohe_names
        except Exception:
            pass
    return names

