"""
src/data.py
-----------
Data loading and saving utilities for the Heart Disease Prediction project.
Handles fetching from UCI ML Repository (ucimlrepo) or local CSV fallback.
"""

import os
import datetime
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num"
]


def load_raw_data(verbose=True):
    """
    Load UCI Heart Disease (Cleveland) dataset.
    First tries ucimlrepo; on failure reads from data/raw/processed.cleveland.data.
    Returns a raw DataFrame with original column names.
    """
    access_date = datetime.date.today().isoformat()

    # --- Try ucimlrepo first ---
    try:
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)
        X = heart.data.features
        y = heart.data.targets
        df = pd.concat([X, y], axis=1)
        # Rename target column to 'num' if needed
        if "num" not in df.columns:
            df = df.rename(columns={df.columns[-1]: "num"})
        if verbose:
            print(f"[data.py] Loaded via ucimlrepo on {access_date}. Shape: {df.shape}")
    except Exception as e:
        if verbose:
            print(f"[data.py] ucimlrepo failed ({e}). Trying local CSV...")
        cleveland_path = os.path.join(RAW_DIR, "processed.cleveland.data")
        if not os.path.exists(cleveland_path):
            raise FileNotFoundError(
                f"Cannot find {cleveland_path}. "
                "Please download it from https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data "
                "and place it in data/raw/."
            )
        df = pd.read_csv(cleveland_path, header=None, names=COLUMNS, na_values="?")
        if verbose:
            print(f"[data.py] Loaded from local CSV on {access_date}. Shape: {df.shape}")

    # Save raw copy
    os.makedirs(RAW_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DIR, "heart_disease_raw.csv")
    df.to_csv(raw_path, index=False)
    if verbose:
        print(f"[data.py] Raw data saved to {raw_path}")
        print(f"[data.py] Access date: {access_date}")

    return df, access_date


def add_binary_target(df, verbose=True):
    """
    Add binary target column 'target':
      0 = no disease (num == 0)
      1 = disease    (num > 0)
    Prints both original and binary distributions.
    """
    df = df.copy()
    df["target"] = (df["num"] > 0).astype(int)

    if verbose:
        print("\n--- Original target distribution (num: 0-4) ---")
        orig_dist = df["num"].value_counts().sort_index()
        for val, cnt in orig_dist.items():
            pct = cnt / len(df) * 100
            print(f"  num={val}: {cnt} ({pct:.1f}%)")

        print("\n--- Binary target distribution ---")
        bin_dist = df["target"].value_counts().sort_index()
        labels = {0: "No Disease", 1: "Disease"}
        for val, cnt in bin_dist.items():
            pct = cnt / len(df) * 100
            print(f"  {val} ({labels[val]}): {cnt} ({pct:.1f}%)")

    return df


def save_processed(df, filename="heart_disease_processed.csv", verbose=True):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path, index=False)
    if verbose:
        print(f"[data.py] Processed data saved to {path}")
    return path

