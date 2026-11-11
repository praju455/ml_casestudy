"""
src/predict.py
--------------
Production prediction interface for the Heart Disease Prediction model.

DISCLAIMER: This tool is for EDUCATIONAL USE ONLY and is NOT medical advice.
Do not use predictions from this model for clinical diagnosis or treatment decisions.
Always consult a qualified medical professional for health concerns.

Usage:
    python src/predict.py --age 55 --sex 1 --cp 0 --trestbps 130 --chol 250 --thalach 160

Input Features (Set A):
    age      : Age in years (integer)
    sex      : Sex (1 = male, 0 = female)
    cp       : Chest pain type (0=typical angina, 1=atypical angina,
                2=non-anginal pain, 3=asymptomatic)
    trestbps : Resting blood pressure in mmHg (integer)
    chol     : Serum cholesterol in mg/dl (integer)
    thalach  : Maximum heart rate achieved (integer)
"""

import os
import sys
import argparse
import pandas as pd
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "models", "best_model.joblib")

DISCLAIMER = (
    "\n⚠️  DISCLAIMER: This prediction is for EDUCATIONAL USE ONLY.\n"
    "   It is NOT medical advice and should NOT be used for clinical diagnosis or treatment.\n"
    "   Please consult a qualified healthcare professional for any medical concerns.\n"
)

FEATURE_COLS = ["age", "sex", "cp", "trestbps", "chol", "thalach"]

CP_LABELS = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-Anginal Pain",
    3: "Asymptomatic",
}


def load_model(model_path=None):
    path = model_path or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run the main pipeline first to train and save the model."
        )
    model = joblib.load(path)
    print(f"[predict.py] Model loaded from {path}")
    return model


def predict(age, sex, cp, trestbps, chol, thalach, model=None):
    """
    Make a single prediction.

    Parameters
    ----------
    age      : int   – patient age in years
    sex      : int   – 1 = male, 0 = female
    cp       : int   – chest pain type (0–3)
    trestbps : int   – resting blood pressure (mmHg)
    chol     : int   – serum cholesterol (mg/dl)
    thalach  : int   – maximum heart rate achieved

    Returns
    -------
    dict with keys: prediction, probability, label, disclaimer
    """
    if model is None:
        model = load_model()

    input_df = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, thalach]],
        columns=FEATURE_COLS,
    )

    pred = int(model.predict(input_df)[0])
    prob = float(model.predict_proba(input_df)[0][1])  # probability of disease
    label = "DISEASE LIKELY" if pred == 1 else "NO DISEASE PREDICTED"

    return {
        "prediction": pred,
        "probability_disease": round(prob, 4),
        "label": label,
        "disclaimer": DISCLAIMER,
    }


def predict_batch(records, model=None):
    """
    Predict for a list of dicts with keys matching FEATURE_COLS.

    Returns a DataFrame with predictions and probabilities appended.
    """
    if model is None:
        model = load_model()
    df = pd.DataFrame(records, columns=FEATURE_COLS)
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]
    df["prediction"] = preds
    df["probability_disease"] = probs.round(4)
    df["label"] = df["prediction"].map({0: "No Disease", 1: "Disease Likely"})
    return df


def _print_result(result, input_dict):
    print("\n" + "=" * 55)
    print("  HEART DISEASE PREDICTION RESULT")
    print("=" * 55)
    print("  Input:")
    for k, v in input_dict.items():
        print(f"    {k:12s}: {v}")
    print(f"\n  Prediction     : {result['label']}")
    print(f"  P(Disease)     : {result['probability_disease']:.4f} ({result['probability_disease']*100:.1f}%)")
    print(result["disclaimer"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heart Disease Prediction (Educational Use Only)"
    )
    parser.add_argument("--age", type=int, required=True, help="Age in years")
    parser.add_argument("--sex", type=int, required=True, choices=[0, 1],
                        help="Sex: 1=male, 0=female")
    parser.add_argument("--cp", type=int, required=True, choices=[0, 1, 2, 3],
                        help="Chest pain type: 0-3")
    parser.add_argument("--trestbps", type=int, required=True, help="Resting BP (mmHg)")
    parser.add_argument("--chol", type=int, required=True, help="Cholesterol (mg/dl)")
    parser.add_argument("--thalach", type=int, required=True, help="Max heart rate")
    parser.add_argument("--model_path", type=str, default=None, help="Path to saved model")

    args = parser.parse_args()
    input_vals = {
        "age": args.age, "sex": args.sex, "cp": args.cp,
        "trestbps": args.trestbps, "chol": args.chol, "thalach": args.thalach
    }
    model = load_model(args.model_path)
    result = predict(**input_vals, model=model)
    _print_result(result, input_vals)

