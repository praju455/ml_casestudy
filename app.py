"""
app.py — Optional Streamlit web app for Heart Disease Prediction.
Run with: streamlit run app.py
DISCLAIMER: Educational use only. Not medical advice.
"""

import os
import sys
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = os.path.join("outputs", "models", "best_model.joblib")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.markdown("> **DISCLAIMER:** This tool is for **educational use only**. It is **not** medical advice. "
            "Always consult a qualified healthcare professional for medical concerns.")

st.markdown("---")
st.subheader("Enter Patient Information (Set A Features)")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=55)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
with col2:
    cp = st.selectbox("Chest Pain Type (cp)",
                      options=[0, 1, 2, 3],
                      format_func=lambda x: {
                          0: "0 – Typical Angina",
                          1: "1 – Atypical Angina",
                          2: "2 – Non-Anginal Pain",
                          3: "3 – Asymptomatic"
                      }[x])
    trestbps = st.number_input("Resting BP (mmHg)", min_value=50, max_value=250, value=130)
with col3:
    chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=600, value=240)
    thalach = st.number_input("Max Heart Rate (thalach)", min_value=50, max_value=250, value=150)

st.markdown("---")

if st.button("🔍 Predict", type="primary"):
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please run the main pipeline first.")
    else:
        model = joblib.load(MODEL_PATH)
        input_df = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, thalach]],
            columns=["age", "sex", "cp", "trestbps", "chol", "thalach"]
        )
        pred = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0][1])

        if pred == 1:
            st.error(f"⚠️ **Disease Likely** — P(Disease) = {prob:.1%}")
        else:
            st.success(f"✅ **No Disease Predicted** — P(Disease) = {prob:.1%}")

        st.metric("Probability of Disease", f"{prob:.1%}")
        st.progress(min(prob, 1.0))

        st.markdown("---")
        st.caption("⚠️ This prediction is for educational purposes only and must not be used "
                   "as a substitute for professional medical advice, diagnosis, or treatment.")

