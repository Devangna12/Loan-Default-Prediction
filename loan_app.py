import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_model.pkl"
SCALER_PATH = BASE_DIR / "loan_scaler.pkl"
TRAIN_CSV = BASE_DIR / "credit_risk_dataset.csv"

st.set_page_config(page_title="Loan Default Predictor")
st.title("Loan Default Prediction")

if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not TRAIN_CSV.exists():
    st.error("Missing model/scaler or training CSV in the app folder. Run training script first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load training data to get category options and defaults
train = pd.read_csv(TRAIN_CSV)
train = train.dropna(subset=[col for col in train.select_dtypes(include=['object']).columns], how='all')

# Minimal set of features (choose based on training script)
numeric_fields = [
    'Age', 'Monthly_Income', 'Loan_Amount', 'Current_Balance'
]
categorical_fields = ['Education_Level', 'Housing_Status']

st.header("Enter applicant details")
with st.form('loan_form'):
    inputs = {}
    for f in numeric_fields:
        default = 0.0
        if f in train.columns and pd.api.types.is_numeric_dtype(train[f]):
            default = float(train[f].median())
        inputs[f] = st.number_input(f, value=default)

    for c in categorical_fields:
        options = sorted(train[c].dropna().astype(str).unique()) if c in train.columns else [""]
        inputs[c] = st.selectbox(c, options)

    submitted = st.form_submit_button('Predict')

if submitted:
    # Build dataframe with columns matching training X
    X = pd.DataFrame([inputs])
    # One-hot encode categorical manually to match training pipeline (basic)
    X_enc = pd.get_dummies(X)
    # Align columns with scaler.feature_names_in_
    cols = list(scaler.feature_names_in_)
    for col in cols:
        if col not in X_enc.columns:
            X_enc[col] = 0.0
    X_enc = X_enc[cols]
    X_scaled = scaler.transform(X_enc)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0,1] if hasattr(model, 'predict_proba') else None

    st.write(f"Predicted default: {pred}")
    if prob is not None:
        st.write(f"Default probability: {prob:.2f}")
