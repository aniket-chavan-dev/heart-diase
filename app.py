import os
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st

from train_model import train_and_save

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
CSV_PATH = os.path.join(BASE_DIR, "heart.csv")

st.set_page_config(page_title="Heart Disease Prediction (SVM)", layout="centered")
st.title("❤️ Heart Disease Prediction — SVM (Streamlit)")

# Ensure model exists
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    st.info("Model not found. Training SVM model now — this may take a few seconds...")
    train_and_save(CSV_PATH)
    st.success("Training complete. Model saved to ./models/")

# Load artifacts with fallback to pickle if necessary
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception:
    import pickle
    # try pickle fallback file names
    model_pickle = MODEL_PATH.replace('.pkl', '_pickle.pkl')
    scaler_pickle = SCALER_PATH.replace('.pkl', '_pickle.pkl')
    if os.path.exists(model_pickle) and os.path.exists(scaler_pickle):
        with open(model_pickle, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_pickle, 'rb') as f:
            scaler = pickle.load(f)
    else:
        st.error('Model artifacts not found. Please run the training script `train_model.py` in this folder to generate the model.')
        st.stop()

# Load data to get feature ranges
df = pd.read_csv(CSV_PATH)
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
FEATURES = [c for c in df.columns if c != "target"]

st.sidebar.header("Patient input")
inputs = {}
with st.sidebar.form(key="input_form"):
    for col in FEATURES:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            lo = float(series.min())
            hi = float(series.max())
            mean = float(series.mean())
            step = (hi - lo) / 100 if hi != lo else 1.0
            inputs[col] = st.number_input(label=col, min_value=lo, max_value=hi, value=mean, step=step)
        else:
            # fallback
            inputs[col] = st.text_input(label=col, value=str(series.mode().iat[0]))
    submit = st.form_submit_button("Predict")

if submit:
    x = np.array([inputs[c] for c in FEATURES], dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    prob = model.predict_proba(x_scaled)[0]

    st.write("### Prediction")
    label = "Heart disease (1)" if pred == 1 else "No heart disease (0)"
    st.write(f"**Result:** {label}")
    st.write(f"**Confidence:** {prob.max():.2f}")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        st.write(f"**Model accuracy (holdout):** {metrics.get('accuracy', 0):.2f}")

st.markdown("---")
st.write("This app uses an SVM model trained on the Cleveland heart disease dataset. Adjust the inputs in the sidebar and press Predict.")

st.caption("Developed using Streamlit — run with `streamlit run app.py`")
