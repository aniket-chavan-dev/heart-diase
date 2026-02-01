# Heart Disease Prediction — Streamlit Frontend (SVM)

Files added:

- `train_model.py` — trains an SVM and saves model + scaler to `./models/`
- `app.py` — Streamlit frontend that loads the trained model and predicts on user input
- `requirements.txt` — Python packages needed

Quick start:

1. (Optional) Create and activate a Python environment (Conda/venv).
2. Install requirements: `pip install -r requirements.txt`.
3. Train model: `python train_model.py` (this will create `./models/svm_model.pkl` and `./models/scaler.pkl` or pickled fallbacks).
4. Run app: `streamlit run app.py` and open the URL shown in your browser.

Notes:

- If `joblib` isn't available, `train_model.py` will save pickle fallbacks named `svm_model_pickle.pkl` and `scaler_pickle.pkl`.
- `app.py` will attempt to load both joblib and pickle artifacts.
