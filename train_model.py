import os
import json
try:
    import joblib
except Exception:
    joblib = None
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_and_save(csv_path: str = "heart.csv"):
    df = pd.read_csv(csv_path)
    # drop unnamed index if present
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    X = df.drop(columns=["target"])
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Use probability=True so frontend can show confidence
    svm = SVC(probability=True, random_state=42)
    svm.fit(x_train_scaled, y_train)

    preds = svm.predict(x_test_scaled)
    acc = accuracy_score(y_test, preds)

    # Save artifacts
    model_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    if joblib is not None:
        joblib.dump(svm, model_path)
        joblib.dump(scaler, scaler_path)
    else:
        # joblib not available in this environment — fallback to pickle files
        with open(model_path.replace('.pkl', '_pickle.pkl'), 'wb') as f:
            pickle.dump(svm, f)
        with open(scaler_path.replace('.pkl', '_pickle.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc}, f)

    print(f"Saved model to {model_path}")
    print(f"Accuracy on holdout set: {acc:.4f}")


if __name__ == "__main__":
    train_and_save()
