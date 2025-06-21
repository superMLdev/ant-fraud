import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

DATA_PATH = "../data/processed/streaming_transactions.csv"
MODEL_OUTPUT = "xgb_fraud_model.pkl"

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    # Encode categorical columns
    df["location"] = df["location"].astype("category").cat.codes
    df["device_type"] = df["device_type"].astype("category").cat.codes

    # Features and target
    X = df[["amount", "user_id", "location", "device_type"]]
    y = df["is_fraud"]

    # Apply SMOTE before train-test split
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n🔍 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    print(f"\n✅ Model saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    train_and_save_model()