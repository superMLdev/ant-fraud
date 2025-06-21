import pandas as pd
import xgboost as xgb
import joblib
from sqlalchemy import create_engine, text
import os
import random
from dotenv import load_dotenv
load_dotenv()

from models.drift_detection import should_retrain_model, evaluate_existing_model_on_recent_data

# Load DB and model
DB_URL = os.getenv("DATABASE_URL")
PROBABILITY_THRESHOLD = os.getenv("PROABILITY_THRESHOLD", 0.5)
MODEL_PATH = os.getenv("MODEL_OUTPUT", "../models/xgb_fraud_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model not found at {MODEL_PATH}. Using fallback random prediction logic.")
        return "fallback"
    return joblib.load(MODEL_PATH)

def fetch_unprocessed_transactions(conn,transaction_id=None):
    print(f"<UNK> Fetching unprocessed transactions from {transaction_id}")
    if transaction_id:
        query = text("SELECT * FROM transactions WHERE transaction_id = :transaction_id AND status = 'received'")
        return pd.read_sql(query, conn, params={"transaction_id": transaction_id})
    else:
        query = text("SELECT * FROM transactions WHERE status = 'received'")
        return pd.read_sql(query, conn)

def preprocess(df):
    # Basic encoding — should match your training logic!
    df = df.copy()
    df["location"] = df["location"].astype("category").cat.codes
    df["device_type"] = df["device_type"].astype("category").cat.codes
    features = ["amount", "user_id", "location", "device_type"]
    return df[features]

def update_prediction(conn, txn_id, prediction, probability):
    stmt = text("""
        UPDATE transactions
        SET prediction = :prediction,
            probability = :probability,
            status = 'predicted'
        WHERE transaction_id = :transaction_id
    """)
    conn.execute(stmt, {
        "transaction_id": txn_id,
        "prediction": int(prediction),
        "probability": float(probability)
    })

def process_transaction(conn, transaction_id):
    engine = create_engine(DB_URL)
    model = load_model()

    df = fetch_unprocessed_transactions(conn,transaction_id)

    if df.empty:
        print("✅ No unprocessed transactions.")
        return

    if model == "fallback":
        preds = [1 if random.random() < 0.02 else 0 for _ in range(len(df))]
        proba = preds  # probabilities same as preds for fallback
    else:
        X = preprocess(df)
        proba = model.predict_proba(X)[:, 1]
        # Convert probabilities to binary predictions
        preds = (proba > PROBABILITY_THRESHOLD).astype(int)

        for txn_id, pred, prob in zip(df["transaction_id"], preds, proba):
            update_prediction(conn, txn_id, pred, prob)
            print(f"🧠 {txn_id} → {'FRAUD' if pred else 'NON-FRAUD'} ({prob:.2f})")

        conn.commit()
        print("✅ All predictions updated using XGBoost.")
        return {
            "transaction_id": transaction_id,
            "prediction": preds[0],
            "probability": proba[0] if model != "fallback" else None
        }



def process_transactions():
    engine = create_engine(DB_URL)
    model = load_model()

    # Evaluate model drift
    drift_needed = False
    if model != "fallback":
        f1_threshold = float(os.getenv("F1_THRESHOLD", 0.90))
        transactions_df = pd.read_sql("SELECT * FROM transactions WHERE feedback IS NOT NULL AND prediction IS NOT NULL", engine)
        drift_needed = should_retrain_model(
            f1_threshold=f1_threshold,
            transactions_df=transactions_df
        )
        if drift_needed:
            print(f"⚠️ F1 score dropped below threshold ({f1_threshold}). Retraining is recommended.")
        else:
            print("✅ No significant model drift detected.")

    if model == "fallback":
        print("🔁 Using fallback fraud predictor with 2% probability.")

    with engine.connect() as conn:
        df = fetch_unprocessed_transactions(conn)

        if df.empty:
            print("✅ No unprocessed transactions.")
            return

        if model == "fallback":
            preds = [1 if random.random() < 0.02 else 0 for _ in range(len(df))]
            proba = preds  # probabilities same as preds for fallback
        else:
            X = preprocess(df)
            proba = model.predict_proba(X)[:, 1]
            preds = (proba > PROBABILITY_THRESHOLD).astype(int)

        for txn_id, pred, prob in zip(df["transaction_id"], preds, proba):
            update_prediction(conn, txn_id, pred, prob)
            print(f"🧠 {txn_id} → {'FRAUD' if pred else 'NON-FRAUD'} ({prob:.2f})")

        conn.commit()
        print("✅ All predictions updated using XGBoost.")

if __name__ == "__main__":
    process_transactions()