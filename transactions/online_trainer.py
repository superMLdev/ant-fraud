import pandas as pd
import xgboost as xgb
import joblib
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import json

load_dotenv()

K_NEIGHBORS = int(os.getenv("K_NEIGHBORS", 5))
TRAINING_WINDOW_SIZE = int(os.getenv("TRAINING_WINDOW_SIZE", 1000))
MIN_FEEDBACK_SAMPLES = int(os.getenv("MIN_FEEDBACK_SAMPLES", 500))
DB_URL = os.getenv("DATABASE_URL")
MODEL_OUTPUT = os.getenv("MODEL_OUTPUT", "../models/xgb_fraud_model.pkl")
MODEL_LOG_PATH = os.getenv("MODEL_LOG_PATH", "../models/training_logs.jsonl")

def fetch_feedback_data():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        labeled_df = pd.read_sql(
            f"""
            SELECT amount, user_id, location, device_type, feedback
            FROM transactions
            WHERE feedback IS NOT NULL
            ORDER BY transaction_id DESC
            LIMIT {TRAINING_WINDOW_SIZE}
            """,
            conn,
        )
        print(f"📥 Labeled feedback rows: {len(labeled_df)}")

        remaining = TRAINING_WINDOW_SIZE - len(labeled_df)
        if remaining > 0:
            print(f"⚠️ Not enough labeled feedback. Using {remaining} pseudo-labeled rows from predictions.")
            unlabeled_df = pd.read_sql(
                f"""
                SELECT amount, user_id, location, device_type, prediction AS feedback
                FROM transactions
                WHERE feedback IS NULL
                ORDER BY transaction_id DESC
                LIMIT {remaining}
                """,
                conn,
            )
            df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
        else:
            df = labeled_df
    print(f"📥 Total records fetched for training: {len(df)}")
    return df

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Drop rows where feedback is missing
    df = df[df["feedback"].notnull()]

    # ✅ Encode categorical columns
    for col in ["location", "device_type"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    y = df["feedback"].astype(int)
    X = df.drop(columns=["feedback"])

    # Remove unwanted columns if they exist
    for col in ["transaction_id", "timestamp"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    print(f"📊 Original class distribution: {y.value_counts().to_dict()}")
    if len(y) < 10:
        print(f"⚠️ Too few samples ({len(y)} total). Skipping training.")
        return None, None

    # ✅ Use SMOTE to balance classes
    k_neighbors = int(os.getenv("K_NEIGHBORS", 3))
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"🧪 After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
    print(f"✅ Preprocessed feedback classes: {set(y_resampled)}")
    return X_resampled, y_resampled

def train_and_save(X, y):
    if len(y) < 10:
        print(f"⚠️ Too few samples ({len(y)}) after preprocessing. Skipping training.")
        return
    if len(set(y)) < 2:
        print("⚠️ Not enough class variety to train — need both 0 and 1 labels.")
        return
    print(f"✅ Preprocessed feedback classes: {set(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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
    print("🔁 Retrained Model Performance:\n", classification_report(y_test, y_pred))

    # Log metrics to JSONL
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": int(len(y)),
        "class_distribution": {int(k): int(v) for k, v in pd.Series(y).value_counts().items()},
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "label_counts_test": {int(k): int(v) for k, v in pd.Series(y_test).value_counts().items()},
        "label_counts_pred": {int(k): int(v) for k, v in pd.Series(y_pred).value_counts().items()}
    }

    with open(MODEL_LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Check for model drift by comparing with rolling average of past F1 scores
    # try:
    #     if os.path.exists(MODEL_LOG_PATH):
    #         with open(MODEL_LOG_PATH, "r") as f:
    #             logs = [json.loads(line) for line in f if line.strip()]
    #             recent_logs = logs[-5:]  # use last 5 runs
    #             f1_scores = [log.get("f1", 0.0) for log in recent_logs if "f1" in log]
    #             if f1_scores:
    #                 avg_f1 = sum(f1_scores) / len(f1_scores)
    #                 f1_drift = abs(f1 - avg_f1)
    #                 log_entry["recent_avg_f1"] = avg_f1
    #                 log_entry["f1_drift"] = f1_drift
    #                 F1_THRESHOLD = float(os.getenv("F1_THRESHOLD", 0.90))
    #                 log_entry["f1_threshold"] = F1_THRESHOLD
    #                 if f1 >= F1_THRESHOLD:
    #                     print(f"📉 F1 score {f1:.4f} is above threshold ({F1_THRESHOLD}). Skipping retrain.")
    #                     log_entry["note"] = "Retraining skipped due to F1 above threshold"
    #                     with open(MODEL_LOG_PATH, "a") as f:
    #                         f.write(json.dumps(log_entry) + "\n")
    #                     return
    #                 else:
    #                     print(f"⚠️ F1 score {f1:.4f} is below threshold ({F1_THRESHOLD}). Proceeding with retraining.")
    # except Exception as e:
    #     print(f"⚠️ Drift check failed: {e}")

    joblib.dump(model, MODEL_OUTPUT)
    print(f"✅ Updated model saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    df = fetch_feedback_data()
    if df.empty:
        print("ℹ️ No feedback data available yet. Skipping retrain.")
    else:
        X, y = preprocess(df)
        if X is not None and y is not None:
            from models.drift_detection import should_retrain_model, evaluate_existing_model_on_recent_data

            model = joblib.load(MODEL_OUTPUT) if os.path.exists(MODEL_OUTPUT) else None
            f1_threshold = float(os.getenv("F1_THRESHOLD", 0.90))

            should_retrain = should_retrain_model(
                f1_threshold=f1_threshold,
                model=model,
                X_new=X,
                y_new=y
            )

            if not should_retrain:
                print("✅ Drift check passed. No retraining needed.")
                exit()

            if len(y) < MIN_FEEDBACK_SAMPLES:
                print(f"⚠️ Only {len(y)} labeled samples — not enough to retrain meaningfully (requires {MIN_FEEDBACK_SAMPLES}). Skipping.")
                exit()

            train_and_save(X, y)