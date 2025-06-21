from sklearn.metrics import f1_score
import os
import joblib

MODEL_OUTPUT = os.getenv("MODEL_OUTPUT", "../models/xgb_fraud_model.pkl")


def should_retrain_model(current_f1=None, recent_avg_f1=None, f1_threshold=None, drift_threshold=None, X_new=None, y_new=None, model=None, transactions_df=None):
    """
    Decide whether to retrain based on F1 score.
    If `f1_threshold` is set, retrain if current F1 is below this value.
    Otherwise, use relative drift threshold.

    Optionally, compute current F1 from new data if model and new labels are provided.
    Or from transactions DataFrame with predictions and feedback.
    """
    if transactions_df is not None and 'prediction' in transactions_df and 'feedback' in transactions_df:
        filtered = transactions_df.dropna(subset=['prediction', 'feedback'])
        if not filtered.empty:
            current_f1 = f1_score(filtered['feedback'], filtered['prediction'])

    elif model is not None and X_new is not None and y_new is not None:
        y_pred = model.predict(X_new)
        current_f1 = f1_score(y_new, y_pred)

    if f1_threshold is not None:
        return current_f1 is not None and current_f1 < f1_threshold
    if drift_threshold is not None and recent_avg_f1 is not None:
        drift = recent_avg_f1 - current_f1
        return drift > drift_threshold
    return False


def evaluate_existing_model_on_recent_data(X, y, model_path=MODEL_OUTPUT):
    if not os.path.exists(model_path):
        print("❌ No existing model found. Cannot evaluate drift.")
        return None

    model = joblib.load(model_path)
    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred, zero_division=0)
    print(f"📉 Evaluated current model F1 on recent data: {f1:.4f}")
    return f1