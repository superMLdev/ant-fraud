from fastapi import FastAPI
app = FastAPI()
from sqlalchemy import create_engine
from llm.explain_with_llm import explain_transaction
from database.insert_transaction import insert_transaction, DATABASE_URL
from models.drift_detection import evaluate_existing_model_on_recent_data
from pipeline.process_transaction import process_transaction
import uuid

@app.post("/api/explain")
def get_explanation(payload: dict):
    return {
        "explanation": explain_transaction(
            transaction_features=payload["features"],
            prediction=payload["prediction"],
            probability=payload.get("probability", None)
        )
    }

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

@app.post("/api/transaction")
def create_transaction(payload: dict):
    tid = str(uuid.uuid4())
    payload["transaction_id"] = tid
    payload["status"] = "received"
    engine = create_engine(DATABASE_URL);
    conn = engine.connect();


    insert_transaction(conn,payload)
    print(f"✅ Transaction {tid} inserted successfully.")
    # Predict fraud
    prediction_result = process_transaction(conn, tid)
    print(prediction_result)
    prediction = prediction_result["prediction"]
    probability = prediction_result.get("probability")
    return {
        "transaction_id": str(tid),
        "prediction": int(prediction),
        "probability": float(probability)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)