import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
load_dotenv()

# Load your DB credentials from env or hardcode for now (replace below)
DB_URL = os.getenv("DATABASE_URL")
print(DB_URL)
CSV_FILE = "../data/processed/streaming_transactions.csv"

def insert_transactions():
    df = pd.read_csv(CSV_FILE)

    # Create DB connection
    engine = create_engine(DB_URL)
    conn = engine.connect()

    inserted = 0
    for _, row in df.iterrows():
        stmt = text("""
        INSERT INTO transactions (
            transaction_id, timestamp, amount, user_id,
            location, device_type, is_fraud, status
        )
        VALUES (
            :transaction_id, :timestamp, :amount, :user_id,
            :location, :device_type, :is_fraud, :status
        )
        ON CONFLICT (transaction_id) DO NOTHING;
        """)

        conn.execute(stmt, {
            "transaction_id": row["transaction_id"],
            "timestamp": row["timestamp"],
            "amount": row["amount"],
            "user_id": row["user_id"],
            "location": row["location"],
            "device_type": row["device_type"],
            "is_fraud": int(row["is_fraud"]),
            "status": "received"
        })
        inserted += 1

    conn.commit()
    conn.close()
    print(f"✅ Inserted {inserted} transactions into database.")

if __name__ == "__main__":
    insert_transactions()