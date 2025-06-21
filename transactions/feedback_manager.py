from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

def submit_feedback(transaction_id: str, actual_label: int):
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        stmt = text("""
            UPDATE transactions
            SET feedback = :feedback
            WHERE transaction_id = :transaction_id
        """)
        conn.execute(stmt, {
            "transaction_id": transaction_id,
            "feedback": actual_label
        })
        conn.commit()
    print(f"✅ Feedback recorded for {transaction_id}: Actual label = {actual_label}")

if __name__ == "__main__":
    fraud_cases = ["5e0e99c3-cf51-4494-a97e-427f031cae98","8d636e5d-8d0d-4776-a09c-e1c481056eb0"
        ,"7e8119ed-b0a1-4f68-9949-053eda6c0a04","6753e05a-8597-4dfa-9350-9bce02ea9f47"]
    # Test example
    for sample_txn_id in fraud_cases:
        submit_feedback(sample_txn_id, 1)  # Mark as fraud
    non_fraud_cases = ["61d2a650-f41d-4809-8460-0ba620bc1f49","3be26918-bc70-4906-902f-b7ab49bf7119"
        ,"bf47bba0-e8df-4e61-bae2-71923f2ab51b","b7ee9996-4650-4689-8ac9-ac91493ff531"]
    for sample_txn_id in non_fraud_cases:
        submit_feedback(sample_txn_id, 0)