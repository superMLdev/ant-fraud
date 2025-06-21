import random
import time
import uuid
from datetime import datetime
import pandas as pd
import os

OUTPUT_FILE = "processed/streaming_transactions.csv"

def generate_transaction():
    return {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "amount": round(random.uniform(1, 1000), 2),
        "user_id": random.randint(1000, 9999),
        "location": random.choice(["NY", "CA", "TX", "FL", "WA"]),
        "device_type": random.choice(["mobile", "desktop", "tablet"]),
        "is_fraud": 0  # default; will overwrite later
    }

def run_stream_simulator(interval_sec=0.01, max_transactions=1000, fraud_rate=0.02):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    transactions = []

    num_frauds = max(1, int(max_transactions * fraud_rate))
    fraud_indices = set(random.sample(range(max_transactions), num_frauds))

    for i in range(max_transactions):
        txn = generate_transaction()
        txn["is_fraud"] = 1 if i in fraud_indices else 0
        transactions.append(txn)
        print(f"Generated: {txn}")
        time.sleep(interval_sec)

    df = pd.DataFrame(transactions)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ {max_transactions} transactions saved to {OUTPUT_FILE}, including {num_frauds} frauds.")

if __name__ == "__main__":
    run_stream_simulator()