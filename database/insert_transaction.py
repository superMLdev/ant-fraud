import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

import uuid

def insert_transaction(conn, transaction: dict):
    """
    Inserts a transaction dictionary into the 'transactions' table.
    Assumes the keys of the dict match the table columns.
    """
    if not conn:
        raise ValueError("Database connection is not established.")

    columns = ', '.join(transaction.keys())
    placeholders = ', '.join([f":{k}" for k in transaction])
    query = text(f"INSERT INTO transactions ({columns}) VALUES ({placeholders})")
    print(query)
    conn.execute(query, transaction)
    conn.commit()

if __name__ == "__main__":
    sample_transaction = {
        "transaction_id": str(uuid.uuid4()),
        "user_id": 1,
        "amount": 150.75,
        "timestamp": "2025-06-20 15:30:00",
        "location": "New York",
        "device_type": "mobile"
    }

    insert_transaction(sample_transaction)
    print("Sample transaction inserted successfully.")