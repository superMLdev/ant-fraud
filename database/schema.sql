CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY,
    timestamp TIMESTAMP,
    amount FLOAT,
    user_id INT,
    location TEXT,
    device_type TEXT,
    is_fraud INT, -- optional: use NULL if unknown
    prediction INT,
    status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);