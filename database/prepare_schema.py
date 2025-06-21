from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
load_dotenv()

# Load your DB credentials from env or hardcode for now (replace below)
DB_URL = os.getenv("DATABASE_URL")

engine = create_engine(DB_URL)

with open("schema.sql", "r") as f:
    sql = f.read()

with engine.connect() as conn:
    conn.execute(sql)