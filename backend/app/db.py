import os
import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://careerpilot:careerpilot@localhost:5432/careerpilot").replace("postgresql+psycopg://", "postgresql://", 1)

def get_db():
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as connection:
        yield connection
