"""
Database connection setup with async support.
"""
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from src.config import DB_URI
from functools import lru_cache

def get_db():
    """Create and return SQLDatabase instance."""
    engine = create_engine(DB_URI)
    return SQLDatabase(engine)

@lru_cache(maxsize=1)
def get_schema(_):
    """Get database schema with caching."""
    db = get_db()
    return db.get_table_info()