# src/database.py
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from src.config import DB_URI
from functools import lru_cache

# -----------------------
# Database setup
# -----------------------
def get_db():
    """Create and return SQLDatabase instance."""
    engine = create_engine(DB_URI)
    return SQLDatabase(engine)

@lru_cache(maxsize=1)
def get_schema(_=None):
    """Get database schema with caching."""
    db = get_db()
    return db.get_table_info()

# -----------------------
# Execute SQL query
# -----------------------
def execute_query(query: str, schema: str = "insurance") -> str:
    """
    Run the SQL query and return formatted results as a string.
    """
    db = get_db()
    
    try:
        # Execute the query
        results = db.run(query)
        
        # Format results nicely for agent output
        if isinstance(results, list):
            formatted = "\n".join(
                " - ".join(str(v) for v in row.values()) if isinstance(row, dict) else str(row)
                for row in results
            )
            return formatted
        return str(results)
    
    except Exception as e:
        return f"Error executing query: {e}"
