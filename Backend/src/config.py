# """
# Configuration for environment variables and constants using openai api .
# """
# import os
# from dotenv import load_dotenv

# load_dotenv(override=True)
# os.environ["PYTHONWARNINGS"] = "ignore"

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF")
# SUPABASE_HOST = os.getenv("SUPABASE_HOST")
# SUPABASE_PORT = os.getenv("SUPABASE_PORT")
# SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")

# DB_URI = f"postgresql://postgres.{SUPABASE_PROJECT_REF}:{SUPABASE_DB_PASSWORD}@{SUPABASE_HOST}:{SUPABASE_PORT}/postgres"
# LLM_MODEL = "gemini-1.5-pro"
"""
Configuration for environment variables and constants using openrouter api .
"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["PYTHONWARNINGS"] = "ignore"
# Database config
SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF")
SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_PORT = os.getenv("SUPABASE_PORT")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_URI = f"postgresql://postgres.{SUPABASE_PROJECT_REF}:{SUPABASE_DB_PASSWORD}@{SUPABASE_HOST}:{SUPABASE_PORT}/postgres"
# OpenRouter config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OpenRouter base URL (fixed, no need to change)
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
# Model name (choose as needed)
LLM_MODEL = "moonshotai/kimi-k2:free"
