"""
Chains for SQL query generation, response formatting, and graph code generation.
"""

from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain

from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.database import get_db
from src.config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_API_BASE
from functools import lru_cache

import psycopg
from langchain_openai import OpenAI

# Initialize LLM with OpenRouter API key and base
llm = OpenAI(
    model_name=LLM_MODEL,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_API_BASE,
    temperature=0,
)

db = get_db()

class StrOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        return text

@lru_cache(maxsize=100)
def validate_sql(query):
    """Validate SQL query to allow only SELECT statements."""
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
    if any(word in query.upper() for word in forbidden):
        raise ValueError("Only SELECT queries are allowed")
    return query

def run_query(query):
    """Execute a validated SQL query."""
    try:
        validate_sql(query)
        if hasattr(db, 'dsn'):
            conn_str = db.dsn
        elif isinstance(db, str):
            conn_str = db
        else:
            raise ValueError("Le format de la base de données 'db' n'est pas reconnu")

        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return cur.fetchall()
    except Exception as e:
        return f"Error executing query: {str(e)}"

def get_schema(_):
    """Get database schema with caching."""
    return db.get_table_info()

# SQL generation chain
sql_template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
sql_prompt = ChatPromptTemplate.from_template(sql_template)

sql_chain = LLMChain(llm=llm, prompt=sql_prompt, output_parser=StrOutputParser())

# Response generation chain
response_template = """Based on the table schema below, question, SQL query, and SQL response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
response_prompt = ChatPromptTemplate.from_template(response_template)

def response_run(inputs):
    schema = inputs["schema"]
    question = inputs["question"]
    query = inputs["query"]
    response = run_query(query)
    if isinstance(response, str):
        return {"response": response}
    formatted_input = {
        "schema": schema,
        "question": question,
        "query": query,
        "response": str(response),
    }
    return full_response_chain.run(formatted_input)

full_response_chain = LLMChain(llm=llm, prompt=response_prompt, output_key="response")

class FullChain(LLMChain):
    """Custom chain to handle SQL generation and response."""

    def __init__(self):
        super().__init__(llm=llm, prompt=sql_prompt, output_key="query")

    def _call(self, inputs):
        query = super()._call(inputs)
        response = run_query(query)
        formatted_input = {
            "schema": inputs["schema"],
            "question": inputs["question"],
            "query": query,
            "response": str(response),
        }
        answer = full_response_chain.run(formatted_input)
        return {"output": answer}

    @property
    def input_keys(self):
        return ["question", "schema"]

    @property
    def output_keys(self):
        return ["output"]

full_chain = FullChain()

# Graph code generation chain (returns code string)
graph_code_template = """Based on the user's question and SQL results, generate Python code using Matplotlib to create the best graph type (e.g., bar for categorical counts, pie for proportions, line for trends, scatter for correlations). Include:
- Imports (matplotlib.pyplot as plt, pandas as pd).
- Use SQL results directly.
- Save figure to buffer (no plt.show()).
- Set titles, labels, and legend.
- Optionally use seaborn (import as sns) for enhanced visuals if appropriate.

Question: {question}
SQL Results: {sql_results}

Generated Code:"""
graph_code_prompt = ChatPromptTemplate.from_template(graph_code_template)

graph_code_chain = LLMChain(llm=llm, prompt=graph_code_prompt, output_parser=StrOutputParser())
