"""
Chains for SQL query generation, response formatting, and graph code generation.
"""

from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from functools import lru_cache

import psycopg
from langchain_openai import OpenAI
from src.database import get_db
from src.config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_API_BASE


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
            raise ValueError("Database format 'db' is not recognized")

        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return cur.fetchall()
    except Exception as e:
        return f"Error executing query: {str(e)}"

def get_schema(_):
    """Get database schema with caching."""
    return db.get_table_info()

# SQL generation prompt and chain
sql_template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
sql_prompt = ChatPromptTemplate.from_template(sql_template)

sql_chain = LLMChain(llm=llm, prompt=sql_prompt, output_parser=StrOutputParser())

# Response generation prompt and chain
response_template = """Based on the table schema below, question, SQL query, and SQL response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
response_prompt = ChatPromptTemplate.from_template(response_template)

full_response_chain = LLMChain(llm=llm, prompt=response_prompt, output_key="response")

# Graph code generation prompt and chain
graph_code_template = """Given the user's question and the SQL query, generate python matplotlib code that creates a relevant graph.

Question: {question}
SQL Query: {sql_query}

Python matplotlib code:"""

graph_code_prompt = ChatPromptTemplate.from_template(graph_code_template)

graph_code_chain = LLMChain(
    llm=llm,
    prompt=graph_code_prompt,
    output_parser=StrOutputParser()
)

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

class FullChain(LLMChain):
    """Custom chain to handle SQL generation and response."""

    def __init__(self):
        super().__init__(llm=llm, prompt=sql_prompt, output_key="query")

    def _call(self, inputs):
        # inputs must contain "question" and "schema"
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

# Updated run_full_chain function that accepts both question and schema
def run_full_chain(inputs):
    """
    inputs should be a dict containing keys:
    - 'question': the user's natural language question
    - 'schema': the database schema string
    """
    try:
        # Pass both question and schema to full_chain
        result = full_chain.invoke(inputs)
        return result["output"]
    except Exception as e:
        return f"Error: {str(e)}"
class FullChain(LLMChain):
    """Custom chain to handle SQL generation and response."""

    def __init__(self):
        super().__init__(llm=llm, prompt=sql_prompt, output_key="query")

    def _call(self, inputs):
        print("Generating SQL query...")
        query = super()._call(inputs)  # This calls the parent class _call, which runs LLM on prompt
        print(f"SQL query generated: {query}")

        print("Running SQL query...")
        response = run_query(query)
        print(f"SQL query response: {response}")

        formatted_input = {
            "schema": inputs["schema"],
            "question": inputs["question"],
            "query": query,
            "response": str(response),
        }

        print("Generating final answer...")
        answer = full_response_chain.run(formatted_input)
        print(f"Final answer: {answer}")

        return {"output": answer}

    @property
    def input_keys(self):
        return ["question", "schema"]

    @property
    def output_keys(self):
        return ["output"]
