from functools import lru_cache
import re
import psycopg
from pydantic import PrivateAttr
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI  # Use ChatOpenAI instead of OpenAI
from src.database import get_db
from src.config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_API_BASE

# -------------------------------
# FinalAnswerParser
# -------------------------------
class FinalAnswerParser(BaseOutputParser):
    """Extracts a clean 'Final Answer' from LLM output."""
    def parse(self, text: str) -> str:
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[*_`]', '', text)
        lines = [
            line for line in text.splitlines()
            if not re.match(r'^\s*(Thought|Action)\s*:', line, re.IGNORECASE)
        ]
        text = "\n".join(lines)
        match = re.search(r'Final Answer\s*:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

# -------------------------------
# Initialize LLM and DB
# -------------------------------
llm = ChatOpenAI(
    model_name=LLM_MODEL,  # Should be set to "moonshot/kimi" or similar in src.config
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_API_BASE,
    temperature=0,
)

db = get_db()

# -------------------------------
# Custom output parser
# -------------------------------
class StrOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        return text

# -------------------------------
# SQL validation & execution
# -------------------------------
@lru_cache(maxsize=100)
def validate_sql(query: str) -> str:
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
    if any(word in query.upper() for word in forbidden):
        raise ValueError("Only SELECT queries are allowed")
    return query

def run_query(query: str):
    try:
        validate_sql(query)
        conn_str = getattr(db, "dsn", db) if isinstance(db, str) or hasattr(db, "dsn") else None
        if not conn_str:
            raise ValueError("Invalid database object format")
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return cur.fetchall()
    except Exception as e:
        return f"Error executing query: {str(e)}"

# -------------------------------
# SQL generation chain
# -------------------------------
sql_template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
sql_prompt = ChatPromptTemplate.from_template(sql_template)

# -------------------------------
# Response generation chain
# -------------------------------
response_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant that converts SQL responses into clean natural language."
    ),
    HumanMessagePromptTemplate.from_template(
        """Based on the table schema, question, SQL query, and SQL response, write a clean natural language response.
Do NOT include tables, markdown, or extra formatting. Just list the data in plain text.

{input}

Final Answer:"""
    )
])

response_chain = RunnableSequence(response_prompt, llm, FinalAnswerParser())

# -------------------------------
# FullChain class
# -------------------------------
class FullChain:
    """Combines SQL generation, execution, and response formatting."""
    def __init__(self):
        self.sql_prompt = sql_prompt
        self.llm = llm
        self._response_chain = response_chain

    def run(self, question: str):
        if not question:
            return {"output": "Error: no question provided"}

        schema = db.get_table_info()

        # Format prompt for ChatOpenAI
        prompt_value = self.sql_prompt.format_prompt(
            question=question,
            schema=schema
        )

        # Generate SQL query using LLM
        query_response = self.llm.invoke(prompt_value)
        query = query_response.content.strip()  # Extract the content from ChatOpenAI response

        sql_response = run_query(query)

        combined_input = (
            f"Schema: {schema}\n"
            f"Question: {question}\n"
            f"SQL Query: {query}\n"
            f"SQL Response: {sql_response}"
        )

        # Format final answer
        answer = self._response_chain.invoke({"input": combined_input})
        return {"output": answer}

full_chain = FullChain()

# -------------------------------
# Graph code generation chain
# -------------------------------
graph_code_template = """Based on the user's question and SQL results, generate Python code using Matplotlib to create the best graph type.
Include:
- Imports (matplotlib.pyplot as plt, pandas as pd)
- Use SQL results directly
- Save figure to buffer (no plt.show())
- Set titles, labels, and legend
- Optionally use seaborn as sns

Question: {question}
SQL Results: {sql_results}

Generated Code:"""
graph_code_prompt = ChatPromptTemplate.from_template(graph_code_template)
graph_code_chain = RunnableSequence(graph_code_prompt, llm, StrOutputParser())