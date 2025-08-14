"""
Chains for SQL query generation, response formatting, and graph code generation.
"""

from functools import lru_cache
from pydantic import PrivateAttr
import psycopg
import re
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema import BaseOutputParser

from langchain_openai import OpenAI
from src.database import get_db
from src.config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_API_BASE

# -------------------------------
# FinalAnswerParser
# -------------------------------
class FinalAnswerParser(BaseOutputParser):
    """
    Extracts a clean 'Final Answer' from LLM output.
    Falls back to full text if no explicit 'Final Answer:' is found.
    Strips Thoughts, Actions, markdown, and URLs.
    """
    def parse(self, text: str) -> str:
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove markdown formatting
        text = re.sub(r'[*_`]', '', text)
        # Remove lines starting with Thought: or Action:
        lines = [line for line in text.splitlines() if not re.match(r'^\s*(Thought|Action)\s*:', line, re.IGNORECASE)]
        text = "\n".join(lines)
        # Extract after 'Final Answer:'
        match = re.search(r'Final Answer\s*:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

# -------------------------------
# Initialize LLM and DB
# -------------------------------
llm = OpenAI(
    model_name=LLM_MODEL,
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
sql_chain = LLMChain(llm=llm, prompt=sql_prompt, output_parser=FinalAnswerParser())

# -------------------------------
# Response generation chain
# -------------------------------
response_template = """Based on the table schema, question, SQL query, and SQL response, write a clean natural language response.
Do NOT include tables, markdown, or extra formatting. Just list the data in plain text.

{input}

Final Answer:"""

response_prompt = PromptTemplate(
    template=response_template,
    input_variables=["input"]  # only one key
)

# Corrected response chain
full_response_chain = LLMChain(
    llm=llm,
    prompt=response_prompt,       # prompt already defines input_variables=["input"]
    output_key="response",
    output_parser=FinalAnswerParser()
)


# -------------------------------
# FullChain class
# -------------------------------
class FullChain(LLMChain):
    _response_chain: LLMChain = PrivateAttr()

    def __init__(self):
        super().__init__(llm=llm, prompt=sql_prompt, output_key="query")
        self._response_chain = full_response_chain

    def _call(self, inputs: dict):
        question = inputs.get("question")
        if not question:
            return {"output": "Error: no question provided"}

        schema = db.get_table_info()
        formatted_prompt = sql_prompt.format_prompt(question=question, schema=schema)
        query = super()._call({"input": str(formatted_prompt)})
        response = run_query(query)

        
        combined_input = f"Schema: {schema}\nQuestion: {question}\nSQL Query: {query}\nSQL Response: {response}"


        answer = self._response_chain.run({"input": combined_input})  # only "input"

        return {"output": answer}

    @property
    def input_keys(self):
        return ["question"]

    @property
    def output_keys(self):
        return ["output"]

full_chain = FullChain()

# -------------------------------
# Graph code generation chain
# -------------------------------
graph_code_template = """Based on the user's question and SQL results, generate Python code using Matplotlib to create the best graph type.
Include:
- Imports (matplotlib.pyplot as plt, pandas as pd).
- Use SQL results directly.
- Save figure to buffer (no plt.show()).
- Set titles, labels, and legend.
- Optionally use seaborn as sns for enhanced visuals.

Question: {question}
SQL Results: {sql_results}

Generated Code:"""
graph_code_prompt = ChatPromptTemplate.from_template(graph_code_template)
graph_code_chain = LLMChain(llm=llm, prompt=graph_code_prompt, output_parser=StrOutputParser())
