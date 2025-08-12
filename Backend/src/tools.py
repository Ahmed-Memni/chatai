"""
Tools for SQL execution, clarification, and graph generation.
"""
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from src.chains import full_chain, sql_chain, graph_code_chain, run_query
from src.database import get_db
import matplotlib.pyplot as plt
import pandas as pd

db = get_db()
repl = PythonREPL()

def run_full_chain(question):
    """Execute SQL query and return natural language response."""
    try:
        return full_chain.invoke({"question": question})
    except Exception as e:
        return f"Error: {str(e)}"

sql_tool = Tool(
    name="sql_query",
    func=run_full_chain,
    description="Use this to answer database queries in natural language by generating and executing SQL."
)

def ask_clarification(query):
    """Request clarification for ambiguous queries."""
    schema = db.get_table_info()
    if "name" in query.lower() and "SELECT name" in schema:
        names = run_query("SELECT DISTINCT name FROM clients LIMIT 5")
        return f"Multiple clients found: {names}. Which one do you mean for '{query}'?"
    return f"Please clarify: {query} (e.g., specify client or policy details)."

clarification_tool = Tool(
    name="ask_clarification",
    func=ask_clarification,
    description="Use this when the query is ambiguous and needs user clarification."
)

def generate_and_execute_graph(inputs):
    """Generate and execute Matplotlib code for graphs."""
    sql_query = sql_chain.invoke({"question": inputs["question"]})
    sql_result = run_query(sql_query)
    code = graph_code_chain.invoke({"question": inputs["question"], "sql_query": sql_query})
    try:
        repl = PythonREPL(sandbox_globals={"plt": plt, "pd": pd})
        repl.run(code)
        fig = plt.gcf()
        plt.close()
        return fig
    except Exception as e:
        return f"Failed to generate graph: {sql_result}"

graph_tool = Tool(
    name="generate_graph",
    func=generate_and_execute_graph,
    description="Use this to generate and execute Matplotlib code for graphs based on query and data."
)

tools = [sql_tool, clarification_tool, graph_tool]