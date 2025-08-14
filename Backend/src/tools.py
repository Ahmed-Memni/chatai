from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from src.chains import full_chain, sql_chain, graph_code_chain, run_query
from src.database import get_db
import matplotlib.pyplot as plt
import pandas as pd

db = get_db()
repl = PythonREPL()

def run_full_chain_tool(inputs):
    """
    Accept either a string or a dict with 'question' or 'input' keys.
    Always passes {'question': ...} to full_chain.
    Returns a plain string, never a dict.
    """
    # Normalize input
    if isinstance(inputs, str):
        question = inputs
    elif isinstance(inputs, dict):
        question = inputs.get("question") or inputs.get("input")
    else:
        return "Error: invalid input type"

    if not question:
        return "Error: no question provided"

    try:
        result = full_chain.run({"question": question})
        # Ensure output is a string
        if isinstance(result, dict):
            return result.get("output", str(result))
        return str(result)
    except Exception as e:
        return f"Error executing SQL: {str(e)}"

sql_tool = Tool(
    name="sql_query",
    func=run_full_chain_tool,
    description="Execute SQL queries and return a plain text result."
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
    question = inputs if isinstance(inputs, str) else inputs.get("question", "")
    if not question:
        return "Error: no question provided for graph"
    try:
        sql_query = sql_chain.invoke({"question": question})
        sql_result = run_query(sql_query)
        code = graph_code_chain.invoke({"question": question, "sql_query": sql_query})
        repl = PythonREPL(sandbox_globals={"plt": plt, "pd": pd})
        repl.run(code)
        fig = plt.gcf()
        plt.close()
        return fig
    except Exception as e:
        return f"Failed to generate graph: {str(e)}"

graph_tool = Tool(
    name="generate_graph",
    func=generate_and_execute_graph,
    description="Generate and execute Matplotlib code based on query and data."
)

def parsing_fallback_tool(error_message):
    """Return a friendly message when parsing fails."""
    return f"Parsing failed, but here’s the last readable message: {error_message}"

parsing_tool = Tool(
    name="parsing_fallback",
    func=parsing_fallback_tool,
    description="Provide a readable fallback output when parsing fails."
)
tools = [sql_tool,graph_tool, clarification_tool, parsing_tool]