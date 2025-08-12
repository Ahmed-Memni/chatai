"""
Utilities for formatting chatbot responses and encoding graphs.
"""
import io
import base64
import matplotlib.pyplot as plt

def format_response(result, is_graph=False):
    """
    Format response as text or base64-encoded graph.

    Args:
        result: The result from agent execution (str or plt.Figure).
        is_graph (bool): Whether to treat the result as a graph.

    Returns:
        str: Text response or base64-encoded PNG string.
    """
    if is_graph and isinstance(result, plt.Figure):
        buf = io.BytesIO()
        result.savefig(buf, format='png')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    if "Please clarify" in result or "Failed to generate graph" in result:
        return result
    return result