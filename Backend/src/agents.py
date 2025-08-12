"""
Agent setup for insurance contract management chatbot.
"""
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.tools import tools
from src.chains import full_chain, graph_code_chain
from langchain_openai import OpenAI
from langchain_openai import OpenAI
from src.config import OPENROUTER_API_KEY, OPENROUTER_API_BASE, LLM_MODEL

llm = OpenAI(
    model_name=LLM_MODEL,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_API_BASE,
    temperature=0,
)



router_template = """Given the user question, decide if it should be answered with text (SQL results) or a graph visualization.
If the question implies visualization (e.g., 'show distribution', 'chart of claims', 'graph premiums'), route to 'graph'.
Otherwise, route to 'text'.

Question: {input}"""
router_prompt = PromptTemplate.from_template(router_template)

destinations = [
    "text: For text-based SQL query responses.",
    "graph: For generating graphs or visualizations."
]
destination_str = "\n".join(destinations)

router_chain = LLMRouterChain.from_llm(
    llm=llm,
    prompt=PromptTemplate(
        template="""{router_template}\n\n{destination_str}\n\nOutput Format:\n{{destination_name}}\n{{input}}""",
        input_variables=["router_template", "destination_str"],
        partial_variables={"router_template": router_template, "destination_str": destination_str},
        output_parser=RouterOutputParser(),
    ),
)

# Correct prompt for create_react_agent, including required variables {tools} and {tool_names}
REACT_PROMPT = PromptTemplate(
    input_variables=["input", "history", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are a database chatbot for insurance contracts. First, decide if the user query needs text or graph output.
Use the available tools accordingly. Remember conversation history for context.

Tools:
{tools}

Tool Names:
{tool_names}

Conversation history:
{history}

Human: {input}
AI: {agent_scratchpad}
"""
)

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

decider_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains={"text": full_chain, "graph": graph_code_chain},
    default_chain=full_chain,
    silent_errors=True
)

def get_agent_executor():
    """Create and return a ReAct agent executor with memory and decider chain."""
    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)
