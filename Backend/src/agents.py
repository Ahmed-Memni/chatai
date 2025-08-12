"""
Agent setup for insurance contract management chatbot.
"""
import json
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.tools import tools
from src.chains import full_chain, graph_code_chain
from langchain_openai import OpenAI
from src.config import OPENROUTER_API_KEY, OPENROUTER_API_BASE, LLM_MODEL

llm = OpenAI(
    model_name=LLM_MODEL,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_API_BASE,
    temperature=0.7,
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

REACT_PROMPT = PromptTemplate(
    input_variables=["input", "schema_json", "history", "tools", "tool_names", "agent_scratchpad"],
    template="""
You are a database assistant.

Schema:
{schema_json}

Tools:
{tools}

Tool Names:
{tool_names}

Conversation history:
{history}

Previous agent reasoning:
{agent_scratchpad}

User question:
{input}

Respond ONLY with:

Thought: your thoughts
Action: <tool_name>
Action Input: <JSON string>

OR

Thought: your thoughts
Final Answer: <answer>
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
    # Define the full schema string here or import it if you have a constant ' talking about the get_db function u can later make it dynamic u modify it here
    schema_text = """Table clients: id (uuid), nom (text), prenom (text), date_naissance (date), email (text), telephone (text), adresse (text), created_at (timestamp with time zone)
Table contrats: id (uuid), client_id (uuid), produit_id (uuid), numero_contrat (text), date_debut (date), date_fin (date), montant_annuel (numeric), statut (text), created_at (timestamp with time zone)
Table paiements: id (uuid), contrat_id (uuid), date_paiement (date), montant (numeric), mode_paiement (text), created_at (timestamp with time zone)
Table produits_assurance: id (uuid), nom (text), description (text), created_at (timestamp with time zone)
Table sinistres: id (uuid), contrat_id (uuid), date_sinistre (date), description (text), montant_estime (numeric), statut (text), created_at (timestamp with time zone)"""

    schema_json = json.dumps(schema_text)  # Properly escaped JSON string

    # Create the agent with the prompt, injecting schema and schema_json
    prompt = REACT_PROMPT.partial(schema=schema_text, schema_json=schema_json)

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)