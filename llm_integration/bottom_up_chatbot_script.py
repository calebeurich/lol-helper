import os, boto3, json
import pandas as pd
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_aws import ChatBedrock  # or langchain_openai.ChatOpenAI, etc.
from dotenv import load_dotenv
from data_processing.sagemaker_aggregation_job.submit_user_data_processing_job import submit_user_processing_job

from llm_integration.config.alias_mapping import ROLES, QUEUES, CHAMPION_CRITERIA, BINARY_REPLIES

# ============================================================
# Environment setup and Literals/Types
# ============================================================
load_dotenv()
MODEL_ID = os.getenv("CHATBOT_LLM_ID")  # Your model ID
REGION = os.getenv("REGION")            # Your region (if AWS)
ALL_ROLES = ["TOP", "MIDDLE", "JUNGLE", "BOTTOM", "UTILITY"]
USER_CHAMPION_SELECTION = ["MOST_PLAYED", "HIGHEST_WR", "MANUAL"]
EXPLORE_OR_OPTIMIZE = ["EXPLORATION", "OPTIMIZATION"]
OPTIMIZATION_SCOPE = ["CLUSTER", "ROLE"]

ALL_CHAMPIONS = json.loads(
    (Path(__file__).resolve().parent / "config" / "champion_aliases.json").read_text(encoding="utf-8")
)

# ============================================================
# LLM SETUP
# ============================================================
bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-2"))

def call_llm(prompt: str) -> str:
    """
    Minimal call to Amazon Nova Micro using Bedrock's Converse API.
    Returns only the assembled text from the response.
    """
    resp = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 100, "temperature": 0},
    )
    # Extract plain text from output blocks
    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    texts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
    return " ".join(t for t in texts if t)


# ============================================================
# Defining shape of state and LLM interaction helper functions
# ============================================================
class State(TypedDict, total=False):
    role: Optional[str]
    use_own_data: Optional[bool]
    user_queue_type: Optional[str]
    user_name: Optional[str]
    user_tag_line: Optional[str]
    champion_criteria: Optional[str]
    user_champion: Optional[str]


class RetryLimitExceeded(Exception):
    """User failed validation too many times."""
    def __init__(self, step: str): super().__init__(step); self.field = step

# Helper function to ask question to user and record answer - replace internals when moving to front end
def ask_and_return(state: State, llm_prompt: str) -> State:
    try:
        question = call_llm(state.get("prompt", llm_prompt))
        reply = input(f"{question}\n").strip().lower()
        return reply
    except Exception as e:
        return {"error": str(e)}   


# Normalize user inputs
def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


# Validate user inputs 
def _choose_valid(
    state: State, prompt: str, mapping: dict[str, str], 
    invalid_prompt: str, step: str
) -> str:
    
    tries = 3
    user_input = ask_and_return(state, prompt)
    while tries > 0:
        key = _norm(user_input)
        if key in mapping:
            return mapping[key]
        tries -= 1
        if tries <= 0:
            raise RetryLimitExceeded(step)
        user_input = ask_and_return(state, invalid_prompt)

# ============================================================
# Nodes
# ============================================================
# ----- MAIN GRAPH -----
def ask_role(state: State) -> State:
    return  {
        "role": _choose_valid(
            state,
            "Say 'What is your desired role?' exactly",
            ROLES,
            "Say 'Please input a valid role' exactly",
            "ask_role"
        )
    }


def ask_use_own_data(state: State) -> State:
    return  {
        "use_own_data": _choose_valid(
            state,
            "Say 'Do you wish to use your own data? (yes/no)' exactly",
            BINARY_REPLIES,
            "Say 'Please answer with yes or no' exactly",
            "ask_use_own_data"
        )
    }


def ask_queue_type(state: State) -> State:
    user_queue_type = _choose_valid(
                state,
                "Say 'Please select a queue type: ranked, draft or both' exactly",
                QUEUES,
                "Say 'Please input a valid queue type' exactly",
                "ask_queue_type"
    )
    #if state.get("use_own_data"):
    user_input = ask_and_return(state, "Say 'Please input your in-game user name and tagline exactly as it appears in your client (e.g. username#tagline)' exactly")
    user_name, user_tag_line = user_input.split("#", 1)
    submit_user_processing_job(user_name, user_tag_line, user_queue_type)
    print("Submitted user data to S3")

    return  {
        "user_queue_type": user_queue_type,
        "user_name": user_name,
        "user_tag_line": user_tag_line
    }


def ask_champion_criteria(state: State) -> State:

    if state.get("use_own_data"):
        criteria = _choose_valid(
            state,
            "Say 'We will select one of your champions to analyze. Choose a criterion: win rate, play rate, or choose one (min 10 games).' exactly",
            CHAMPION_CRITERIA,
            "Say 'Please input a valid option: win rate, play rate, or choose one' exactly",
            "ask_champion_criteria"
        )
        if criteria == "user_choice":
            champion = _choose_valid(
                state,
                "Say 'Please state your desired champion' exactly",
                ALL_CHAMPIONS,
                "Say 'Please input a valid champion name' exactly",
                "ask_user_champion_choice"
            )
            return {"user_champion": champion}
        return {"champion_criteria": criteria}
    
    champion = _choose_valid(
        state,
        "Say 'Please select a champion whose playstyle most represents you' exactly",
        ALL_CHAMPIONS,
        "Say 'Please input a valid champion name' exactly",
        "ask_user_champion_choice"
    )
    return {"user_champion": champion}


def ask_user_name_and_tag(state: State) -> State:
    #if state.get("use_own_data"):
    user_input = ask_and_return(state, "Say 'Please input your in-game user name and tagline exactly as it appears in your client (e.g. username#tagline)' exactly")
    user_name, user_tag_line = user_input.split("#", 1)
    submit_user_processing_job(user_name, user_tag_line)
    print("Submitted user data to S3")

# ============================================================
# Recommender System Functions
# ============================================================


# ============================================================
# Graph
# ============================================================
graph = StateGraph(State)
graph.add_node("ask_role", ask_role)
graph.add_node("ask_use_own_data", ask_use_own_data)
graph.add_node("ask_queue_type", ask_queue_type)
graph.add_node("ask_champion_criteria", ask_champion_criteria)

graph.add_edge(START, "ask_role")
graph.add_edge("ask_role", "ask_use_own_data")
graph.add_conditional_edges(
    "ask_use_own_data",
    lambda state: "go_queue_type" if state.get("use_own_data") else "skip_to_champion",
    {"go_queue_type": "ask_queue_type", "skip_to_champion": "ask_champion_criteria"}
)
graph.add_edge("ask_queue_type", "ask_champion_criteria")
graph.add_edge("ask_champion_criteria", END)
app = graph.compile()
g = app.get_graph()
g.print_ascii()
print()


# ============================================================
# Driver
# ============================================================
if __name__ == "__main__":
    result = app.invoke({})
    print(">>> Final state:", result)