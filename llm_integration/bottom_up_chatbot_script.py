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

from config.alias_mapping import ROLES, QUEUES, CHAMPION_CRITERIA

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
        inferenceConfig={"maxTokens": 100, "temperature": 0.1},
    )
    # Extract plain text from output blocks
    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    texts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
    return " ".join(t for t in texts if t)


# ============================================================
# Defining shape of state and LLM interaction helper function
# ============================================================
class State(TypedDict, total=False):
    role: Optional[str]
    use_own_data: Optional[bool]
    queue_type: Optional[str]
    champion_criteria: Optional[str]
    user_champion: Optional[str]

# Helper function to ask question to user and record answer - replace internals when moving to front end
def ask_and_return(state: State, llm_prompt: str) -> State:
    try:
        question = call_llm(state.get("prompt", llm_prompt))
        reply = input(f"{question}\n").strip().lower()
        return reply
    except Exception as e:
        return {"error": str(e)}   


# ============================================================
# Nodes
# ============================================================
# ----- MAIN GRAPH -----
def ask_role(state: State) -> State:
    reply = ask_and_return(state, "Say 'What is your desired role?' exactly")
    while True:
        if reply in ROLES:
            return {"role": ROLES[reply]}
        else:
            reply = ask_and_return(state, "Say 'Please input a valid role' exactly")


def ask_use_own_data(state: State) -> State:
    reply = ask_and_return(state, "Say 'Do you wish to use your own data? (yes/no)' exactly")
    while True:
        if reply == "yes" or reply == "y":
            return {"use_own_data": True}
        elif reply == "no" or reply == "n":
            return {"use_own_data": False}
        else:
            reply = ask_and_return(state, "Say 'Please answer with yes or no' exactly")


def ask_queue_type(state: State) -> State:
    reply = ask_and_return(state, "Say 'Please select a queue type: ranked, draft or both' exactly")
    while True:
        if reply in QUEUES:
            return {"queue_type": QUEUES[reply]}
        else:
            reply = ask_and_return(state, "Say 'Please input a valid queue type' exactly")



def ask_champion_criteria(state: State) -> State:

    def user_select_champion():
        champion = ask_and_return(state, "Say 'Please state your desired champion' exactly")
        while True:
            if champion in ALL_CHAMPIONS:
                return ALL_CHAMPIONS[champion]
            else:
                champion = ask_and_return(state, "Say 'Please input a valid champion name' exactly")

    if state.get("use_own_data"):
        reply = ask_and_return(
            state, """Say 'We will now select one of your champions' data to analyze. 
            Shall we select one based on win rate, play rate or do you wish to choose one? 
            (Minimum 10 games)' exactly"""
        )
        while True:
            if reply in CHAMPION_CRITERIA:
                if CHAMPION_CRITERIA[reply] == "user_choice":
                    return {"user_champion": user_select_champion()}
                else:
                    return {"champion_criteria": CHAMPION_CRITERIA[reply]}
            else:
                reply = ask_and_return(state, "Say 'Please input a valid queue type' exactly")
    
    else: # Consider pulling the champions list from chosen role df
        reply = ask_and_return(
            state, "Say 'Please select a champion who's playstyle most represent you' exactly"
        )
        while True:
            if reply in ALL_CHAMPIONS:
                return {"user_champion": CHAMPION_CRITERIA[reply]}
            else:
                reply = ask_and_return(state, "Say 'Please input a valid champion name' exactly")

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
    lambda state: "go_queue_type" if state.get("ask_use_own_data") else "skip_to_champion",
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
    result = app.invoke({"ask_role": "none"})
    print(">>> Final state:", result)