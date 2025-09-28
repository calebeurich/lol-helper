import os, io, boto3, json, asyncio, functools, weakref
import pandas as pd
from pathlib import Path
from botocore.config import Config
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from io import StringIO
from llm_integration.data_processing.user_data_compiling.data_collection import compile_user_data
from data_processing.user_data_compiling.pandas_user_data_aggregation import main_aggregator, InsufficientSampleError
from data_processing.recommender_system.rec_sys_functions import filter_by_champion, filter_by_criterion

from config.alias_mapping import ROLES, QUEUES, CHAMPION_CRITERIA, BINARY_REPLIES

# ============================================================
# Environment setup and Literals/Types
# ============================================================
PATCH = "patch_15_6"
load_dotenv()
# LLM bedrock variables
MODEL_ID = os.getenv("CHATBOT_LLM_ID")  
REGION = os.getenv("REGION")     
# S3 variables
BUCKET = os.getenv("BUCKET")
PREFIX  = os.getenv("PROCESSED_DATA_FOLDER")
cfg = Config(read_timeout=120, retries={"max_attempts": 3, "mode": "adaptive"})
# Constants
ALL_ROLES = ["TOP", "MIDDLE", "JUNGLE", "BOTTOM", "UTILITY"]
USER_CHAMPION_SELECTION = ["MOST_PLAYED", "HIGHEST_WR", "MANUAL"]
EXPLORE_OR_OPTIMIZE = ["EXPLORATION", "OPTIMIZATION"]
OPTIMIZATION_SCOPE = ["CLUSTER", "ROLE"]

ALL_CHAMPIONS = json.loads(
    (Path(__file__).resolve().parent / "config" / "champion_aliases.json").read_text(encoding="utf-8")
)
def get_processed_dataframe(role: str, req_type: str) -> pd.DataFrame:
    """req_type = champion_residuals or clusters"""
    key = f"{PREFIX}/clusters/{PATCH}/{role.lower()}_{req_type}_df.csv"
    s3  = boto3.client("s3", region_name=REGION, config=cfg)
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))
# ============================================================
# S3 DATA LOADING AND CACHING
# ============================================================

class S3Paths:
    def __init__(self, role: str, patch = PATCH):
        self.role = role
        self.paths = {
            "champ_semantic_tags_and_desc": f"{PREFIX}/bedrock_output/{patch}/{role}_champion_semantic_tags_and_descriptions.csv",
            "cluster_semantic_tags_and_desc": f"{PREFIX}/bedrock_output/{patch}/{role}_cluster_semantic_tags_and_descriptions.csv",
            "champion_x_role_x_user_agg": f"{PREFIX}/champion_x_role_x_user/{patch}/champion_x_role_x_user_aggregated_data.csv",    
            "champion_x_role_agg" : f"{PREFIX}/champion_x_role/{patch}/champion_x_role_x_user_aggregated_data.csv",
            "champion_residuals ": f"{PREFIX}/clusters/{patch}/{role}_champion_residuals_df.csv",
            "clusters ": f"{PREFIX}/clusters/{patch}/{role}_clusters_df.csv",
            "counter_stats_dfs_by_role": f"{PREFIX}/counter_stats_dfs_by_role/{patch}/{role}/{role}_counter_stats.csv"
        }

class S3CSVCache: 
    """Simple S3 CSV cache for multiple dataframes."""
    
    def __init__(self, bucket: str = BUCKET, region: str = REGION):

        self.bucket = bucket
        self.cache = {}  # {champion_id: {data_type: df}}

        self.s3 = boto3.client("s3", region_name=region, config=cfg)
    
    def get_all_data(self, role: str, patch: str = PATCH) -> dict[str, pd.DataFrame]:
        

        # Check if we already have all data cached
        if role in self.cache:
            return self.cache[role]
        
        paths = S3Paths(role)

        # Load all CSVs
        result = {}
        for data_type, s3_key in paths.items():
            try:
                # Download from S3
                response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
                df = pd.read_csv(io.BytesIO(response["Body"].read()))
                print(f"Loaded {s3_key}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {s3_key}: {e}")
                result[data_type] = None
        
        # Cache all data for this champion
        self.cache[role] = result
        
        return result
    
    def clear(self, role: str): # Use when restarting with same user
        """Clear cache for a champion."""
        if role in self.cache:
            del self.cache[role]
            print(f"Cleared cache for {role}")

# Initialize cache
cache = S3CSVCache()

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
    role: str
    use_own_data: Optional[bool]
    user_queue_type: Optional[str]
    user_name: Optional[str]
    user_tag_line: Optional[str]
    filtering_criteria: Optional[str]
    champion_criteria: Optional[str]
    user_champion: Optional[str]
    user_data: Optional[pd.DataFrame]
    champion_row: Optional[dict]




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

def load_s3_data(state: State) -> State:
    role = state["role"]

def ask_use_own_data(state: State) -> State:
    use_own_data = _choose_valid(
            state,
            "Say 'Do you wish to use your own data? (yes/no)' exactly",
            BINARY_REPLIES,
            "Say 'Please answer with yes or no' exactly",
            "ask_use_own_data"
    )

    if use_own_data == True:
        user_queue_type = _choose_valid(
                    state,
                    "Say 'Please select a queue type: ranked, draft or both' exactly",
                    QUEUES,
                    "Say 'Please input a valid queue type' exactly",
                    "ask_queue_type"
        )

        user_info = ask_and_return(
            state, "Say 'Please input your in-game user name and tagline exactly as it appears in your client (e.g. username#tagline)' exactly"
        )
        user_name, user_tag_line = user_info.split("#", 1)

        return {
            "use_own_data": use_own_data,
            "user_name": user_name,
            "user_tag_line": user_tag_line,
            "user_queue_type": user_queue_type
        }
    
    return {"use_own_data": use_own_data}

# Conditional
def compile_raw_user_df(state: State) -> State:
    user_name = state["user_name"]
    user_tag_line = state["user_tag_line"]
    user_queue_type = state["user_queue_type"]

    match_data_df, items_dict, user_puuid = compile_user_data(user_name, user_tag_line, user_queue_type)
    user_df = main_aggregator(raw_master_df=match_data_df, queue_type=user_queue_type, items_dict=items_dict, user_puuid=user_puuid)
    user_df_dict = user_df.to_dict(orient="records")

    return {"user_data": user_df_dict}

# Decide if we want it as a node or as an if statement later
def compile_raw_default_df(state: State) -> State:
    return


# Conditional
def ask_filtering_criteria(state: State) -> State:
    
    return {
        "filtering_criteria": _choose_valid(
        state,
        "Say 'We will select one of your champions to analyze. Choose a criterion: win rate, play rate, or choose one (min 10 games).' exactly",
        CHAMPION_CRITERIA,
        "Say 'Please input a valid option: win rate, play rate, or choose one' exactly",
        "ask_filtering_criteria"
        )
    }

# Conditional on ask_filtering_criteria and position conditional on use_own_data
def choose_champion(state: State) -> State:
    
    champion = _choose_valid(
        state,
        "Say 'Please state your desired champion' exactly",
        ALL_CHAMPIONS,
        "Say 'Please input a valid champion name' exactly",
        "choose_champion"
    )
    return {"user_champion": champion}

# Maybe not conditional?
def filter_dataframe(state: State) -> State:
    return

# Delete
def ask_queue_and_user_info(state: State) -> State:
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
    match_data_df, items_dict, user_puuid = compile_user_data(user_name, user_tag_line, user_queue_type)

    user_df = main_aggregator(raw_master_df=match_data_df, queue_type=user_queue_type, items_dict=items_dict, user_puuid=user_puuid)
    user_df = user_df.to_dict(orient="records")

    return  {
        "user_queue_type": user_queue_type,
        "user_name": user_name,
        "user_tag_line": user_tag_line,
        "user_data": user_df
    }

# Delete soon
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

# Delete
def extract_champion_data(state: State) -> State:
    champion_name = state["user_champion"]
    user_dict = state["user_data"]
    user_df = pd.DataFrame(user_dict)

    filtered_row = user_df[user_df["champion_name"] == champion_name]
    if filtered_row.empty or int(filtered_row["total_games_played_in_role"]) < MINIMUM_GAMES:
        raise InsufficientSampleError("champion games")
    
    if len(filtered_row) > 1:
        # This shouldn't happen if champions per role are unique
        raise ValueError(f"Data integrity issue: Multiple rows with champion_name {champion_name}")
    
    return {"champion_row": filtered_row.iloc[0].to_dict()}

# ============================================================
# Recommender System Functions
# ============================================================


# ============================================================
# Graph
# ============================================================
graph = StateGraph(State)
graph.add_node("ask_role", ask_role)
graph.add_node("ask_use_own_data", ask_use_own_data)
graph.add_node("ask_queue_and_user_info", ask_queue_and_user_info)
graph.add_node("ask_champion_criteria", ask_champion_criteria)

graph.add_edge(START, "ask_role")
graph.add_edge("ask_role", "ask_use_own_data")
graph.add_conditional_edges(
    "ask_use_own_data",
    lambda state: "go_queue_type" if state.get("use_own_data") else "skip_to_champion",
    {"go_queue_type": "ask_queue_and_user_info", "skip_to_champion": "ask_champion_criteria"}
)
graph.add_edge("ask_queue_and_user_info", "ask_champion_criteria")
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