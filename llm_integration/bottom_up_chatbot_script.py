import os, io, boto3, json, asyncio, functools, weakref
import pandas as pd
from pathlib import Path
from botocore.config import Config
from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from io import StringIO
from llm_integration.data_processing.user_data_compiling.data_collection import compile_user_data
from data_processing.user_data_compiling.pandas_user_data_aggregation import find_valid_queues, aggregate_user_data, InsufficientSampleError
from data_processing.recommender_system.rec_sys_functions import extract_vector

from config.alias_mapping import ROLES, QUEUES, CHAMPION_CRITERIA, BINARY_REPLIES

# ============================================================
# Environment setup and Literals/Types
# ============================================================
PATCH = "patch_15_6"
# In production these will need to be taken as inputs 
CURRENT_PATCH = "15.6"
patch_naming = f"patch_{CURRENT_PATCH.replace(".", "_")}"
PATCH_START_TIME = "1742342400" # March 19th, 2025 in timestamp seconds
PATCH_END_TIME = "1743552000" # APRIL 4TH, 2025 in timestamp
MINIMUM_GAMES = 10
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
    key = f"{PREFIX}/clusters/{patch_naming}/{role.lower()}_{req_type}_df.csv"
    s3  = boto3.client("s3", region_name=REGION, config=cfg)
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))
# ============================================================
# S3 DATA LOADING AND CACHING
# ============================================================

class S3Paths:
    def __init__(self, role: str, patch = patch_naming):
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
    
    def get_global_data(self, role: str, patch: str = patch_naming) -> Dict[str, pd.DataFrame]:
        
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
                dict_df = df.to_dict(orient="records")
                print(f"Loaded {s3_key}: {len(df)} rows")
                result[data_type] = dict_df 
            except Exception as e:
                print(f"Error loading {s3_key}: {e}")
                result[data_type] = None
        
        # Cache all data for this champion
        self.cache[role] = result
        
        return result
    
    def add_dict(self, role: str, payload: dict[str, Any]) -> None:
        """Merge arbitrary dict into the role’s cache bucket."""
        self.cache.setdefault(role, {}).update(payload)

    def put(self, role: str, key: str, value: Any) -> None:
        self.cache.setdefault(role, {})[key] = value

    def get(self, role: str, key: str | None = None) -> Any:
        bucket = self.cache.get(role, {})
        return bucket if key is None else bucket.get(key)

    def clear(self, role: str) -> None:
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
    dead_end: Optional[bool]
    user_puuid: Optional[str]
    user_queue_type: Optional[str]
    valid_queues: Optional[dict]
    user_name: Optional[str]
    user_tag_line: Optional[str]
    selection_criterion: Optional[str]
    user_champion: Optional[str]
    desired_sample: Optional[pd.DataFrame]
    champion_row: Optional[dict]

    # Data process tracking
    s3_data_loaded = Optional[bool]
    user_api_data_loaded = Optional[bool]


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
    state: State, prompt: str, mapping: Dict[str, str], 
    invalid_prompt: str, step: str
) -> str:
    
    tries = 3
    user_input = ask_and_return(state, prompt)
    while tries > 0:
        key = _norm(user_input)
        if key in mapping:
            return mapping[key] if isinstance(mapping, dict) else key
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

# Preload s3 data and cache it
def load_and_cache_s3_data(state: State) -> State:
    role = state["role"]
    cache.get_global_data(role=role)
    return {"s3_data_loaded":True}


def ask_use_own_data(state: State) -> State:
    use_own_data = _choose_valid(
            state,
            "Say 'Do you wish to use your own data (yes/no)? A minimum of 10 games (any Summoner's Rift queue) with at least 1 champion in the desired patch is required' exactly",
            BINARY_REPLIES,
            "Say 'Please answer with yes or no' exactly",
            "ask_use_own_data"
    )

    if use_own_data == True:

        user_info = ask_and_return(
            state, "Say 'Please input your in-game user name and tagline exactly as it appears in your client (e.g. username#tagline)' exactly"
        )
        user_name, user_tag_line = user_info.split("#", 1)

        return {
            "use_own_data": use_own_data,
            "user_name": user_name,
            "user_tag_line": user_tag_line,
            "desired_sample": "user_api_df"
        }
    
    return {"use_own_data": use_own_data, "desired_sample": "champion_x_role_agg"}


# Conditional
def find_valid_queues(state: State) -> State:

    if not state["use_own_data"]: # Skip process if not using user data
        return {}

    user_name = state["user_name"]
    user_tag_line = state["user_tag_line"]
    role = state["role"].upper()
    # Consider caching compiled data
    try:
        match_data_df, items_dict, user_puuid = compile_user_data(
            user_name, user_tag_line, PATCH_START_TIME, PATCH_END_TIME, MINIMUM_GAMES, CURRENT_PATCH
        )
    except InsufficientSampleError:
        return {"user_api_data_loaded" : "Insufficient total games"}
    
    merged_df, valid_queues, all_item_tags, all_summoner_spells = find_valid_queues(match_data_df, items_dict, role, MINIMUM_GAMES)
    cache.put(role=state["role"], key="all_item_tags", value=all_item_tags)
    cache.put(role=state["role"], key="all_summoner_spells", value=all_summoner_spells) 

    if valid_queues:
        user_queue_type = _choose_valid(
            state,
            f"Say 'You have enough data for the following queue(s): {", ".join(valid_queues)}. Please select one.' exactly",
            valid_queues, # Need to account for spelling mistakes
            "Say 'Please input a valid queue type' exactly",
            "compile_user_df"
        )
    else:
        key = _choose_valid(
            state,
            f"Say 'The account `{user_name}` does not meet the minimum requirement of 10 games in this patch for the analysis, would you like to proceed using global data instead?' exactly",
            BINARY_REPLIES,
            "Say 'Please answer with yes or no' exactly",
            "compile_user_df"
        )
        return {
            "use_own_data": False, "user_api_data_loaded" : "insufficient_total_games", "desired_sample": "champion_x_role_agg"
        } if key else {"dead_end": True}

    user_df_dict = merged_df.to_dict(orient="records")

    cache.put(role=state["role"], key="user_api_df", value=user_df_dict)

    return {"user_queue_type": user_queue_type, "user_api_data_loaded" : True, "valid_queues": valid_queues, "user_puuid": user_puuid}


def ask_selection_criterion(state: State) -> State: # Also compile_user_df?

    def choose_champion(valid_champions):
        return _choose_valid(
            state,
            f"Say 'The following champions have sufficient data, please select one: {", ".join(valid_champions)}' exactly",
            valid_champions,
            "Say 'Please input a valid champion name' exactly",
            "ask_selection_criterion"
        )

    user_queue_type = state["user_queue_type"]
    data_loaded = state["user_api_data_loaded"]
    role = state["role"]
    dataset = state["desired_sample"]
    user_puuid = state["user_puuid"]

    criterion = _choose_valid(
        state,
        "Say 'We will select a representative champions to analyze. Choose a criterion: win rate, play rate, or choose one (min 10 games).' exactly",
        CHAMPION_CRITERIA,
        "Say 'Please input a valid option: win rate, play rate, or choose one' exactly",
        "ask_selection_criterion"
    )

    df = pd.DataFrame(cache.get(role, dataset))
    if data_loaded:
        all_item_tags = cache.get(role, "all_item_tags")
        all_summoner_spells = cache.get(role, "all_summoner_spells")

        df = aggregate_user_data(df, all_item_tags, all_summoner_spells, user_puuid, user_queue_type, MINIMUM_GAMES)

    valid_champions = df["champion_name"].tolist()

    valid_champions = [ALL_CHAMPIONS[champ_name] for champ_name in valid_champions]

    if criterion == "user_choice":
        return {"user_champion": choose_champion(valid_champions), "selection_criterion": criterion}
    
    # Consider adding function to extract user vector here

    return {
        "selection_criterion": _choose_valid(
        state,
        "Say 'We will select a representative champions to analyze. Choose a criterion: win rate, play rate, or choose one (min 10 games).' exactly",
        CHAMPION_CRITERIA,
        "Say 'Please input a valid option: win rate, play rate, or choose one' exactly",
        "ask_selection_criterion"
        )
    }

# Maybe not conditional?
def extract_user_vector(state: State) -> State:
    criterion, role = state["selection_criterion"], state["role"]
    user_name = state["user_name"]
    
    if state["user_api_data_loaded"] == True:
        key = state["desired_sample"]
    elif state["user_api_data_loaded"] == "Insufficient total games":
        key = _choose_valid(
            state,
            f"Say 'The account `{user_name}` does not meet the minimum requirement of 10 games (ranked and draft combined) in this patch for the analysis, would you like to proceed using global data instead?' exactly",
            BINARY_REPLIES,
            "Say 'Please answer with yes or no' exactly",
            "extract_user_vector"
        )
    df = cache.get(role, key)

    tries = 0
    while tries < 3:
        try:
            user_vector = extract_vector(df, criterion, MINIMUM_GAMES)
        except InsufficientSampleError as e:
            print(e)
            

    return




# ============================================================
# Recommender System Functions
# ============================================================



# ============================================================
# Graph
# ============================================================
graph = StateGraph(State)
graph.add_node("ask_role", ask_role)
graph.add_node("load_and_cache_s3_data", load_and_cache_s3_data)
graph.add_node("ask_use_own_data", ask_use_own_data)
graph.add_node("ask_selection_criterion", ask_selection_criterion)
graph.add_node("compile_user_df", compile_user_df)
graph.add_node("extract_user_vector", extract_user_vector)

graph.add_edge(START, "ask_role")
# Run in parallel so S3 data pull does not slow process down
graph.add_edge("ask_role", "load_and_cache_s3_data") # Dead end
graph.add_edge("ask_role", "ask_use_own_data")
# Run in parallel with compile_user_df being conditional on use_own_data
graph.add_edge("ask_use_own_data", "compile_user_df")
graph.add_edge("ask_use_own_data", "ask_selection_criterion")
# Merging parallel nodes
graph.add_conditional_edges(
    "compile_user_df",
    lambda state: "go_end" if state.get("dead_end") else "skip_to_vector",
    {"go_end": END, "skip_to_vector": "extract_user_vector"}
)
graph.add_edge("ask_selection_criterion", "extract_user_vector")


graph.add_conditional_edges(
    "ask_use_own_data",
    lambda state: "go_queue_type" if state.get("use_own_data") else "skip_to_champion",
    {"go_queue_type": "ask_queue_and_user_info", "skip_to_champion": "ask_champion_criterion"}
)
graph.add_edge("ask_queue_and_user_info", "ask_champion_criterion")
graph.add_edge("ask_champion_criterion", END)
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