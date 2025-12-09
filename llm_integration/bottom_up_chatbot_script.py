# ──────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Union

# ──────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────
import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

# ──────────────────────────────────────────
# Local Application Imports
# ──────────────────────────────────────────
from llm_integration.config.alias_mapping import (
    BINARY_REPLIES,
    CHAMPION_CRITERIA,
    METHODOLOGIES,
    ROLES,
)
from llm_integration.data_processing.recommender_system.rec_sys_functions import (
    extract_vector,
    filter_user_and_global_dfs,
    find_best_alternative,
    find_cluster_representatives,
    find_recs_within_cluster,
    recommend_champions_from_main,
    similar_playstyle_users,
)
from llm_integration.data_processing.user_data_compiling.data_collection import (
    compile_user_data,
)
from llm_integration.data_processing.user_data_compiling.pandas_user_data_aggregation import (
    InsufficientSampleError,
    aggregate_user_data,
    find_valid_queues,
)

# ──────────────────────────────────────────
# Rapidfuzz Setup
# ──────────────────────────────────────────
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    _HAS_RAPIDFUZZ = False

# ──────────────────────────────────────────
# Environment setup and Literals/Types
# ──────────────────────────────────────────
PATCH = "patch_15_6"
# In production these will need to be taken as inputs 
CURRENT_PATCH = "15.6"
patch_naming = f"patch_{CURRENT_PATCH.replace('.', '_')}"
PATCH_START_TIME = "1742342400" # March 19th, 2025 in timestamp seconds
PATCH_END_TIME = "1743552000" # APRIL 4TH, 2025 in timestamp

MINIMUM_GAMES = 1
TOP_K = 3 # Number of recommended champions

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
# ──────────────────────────────────────────
# S3 DATA LOADING AND CACHING
# ──────────────────────────────────────────

class S3Paths:
    def __init__(self, role: str, patch = patch_naming):
        self.role = role
        self.paths = {
            "champ_semantic_tags_and_desc": f"{PREFIX}/bedrock_output/{patch}/{role.lower()}_champion_semantic_tags_and_descriptions.parquet",
            "cluster_semantic_tags_and_desc": f"{PREFIX}/bedrock_output/{patch}/{role.lower()}_cluster_semantic_tags_and_descriptions.parquet",
            "champion_x_role_x_user_agg": f"{PREFIX}/champion_x_role_x_user/{patch}/{role.lower()}_champion_x_role_x_user_aggregated_data.parquet",    
            "champion_x_role_agg": f"{PREFIX}/champion_x_role/{patch}/{role.lower()}_champion_x_role_aggregated_data.parquet",
            "champion_residuals": f"{PREFIX}/clusters/{patch}/{role.lower()}_champion_residuals_df.parquet",
            "clusters": f"{PREFIX}/clusters/{patch}/{role.lower()}_clusters_df.parquet",
            "counter_stats_dfs_by_role": f"{PREFIX}/counter_stats_dfs_by_role/{patch}/{role.lower()}/{role.lower()}_counter_stats.parquet",
            "items_dict": "data_mapping/item_id_tags.json"
        }

    def items(self):
        return self.paths.items()

class S3Cache: 
    """Simple S3 parquet cache for multiple dataframes."""
    
    def __init__(self, bucket: str = BUCKET, region: str = REGION):
        self.bucket = bucket
        self.cache = {}  # {role: {data_type: df}}
        self.s3 = boto3.client("s3", region_name=region, config=cfg)
    
    def get_global_data(self, role: str, patch: str = patch_naming):
        # Return if already cached
        if role in self.cache:
            return self.cache[role]
        
        paths = S3Paths(role, patch)  # use the patch param

        result = {}
        for data_type, s3_key in paths.items():
            try:
                response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            except Exception as e:
                print(f"Error loading {s3_key}: {e}")
                result[data_type] = None
                continue  # don't use an undefined response

            if s3_key.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(response["Body"].read()))
                print(f"Loaded DataFrame {s3_key}: {len(df)} rows")
                result[data_type] = df
            else:
                data = json.load(io.BytesIO(response["Body"].read()))
                print(f"Loaded Dict {s3_key}")
                result[data_type] = data
        
        self.cache[role] = result
        return result

    def get(self, role: str, data_type: str):
        return self.cache.get(role).get(data_type)
    
    def put(self, role: str, key: str, value: Any):
        self.cache.setdefault(role)[key] = value 

# Initialize cache
cache = S3Cache()

# ──────────────────────────────────────────
# LLM SETUP
# ──────────────────────────────────────────
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


def find_similar_champions_llm(
    filtered_df: pd.DataFrame,
    target_tags: str,
    target_description: str,
    top_k: int = 3
):
    similarities = []

    for _, row in filtered_df.iterrows():
        champion_name = row["champion_name"]
        if pd.isna(champion_name):
            continue

        champ_tags = row.get("tags", "")
        champ_description = row.get("description", "")

        prompt = f"""
You are evaluating semantic similarity between League of Legends champions using only English-language descriptions.

TARGET CHAMPION:
Tags: {target_tags}
Description: {target_description}

CANDIDATE CHAMPION:
Name: {champion_name}
Tags: {champ_tags}
Description: {champ_description}

Task:
Rate how semantically similar the CANDIDATE champion is to the TARGET champion.
Focus ONLY on meaning, themes, playstyle concepts, and semantic overlap.

Output:
Respond ONLY with a number between 0 and 1, where:
- 1 = extremely similar
- 0 = completely unrelated
"""

        response = call_llm(prompt)

        # Extract numeric score safely
        try:
            score = float(response.strip())
        except:
            score = 0.0

        similarities.append((champion_name, score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# ─────────────────────────────────────────────────────────────
# Defining shape of state and LLM interaction helper functions
# ─────────────────────────────────────────────────────────────
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
    user_vector: Optional[dict]
    decision_making_method: Optional[str]
    champion_description: Optional[str]
    champion_tags: Optional[str]
    cluster_description : Optional[str]
    cluster_tags : Optional[str]
    cluster_id: Optional[str]

    similar_playstyle_rec: Optional[list]
    same_main_rec: Optional[list]

    # Data process tracking
    s3_data_loaded: Optional[bool]
    user_api_data_loaded: Optional[bool]
    
    final_rec: Optional[Sequence]
    end: Optional[bool]


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
def _norm(s):
    if isinstance(s, str):
        return " ".join(s.lower().strip().split())
    return s


def _best_matches(user_text: str, choices: list[str], topn: int = 3):
    """
    Evaluate spelling closeness to choices
    Uses RapidFuzz if available; otherwise difflib ratio in 0..100.
    """
    if _HAS_RAPIDFUZZ:
        scores = []
        for c in choices:
            # Use simple ratio for better character-level matching
            s1 = fuzz.ratio(user_text, c)
            
            # Only use token-based for longer strings where it makes sense
            if len(user_text) > 6 and len(c) > 6:
                s2 = fuzz.WRatio(user_text, c)
                s3 = fuzz.token_set_ratio(user_text, c)
                scores.append((c, max(s1, s2, s3)))
            else:
                # For short strings, just use simple ratio
                scores.append((c, s1))
                
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:topn]
    else:
        # difflib
        candidates = difflib.get_close_matches(user_text, choices, n=topn, cutoff=0.0)
        scores = []
        for c in candidates:
            score = difflib.SequenceMatcher(None, user_text, c).ratio() * 100.0
            scores.append((c, score))
        # pad with other choices if needed
        if len(scores) < topn:
            pool = [c for c in choices if c not in {x for x, _ in scores}]
            for c in pool[: topn - len(scores)]:
                score = difflib.SequenceMatcher(None, user_text, c).ratio() * 100.0
                scores.append((c, score))
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:topn]

# Validate user inputs 
def _choose_valid(
    state: State,
    prompt: str,
    mapping: Union[Dict[str, str], Sequence[str]],
    invalid_prompt: str,
    step: str
) -> str:
    """
    Supports:
      - dict[str, str]: match on keys, return mapping[key]
      - sequence[str]:  match on items, return the chosen item
    """
    # Build normalized lookup tables
    if isinstance(mapping, dict):
        # Normalized key -> (display_key, return_value)
        norm_to_pair = {_norm(k): (k, v) for k, v in mapping.items()}
        choices_norm = list(norm_to_pair.keys())
        display_from_norm = lambda nk: norm_to_pair[nk][0]
        value_from_norm   = lambda nk: norm_to_pair[nk][1]
    else:
        # Treat as list/tuple of choices
        norm_to_val = {}
        for item in mapping:
            norm_to_val[_norm(item)] = item  
        choices_norm = list(norm_to_val.keys())
        display_from_norm = lambda nk: norm_to_val[nk]
        value_from_norm   = lambda nk: norm_to_val[nk]

    tries = 3
    first = True
    while tries > 0:
        user_input = ask_and_return(state, prompt if first else invalid_prompt)
        first = False
        key_norm = _norm(user_input)

        # exact normalized match
        if str(key_norm) in choices_norm:
            return value_from_norm(key_norm)

        # Check for exact prefix matches first (handles "hwei" -> "Hwei" perfectly)
        prefix_matches = [c for c in choices_norm if c.startswith(key_norm)]
        if len(prefix_matches) == 1:
            return value_from_norm(prefix_matches[0])

        # fuzzy suggestions over normalized choices
        candidates = _best_matches(key_norm, choices_norm, topn=3)
        if candidates:
            best_norm, best_score = candidates[0]
            second_score = candidates[1][1] if len(candidates) > 1 else 0.0

            # For short strings (like champion names), be much stricter
            if len(key_norm) <= 6:  # Short input
                # Check if first two characters match
                first_chars_match = (len(key_norm) >= 2 and len(best_norm) >= 2 and 
                                   key_norm[:2] == best_norm[:2])
                
                if first_chars_match:
                    # More lenient if prefix matches
                    min_score = 85
                    min_gap = 10
                else:
                    # Very strict if prefix doesn't match
                    min_score = 95
                    min_gap = 15
            else:
                # Original thresholds for longer strings
                min_score = 90
                min_gap = 5

            # Auto-accept with dynamic thresholds
            if best_score >= min_score or (best_score >= (min_score - 10) and (best_score - second_score) >= min_gap):
                return value_from_norm(best_norm)

            # Don't offer disambiguation for poor matches
            if best_score < 70:
                tries -= 1
                continue

            # disambiguate with top-3
            options = [display_from_norm(nk) for nk, _ in candidates]
            listing = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
            choice = ask_and_return(
                state,
                f"I found similar options. Which did you mean?\n{listing}\nType 1-{len(options)}, or retype your choice:"
            )
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(options):
                    # map back through normalization to be consistent
                    return options[idx - 1]

        tries -= 1

    raise RetryLimitExceeded(step)


# ──────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────
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
            f"Say 'Do you wish to use your own data (yes/no)? A minimum of {MINIMUM_GAMES} games (any Summoner's Rift queue) with at least 1 champion in the desired patch is required' exactly",
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
def get_user_queue(state: State) -> State:

    if not state["use_own_data"]: # Skip process if not using user data
        return {}

    user_name = state["user_name"]
    user_tag_line = state["user_tag_line"]
    role = state["role"]
    # Consider caching compiled data
    try:
        match_data_df, user_puuid, num_games_per_queue = compile_user_data(
            user_name, user_tag_line, PATCH_START_TIME, PATCH_END_TIME, CURRENT_PATCH
        )
    except InsufficientSampleError:
        key = _choose_valid(
            state,
            f"Say 'The account `{user_name}` does not meet the minimum requirement of {MINIMUM_GAMES} games in this patch for the analysis, would you like to proceed using global data instead?' exactly",
            BINARY_REPLIES,
            "Say 'Please answer with yes or no' exactly",
            "compile_user_df"
        )
        return {
            "use_own_data": False, "user_api_data_loaded" : "insufficient_total_games", "desired_sample": "champion_x_role_agg"
        } if key else {"dead_end": True}
    
    items_dict = cache.get(role, "items_dict")

    merged_df, valid_queues, all_item_tags, all_summoner_spells = find_valid_queues(match_data_df, items_dict, user_puuid, role, MINIMUM_GAMES)
    cache.put(role=state["role"], key="all_item_tags", value=all_item_tags)
    cache.put(role=state["role"], key="all_summoner_spells", value=all_summoner_spells) 

    valid_queues = list(valid_queues.keys())
    if valid_queues:
        user_queue_type = _choose_valid(
            state,
            f"Say 'You have enough data for the following queue(s): {', '.join(valid_queues)}. Please select one.' exactly",
            valid_queues, 
            "Say 'Please input a valid queue type' exactly",
            "compile_user_df"
        )
        aggregated_df = aggregate_user_data(merged_df, all_item_tags, all_summoner_spells, user_queue_type, MINIMUM_GAMES)
        cache.put(role=state["role"], key="user_api_df", value=aggregated_df)
        return {"user_queue_type": user_queue_type, "user_api_data_loaded" : True, "valid_queues": valid_queues, "user_puuid": user_puuid}
    
    else:
        key = _choose_valid(
            state,
            f"Say 'The account `{user_name}` does not meet the minimum requirement of {MINIMUM_GAMES} games in this patch for the analysis, would you like to proceed using global data instead?' exactly",
            BINARY_REPLIES,
            "Say 'Please answer with yes or no' exactly",
            "compile_user_df"
        )
        return {
            "use_own_data": False, "user_api_data_loaded" : "insufficient_total_games", "desired_sample": "champion_x_role_agg"
        } if key else {"dead_end": True}


def compile_user_vector(state: State) -> State: # Also compile_user_df?

    def choose_champion(valid_champions):
        return _choose_valid(
            state,
            f"Say 'The following champions have sufficient data, please select one: {', '.join(valid_champions)}' exactly",
            valid_champions,
            "Say 'Please input a valid champion name' exactly",
            "compile_user_vector"
        )

    role = state["role"]
    dataset = state["desired_sample"]

    criterion = _choose_valid(
        state,
        f"Say 'We will select a representative champions to analyze. Choose a criterion: win rate, play rate, or manual selection (min {MINIMUM_GAMES} games).' exactly",
        CHAMPION_CRITERIA,
        "Say 'Please input a valid option: win rate, play rate, or choose a champion' exactly",
        "compile_user_vector"
    )

    df = pd.DataFrame(cache.get(role, dataset))

    if criterion == "user_choice":
        valid_champions = df["champion_name"].tolist()
        valid_champions = [ALL_CHAMPIONS[champ_name] for champ_name in valid_champions]
        user_champion = choose_champion(valid_champions)
        user_vector, _ = extract_vector(df, criterion, user_champion, MINIMUM_GAMES)

    else:
        user_vector_or_names, n = extract_vector(df, criterion, None, MINIMUM_GAMES)
        if n > 1:
            names = user_vector_or_names
            chosen = _choose_valid(
                state,
                f"Say 'There are more than one champions meeting this criteria. Please choose one of the following: {', '.join(names)}' exactly",
                names,
                f"Say 'Please input a valid option: {', '.join(names)}' exactly",
                "compile_user_vector"
            )
            user_vector = df.loc[df["champion_name"] == chosen].iloc[[0]]
            user_champion = chosen
        else:
            user_vector = user_vector_or_names  
            user_champion = user_vector.iloc[0]["champion_name"]
    

    return {"user_champion": user_champion, "selection_criterion": criterion, "user_vector": user_vector}#.to_dict(orient="records")}


def pull_tags_and_descriptions(state: State) -> State:
    role = state["role"]
    user_champion = state["user_champion"]

    champion_semantics_df = pd.DataFrame(cache.get(role, "champ_semantic_tags_and_desc"))
    champion_description, champion_tags = champion_semantics_df.loc[
        champion_semantics_df["id"] == f"{user_champion}__{role}", ["description", "tags"]
    ].squeeze()

    _ = _choose_valid( # Placeholder, should be replaced with button on front end
        state,
        f"""Say 'The champion {user_champion} has the following tags:
        {champion_tags}
        And the following description:
        {champion_description}' exactly.
        Please type Continue when ready.""",
        ["Continue"],
        "Say 'Please type Continue' exactly",
        "compile_user_df"
    )

    cluster_df = pd.DataFrame(cache.get(role, "cluster_semantic_tags_and_desc"))
    cluster_df["id"] = pd.to_numeric(cluster_df["id"], errors="coerce")
    champion_residuals_df = pd.DataFrame(cache.get(role, "champion_residuals"))

    cluster_id = int(champion_residuals_df.loc[
        champion_residuals_df["champion_name"] == user_champion, ["cluster"]
    ].squeeze())
    cluster_df["id"] = pd.to_numeric(cluster_df["id"], errors="coerce")

    row = cluster_df.loc[cluster_df["id"] == cluster_id, ["description", "tags"]]

    if row.empty:
        raise ValueError(f"No cluster found for id={cluster_id}")

    cluster_description, cluster_tags = row.iloc[0]

    _ = _choose_valid( # Placeholder, should be replaced with button on front end
        state,
        f"""Say 'The champion {user_champion} belongs to the cluster with the following tags:
        {cluster_tags}
        And the following description:
        {cluster_description}.
        Please type Continue when ready' exactly.""",
        ["Continue"],
        "Say 'Please type Continue' exactly",
        "compile_user_df"
    )

    return {
        "champion_description": champion_description, "champion_tags": champion_tags,
        "cluster_description": cluster_description, "cluster_tags": cluster_tags,
        "cluster_id": cluster_id
    }


def decision_making_method(state: State) -> State:
    
    user_choice = _choose_valid(
        state,
        f"Say 'We will now begin the recommendation analysis. Please choose one of the following methodologies: {', '.join(METHODOLOGIES)}' exactly",
        METHODOLOGIES,
        f"Say 'Please input a valid option: {', '.join(METHODOLOGIES)}' exactly",
        "decision_making_method"
    )
    
    return {"decision_making_method": user_choice}


def collaborative_filtering(state: State) -> State:
    role = state["role"]
    user_champion = state["user_champion"]
    global_users_df = pd.DataFrame(cache.get(role, "champion_x_role_x_user_agg"))
    user_vector = state["user_vector"]

    filtered_user_vector, filtered_global_user_df = filter_user_and_global_dfs(role, user_vector, global_users_df, MINIMUM_GAMES)
    similar_playstyle_rec = similar_playstyle_users(filtered_user_vector, filtered_global_user_df, TOP_K)
    same_main_rec = recommend_champions_from_main(user_champion, filtered_global_user_df, TOP_K)
    print(f"similar_playstyle_rec: {similar_playstyle_rec}")
    print(f"same_main_rec: {same_main_rec}")
    recommendation = {
        "similiar playstyle": similar_playstyle_rec,
        "same mains": same_main_rec                
    }

    return {
        "similar_playstyle_rec": similar_playstyle_rec, "same_main_rec":same_main_rec, 
            "end": True, "final_rec": recommendation
        }

def mathematical_optimization(state: State) -> State:

    role = state["role"]
    cluster_id = state["cluster_id"]
    user_champion = state["user_champion"]
    champion_residuals_df = pd.DataFrame(cache.get(role, "champion_residuals"))

    scope = _choose_valid(
        state,
        f"Say 'Do we want to stay within cluster scope or look at whole role' exactly",
        {"Within cluster": "cluster_scope", "Whole role": "role_scope"},
        f"Say 'Please input a valid option: {', '.join(['Within cluster', 'Whole role'])}' exactly",
        "mathematical_optimization"
    )

    filtered_residuals_df = champion_residuals_df.loc[
            champion_residuals_df["cluster"] == cluster_id
    ] if scope == "cluster_scope" else champion_residuals_df
    
    cache.put(role=state["role"], key="filtered_residuals_df", value=filtered_residuals_df)

    win_rate = _choose_valid(
        state,
        f"Say 'Do we want to round your champion pool based on win rate?' exactly",
        BINARY_REPLIES,
        f"Say 'Please answer with yes or no' exactly",
        "mathematical_optimization"
    )
    if win_rate:
        counter_stats_df = pd.DataFrame(cache.get(role, "counter_stats_dfs_by_role"))

        if scope == "cluster_scope":
            counter_stats_df = counter_stats_df.loc[counter_stats_df["champion_name"].isin(filtered_residuals_df["champion_name"])]
        recommendation = find_best_alternative(counter_stats_df, user_champion, MINIMUM_GAMES)
        print(recommendation)
        return {"end": True, "final_rec": recommendation}
    
    else:
        champion_x_role_df = pd.DataFrame(cache.get(role, "champion_x_role_agg"))
        champion_x_role_df = champion_x_role_df[champion_x_role_df["champion_name"].isin(filtered_residuals_df["champion_name"])]
        similar_champs, different_champs = find_recs_within_cluster(champion_x_role_df, user_champion)
        return {
            "end": True, "final_rec": {"similar":similar_champs, "different":different_champs}
        }


def natural_language_exploration(state: State) -> State:
    role = state["role"]
    user_champion = state["user_champion"]

    cluster_semantics_df = pd.DataFrame(cache.get(role, "cluster_semantic_tags_and_desc"))
    cluster_semantics_df["id"] = cluster_semantics_df["id"].astype(str)
    
    print(f"Please find all {role.capitalize()} clusters with their descriptions and semantic tags below:")
    print(cluster_semantics_df)

    cluster_id = _choose_valid(
        state,
        f"Ask user to select the cluster id of the cluster they preferred",
        cluster_semantics_df["id"].tolist(),
        f"Say 'Please select a valid cluster id' exactly",
        "natural_language_exploration"
    )

    champion_residuals_df = pd.DataFrame(cache.get(role, "champion_residuals"))

    filtered_residuals_df = champion_residuals_df.loc[
        (champion_residuals_df["cluster"] == cluster_id) |
        (champion_residuals_df["champion_name"] == user_champion)
    ]
    cluster_champions = filtered_residuals_df["champion_name"].tolist()

    similarity_criterion = _choose_valid(
        state,
        f"Ask user if they would prefer champions from this cluster that are most similar to theirs",
        BINARY_REPLIES,
        f"Say 'Please answer with yes or no' exactly",
        "natural_language_exploration"
    )

    if similarity_criterion:
        champion_semantics_df = pd.DataFrame(cache.get(role, "champ_semantic_tags_and_desc"))
        champion_semantics_df["champion_name"] = champion_semantics_df["id"].str.split("__", n=1).str[0]

        champion_semantics_df = (
            champion_semantics_df[champion_semantics_df["champion_name"].isin(cluster_champions) & 
                                  champion_semantics_df["champion_name"] != user_champion
            ]
        )

        champion_description = state["champion_description"]
        champion_tags = state["champion_tags"]

        recommendation = find_similar_champions_llm(
            champion_semantics_df,
            champion_description,
            champion_tags,
            top_k=3
        )

    else:
        recommendation = find_cluster_representatives(filtered_residuals_df, cluster_id, user_champion, top_k=3)

    return {
            "end": True, "final_rec": recommendation
    }

#global_users_df = pd.DataFrame(cache.get(role, "champion_residuals"))

#global_users_df = pd.DataFrame(cache.get(role, "clusters"))
    
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
graph.add_node("get_user_queue", get_user_queue)
graph.add_node("compile_user_vector", compile_user_vector)
graph.add_node("pull_tags_and_descriptions", pull_tags_and_descriptions)
graph.add_node("decision_making_method", decision_making_method)
graph.add_node("collaborative_filtering", collaborative_filtering)
graph.add_node("mathematical_optimization", mathematical_optimization)
graph.add_node("natural_language_exploration", natural_language_exploration)
#graph.add_node("extract_user_vector", extract_user_vector)

graph.add_edge(START, "ask_role")
graph.add_edge("ask_role", "load_and_cache_s3_data") # Dead end
graph.add_edge("load_and_cache_s3_data", "ask_use_own_data")
# Run in parallel with compile_user_vector being conditional on use_own_data
graph.add_conditional_edges(
    "ask_use_own_data",
    lambda state: "go_queue" if state.get("use_own_data") else "skip_to_vector",
    {"go_queue": "get_user_queue", "skip_to_vector": "compile_user_vector"}
)
graph.add_conditional_edges(
    "get_user_queue",
    lambda state: "go_vector" if not state.get("dead_end") else "end",
    {"go_vector": "compile_user_vector", "end": END}
)
graph.add_edge("compile_user_vector", "pull_tags_and_descriptions")
graph.add_edge("pull_tags_and_descriptions", "decision_making_method")
graph.add_conditional_edges(
    "decision_making_method",
    lambda state: state.get("decision_making_method"),
    {
        "collaborative_filtering": "collaborative_filtering", 
        "mathematical_optimization": "mathematical_optimization", 
        "natural_language_exploration": "natural_language_exploration"
    }
)
graph.add_edge("collaborative_filtering", END)
graph.add_edge("mathematical_optimization", END)
graph.add_edge("natural_language_exploration", END)
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