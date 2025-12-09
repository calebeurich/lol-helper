# ──────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────
import io
import json
import os
import requests
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────
import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv

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
# Environment setup and Constants
# ──────────────────────────────────────────
CURRENT_PATCH = "15.6"
patch_naming = f"patch_{CURRENT_PATCH.replace('.', '_')}"
PATCH_START_TIME = "1742342400"
PATCH_END_TIME = "1743552000"

MINIMUM_GAMES = 1
TOP_K = 3
BINARY_REPLY = ["Yes", "No"]

load_dotenv()
MODEL_ID = os.getenv("CHATBOT_LLM_ID")
REGION = os.getenv("REGION")
BUCKET = os.getenv("BUCKET")
PREFIX = os.getenv("PROCESSED_DATA_FOLDER")
cfg = Config(read_timeout=120, retries={"max_attempts": 3, "mode": "adaptive"})

ALL_CHAMPIONS = json.loads(
    (Path(__file__).resolve().parent / "config" / "champion_aliases.json").read_text(encoding="utf-8")
)


# ──────────────────────────────────────────
# S3 Cache (unchanged)
# ──────────────────────────────────────────
class S3Paths:
    def __init__(self, role: str, patch=patch_naming):
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
    def __init__(self, bucket: str = BUCKET, region: str = REGION):
        self.bucket = bucket
        self.cache = {}
        self.s3 = boto3.client("s3", region_name=region, config=cfg)

    def get_global_data(self, role: str, patch: str = patch_naming):
        if role in self.cache:
            return self.cache[role]

        paths = S3Paths(role, patch)
        result = {}
        for data_type, s3_key in paths.items():
            try:
                response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            except Exception as e:
                print(f"Error loading {s3_key}: {e}")
                result[data_type] = None
                continue

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
        role_cache = self.cache.get(role)
        if role_cache:
            return role_cache.get(data_type)
        return None

    def put(self, role: str, key: str, value: Any):
        if role not in self.cache:
            self.cache[role] = {}
        self.cache[role][key] = value


# ──────────────────────────────────────────
# LLM Setup (unchanged)
# ──────────────────────────────────────────
bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-2"))


def call_llm(prompt: str) -> str:
    resp = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 100, "temperature": 0},
    )
    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    texts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
    return " ".join(t for t in texts if t)


def find_similar_champions_llm(filtered_df: pd.DataFrame, target_tags: str, target_description: str, top_k: int = 3):
    similarities = []
    for _, row in filtered_df.iterrows():
        champion_name = row["champion_name"]
        if pd.isna(champion_name):
            continue

        champ_tags = row.get("tags", "")
        champ_description = row.get("description", "")

        prompt = f"""
You are evaluating semantic similarity between League of Legends champions.

TARGET CHAMPION:
Tags: {target_tags}
Description: {target_description}

CANDIDATE CHAMPION:
Name: {champion_name}
Tags: {champ_tags}
Description: {champ_description}

Rate similarity from 0 to 1. Respond ONLY with a number.
"""
        response = call_llm(prompt)
        try:
            score = float(response.strip())
        except:
            score = 0.0
        similarities.append((champion_name, score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in similarities[:top_k]]


# ──────────────────────────────────────────
# Step Enum - All possible wizard steps
# ──────────────────────────────────────────
class Step(Enum):
    # Initial steps
    ASK_ROLE = auto()
    LOAD_S3_DATA = auto()
    ASK_USE_OWN_DATA = auto()
    
    # User data flow
    ASK_USERNAME = auto()
    FETCH_USER_DATA = auto()
    ASK_QUEUE_TYPE = auto()
    AGGREGATE_USER_DATA = auto()
    ASK_USE_GLOBAL_DATA_FALLBACK = auto()
    
    # Champion selection flow
    ASK_CHAMPION_CRITERIA = auto()
    MANUAL_CHAMPION_SELECTION = auto()
    COMPILE_USER_VECTOR = auto()
    ASK_MULTIPLE_CHAMPIONS = auto()
    RECOMPILE_USER_VECTOR = auto()
    
    # Tags and descriptions
    PULL_TAGS_AND_DESCRIPTIONS = auto()
    SHOW_CHAMPION_INFO = auto()
    
    # Method selection
    ASK_DECISION_METHOD = auto()
    
    # Collaborative filtering
    COMPUTE_COLLABORATIVE_FILTERING = auto()
    
    # Mathematical optimization
    ASK_MATH_OPT_SCOPE = auto()
    ASK_MATH_OPT_BY_WR = auto()
    COMPUTE_MATH_OPTIMIZATION = auto()
    
    # Natural language exploration
    ASK_NL_CLUSTER = auto()
    ASK_NL_SIMILARITY = auto()
    COMPUTE_NL_EXPLORATION = auto()
    
    # Final
    SHOW_RECOMMENDATIONS = auto()
    DONE = auto()
    ERROR = auto()


# ──────────────────────────────────────────
# UI Configuration dataclass
# ──────────────────────────────────────────
@dataclass
class UIConfig:
    """Configuration for what to display in the UI"""
    input_type: Optional[str] = None  # "button", "text", "dropdown", "selectbox", "continue", "none"
    question: Optional[str] = None
    choices: Optional[List[str]] = None
    placeholder: Optional[str] = None
    dataframe: Optional[List[dict]] = None
    error_message: Optional[str] = None
    is_loading: bool = False


# ──────────────────────────────────────────
# Wizard State dataclass
# ──────────────────────────────────────────
@dataclass
class WizardState:
    current_step: Step = Step.ASK_ROLE
    
    # User inputs
    role: Optional[str] = None
    use_own_data: Optional[str] = None  # "Yes" or "No"
    user_name_and_tag: Optional[str] = None
    user_queue_type: Optional[str] = None
    use_global_data: Optional[str] = None  # "Yes" or "No" (fallback)
    selection_criterion: Optional[str] = None
    user_champion: Optional[str] = None
    decision_making_method: Optional[str] = None
    math_opt_scope: Optional[str] = None
    math_opt_by_wr: Optional[str] = None
    selected_cluster_id: Optional[str] = None
    nat_lang_expl_similarity: Optional[str] = None
    
    # Computed data
    desired_sample: Optional[str] = None
    user_puuid: Optional[str] = None
    valid_queues: Optional[List[str]] = None
    user_vector: Optional[Any] = None
    multiple_champions: Optional[bool] = None
    multiple_champion_names: Optional[List[str]] = None
    
    champion_description: Optional[str] = None
    champion_tags: Optional[str] = None
    cluster_description: Optional[str] = None
    cluster_tags: Optional[str] = None
    cluster_id: Optional[str] = None
    
    recommendations: Optional[dict] = None
    
    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0


# ──────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────
def fetch_descriptions(iterable, role, cache: S3Cache) -> dict:
    champion_semantics_df = pd.DataFrame(cache.get(role, "champ_semantic_tags_and_desc"))
    output = {}
    for champion in iterable:
        result = champion_semantics_df.loc[
            champion_semantics_df["id"] == f"{champion}__{role}", ["description"]
        ]
        if not result.empty:
            output[champion] = result.squeeze()
        else:
            output[champion] = ""
    return output


# ──────────────────────────────────────────
# Main Wizard Class
# ──────────────────────────────────────────
class RecommenderWizard:
    def __init__(self, state: Optional[WizardState] = None, cache: Optional[S3Cache] = None):
        self.state = state or WizardState()
        self.cache = cache or S3Cache()
    
    def get_ui_config(self) -> UIConfig:
        """Get the UI configuration for the current step"""
        step = self.state.current_step
        
        # ─── Initial Steps ───
        if step == Step.ASK_ROLE:
            return UIConfig(
                input_type="button",
                question="What is your desired role?",
                choices=["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
            )
        
        elif step == Step.LOAD_S3_DATA:
            return UIConfig(is_loading=True, question="Loading data...")
        
        elif step == Step.ASK_USE_OWN_DATA:
            return UIConfig(
                input_type="button",
                question=f"Do you wish to use your own data? A minimum of {MINIMUM_GAMES} games is required.",
                choices=BINARY_REPLY
            )
        
        # ─── User Data Flow ───
        elif step == Step.ASK_USERNAME:
            return UIConfig(
                input_type="text",
                question="Enter your account Username and Tag:",
                placeholder="Username#TAG"
            )
        
        elif step == Step.FETCH_USER_DATA:
            return UIConfig(is_loading=True, question="Fetching your account data...")
        
        elif step == Step.ASK_QUEUE_TYPE:
            return UIConfig(
                input_type="button",
                question="You have enough data for the following queue(s), please select one:",
                choices=self.state.valid_queues
            )
        
        elif step == Step.AGGREGATE_USER_DATA:
            return UIConfig(is_loading=True, question="Processing your data...")
        
        elif step == Step.ASK_USE_GLOBAL_DATA_FALLBACK:
            return UIConfig(
                input_type="button",
                question=self.state.error_message or "Would you like to proceed using global data instead?",
                choices=BINARY_REPLY
            )
        
        # ─── Champion Selection Flow ───
        elif step == Step.ASK_CHAMPION_CRITERIA:
            return UIConfig(
                input_type="button",
                question="We will select a representative champion to analyze. Choose a criterion:",
                choices=list(CHAMPION_CRITERIA.keys())
            )
        
        elif step == Step.MANUAL_CHAMPION_SELECTION:
            role = self.state.role
            dataset = self.state.desired_sample
            champion_df = pd.DataFrame(self.cache.get(role, dataset))
            valid_champions = champion_df["champion_name"].tolist()
            display_names = [ALL_CHAMPIONS.get(name, name) for name in valid_champions]
            
            return UIConfig(
                input_type="dropdown",
                question="The following champions have sufficient data, please select one:",
                choices=display_names,
                placeholder="Select a champion"
            )
        
        elif step == Step.COMPILE_USER_VECTOR:
            return UIConfig(is_loading=True, question="Analyzing champion data...")
        
        elif step == Step.ASK_MULTIPLE_CHAMPIONS:
            return UIConfig(
                input_type="button",
                question="Multiple champions meet this criteria. Please choose one:",
                choices=self.state.multiple_champion_names
            )
        
        elif step == Step.RECOMPILE_USER_VECTOR:
            return UIConfig(is_loading=True, question="Processing selection...")
        
        # ─── Tags and Descriptions ───
        elif step == Step.PULL_TAGS_AND_DESCRIPTIONS:
            return UIConfig(is_loading=True, question="Loading champion details...")
        
        elif step == Step.SHOW_CHAMPION_INFO:
            user_champion = self.state.user_champion
            return UIConfig(
                input_type="continue",
                question=f"""**Champion: {user_champion}**

**Tags:** {self.state.champion_tags}

**Description:** {self.state.champion_description}

**Cluster Tags:** {self.state.cluster_tags}

**Cluster Description:** {self.state.cluster_description}

Press Continue when ready for the next step."""
            )
        
        # ─── Method Selection ───
        elif step == Step.ASK_DECISION_METHOD:
            return UIConfig(
                input_type="button",
                question="Choose a recommendation methodology:",
                choices=list(METHODOLOGIES.keys())
            )
        
        # ─── Collaborative Filtering ───
        elif step == Step.COMPUTE_COLLABORATIVE_FILTERING:
            return UIConfig(is_loading=True, question="Computing recommendations...")
        
        # ─── Mathematical Optimization ───
        elif step == Step.ASK_MATH_OPT_SCOPE:
            return UIConfig(
                input_type="button",
                question="Do we want to stay within cluster scope or look at the whole role?",
                choices=["Within cluster", "Whole role"]
            )
        
        elif step == Step.ASK_MATH_OPT_BY_WR:
            return UIConfig(
                input_type="button",
                question="Do we want to round your champion pool based on win rate?",
                choices=BINARY_REPLY
            )
        
        elif step == Step.COMPUTE_MATH_OPTIMIZATION:
            return UIConfig(is_loading=True, question="Computing recommendations...")
        
        # ─── Natural Language Exploration ───
        elif step == Step.ASK_NL_CLUSTER:
            role = self.state.role
            cluster_df = pd.DataFrame(self.cache.get(role, "cluster_semantic_tags_and_desc"))
            cluster_df["id"] = cluster_df["id"].astype(str)
            
            return UIConfig(
                input_type="selectbox",
                question=f"Select a cluster to explore for {role.capitalize()}:",
                choices=cluster_df["id"].tolist(),
                dataframe=cluster_df.to_dict("records")
            )
        
        elif step == Step.ASK_NL_SIMILARITY:
            return UIConfig(
                input_type="button",
                question=f"Do you prefer champions most similar to {self.state.user_champion}?",
                choices=BINARY_REPLY
            )
        
        elif step == Step.COMPUTE_NL_EXPLORATION:
            return UIConfig(is_loading=True, question="Computing recommendations...")
        
        # ─── Final ───
        elif step == Step.SHOW_RECOMMENDATIONS:
            return UIConfig(input_type="none")
        
        elif step == Step.ERROR:
            return UIConfig(
                input_type="none",
                error_message=self.state.error_message or "An error occurred."
            )
        
        elif step == Step.DONE:
            return UIConfig(input_type="none")
        
        return UIConfig(error_message=f"Unknown step: {step}")
    
    def submit_input(self, value: Any) -> None:
        """Handle user input and advance to next step"""
        step = self.state.current_step
        
        if step == Step.ASK_ROLE:
            self.state.role = value
            self.state.current_step = Step.LOAD_S3_DATA
        
        elif step == Step.ASK_USE_OWN_DATA:
            self.state.use_own_data = value
            if value == "Yes":
                self.state.desired_sample = "user_api_data"
                self.state.current_step = Step.ASK_USERNAME
            else:
                self.state.desired_sample = "champion_x_role_agg"
                self.state.current_step = Step.ASK_CHAMPION_CRITERIA
        
        elif step == Step.ASK_USERNAME:
            self.state.user_name_and_tag = value
            self.state.current_step = Step.FETCH_USER_DATA
        
        elif step == Step.ASK_QUEUE_TYPE:
            self.state.user_queue_type = value
            self.state.current_step = Step.AGGREGATE_USER_DATA
        
        elif step == Step.ASK_USE_GLOBAL_DATA_FALLBACK:
            self.state.use_global_data = value
            if value == "Yes":
                self.state.desired_sample = "champion_x_role_agg"
                self.state.current_step = Step.ASK_CHAMPION_CRITERIA
            else:
                self.state.current_step = Step.ERROR
                self.state.error_message = "Cannot proceed without data."
        
        elif step == Step.ASK_CHAMPION_CRITERIA:
            self.state.selection_criterion = value
            # Check against display name since that's what the button shows
            if value == "Manual Champion Selection":
                self.state.current_step = Step.MANUAL_CHAMPION_SELECTION
            else:
                self.state.current_step = Step.COMPILE_USER_VECTOR
        
        elif step == Step.MANUAL_CHAMPION_SELECTION:
            # Convert display name back to internal name if needed
            reverse_mapping = {v: k for k, v in ALL_CHAMPIONS.items()}
            self.state.user_champion = reverse_mapping.get(value, value)
            self.state.current_step = Step.COMPILE_USER_VECTOR
        
        elif step == Step.ASK_MULTIPLE_CHAMPIONS:
            self.state.user_champion = value
            self.state.current_step = Step.RECOMPILE_USER_VECTOR
        
        elif step == Step.SHOW_CHAMPION_INFO:
            self.state.current_step = Step.ASK_DECISION_METHOD
        
        elif step == Step.ASK_DECISION_METHOD:
            self.state.decision_making_method = value
            # Use the actual display names from your buttons
            if value == "Compare with similar players":
                self.state.current_step = Step.COMPUTE_COLLABORATIVE_FILTERING
            elif value == "Champion pool optimization":
                self.state.current_step = Step.ASK_MATH_OPT_SCOPE
            elif value == "Qualitative exploration":
                self.state.current_step = Step.ASK_NL_CLUSTER
        
        elif step == Step.ASK_MATH_OPT_SCOPE:
            self.state.math_opt_scope = value
            self.state.current_step = Step.ASK_MATH_OPT_BY_WR
        
        elif step == Step.ASK_MATH_OPT_BY_WR:
            self.state.math_opt_by_wr = value
            self.state.current_step = Step.COMPUTE_MATH_OPTIMIZATION
        
        elif step == Step.ASK_NL_CLUSTER:
            self.state.selected_cluster_id = str(value)
            self.state.current_step = Step.ASK_NL_SIMILARITY
        
        elif step == Step.ASK_NL_SIMILARITY:
            self.state.nat_lang_expl_similarity = value
            self.state.current_step = Step.COMPUTE_NL_EXPLORATION
    
    def process_compute_step(self) -> bool:
        """
        Process computation steps (no user input needed).
        Returns True if still computing, False if done with this step.
        """
        step = self.state.current_step
        
        if step == Step.LOAD_S3_DATA:
            self._load_s3_data()
            self.state.current_step = Step.ASK_USE_OWN_DATA
            return False
        
        elif step == Step.FETCH_USER_DATA:
            success = self._fetch_user_data()
            return False
        
        elif step == Step.AGGREGATE_USER_DATA:
            self._aggregate_user_data()
            self.state.current_step = Step.ASK_CHAMPION_CRITERIA
            return False
        
        elif step == Step.COMPILE_USER_VECTOR:
            self._compile_user_vector()
            return False
        
        elif step == Step.RECOMPILE_USER_VECTOR:
            self._recompile_user_vector()
            self.state.current_step = Step.PULL_TAGS_AND_DESCRIPTIONS
            return False
        
        elif step == Step.PULL_TAGS_AND_DESCRIPTIONS:
            self._pull_tags_and_descriptions()
            self.state.current_step = Step.SHOW_CHAMPION_INFO
            return False
        
        elif step == Step.COMPUTE_COLLABORATIVE_FILTERING:
            self._compute_collaborative_filtering()
            self.state.current_step = Step.SHOW_RECOMMENDATIONS
            return False
        
        elif step == Step.COMPUTE_MATH_OPTIMIZATION:
            self._compute_math_optimization()
            self.state.current_step = Step.SHOW_RECOMMENDATIONS
            return False
        
        elif step == Step.COMPUTE_NL_EXPLORATION:
            self._compute_nl_exploration()
            self.state.current_step = Step.SHOW_RECOMMENDATIONS
            return False
        
        return False
    
    def is_compute_step(self) -> bool:
        """Check if current step requires computation (no user input)"""
        compute_steps = {
            Step.LOAD_S3_DATA,
            Step.FETCH_USER_DATA,
            Step.AGGREGATE_USER_DATA,
            Step.COMPILE_USER_VECTOR,
            Step.RECOMPILE_USER_VECTOR,
            Step.PULL_TAGS_AND_DESCRIPTIONS,
            Step.COMPUTE_COLLABORATIVE_FILTERING,
            Step.COMPUTE_MATH_OPTIMIZATION,
            Step.COMPUTE_NL_EXPLORATION,
        }
        return self.state.current_step in compute_steps
    
    def is_done(self) -> bool:
        """Check if wizard is complete"""
        return self.state.current_step in {Step.SHOW_RECOMMENDATIONS, Step.DONE, Step.ERROR}
    
    # ─────────────────────────────────────
    # Private computation methods
    # ─────────────────────────────────────
    def _load_s3_data(self):
        self.cache.get_global_data(role=self.state.role)
    
    def _fetch_user_data(self) -> bool:
        try:
            user_info = self.state.user_name_and_tag
            user_name, user_tag_line = user_info.split("#", 1)
            role = self.state.role
            
            match_data_df, user_puuid, num_games_per_queue = compile_user_data(
                user_name, user_tag_line, PATCH_START_TIME, PATCH_END_TIME, CURRENT_PATCH
            )
            
            items_dict = self.cache.get(role, "items_dict")
            merged_df, valid_queues, all_item_tags, all_summoner_spells = find_valid_queues(
                match_data_df, items_dict, user_puuid, role, MINIMUM_GAMES
            )
            
            self.cache.put(role=role, key="all_item_tags", value=all_item_tags)
            self.cache.put(role=role, key="all_summoner_spells", value=all_summoner_spells)
            self.cache.put(role=role, key="merged_df", value=merged_df)
            
            valid_queue_list = list(valid_queues.keys())
            if valid_queue_list:
                self.state.valid_queues = valid_queue_list
                self.state.user_puuid = user_puuid
                self.state.current_step = Step.ASK_QUEUE_TYPE
                return True
            else:
                raise InsufficientSampleError()
        
        except InsufficientSampleError:
            self.state.error_message = f"Insufficient games found. Use global data instead?"
            self.state.current_step = Step.ASK_USE_GLOBAL_DATA_FALLBACK
            return False
        
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                self.state.retry_count += 1
                if self.state.retry_count < 3:
                    self.state.error_message = f"User not found. Attempt {self.state.retry_count}/3."
                    self.state.user_name_and_tag = None
                    self.state.current_step = Step.ASK_USERNAME
                else:
                    self.state.error_message = "Maximum retries exceeded. Use global data instead?"
                    self.state.current_step = Step.ASK_USE_GLOBAL_DATA_FALLBACK
            elif e.response.status_code in (400, 403):
                self.state.error_message = "API key expired or invalid."
                self.state.current_step = Step.ERROR
            else:
                self.state.error_message = f"HTTP error: {e.response.status_code}"
                self.state.current_step = Step.ERROR
            return False
    
    def _aggregate_user_data(self):
        role = self.state.role
        user_queue_type = self.state.user_queue_type
        merged_df = self.cache.get(role, "merged_df")
        all_item_tags = self.cache.get(role, "all_item_tags")
        all_summoner_spells = self.cache.get(role, "all_summoner_spells")
        
        aggregated_df = aggregate_user_data(
            merged_df, all_item_tags, all_summoner_spells, user_queue_type, MINIMUM_GAMES
        )
        self.cache.put(role=role, key="user_api_data", value=aggregated_df)
    
    def _compile_user_vector(self):
        role = self.state.role
        dataset = self.state.desired_sample
        criterion_display = self.state.selection_criterion
        
        # Map display name to internal criterion value
        criterion = CHAMPION_CRITERIA.get(criterion_display, criterion_display)
        # DEBUG: Print available columns

        df = pd.DataFrame(self.cache.get(role, dataset))
        # Handle manual selection separately - champion is already chosen
        if criterion == "manual_selection":
            user_champion = self.state.user_champion
            # Just extract the vector for the selected champion
            user_vector = df.loc[df["champion_name"] == user_champion]
            
            if user_vector.empty:
                # Try matching with the display name mapping
                reverse_mapping = {v: k for k, v in ALL_CHAMPIONS.items()}
                internal_name = reverse_mapping.get(user_champion, user_champion)
                user_vector = df.loc[df["champion_name"] == internal_name]
                if not user_vector.empty:
                    self.state.user_champion = internal_name  # Update to internal name
            
            if user_vector.empty:
                self.state.error_message = f"Champion '{user_champion}' not found in data."
                self.state.current_step = Step.ERROR
                return
            
            self.state.multiple_champions = False
            self.state.user_vector = user_vector.iloc[[0]]
            self.state.current_step = Step.PULL_TAGS_AND_DESCRIPTIONS
            return
        
        # For non-manual selection (win_rate, play_rate)
        user_vector_or_names, n = extract_vector(df, criterion, None, MINIMUM_GAMES)
        
        if n > 1:
            self.state.multiple_champions = True
            self.state.multiple_champion_names = user_vector_or_names
            self.cache.put(role=role, key="multiple_champions", value=user_vector_or_names)
            self.state.current_step = Step.ASK_MULTIPLE_CHAMPIONS
        else:
            self.state.multiple_champions = False
            self.state.user_vector = user_vector_or_names
            self.state.user_champion = user_vector_or_names.iloc[0]["champion_name"]
            self.state.current_step = Step.PULL_TAGS_AND_DESCRIPTIONS
    
    def _recompile_user_vector(self):
        df = pd.DataFrame(self.cache.get(self.state.role, self.state.desired_sample))
        user_champion = self.state.user_champion
        self.state.user_vector = df.loc[df["champion_name"] == user_champion].iloc[[0]]
    
    def _pull_tags_and_descriptions(self):
        role = self.state.role
        user_champion = self.state.user_champion
        
        champion_semantics_df = pd.DataFrame(self.cache.get(role, "champ_semantic_tags_and_desc"))
        result = champion_semantics_df.loc[
            champion_semantics_df["id"] == f"{user_champion}__{role}", ["description", "tags"]
        ]
        
        if not result.empty:
            self.state.champion_description, self.state.champion_tags = result.squeeze()
        
        cluster_df = pd.DataFrame(self.cache.get(role, "cluster_semantic_tags_and_desc"))
        cluster_df["id"] = pd.to_numeric(cluster_df["id"], errors="coerce")
        champion_residuals_df = pd.DataFrame(self.cache.get(role, "champion_residuals"))
        
        cluster_result = champion_residuals_df.loc[
            champion_residuals_df["champion_name"] == user_champion, ["cluster"]
        ]
        
        if not cluster_result.empty:
            self.state.cluster_id = str(cluster_result.squeeze())
            
            cluster_row = cluster_df.loc[cluster_df["id"] == self.state.cluster_id, ["description", "tags"]]
            if not cluster_row.empty:
                self.state.cluster_description, self.state.cluster_tags = cluster_row.iloc[0]
    
    def _compute_collaborative_filtering(self):
        role = self.state.role
        user_champion = self.state.user_champion
        global_users_df = pd.DataFrame(self.cache.get(role, "champion_x_role_x_user_agg"))
        user_vector = self.state.user_vector
        
        filtered_user_vector, filtered_global_user_df = filter_user_and_global_dfs(
            role, user_vector, global_users_df, MINIMUM_GAMES
        )
        similar_playstyle_recs = similar_playstyle_users(filtered_user_vector, filtered_global_user_df, TOP_K)
        same_main_recs = recommend_champions_from_main(user_champion, filtered_global_user_df, TOP_K)
        
        similar_playstyle_desc = fetch_descriptions(similar_playstyle_recs, role, self.cache)
        same_main_desc = fetch_descriptions(same_main_recs, role, self.cache)
        
        self.state.recommendations = {
            "similar_playstyle": {
                "title": "Champions played by users with a similar playstyle to yours",
                "champions": similar_playstyle_recs,
                "descriptions": similar_playstyle_desc
            },
            "same_mains": {
                "title": f"Champions played by users that main {user_champion}",
                "champions": same_main_recs,
                "descriptions": same_main_desc
            }
        }
    
    def _compute_math_optimization(self):
        role = self.state.role
        cluster_id = self.state.cluster_id
        user_champion = self.state.user_champion
        scope = self.state.math_opt_scope
        win_rate = self.state.math_opt_by_wr == "Yes"
        
        champion_residuals_df = pd.DataFrame(self.cache.get(role, "champion_residuals"))
                
        if scope == "Within cluster":
            filtered_residuals_df = champion_residuals_df.loc[
                champion_residuals_df["cluster"] == cluster_id
            ]
            print("CLUSTER CHAMPS")
            print(filtered_residuals_df["champion_name"].tolist())
        else:
            filtered_residuals_df = champion_residuals_df
        
        if win_rate:
            counter_stats_df = pd.DataFrame(self.cache.get(role, "counter_stats_dfs_by_role"))
            
            # Debug: print available champion names in counter stats
            print(f"DEBUG: available {role} champions in counter_stats = {counter_stats_df['champion_name'].unique().tolist()}")
            
            if scope == "Within cluster":
                counter_stats_df = counter_stats_df.loc[
                    counter_stats_df["champion_name"].isin(filtered_residuals_df["champion_name"])
                ]
            
            # Check if user_champion exists in the data
            if user_champion not in counter_stats_df["champion_name"].values:
                # Try to find a matching name
                available_names = counter_stats_df["champion_name"].unique().tolist()
                print(f"DEBUG: Looking for '{user_champion}' in {available_names}")
                
                # Try case-insensitive match
                for name in available_names:
                    if name.lower().replace(" ", "").replace("'", "") == user_champion.lower().replace(" ", "").replace("'", ""):
                        user_champion = name
                        print(f"DEBUG: Found match: {name}")
                        break
            
            recommended_champions = find_best_alternative(counter_stats_df, user_champion, MINIMUM_GAMES)
            recommended_desc = fetch_descriptions(recommended_champions, role, self.cache)
            
            self.state.recommendations = {
                "default": {
                    "title": f"Champions that round out your pool (better in {self.state.user_champion}'s bad matchups)",
                    "champions": recommended_champions,
                    "descriptions": recommended_desc
                }
            }
        else:
            champion_x_role_df = pd.DataFrame(self.cache.get(role, "champion_x_role_agg"))
            champion_x_role_df = champion_x_role_df[
                champion_x_role_df["champion_name"].isin(filtered_residuals_df["champion_name"])
            ]
            
            # Check if user_champion exists in the data
            if user_champion not in champion_x_role_df["champion_name"].values:
                available_names = champion_x_role_df["champion_name"].unique().tolist()
                print(f"DEBUG: Looking for '{user_champion}' in {available_names[:20]}")
                
                for name in available_names:
                    if name.lower().replace(" ", "").replace("'", "") == user_champion.lower().replace(" ", "").replace("'", ""):
                        user_champion = name
                        print(f"DEBUG: Found match: {name}")
                        break
            
            similar_champs, different_champs = find_recs_within_cluster(champion_x_role_df, user_champion)
            similar_desc = fetch_descriptions(similar_champs, role, self.cache)
            different_desc = fetch_descriptions(different_champs, role, self.cache)
            
            self.state.recommendations = {
                "similar": {
                    "title": f"Champions with similar playstyle to {self.state.user_champion}",
                    "champions": similar_champs,
                    "descriptions": similar_desc
                },
                "different": {
                    "title": f"Champions with different playstyle from {self.state.user_champion}",
                    "champions": different_champs,
                    "descriptions": different_desc
                }
            }
    
    def _compute_nl_exploration(self):
        role = self.state.role
        user_champion = self.state.user_champion
        cluster_id = str(self.state.selected_cluster_id)
        similarity = self.state.nat_lang_expl_similarity == "Yes"
        
        champion_residuals_df = pd.DataFrame(self.cache.get(role, "champion_residuals"))
        champion_residuals_df["cluster"] = champion_residuals_df["cluster"].astype(str)
        
        filtered_residuals_df = champion_residuals_df.loc[
            (champion_residuals_df["cluster"] == cluster_id) |
            (champion_residuals_df["champion_name"] == user_champion)
        ]
        cluster_champions = filtered_residuals_df["champion_name"].tolist()
        
        if similarity:
            champion_semantics_df = pd.DataFrame(self.cache.get(role, "champ_semantic_tags_and_desc"))
            champion_semantics_df["champion_name"] = champion_semantics_df["id"].str.split("__", n=1).str[0]
            
            champion_semantics_df = champion_semantics_df[
                champion_semantics_df["champion_name"].isin(cluster_champions) &
                (champion_semantics_df["champion_name"] != user_champion)
            ]
            
            recommended_champions = find_similar_champions_llm(
                champion_semantics_df,
                self.state.champion_tags,
                self.state.champion_description,
                top_k=3
            )
            title = f"Champions in cluster #{cluster_id} similar to {user_champion}"
        else:
            recommended_champions = find_cluster_representatives(
                filtered_residuals_df, cluster_id, user_champion, top_k=3
            )
            title = f"Champions closest to cluster #{cluster_id}'s identity"
        
        rec_desc = fetch_descriptions(recommended_champions, role, self.cache)
        
        self.state.recommendations = {
            "default": {
                "title": title,
                "champions": recommended_champions,
                "descriptions": rec_desc
            }
        }