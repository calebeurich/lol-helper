from lambda_compute.user_data_collection_and_processing.pandas_user_data_aggregation import main_aggregator
from dotenv import load_dotenv
from functools import lru_cache
import pandas as pd
import time, json, argparse, os, glob, requests, sys, gc, re


# Load environment variables and set up
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
HEADERS = {
    "X-Riot-Token": RIOT_API_KEY
}

# In production these will need to be taken as inputs 
CURRENT_PATCH = "15.6"
PATCH_START_TIME = "1742342400" # March 19th, 2025 in timestamp seconds
PATCH_END_TIME = "1743552000" # APRIL 4TH, 2025 in timestamp

BATCH_SIZE = 100
SAVE_FREQUENCY = 50
REQUEST_TIMEOUT = 10

class QueueTypeError(Exception):
    """Incorrect queue type input."""


# --- fast camelCase/PascalCase -> snake_case ---
_CAMEL1 = re.compile(r'(.)([A-Z][a-z]+)')
_CAMEL2 = re.compile(r'([a-z0-9])([A-Z])')

@lru_cache(maxsize=4096)
def camel_to_snake(name: str) -> str:
    s1 = _CAMEL1.sub(r'\1_\2', name)
    s2 = _CAMEL2.sub(r'\1_\2', s1)
    return s2.lower()

def rename_keys(obj):
    if isinstance(obj, dict):
        return {camel_to_snake(k): rename_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rename_keys(v) for v in obj]
    return obj


def handle_rate_limit(api_calls: int, start_time: float, buffer: int = 5) -> tuple[int, float]:
    """Handle Riot API rate limiting"""
    if api_calls >= (100 - buffer):  # 100 requests per 2 min, with buffer
        elapsed = time.time() - start_time
        if elapsed < 120:  # 2 minute window
            sleep_time = 120 - elapsed + 1  # Add 1 second buffer
            print(f'Rate limit approaching - sleeping for {sleep_time:.1f} seconds')
            time.sleep(sleep_time)
        start_time = time.time()
        api_calls = 0
    return api_calls, start_time


def get_puuid(user_name: str, user_tag_line: str) -> str:
    api_url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{user_name}/{user_tag_line}?api_key=RGAPI-d2dfae0b-7762-478c-84a5-ca2e61ef0914"
    try:
        response = requests.get(api_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        user_account_data = response.json()
    
    except requests.Timeout:
        sys.exit("Request Timeout Error")

    except requests.HTTPError as e:
        sys.exit(f"HTTP Error: {e}")
        
    except Exception as e:
        sys.exit(f"Error: {str(e)}")

    return user_account_data.get("puuid", "Puuid not found in JSON response")


def get_match_ids(
    puuid: str,
    queue_type: str = "ranked",
    patch_start_time: str = PATCH_START_TIME, 
    patch_end_time: str = PATCH_END_TIME, 
    matches_per_summoner: int = 100,
) -> pd.DataFrame:

    if queue_type == "ranked":
        api_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={patch_start_time}&endTime={patch_end_time}&queue=420&start=0&count={matches_per_summoner}"
    elif queue_type == "draft":
        api_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={patch_start_time}&endTime={patch_end_time}&queue=400&start=0&count={matches_per_summoner}"
    elif queue_type == "both": # Alternative is to process all matches, can be easily tuned to accept different queue types (ARAM, etc)
        api_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={patch_start_time}&endTime={patch_end_time}&start=0&count={matches_per_summoner}"
    else:
        raise QueueTypeError

    api_calls = 0
    start_time = time.time()

    api_calls, start_time = handle_rate_limit(api_calls, start_time)

    try:
        response = requests.get(
            api_url, headers=HEADERS, timeout=10
            )
        
        response.raise_for_status()
        match_history = response.json()

    except requests.Timeout:
        sys.exit("Request Timeout Error")

    except requests.HTTPError as e:
        sys.exit(f"HTTP Error: {e}")
        
    except Exception as e:
        sys.exit(f"Error: {str(e)}")
    
    if match_history:
        match_history_df = pd.DataFrame({
            "puuid": puuid,
            "match_id": match_history
        })
    
    else:
        sys.exit("No matches found for specified time period.") 

    return match_history_df


def get_match_data(match_history_df: pd.DataFrame, current_patch: str):

    def ensure_parsed(obj):
        # Return a Python object (dict/list) no matter what was passed
        if isinstance(obj, (dict, list)):
            return obj
        if isinstance(obj, str):
            return json.loads(obj)  # parse JSON string
        return obj  # or raise TypeError if you want to be strict

    api_calls  = 0
    start_time = time.time()
    match_data_df = pd.DataFrame(columns=["puuid", "match_id", "match_data"])
    rows = []

    match_history_df.to_csv("match_history.csv")
    # iterate directly over the match_id column, not iterrows()
    for row in match_history_df.itertuples(index=False):

        match_id = row.match_id
        api_calls, start_time = handle_rate_limit(api_calls, start_time)

        url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        match_data_raw = response.json()
        match_data_parsed = ensure_parsed(match_data_raw)

        match_data = rename_keys(match_data_parsed)

        info = match_data.get("info", {})
        game_version = info.get("game_version")
        if isinstance(game_version, str):
            game_patch = ".".join(game_version.split(".")[:2])
            if game_patch == current_patch:
                rows.append({
                    "puuid": row.puuid,
                    "match_id": match_id,
                    "match_data": match_data
                })

        api_calls += 1

    gc.collect()

    match_data_df = pd.DataFrame.from_records(rows, columns=["puuid", "match_id", "match_data"])
    match_data_df.rename(columns=camel_to_snake, inplace=True)

    return match_data_df


def compile_and_aggregate_user_data(user_name, user_tag_line, queue_type):

    puuid = get_puuid(user_name=user_name, user_tag_line=user_tag_line)
    match_history_df = get_match_ids(puuid=puuid, queue_type=queue_type)

    match_data_df = get_match_data(match_history_df=match_history_df, current_patch=CURRENT_PATCH)
    match_data_df.to_csv("match_data_test1.csv")

    try:
        with open("item_id_tags.json", "r") as f:
            items_dict = json.load(f)
    except FileNotFoundError:
        print("Items dict json not found")
    match_data_df.to_csv("match_data_test.csv")

    user_df = main_aggregator(raw_master_df=match_data_df, queue_type=queue_type, items_dict=items_dict, user_puuid=puuid)

    return user_df