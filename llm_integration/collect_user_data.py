import pandas as pd
import os, sys, gc
import requests
import json
from dotenv import load_dotenv
import time
from tqdm import tqdm
import gc
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

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

spark = (
    SparkSession.builder
        .appName("user_data_aggregation")
        .getOrCreate()
)

class QueueTypeError(Exception):
    """Incorrect queue type input."""


def handle_rate_limit(api_calls: int, start_time: float, buffer: int = 5) -> tuple[int, float]:
    """Handle Riot API rate limiting"""
    if api_calls >= (100 - buffer):  # 100 requests per 2 min, with buffer
        elapsed = time.time() - start_time
        if elapsed < 120:  # 2 minute window
            sleep_time = 120 - elapsed + 1  # Add 1 second buffer
            tqdm.write(f'Rate limit approaching - sleeping for {sleep_time:.1f} seconds')
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
    patch_start_time: str = PATCH_START_TIME, 
    patch_end_time: str = PATCH_END_TIME, 
    matches_per_summoner: int = 100,
    queue_type: str = "ranked"
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
            "matchId": match_history
        })
    
    else:
        sys.exit("No matches found for specified time period.") 

    return match_history_df

def get_match_data(match_history_df: pd.DataFrame, current_patch: str):

    api_calls  = 0
    start_time = time.time()
    match_data_df = pd.DataFrame(columns=["puuid", "match_id", "match_data"])

    # iterate directly over the match_id column, not iterrows()
    for _, row in tqdm(match_history_df.iterrows(),
                         total=len(match_history_df),
                         desc="Fetching match data"):
        
        match_id = row["match_id"]
        api_calls, start_time = handle_rate_limit(api_calls, start_time)

        url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        match_data = response.json()

        if "info" in match_data and "gameVersion" in match_data["info"]:
            game_patch = ".".join(match_data["info"]["gameVersion"].split(".")[:2])
            if game_patch == current_patch:
                new_row = {
                    "puuid": row["puuid"],
                    "match_id": match_id,
                    "match_data": json.dumps(match_data)
                }
                match_data_df = pd.concat([match_data_df, pd.DataFrame([new_row])], ignore_index=True)

        api_calls += 1

    gc.collect()

    match_data_df = spark.createDataFrame(match_data_df)

    gc.collect()

    return match_data_df

def compile_user_raw_data(user_name, user_tag_line, queue_type):

    puuid = get_puuid(user_name, user_tag_line)
    match_history = get_match_ids(puuid, queue_type)
    match_data_df = get_match_data(match_history, CURRENT_PATCH)

    return match_data_df
