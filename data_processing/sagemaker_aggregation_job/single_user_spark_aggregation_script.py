from dotenv import load_dotenv
from pyspark.sql import SparkSession, DataFrame
from spark_champion_aggregation import main_aggregator
import pandas as pd
import time, json, argparse, os, glob, requests, sys, gc


# Load environment variables and set up
load_dotenv()

PATCH = "15_6"

RIOT_API_KEY = os.getenv("RIOT_API_KEY")
HEADERS = {
    "X-Riot-Token": RIOT_API_KEY
}

BATCH_SIZE = 100
SAVE_FREQUENCY = 50
REQUEST_TIMEOUT = 10

# In production these will need to be taken as inputs 
CURRENT_PATCH = "15.6"
PATCH_START_TIME = "1742342400" # March 19th, 2025 in timestamp seconds
PATCH_END_TIME = "1743552000" # APRIL 4TH, 2025 in timestamp

class QueueTypeError(Exception):
    """Incorrect queue type input."""

def handle_rate_limit(api_calls: int, start_time: float, buffer: int = 5) -> tuple[int, float]:
    """Handle Riot API rate limiting"""
    if api_calls >= (100 - buffer):  # 100 requests per 2 min, with buffer
        elapsed = time.time() - start_time
        if elapsed < 120:  # 2 minute window
            sleep_time = 120 - elapsed + 1  # Add 1 second buffer
            print(f"Rate limit approaching - sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        start_time = time.time()
        api_calls = 0
    return api_calls, start_time


def get_puuid(user_name: str, user_tag_line: str) -> str:

    api_url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{user_name}/{user_tag_line}"
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
    user_queue_type: str = "ranked"
) -> pd.DataFrame:

    if user_queue_type == "ranked":
        api_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={patch_start_time}&endTime={patch_end_time}&queue=420&start=0&count={matches_per_summoner}"
    elif user_queue_type == "draft":
        api_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={patch_start_time}&endTime={patch_end_time}&queue=400&start=0&count={matches_per_summoner}"
    elif user_queue_type == "both": # Alternative is to process all matches, can be easily tuned to accept different queue types (ARAM, etc)
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

def get_match_data(match_history_df: pd.DataFrame, current_patch: str, spark: SparkSession) -> DataFrame:

    api_calls  = 0
    start_time = time.time()
    match_data_df = pd.DataFrame(columns=["puuid", "match_id", "match_data"])

    # iterate directly over the match_id column, not iterrows()
    for _, row in match_history_df.iterrows():
        
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

    match_data_sdf = spark.createDataFrame(match_data_df.to_dict("records"))

    gc.collect()

    return match_data_sdf

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--input-path",  type=str, required=True)
    parser.add_argument("--user_name", type=str, required=True)
    parser.add_argument("--user_tag_line", type=str, required=True)
    parser.add_argument("--user_queue_type", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--items-json-path", type=str, required=True)
    args = parser.parse_args()
    user_name, user_tag_line, user_queue_type = args.user_name, args.user_tag_line, args.user_queue_type
    
    print(f"Starting Spark job")

    spark = (
        SparkSession.builder
            .appName("user_data_aggregation")
            .getOrCreate()
    )

    user_puuid = get_puuid(user_name, user_tag_line)
    print(user_puuid)

    user_match_history_df = get_match_ids(
        puuid=user_puuid, patch_start_time=PATCH_START_TIME, 
        patch_end_time=PATCH_END_TIME, matches_per_summoner=100, user_queue_type=user_queue_type
    )

    user_match_data_df = get_match_data(match_history_df=user_match_history_df, current_patch=CURRENT_PATCH, spark=spark)

    # Hand SparkSession + folder‑prefix into aggregator
    single_user_df = main_aggregator(
        spark, 
        user_match_data_df,
        args.items_json_path,
        single_user_flag = True,
        single_user_puuid = user_puuid,
        desired_queue_id = 420
    )

    print(f"Aggregation completed; writing output as csv to {args.output_path}")

    # Create a subdirectory that Spark can manage
    single_user_user_output_path = f"{args.output_path}/single_user_data_[{user_name}#{user_tag_line}]/patch_{PATCH}"
    print(f"Writing to: file://{single_user_user_output_path}")
    
    (single_user_df
        .coalesce(1)
        .write
        .option("header", True)
        .mode("overwrite")
        .csv(f"file://{single_user_user_output_path}")) # "file://" prefix needed for Spark to output to S3

    print("Spark aggregation complete.")

    # Find the part file
    part_files = glob.glob(f"{single_user_user_output_path}/part-*.csv")
    if part_files:
        part_file = part_files[0]
        new_name = f"{single_user_user_output_path}/single_user_aggregated_data.csv"

        print(f"Renaming {part_file} to {new_name}")
        os.rename(part_file, new_name)

        # Remove .crc files
        for crc_file in glob.glob(f"{single_user_user_output_path}/.*.crc"):
            os.remove(crc_file)

        print("File renamed successfully")
    else:
        print("Warning: No part file found to rename")

    spark.stop()    

if __name__ == "__main__":
    # Use --input-path and --output-path from CLI
    main()