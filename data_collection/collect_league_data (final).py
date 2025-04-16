import pandas as pd
import os
import requests
import json
from dotenv import load_dotenv
import time
from tqdm import tqdm
from typing import List, Optional
import numpy as np
import gc
import csv

# Load environment variables and set up
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
HEADERS = {
    "X-Riot-Token": RIOT_API_KEY
}

# Constants
CURRENT_PATCH = "15.6"
PATCH_START_TIME = "1742342400" # March 19th, 2025 in timestamp seconds
PATCH_END_TIME = "1743552000" # APRIL 4TH, 2025 in timestamp
APEX_TIERS = ['challengerleagues', 'grandmasterleagues', 'masterleagues']
DIVISIONS = ['DIAMOND']  # Can be expanded to ['DIAMOND', 'EMERALD']
TIERS = ["I", "II", "III", "IV"] # can be expanded to ["I", "II", "III", "IV"]

BATCH_SIZE = 100
SAVE_FREQUENCY = 50
REQUEST_TIMEOUT = 10 

# File paths
SUMMONER_IDS_FILE = "summoner_ids.csv"
SUMMONER_PUUID_FILE = "puuids_and_summids.csv"
MATCH_IDS_FILE = "match_ids.csv"
MATCH_DATA_FILE = "match_data.csv"

pd.set_option('display.max_columns', None)

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

def get_summoner_ids_from_api(api_url: str, req_type: str) -> List[str]:
    """Fetch summoner IDs from Riot API"""
    time.sleep(0.05)  # Ensure we don't exceed 20 requests/sec
    summoner_ids = []
    try:
        resp = requests.get(api_url, headers=HEADERS)
        resp.raise_for_status()
        response = resp.json()
        
        if req_type == "apex":
            entries = response.get('entries', [])
            summoner_ids = [entry['summonerId'] for entry in entries]
        elif req_type == "regular":
            summoner_ids = [entry['summonerId'] for entry in response]
            
    except requests.HTTPError:
        tqdm.write("Couldn't complete request")   
    
    return summoner_ids

def get_sums_for_apex_leagues() -> List[str]:
    """Collect summoner IDs from apex leagues"""
    apex_summoner_ids = []
    for apex_league in tqdm(APEX_TIERS, desc="Fetching apex leagues"):
        api_url = f"https://na1.api.riotgames.com/lol/league/v4/{apex_league}/by-queue/RANKED_SOLO_5x5"
        summoner_ids = get_summoner_ids_from_api(api_url, "apex")
        apex_summoner_ids.extend(summoner_ids)
    return apex_summoner_ids

def get_sums_for_reg_divisions(start_time: float) -> List[str]:
    """Collect summoner IDs from regular divisions"""
    api_calls = 3  # starts at 3 since there should be 3 apex tier calls before this
    reg_division_ids = []
    
    for division in tqdm(DIVISIONS, desc="Processing divisions"):
        for tier in tqdm(TIERS, desc=f"Processing {division} tiers", leave=False):
            page = 1
            while True:
                api_calls, start_time = handle_rate_limit(api_calls, start_time)
                
                api_url = f'https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{division}/{tier}?page={page}'
                output_list = get_summoner_ids_from_api(api_url, "regular")
                
                if not output_list:
                    break
                    
                reg_division_ids.extend(output_list)
                page += 1
                api_calls += 1
                
    return reg_division_ids

def collect_summoner_ids() -> None:
    """Collect and save summoner IDs"""
    start = time.time()
    tqdm.write("Collecting summoner IDs...")
    
    all_sum_ids = get_sums_for_apex_leagues()
    #all_sum_ids.extend(get_sums_for_reg_divisions(start))
    
    # Remove duplicates and save
    all_sum_ids = list(set(all_sum_ids))
    pd.DataFrame({"summonerId": all_sum_ids}).to_csv(SUMMONER_IDS_FILE, index=False)
    
    tqdm.write(f"Total unique summoner IDs collected: {len(all_sum_ids)}")

def get_puuid(summoner_ids_file: str) -> None:
    """Fetch PUUIDs for summoner IDs"""
    df_summids = pd.read_csv(summoner_ids_file)

    # Check if match_data csv file/dataframe already exists and create it if it does not
    try:
        existing_data = pd.read_csv(SUMMONER_PUUID_FILE)
        # Find already processed matches to skip them
        processed_summids = set(existing_data[existing_data['puuid'].notna()]['summonerId'])
    except FileNotFoundError:
        existing_data = df_summids['summonerId'].copy()
        processed_summids = set()

    # Open file in append mode to avoid reading/writing the whole file
    with open(SUMMONER_PUUID_FILE, 'a+', newline='') as f_output:
        writer = csv.writer(f_output)
        # Write header if file is empty
        if f_output.tell() == 0:
            writer.writerow(['summonerId', 'puuid'])

        row_counter = 0
        api_calls = 0
        start_time = time.time()

        # Only process matches not already in the output file
        summids_to_process = [summid for summid in df_summids['summonerId'] if summid not in processed_summids]

        try:
            for summoner_id in tqdm(summids_to_process, desc='Fetching puuids'):
                api_calls, start_time = handle_rate_limit(api_calls, start_time)
                
                api_url = f"https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"

                try:
                    # Add timeout to requests
                    response = requests.get(api_url, headers=HEADERS, timeout=10)
                    response.raise_for_status()
                    summoner_data = response.json()

                    # Check response structure and patch
                    if 'puuid' in summoner_data:
                        # Write directly into file instead of a growing dataframe
                        writer.writerow([
                            summoner_id,
                            summoner_data.get("puuid", "Puuid not found in JSON response")
                        ])
                    else:
                        writer.writerow([
                            summoner_id,
                            summoner_data.get("puuid", "Puuid not in JSON response")
                        ])

                    time.sleep(0.05)
                    api_calls += 1
                    row_counter += 1

                    # Force file write and clear memory periodically
                    if row_counter % 100 == 0:
                        f_output.flush()
                        gc.collect()
                
                except requests.Timeout:
                    print(f'Request timeout for summoner {summoner_id}, continuing...')
                    continue

                except requests.HTTPError as e:
                    print(f'HTTP Error for summoner {summoner_id}: {e}')
                    if e.response.status_code == 403 or e.response.status_code == 400:
                        print('Probably expired API key')
                        return
                    elif e.response.status_code == 429:
                        wait = time.sleep(121 - (time.time() - start_time))
                        print(f'Rate limit exceeded, waiting {wait} seconds')
                        api_calls = 0
                        start_time = time.time()
                    # Handle gateway timeouts
                    elif e.response.status_code == 504:
                        print(f'Gateway timeout for summoner {summoner_id}, waiting 30 seconds and retrying...')
                        time.sleep(30)
                        # Retry same match matchdata pull
                        continue

                    
                except Exception as e:
                    print(f"Error with summoner {summoner_id}: {str(e)}")
                    continue

        except KeyboardInterrupt:
            print('Process interrupted by user')
            return
        
        print(f'Completed processing {row_counter} summoners')


def collect_match_ids(summoner_data_file: str, patch_start_time: str, patch_end_time: str, matches_per_summoner: int = 100) -> None:
    """Collect match IDs for summoners
    Args:
        summoner_data_file: Path to file containing summoner data
        matches_per_summoner: Number of most recent matches to collect per summoner (default: 5)
    """
    df_summoners = pd.read_csv(summoner_data_file)
    #df_match_ids = pd.DataFrame(columns=["summonerId", "puuid", "matchId"])
    
    # Check if match_id csv file already exists and create it if not
    try:
        existing_data = pd.read_csv(MATCH_IDS_FILE)
        # Find already processed summoner ids to skip them
        processed_summids = set(existing_data[existing_data['matchId'].notna()]['summonerId'])
    except FileNotFoundError:
        existing_data = df_summoners[['summonerId', 'puuid']].copy()
        processed_summids = set()

    # Open file in append mode to avoide reading/writing the whole file
    with open(MATCH_IDS_FILE, 'a+', newline='') as f_output:
        writer = csv.writer(f_output)
        # Write header if file is empty
        if f_output.tell() == 0:
            writer.writerow(['summonerId', 'puuid', 'matchId'])

        api_calls = 0
        start_time = time.time()
        row_counter = 0
    
        # Only process summids not already in the output file
        summids_to_process = [summid for summid in df_summoners['summonerId'] if summid not in processed_summids]

        try:
            for summ_id in tqdm(summids_to_process, desc='Fetching match ids'):
                api_calls, start_time = handle_rate_limit(api_calls, start_time)

                match_history_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{row['puuid']}/ids?startTime={patch_start_time}&endTime={patch_end_time}&queue=420&start=0&count={matches_per_summoner}" 

                try:
                    # Add timeout to requests
                    response = requests.get(match_history_url, headers=HEADERS, timeout=10)
                    response.raise_for_status()
                    match_history = response.json()

                    row_data = df_summoners[df_summoners['summonerId'] == summ_id].iloc[0]

                    writer.writerow([
                                row_data['summonerId'],
                                row_data['puuid'],
                                json.dumps(match_history)
                            ])

                    time.sleep(0.05)
                    api_calls += 1
                    row_counter += 1        

                    # Force file write and clear memory periodically
                    if row_counter % 100 == 0:
                        f_output.flush()
                        gc.collect()

                except requests.Timeout:
                    print(f'Request timeout fo summoner {summ_id}, continuing...')
                    continue

                except requests.HTTPError as e:
                    print(f'HTTP Error for summoner {summ_id}: {e}')
                    if e.response.status_code == 403 or e.response.status_code == 400:
                        print('Probably expired API key')
                        return
                    elif e.response.status_code == 429:
                        wait = time.sleep(121 - (time.time() - start_time))
                        print(f'Rate limit exceeded, waiting {wait} seconds')
                        api_calls = 0
                        start_time = time.time()
                    # Handle gateway timeouts
                    elif e.response.status_code == 504:
                        print(f'Gateway timeout for summoner {summ_id}, waiting 30 seconds and retrying...')
                        time.sleep(30)
                        # Retry same match matchdata pull
                        continue

                except Exception as e:
                    print(f"Error with match {summ_id}: {str(e)}")
                    continue

        except KeyboardInterrupt:
            print('Process interrupted by user')
            return
        
        print(f'Completed processing {row_counter} summoners')

def collect_match_data(match_ids_file: str, current_patch: str) -> None:
    """Collect detailed match data for match IDs"""
    df_match_ids = pd.read_csv(match_ids_file)

    # Check if match_data csv file/dataframe already exists and create it if it does not
    try:
        existing_data = pd.read_csv(MATCH_DATA_FILE)
        # Find already processed matches to skip them
        processed_match_ids = set(existing_data[existing_data['matchData'].notna()]['matchId'])
    except FileNotFoundError:
        existing_data = df_match_ids[['summonerId', 'puuid', 'matchId']].copy()
        processed_match_ids = set()
        
    # Open file in append mode to avoid reading/writing the whole file
    with open(MATCH_DATA_FILE, 'a+', newline='') as f_output:
        writer = csv.writer(f_output)
        # Write header if file is empty
        if f_output.tell() == 0:
            writer.writerow(['summonerId', 'puuid', 'matchId', 'matchData'])

        row_counter = 0
        api_calls = 0
        start_time = time.time()

        # Only process matches not already in the output file
        matches_to_process = [match for match in df_match_ids['matchId'] if match not in processed_match_ids]

        try:
            for match_id in tqdm(matches_to_process, desc='Fetching match data'):
                api_calls, start_time = handle_rate_limit(api_calls, start_time)

                match_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"

                try:
                    # Add timeout to requests
                    response = requests.get(match_url, headers=HEADERS, timeout=10)
                    response.raise_for_status()
                    match_data = response.json()

                    # Check response structure and patch
                    if 'info' in match_data and 'gameVersion' in match_data['info']:
                        #game_patch = (match_data["info"]["gameVersion"].split("."))[0] + "." + (match_data["info"]["gameVersion"].split("."))[1]
                        version_parts = match_data["info"]["gameVersion"].split(".")
                        game_patch = version_parts[0] + "." + version_parts[1]

                        # Get corresponding row
                        row_data = df_match_ids[df_match_ids['matchId'] == match_id].iloc[0]

                        if game_patch == current_patch:
                            # Write directly into file instead of a growing dataframe
                            writer.writerow([
                                row_data['summonerId'],
                                row_data['puuid'],
                                match_id,
                                json.dumps(match_data)
                            ])
                        else:
                            writer.writerow([
                                row_data['summonerId'],
                                row_data['puuid'],
                                match_id,
                                f'Patch Error: Patch {game_patch}'
                            ])

                    time.sleep(0.05)
                    api_calls += 1
                    row_counter += 1

                    # Force file write and clear memory periodically
                    if row_counter % 100 == 0:
                        f_output.flush()
                        gc.collect()
                
                except requests.Timeout:
                    print(f'Request timeout for match {match_id}, continuing...')
                    continue

                except requests.HTTPError as e:
                    print(f'HTTP Error for match {match_id}: {e}')
                    if e.response.status_code == 403 or e.response.status_code == 400:
                        print('Probably expired API key')
                        return
                    elif e.response.status_code == 429:
                        wait = time.sleep(121 - (time.time() - start_time))
                        print(f'Rate limit exceeded, waiting {wait} seconds')
                        api_calls = 0
                        start_time = time.time()
                    # Handle gateway timeouts
                    elif e.response.status_code == 504:
                        print(f'Gateway timeout for match {match_id}, waiting 30 seconds and retrying...')
                        time.sleep(30)
                        # Retry same match matchdata pull
                        continue

                    
                except Exception as e:
                    print(f"Error with match {match_id}: {str(e)}")
                    continue

        except KeyboardInterrupt:
            print('Process interrupted by user')
            return
        
        print(f'Completed processing {row_counter} matches')

def create_small_match_ids(match_ids_file: str, matches_per_summoner: int = 5, sample_fraction: float = 0.25) -> None:
    """Create a smaller match IDs file with limited matches per summoner and sampled PUUIDs
    Args:
        match_ids_file: Path to original match IDs file
        matches_per_summoner: Number of most recent matches to keep per summoner
        sample_fraction: Fraction of PUUIDs to keep (default: 0.25 for 25%)
    """
    df = pd.read_csv(match_ids_file)
    
    # Get unique PUUIDs and randomly sample a fraction of them
    unique_puuids = df['puuid'].unique()
    sample_size = int(len(unique_puuids) * sample_fraction)
    sampled_puuids = np.random.choice(unique_puuids, size=sample_size, replace=False)
    
    # Filter dataframe to only include sampled PUUIDs
    df_sampled = df[df['puuid'].isin(sampled_puuids)]
    
    # Sort matches within each PUUID group to ensure most recent are first
    # Match IDs are chronological - higher values are more recent
    df_sampled = df_sampled.sort_values(['puuid', 'matchId'], ascending=[True, False])
    
    # Group by puuid and keep only the first n matches (most recent) for each
    df_small = df_sampled.groupby('puuid').head(matches_per_summoner)
    
    # Save the smaller dataset
    df_small.to_csv("match_ids_small.csv", index=False)
    tqdm.write(f"Created smaller match IDs file with {len(df_small)} matches")
    tqdm.write(f"Unique summoners: {df_small['puuid'].nunique()} (from original {len(unique_puuids)})")

def main(start_phase: str = "summoner_ids") -> None:
    """Main execution function with configurable starting phase"""
    if start_phase == "create_small":
        create_small_match_ids(MATCH_IDS_FILE)
        collect_match_data("match_ids_small.csv", CURRENT_PATCH)
    
    elif start_phase == 'summoner_ids':
        collect_summoner_ids()
        #time.sleep(120)  # Rate limit break
        
    elif start_phase == "puuids":       
        get_puuid(SUMMONER_IDS_FILE)
        #time.sleep(120)  # Rate limit break
        
        #collect_match_ids(SUMMONER_PUUID_FILE, PATCH_START_TIME, PATCH_END_TIME, matches_per_summoner=100)
        #time.sleep(120)  # Rate limit break
        
        #collect_match_data("match_ids_small.csv", CURRENT_PATCH)
    
    elif start_phase == "match_ids":
        collect_match_ids(SUMMONER_PUUID_FILE, PATCH_START_TIME, PATCH_END_TIME, matches_per_summoner=100)
        #time.sleep(120)  # Rate limit break
        
        #collect_match_data("match_ids_small.csv", CURRENT_PATCH)
    
    elif start_phase == "match_data":
        collect_match_data(MATCH_IDS_FILE, CURRENT_PATCH)

if __name__ == "__main__":
    main(start_phase="puuids")  # This will create small file and then collect match data 