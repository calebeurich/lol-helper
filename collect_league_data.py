import pandas as pd
import os
import requests
import json
from dotenv import load_dotenv
import time
from tqdm import tqdm
from typing import List, Optional

# Load environment variables and set up
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
HEADERS = {
    "X-Riot-Token": RIOT_API_KEY
}

# Constants
CURRENT_PATCH = "15.3"
APEX_TIERS = ['challengerleagues', 'grandmasterleagues', 'masterleagues']
DIVISIONS = ['DIAMOND']  # Can be expanded to ['DIAMOND', 'EMERALD']
TIERS = ["I", "II"] # can be expanded to ["I", "II", "III", "IV"]

# File paths
SUMMONER_IDS_FILE = "summoner_ids.csv"
SUMMONER_PUUID_FILE = "summoner_and_puuids.csv"
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
    all_sum_ids.extend(get_sums_for_reg_divisions(start))
    
    # Remove duplicates and save
    all_sum_ids = list(set(all_sum_ids))
    pd.DataFrame({"summonerId": all_sum_ids}).to_csv(SUMMONER_IDS_FILE, index=False)
    
    tqdm.write(f"Total unique summoner IDs collected: {len(all_sum_ids)}")

def get_puuid(summoner_ids_file: str) -> None:
    """Fetch PUUIDs for summoner IDs"""
    df = pd.read_csv(summoner_ids_file)
    api_calls = 0
    start_time = time.time()
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Fetching PUUIDs"):
        api_calls, start_time = handle_rate_limit(api_calls, start_time)
        
        summoner_id = row["summonerId"]
        api_url = f"https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
        
        try:
            resp = requests.get(api_url, headers=HEADERS)
            resp.raise_for_status()
            summoner_info = resp.json()
            df.at[index, "puuid"] = summoner_info.get("puuid", "Not Found")
        except requests.RequestException as e:
            tqdm.write(f"Error fetching puuid for {summoner_id}: {e}")
        
        time.sleep(0.05)
        api_calls += 1
    
    df.to_csv(summoner_ids_file, index=False)

def collect_match_ids(summoner_data_file: str) -> None:
    """Collect match IDs for summoners"""
    df_summoners = pd.read_csv(summoner_data_file)
    df_match_ids = pd.DataFrame(columns=["summonerId", "puuid", "matchId"])
    matches_per_summoner = 100
    
    api_calls = 0
    start_time = time.time()
    
    for _, row in tqdm(df_summoners.iterrows(), total=len(df_summoners), desc="Fetching match IDs"):
        api_calls, start_time = handle_rate_limit(api_calls, start_time)
        
        match_history_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{row['puuid']}/ids?start=0&count={matches_per_summoner}"
        
        try:
            match_ids = requests.get(match_history_url, headers=HEADERS).json()
            
            # Create DataFrame rows for each match ID
            if match_ids:
                new_rows = []
                for match_id in match_ids:
                    new_rows.append({
                        "summonerId": row["summonerId"],
                        "puuid": row["puuid"],
                        "matchId": match_id
                    })
                df_match_ids = pd.concat([df_match_ids, pd.DataFrame(new_rows)], ignore_index=True)
            
            time.sleep(0.05)
            api_calls += 1
                
        except requests.RequestException as e:
            tqdm.write(f"Error fetching match IDs for {row['puuid']}: {e}")
    
    df_match_ids = df_match_ids.drop_duplicates("matchId")
    df_match_ids.to_csv(MATCH_IDS_FILE, index=False)
    tqdm.write(f"Match ID statistics:\n{df_match_ids.nunique()}")

def collect_match_data(match_ids_file: str, current_patch: str) -> None:
    """Collect detailed match data for match IDs"""
    df_match_ids = pd.read_csv(match_ids_file)
    df_match_data = pd.DataFrame(columns=["matchId", "matchData"])
    
    api_calls = 0
    start_time = time.time()
    
    for _, row in tqdm(df_match_ids.iterrows(), total=len(df_match_ids), desc="Fetching match data"):
        api_calls, start_time = handle_rate_limit(api_calls, start_time)
        
        match_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{row['matchId']}"
        
        try:
            match_data = requests.get(match_url, headers=HEADERS).json()
            
            # Check if match is from current patch
            game_patch = (match_data["info"]["gameVersion"].split("."))[0] + "." + (match_data["info"]["gameVersion"].split("."))[1]
            if game_patch == current_patch:
                new_row = {
                    "matchId": row["matchId"],
                    "matchData": json.dumps(match_data)
                }
                df_match_data = pd.concat([df_match_data, pd.DataFrame([new_row])], ignore_index=True)
            
            time.sleep(0.05)
            api_calls += 1
                
        except requests.RequestException as e:
            tqdm.write(f"Error fetching match data for {row['matchId']}: {e}")
    
    df_match_data.to_csv(MATCH_DATA_FILE, index=False)
    tqdm.write(f"Match data statistics:\n{df_match_data.nunique()}")

def main(start_phase: str = "summoner_ids") -> None:
    """Main execution function with configurable starting phase"""
    if start_phase == "summoner_ids":
        collect_summoner_ids()
        time.sleep(120)  # Rate limit break
        
        get_puuid(SUMMONER_IDS_FILE)
        time.sleep(120)  # Rate limit break
        
        collect_match_ids(SUMMONER_PUUID_FILE)
        time.sleep(120)  # Rate limit break
        
        collect_match_data(MATCH_IDS_FILE, CURRENT_PATCH)
    
    elif start_phase == "match_ids":
        collect_match_ids(SUMMONER_PUUID_FILE)
        time.sleep(120)  # Rate limit break
        
        collect_match_data(MATCH_IDS_FILE, CURRENT_PATCH)
    
    elif start_phase == "match_data":
        collect_match_data(MATCH_IDS_FILE, CURRENT_PATCH)

if __name__ == "__main__":
    main(start_phase="match_ids")  # Change this to control where to start 