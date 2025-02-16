import pandas as pd
import os
import requests
import json
from dotenv import load_dotenv
import time
from tqdm import tqdm

# Load environment variables and set up
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
headers = {
    "X-Riot-Token": RIOT_API_KEY
}

# Constants
CURRENT_PATCH = "15.3"
apex_tiers = ['challengerleagues', 'grandmasterleagues', 'masterleagues']
divisions = ['DIAMOND']  # Can be expanded to ['DIAMOND', 'EMERALD']
tiers = ["I", "II", "III", "IV"]

pd.set_option('display.max_columns', None)

def get_summoner_ids_from_api(api_url, req_type):
    time.sleep(0.05)  # Ensure we don't exceed 20 requests/sec
    summoner_ids = []
    try:
        resp = requests.get(api_url, headers=headers)
        resp.raise_for_status()
    except requests.HTTPError:
        tqdm.write("Couldn't complete request")   
        return summoner_ids
    
    response = resp.json()
    if req_type == "apex":
        entries = response.get('entries', [])
        for entry in entries:
            summoner_ids.append(entry['summonerId'])
    if req_type == "regular":
        for entry in response:
            summoner_ids.append(entry['summonerId'])
    return summoner_ids

def get_sums_for_apex_leagues():
    apex_summoner_ids = []
    for apex_league in tqdm(apex_tiers, desc="Fetching apex leagues"):
        api_url = f"https://na1.api.riotgames.com/lol/league/v4/{apex_league}/by-queue/RANKED_SOLO_5x5"
        summoner_ids = get_summoner_ids_from_api(api_url, "apex")
        apex_summoner_ids += summoner_ids
    return apex_summoner_ids

def get_sums_for_reg_divisions(start_time):
    inner_start_time = start_time
    total_loops = 3  # starts at 3 since there should be 3 apex tier calls before this
    reg_division_ids = []
    
    division_pbar = tqdm(divisions, desc="Processing divisions")
    for division in division_pbar:
        division_pbar.set_postfix_str(division)
        tier_pbar = tqdm(tiers, desc="Processing tiers", leave=False)
        for tier in tier_pbar:
            tier_pbar.set_postfix_str(tier)
            page = 1
            more_pages = True
            while more_pages:
                # Rate limit handling
                if total_loops >= 95:  # Buffer of 5 requests for safety
                    elapsed = time.time() - inner_start_time
                    if elapsed < 120:  # 2 minute window
                        sleep_time = 120 - elapsed + 1  # Add 1 second buffer
                        tqdm.write(f'Rate limit approaching - sleeping for {sleep_time:.1f} seconds')
                        time.sleep(sleep_time)
                    inner_start_time = time.time()
                    total_loops = 0

                api_url = f'https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{division}/{tier}?page={page}'
                output_list = get_summoner_ids_from_api(api_url, "regular")
                
                if output_list:
                    page += 1
                    reg_division_ids += output_list
                else:
                    more_pages = False
                total_loops += 1
    return reg_division_ids

def get_all_wanted_sums():
    all_sum_ids = []
    start = time.time()
    tqdm.write("Collecting summoner IDs...")
    all_sum_ids += get_sums_for_apex_leagues()
    all_sum_ids += get_sums_for_reg_divisions(start)
    
    # Remove duplicates
    tqdm.write(f"Total summoner IDs collected: {len(all_sum_ids)}")
    all_sum_ids = list(set(all_sum_ids))
    tqdm.write(f"Unique summoner IDs: {len(all_sum_ids)}")
    
    return all_sum_ids

def get_puuid(csv_only_summ_ids):
    df_only_summ_ids = pd.read_csv(csv_only_summ_ids, index_col=False)
    
    # Track API calls for rate limiting
    start_time = time.time()
    api_calls = 0
    
    # Create progress bar
    pbar = tqdm(total=len(df_only_summ_ids), desc="Fetching PUUIDs")
    
    for index, row in df_only_summ_ids.iterrows():
        # Rate limit handling
        api_calls += 1
        if api_calls >= 95:  # Buffer of 5 requests
            elapsed = time.time() - start_time
            if elapsed < 120:
                sleep_time = 120 - elapsed + 1
                tqdm.write(f'Rate limit approaching - sleeping for {sleep_time:.1f} seconds')
                time.sleep(sleep_time)
            start_time = time.time()
            api_calls = 0
            
        summoner_id = row["summonerId"]
        api_url = f"https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
        
        try:
            resp = requests.get(api_url, headers=headers)
            resp.raise_for_status()
            summoner_info = resp.json()
            df_only_summ_ids.at[index, "puuid"] = summoner_info.get("puuid", "Not Found")
        except requests.RequestException as e:
            tqdm.write(f"Error fetching puuid for {summoner_id}: {e}")
        
        pbar.update(1)
        time.sleep(0.05)  # Ensure we don't exceed 20 requests/sec
    
    pbar.close()
    df_only_summ_ids.to_csv(csv_only_summ_ids, index=False)

def get_match_ids(csv_summids_and_puuids, current_patch):
    df_summoners = pd.read_csv(csv_summids_and_puuids)
    df_all_matches = pd.DataFrame(columns=["summonerId", "puuid", "matchId", "matchData"])
    batch_size = 100  # Max matches we can request at once
    
    # Track API calls for rate limiting
    start_time = time.time()
    api_calls = 0
    
    # Create progress bar
    pbar = tqdm(total=len(df_summoners), desc="Fetching matches")
    
    for _, row in df_summoners.iterrows():
        summoner_id = row["summonerId"]
        puuid = row["puuid"]
        start_index = 0
        found_old_patch = False
        
        while not found_old_patch:
            # Rate limit handling for match history request
            api_calls += 1
            if api_calls >= 95:
                elapsed = time.time() - start_time
                if elapsed < 120:
                    sleep_time = 120 - elapsed + 1
                    tqdm.write(f'Rate limit approaching - sleeping for {sleep_time:.1f} seconds')
                    time.sleep(sleep_time)
                start_time = time.time()
                api_calls = 0
                
            match_history_api_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start_index}&count={batch_size}"
            
            try:
                resp = requests.get(match_history_api_url, headers=headers)
                resp.raise_for_status()
                match_ids = resp.json()
                
                # If no more matches found, break
                if not match_ids:
                    break
                
                for match_id in match_ids:
                    # Rate limit handling for match data request
                    api_calls += 1
                    if api_calls >= 95:
                        elapsed = time.time() - start_time
                        if elapsed < 120:
                            sleep_time = 120 - elapsed + 1
                            tqdm.write(f'Rate limit approaching - sleeping for {sleep_time:.1f} seconds')
                            time.sleep(sleep_time)
                        start_time = time.time()
                        api_calls = 0
                        
                    match_data_api_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
                    response = requests.get(match_data_api_url, headers=headers)
                    match_data = response.json()
                    
                    game_patch = (match_data["info"]["gameVersion"].split("."))[0] + "." + (match_data["info"]["gameVersion"].split("."))[1]
                    
                    if game_patch == current_patch:
                        df_mini = pd.DataFrame({
                            "summonerId": [summoner_id],
                            "puuid": [puuid],
                            "matchId": [match_id],
                            "matchData": [json.dumps(match_data)]
                        })
                        df_all_matches = pd.concat([df_all_matches, df_mini], ignore_index=True)
                    elif float(game_patch) < float(current_patch):
                        # Found an older patch, no need to look further for this player
                        found_old_patch = True
                        break
                    
                    time.sleep(0.05)  # Ensure we don't exceed 20 requests/sec
                
                if found_old_patch:
                    break
                    
                # Move to next batch of matches
                start_index += batch_size
                    
            except requests.RequestException as e:
                tqdm.write(f"Error fetching matches for {puuid}: {e}")
                break  # Move to next player if there's an error
        
        pbar.update(1)
    
    pbar.close()
    df_all_matches = df_all_matches.drop_duplicates("matchId")
    df_all_matches.to_csv("all_match_data.csv", index=False)
    tqdm.write(f"Match statistics:\n{df_all_matches.nunique()}")
    return df_all_matches

def create_match_df(dataframe_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_all_matches = pd.read_csv(dataframe_csv)
    df_all_matches["matchData"] = df_all_matches["matchData"].apply(json.loads)
    df_participants = create_df_participants(df_all_matches)
    df_teams = create_df_teamdata(df_all_matches)
    return df_participants, df_teams

def create_df_participants(df_all_matches: pd.DataFrame) -> pd.DataFrame:
    df_all_matches["participants"] = df_all_matches["matchData"].apply(lambda m: m["info"]["participants"])
    df_exploded = df_all_matches.explode("participants").reset_index(drop=True)
    df_participants = pd.json_normalize(df_exploded["participants"], sep="_")
    df_participants["matchId"] = df_exploded["matchData"].apply(lambda m: m["metadata"]["matchId"])
    return df_participants

def create_df_teamdata(df_all_matches: pd.DataFrame) -> pd.DataFrame:
    df_all_matches["teams"] = df_all_matches["matchData"].apply(lambda m: m["info"]["teams"])
    df_exploded = df_all_matches.explode("teams").reset_index(drop=True)
    df_teams = pd.json_normalize(df_exploded["teams"], sep="_")
    df_teams["matchId"] = df_exploded["matchData"].apply(lambda m: m["metadata"]["matchId"])
    return df_teams

def aggregate_champion_data(df_participants: pd.DataFrame, df_teams: pd.DataFrame) -> pd.DataFrame:
    # Create a teamId column in participants based on participantId
    df_participants['teamId'] = df_participants['participantId'].apply(lambda x: 100 if x <= 5 else 200)
    
    # Merge participants with their team data
    merged_df = df_participants.merge(
        df_teams,
        on=['matchId', 'teamId'],
        how='left'
    )
    
    # Aggregate by champion
    champion_stats = merged_df.groupby(['championId', 'championName']).agg({
        # Core stats
        'kills': 'mean',
        'deaths': 'mean',
        'assists': 'mean',
        'matchId': 'count',
        
        # Combat stats
        'totalDamageDealtToChampions': 'mean',
        'totalDamageTaken': 'mean',
        'magicDamageDealtToChampions': 'mean',
        'physicalDamageDealtToChampions': 'mean',
        'trueDamageDealtToChampions': 'mean',
        'largestCriticalStrike': 'mean',
        'timeCCingOthers': 'mean',
        'totalHealsOnTeammates': 'mean',
        'totalDamageShieldedOnTeammates': 'mean',
        
        # Economy stats
        'goldEarned': 'mean',
        'goldSpent': 'mean',
        'totalMinionsKilled': 'mean',
        'neutralMinionsKilled': 'mean',
        
        # Vision stats
        'visionScore': 'mean',
        'wardsPlaced': 'mean',
        'wardsKilled': 'mean',
        'visionWardsBoughtInGame': 'mean',
        
        # Team objectives
        'objectives_baron_kills': 'mean',
        'objectives_champion_kills': 'mean',
        'objectives_dragon_kills': 'mean',
        'objectives_inhibitor_kills': 'mean',
        'objectives_riftHerald_kills': 'mean',
        'objectives_tower_kills': 'mean',
        
        # First objective rates
        'objectives_baron_first': 'mean',
        'objectives_dragon_first': 'mean',
        'objectives_inhibitor_first': 'mean',
        'objectives_riftHerald_first': 'mean',
        'objectives_tower_first': 'mean',
        
        # Game length related
        'timePlayed': 'mean',
        'longestTimeSpentLiving': 'mean',
        
        # Multikill stats
        'doubleKills': 'mean',
        'tripleKills': 'mean',
        'quadraKills': 'mean',
        'pentaKills': 'mean',
        
        # Combat achievements
        'firstBloodKill': 'mean',
        'firstTowerKill': 'mean',
        'turretKills': 'mean',
        'inhibitorKills': 'mean',
    }).round(2)
    
    # Rename columns for clarity
    champion_stats = champion_stats.rename(columns={
        'matchId': 'games_played',
        'objectives_baron_first': 'first_baron_rate',
        'objectives_dragon_first': 'first_dragon_rate',
        'objectives_inhibitor_first': 'first_inhibitor_rate',
        'objectives_riftHerald_first': 'first_herald_rate',
        'objectives_tower_first': 'first_tower_rate',
        'firstBloodKill': 'first_blood_rate',
        'firstTowerKill': 'first_tower_kill_rate'
    })
    
    # Calculate derived stats
    champion_stats['kda'] = ((champion_stats['kills'] + champion_stats['assists']) / 
                            champion_stats['deaths']).round(2)
    champion_stats['cs_per_minute'] = ((champion_stats['totalMinionsKilled'] + champion_stats['neutralMinionsKilled']) / 
                                      (champion_stats['timePlayed'] / 60)).round(2)
    champion_stats['gold_per_minute'] = (champion_stats['goldEarned'] / 
                                        (champion_stats['timePlayed'] / 60)).round(2)
    champion_stats['damage_per_minute'] = (champion_stats['totalDamageDealtToChampions'] / 
                                          (champion_stats['timePlayed'] / 60)).round(2)
    
    return champion_stats

def main():
    # 1. Collect summoner IDs
    all_wanted_sum_ids = get_all_wanted_sums()
    sums_df = pd.DataFrame({"summonerId": all_wanted_sum_ids})
    sums_df.to_csv("all_wanted_summoner_ids.csv", index=False)
    tqdm.write("Saved summoner IDs to CSV")

    # 2. Get PUUIDs for all summoners
    get_puuid("all_wanted_summoner_ids.csv")
    tqdm.write("Added PUUIDs to CSV")

    # 3. Get match data
    get_match_ids("all_wanted_summoner_ids.csv", CURRENT_PATCH)
    tqdm.write("Saved match data to CSV")

    # 4. Process match data
    df_participants, df_teams = create_match_df("all_match_data.csv")
    tqdm.write("Created participant and team dataframes")

    # 5. Aggregate champion statistics
    champion_stats = aggregate_champion_data(df_participants, df_teams)
    champion_stats.to_csv("champion_statistics.csv")
    tqdm.write("Saved champion statistics to CSV")
    
    # Display summary
    tqdm.write("\nFinal champion statistics:")
    tqdm.write(f"Total champions analyzed: {len(champion_stats)}")
    tqdm.write("\nTop 5 most played champions:")
    print(champion_stats.sort_values('games_played', ascending=False).head())

if __name__ == "__main__":
    main() 