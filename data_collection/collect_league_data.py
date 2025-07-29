import pandas as pd
import os
import requests
import json
from dotenv import load_dotenv
import time
from tqdm import tqdm
from typing import List, Optional, Tuple
import numpy as np
import gc
import csv

# Load environment variables and set up
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
HEADERS = {
    "X-Riot-Token": RIOT_API_KEY
}

# Constants - add constants for strings that are used commonly ('puuid', etc.) and consider having constants in separate file
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


def handle_rate_limit(api_calls: int, start_time: float, buffer: int = 5) -> tuple[int, float]:
    """
    Pause execution to respect the Riot API rate limit window if nearing the allowed number of calls.

    Parameters
    ----------
    api_calls : int
        Number of API calls made so far in the current window.
    start_time : float
        Epoch timestamp when the current rate-limit window began.
    buffer : int, optional
        Safety buffer to stay below the hard limit (default is 5).

    Returns
    -------
    api_calls : int
        Reset to 0 if we slept; otherwise unchanged.
    start_time : float
        Updated to now if we slept; otherwise unchanged.
    """
    LIMIT = 100
    WINDOW = 120 # Seconds

    if api_calls >= (LIMIT - buffer):  # 100 requests per 2 min, with buffer
        elapsed = time.time() - start_time
        if elapsed < WINDOW:  # 2 minute window
            sleep_time = WINDOW - elapsed + 1  # Add 1 second buffer
            tqdm.write(f"Rate limit approaching - sleeping for {sleep_time:.1f}s")
            time.sleep(sleep_time)
        start_time = time.time()
        api_calls = 0

    return api_calls, start_time


class RiotAPIClient:
    """
    Client for making rate-limited requests to Riot Games API.
    
    Handles common error scenarios and rate limiting logic to ensure
    compliance with API usage limits.
    
    Attributes
    ----------
    headers : dict
        HTTP headers including API key for authentication.
    api_calls : int
        Current count of API calls in the rate limit window.
    start_time : float
        Unix timestamp marking the start of current rate limit window.
    max_retries : int
        Maximum number of retries for transient errors (default: 2).
        
    Notes
    -----
    - Implements 100 requests per 2 minutes rate limiting
    - 429 errors retry indefinitely (don't count towards max_retries)
    - 504, timeout, and other errors retry up to max_retries times
    - 404 errors don't retry
    - Returns specific error info for failed requests
    """
    
    def __init__(self, headers: dict, initial_api_calls: int = 0, max_retries: int = 2):
        """
        Initialize the API client.
        
        Parameters
        ----------
        headers : dict
            HTTP headers including API key.
        initial_api_calls : int, optional
            Starting count for API calls (default: 0).
        max_retries : int, optional
            Maximum retry attempts for transient errors (default: 2).
        """
        self.headers = headers
        self.api_calls = initial_api_calls
        self.start_time = time.time()
        self.max_retries = max_retries
        
    def request(self, url: str, timeout: int = 10, 
                context: str = "request") -> Tuple[Optional[dict], Optional[str]]:
        """
        Make a rate-limited API request with error handling and retries.
        
        Parameters
        ----------
        url : str
            The API endpoint URL.
        timeout : int, optional
            Request timeout in seconds (default: 10).
        context : str, optional
            Context identifier for error messages (e.g., match_id, puuid).
            
        Returns
        -------
        Tuple[Optional[dict], Optional[str]]
            (response_data, error_type) where:
            - response_data: JSON response if successful, None if failed
            - error_type: None if successful, error description if failed
            
        Raises
        ------
        SystemExit
            When authentication errors (400/403) occur, suggesting expired API key.
        """
        retry_count = 0
        
        while True:
            # Apply rate limiting
            self.api_calls, self.start_time = handle_rate_limit(self.api_calls, self.start_time)
            
            try:
                response = requests.get(url, headers=self.headers, timeout=timeout)
                response.raise_for_status()
                
                # Successful request
                time.sleep(0.05)  # Respect per-second rate limit
                self.api_calls += 1
                return response.json(), None
                
            except requests.Timeout:
                retry_count += 1
                if retry_count <= self.max_retries:
                    tqdm.write(
                        f"Request timeout for {context}, "
                        f"retrying ({retry_count}/{self.max_retries})..."
                    )
                    time.sleep(1 * retry_count)  # Backoff delay
                    continue
                else:
                    tqdm.write(
                        f"Request timeout for {context}, max retries exceeded."
                    )
                    return None, "timeout_error"
                    
            except requests.HTTPError as e:
                if e.response.status_code in (400, 403):
                    tqdm.write('Expired API key')
                    raise SystemExit(1)
                    
                elif e.response.status_code == 429:
                    # Rate limit - retry indefinitely without counting
                    wait_time = 121 - (time.time() - self.start_time)
                    if wait_time < 0:
                        wait_time = 5  # Minimum wait
                    tqdm.write(f"Rate limit exceeded, waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                    self.api_calls = 0
                    self.start_time = time.time()
                    continue  # Don't increment retry_count
                    
                elif e.response.status_code == 504:
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        wait_time = 5 * retry_count  # Progressive backoff
                        tqdm.write(
                            f"Gateway timeout for {context}, "
                            f"waiting {wait_time} seconds and retrying..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        tqdm.write(
                            f"Gateway timeout for {context}, max retries exceeded."
                        )
                        return None, "504_gateway_timeout"
                        
                elif e.response.status_code == 404:
                    # Don't retry 404s
                    tqdm.write(f"Not found: {context}")
                    return None, "404_not_found"
                    
                else:
                    # Other HTTP errors - try again
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        tqdm.write(
                            f"HTTP error {e.response.status_code} for {context}, "
                            f"retrying..."
                        )
                        time.sleep(2)
                        continue
                    else:
                        tqdm.write(
                            f"HTTP error for {context}, max retries exceeded"
                        )
                        return None, f"{e.response.status_code}_http_error"
                        
            except Exception as e:
                retry_count += 1
                if retry_count <= self.max_retries:
                    tqdm.write(
                        f"Error with {context}: {str(e)}, retrying..."
                    )
                    time.sleep(2)
                    continue
                else:
                    tqdm.write(
                        f"Error with {context}: {str(e)}, max retries exceeded"
                    )
                    return None, f"{type(e).__name__}_exception"


def get_summoner_ids_from_api(api_url: str, req_type: str) -> List[str]:
    """
    Fetch summoner IDs from Riot API for a specific league division.
    
    Parameters
    ----------
    api_url : str
        The complete API endpoint URL to fetch summoner data from.
    req_type : str
        Type of league request - either 'apex' (Master, Grandmaster, Challenger)
        or 'regular' (Diamond divisions). Determines the response structure parsing.
    
    Returns
    -------
    List[str]
        List of summoner IDs retrieved from the API response.
        Returns empty list if the request fails.
    
    Notes
    -----
    - Apex leagues return all entries in a single call under 'entries' key
    - Regular leagues return entries directly as a list (one page at a time)
    - Implements a 50ms delay to respect rate limits
    - Uses global HEADERS for API authentication
    """
    time.sleep(0.05)  # Rate limit protection
    
    try:
        response = requests.get(api_url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        if req_type == "apex":
            entries = data.get("entries", [])
            return [entry["summonerId"] for entry in entries]
        elif req_type == "regular":
            return [entry["summonerId"] for entry in response]
        elif req_type is None:
            raise ValueError("Missing request type")
        else:
            raise ValueError(f"Request type must be either 'apex' or 'regular', instead got {req_type}")

    except requests.exceptions.RequestException as e:
        tqdm.write(f"API request failed: {type(e).__name__} - {str(e)}")
        return []
    except (KeyError, ValueError) as e:
        tqdm.write(f"Failed to parse response: {type(e).__name__} - {str(e)}")
        return []
    

def get_sums_for_apex_leagues() -> List[str]:
    """
    Collect summoner IDs from all apex tier leagues.
    
    Retrieves summoner IDs from Master, Grandmaster, and Challenger leagues
    by making one API call per league tier.
    
    Returns
    -------
    List[str]
        Combined list of summoner IDs from all apex leagues.
        
    Notes
    -----
    - Uses global APEX_TIERS constant for league iteration
    - Each apex league returns all entries in a single API call
    - Shows progress using tqdm progress bar
    """
    apex_summoner_ids = []

    for apex_league in tqdm(APEX_TIERS, desc="Fetching apex leagues"):
        api_url = f"https://na1.api.riotgames.com/lol/league/v4/{apex_league}/by-queue/RANKED_SOLO_5x5"
        summoner_ids = get_summoner_ids_from_api(api_url, "apex")
        apex_summoner_ids.extend(summoner_ids)

    return apex_summoner_ids


def get_sums_for_reg_divisions(start_time: float) -> List[str]:
    """
    Collect summoner IDs from regular (non-apex) divisions with pagination.
    
    Iterates through Diamond divisions (IV, III, II, I) and retrieves all
    summoner IDs using paginated API calls, respecting rate limits.
    
    Parameters
    ----------
    start_time : float
        Unix timestamp marking the start of the current rate limit window.
        Used to track and enforce API rate limiting.
        
    Returns
    -------
    List[str]
        Combined list of summoner IDs from all regular divisions.
        
    Notes
    -----
    - Starts with api_calls=3 to account for preceding apex league calls
    - Implements pagination for each division/tier combination
    - Uses handle_rate_limit to ensure compliance with 100 requests/2 minutes
    - Shows nested progress bars for divisions and tiers
    """
    api_calls = 3  # starts at 3 since there should be 3 apex tier calls before this
    reg_division_ids = []
    
    for division in tqdm(DIVISIONS, desc="Processing divisions"):
        for tier in tqdm(TIERS, desc=f"Processing {division} tiers", leave=False):
            page = 1

            while True:
                api_calls, start_time = handle_rate_limit(api_calls, start_time)
                
                api_url = f'https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{division}/{tier}?page={page}'
                summoner_ids = get_summoner_ids_from_api(api_url, "regular")
                
                if not summoner_ids:
                    break
                    
                reg_division_ids.extend(summoner_ids)
                page += 1
                api_calls += 1
                
    return reg_division_ids


def collect_summoner_ids() -> None:
    """
    Retrieve summoner IDs from Apex leagues and regular Diamond+ divisions, deduplicate them,
    and save the unique IDs to a CSV file.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    start_time = time.time()
    tqdm.write("Collecting summoner IDs...")
    
    # Fetch from Apex leagues and regular divisions
    apex_ids = get_sums_for_apex_leagues()
    regular_ids = get_sums_for_reg_divisions(start_time)

    # Deduplicate
    unique_ids = set(apex_ids)
    unique_ids.update(regular_ids)

    # Persist to disk
    df = pd.DataFrame({"summonerId": list(unique_ids)})
    df.to_csv(SUMMONER_IDS_FILE, index=False)
    
    tqdm.write(f"Total unique summoner IDs collected: {len(unique_ids)}")


def get_puuid(summoner_ids_file: str) -> None:
    """
    Fetch PUUIDs for summoner IDs and save to CSV file.
    
    Reads summoner IDs from input file, fetches corresponding PUUIDs from 
    Riot API, and appends results to output CSV. Implements checkpoints 
    to avoid reprocessing already fetched PUUIDs.
    
    Parameters
    ----------
    summoner_ids_file : str
        Path to CSV file containing summoner IDs to process.
        Expected to have a 'summonerId' column.
     
    Returns
    -------
    None
        Results are written directly to SUMMONER_PUUID_FILE.
        
    Notes
    -----
    - Skips summoner IDs that have already been processed
    - Implements rate limiting (100 requests per 2 minutes)
    - Writes to file every 100 records for data safety
    - Handles various API errors with appropriate retry logic
    - Uses append mode to avoid memory issues with large datasets
    """
    summoner_ids_df = pd.read_csv(summoner_ids_file)
    
    # Check for existing processed summoner IDs
    try:
        existing_data = pd.read_csv(SUMMONER_PUUID_FILE)
        processed_summoner_ids = set(existing_data[existing_data["puuid"].notna()]["summonerId"])
    except FileNotFoundError:
        processed_summoner_ids = set()

    # Open file in append mode for efficient writing
    with open(SUMMONER_PUUID_FILE, "a+", newline="") as f_output:
        writer = csv.writer(f_output)

        # Write header if file is empty
        if f_output.tell() == 0:
            writer.writerow(["summonerId", "puuid"])

        row_counter = 0
        api_calls = 0
        start_time = time.time()

        # Filter out already processed summoner IDs
        summoner_ids_to_process = [
            summid for summid in summoner_ids_df["summonerId"] 
            if summid not in processed_summoner_ids
        ]

        try:
            for summoner_id in tqdm(summoner_ids_to_process, desc="Fetching puuids"):
                api_calls, start_time = handle_rate_limit(api_calls, start_time)
                
                api_url = f"https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"

                try:
                    # Add timeout to requests
                    response = requests.get(api_url, headers=HEADERS, timeout=10)
                    response.raise_for_status()
                    summoner_data = response.json()

                    # Extract PUUID and write to file

                    puuid = summoner_data.get("puuid", "PUUID not found")
                    writer.writerow([summoner_id, puuid])

                    time.sleep(0.05) # Respect per-second rate limit
                    api_calls += 1
                    row_counter += 1

                    # Force file write and clear memory periodically
                    if row_counter % 100 == 0:
                        f_output.flush()
                        gc.collect()
                
                except requests.Timeout:
                    tqdm.write(f"Request timeout for summoner {summoner_id}, skipping...")
                    continue

                except requests.HTTPError as e:
                    if e.response.status_code in (400, 403):
                        tqdm.write(f"Authentication error ({e.response.status_code}): Check API key")
                        return
                    elif e.response.status_code == 429:
                        wait_time = max(0, 121 - (time.time() - start_time))
                        tqdm.write(f"Rate limit exceeded, waiting {wait_time:.1f} seconds")
                        time.sleep(wait_time)
                        api_calls = 0
                        start_time = time.time()
                    elif e.response.status_code == 504:
                        tqdm.write(f"Gateway timeout for summoner {summoner_id}, retrying after 30s...")
                        time.sleep(30)
                        continue
                    else:
                        tqdm.write(f"HTTP Error {e.response.status_code} for summoner {summoner_id}")
                        continue
                    
                except Exception as e:
                    tqdm.write(f"Unexpected error for summoner {summoner_id}: {type(e).__name__}")
                    continue

        except KeyboardInterrupt:
            tqdm.write("\nProcess interrupted by user")
            return
        
        tqdm.write(f"Completed processing {row_counter} summoners")


def collect_match_ids(summoner_data_file: str, patch_start_time: str,
                      patch_end_time: str, matches_per_summoner: int = 100) -> None:
    """
    Collect match IDs for summoners within a specific patch timeframe.
    
    Reads summoner data from CSV, fetches match history for each summoner
    during the specified patch period, and saves match IDs to a CSV file.
    Implements checkpointing to avoid reprocessing summoners.
    
    Parameters
    ----------
    summoner_data_file : str
        Path to CSV file containing summoner data with 'summonerId' and 'puuid' columns.
    patch_start_time : str
        Unix timestamp (as string) marking the start of the patch period.
    patch_end_time : str
        Unix timestamp (as string) marking the end of the patch period.
    matches_per_summoner : int, optional
        Maximum number of matches to fetch per summoner (default: 100).
        
    Returns
    -------
    None
        Results are written directly to MATCH_IDS_FILE.
        
    Notes
    -----
    - Only fetches ranked matches (queue=420)
    - Skips summoners that have already been processed
    - Writes to file every 100 summoners for data safety
    - Uses Americas routing for match-v5 API endpoints
    """
    current_patch = CURRENT_PATCH
    df_summoners = pd.read_csv(summoner_data_file)
    

    # Check if match_id csv file already exists and create it if not
    try:
        existing_data = pd.read_csv(MATCH_IDS_FILE)
        # Find already processed summoner ids to skip them
        processed_puuids = set(existing_data[existing_data["matchId"].notna()]["puuid"])
    except FileNotFoundError:
        processed_puuids = set()

    # Initialize API client
    api_client = RiotAPIClient(HEADERS)

    # Open file in append mode to avoide reading/writing the whole file
    with open(MATCH_IDS_FILE, "a+", newline="") as f_output:
        writer = csv.writer(f_output)
        
        # Write header if file is empty
        if f_output.tell() == 0:
            writer.writerow(["summonerId", "puuid", "matchId"])

        row_counter = 0
    
        # Only process summids not already in the output file
        puuids_to_process = [
            puuid for puuid in df_summoners["puuid"]
            if puuid not in processed_puuids
        ]

        try:
            for puuid in tqdm(puuids_to_process, desc="Fetching match IDs"):
                api_calls, start_time = handle_rate_limit(api_calls, start_time)

                match_history_url = (
                    f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/"
                    f"{puuid}/ids?startTime={patch_start_time}&endTime={patch_end_time}"
                    f"&queue=420&start=0&count={matches_per_summoner}"
                ) 

                match_history, error_type = api_client.request(
                    match_history_url,
                    context=f"puuid {puuid}"
                )

                if error_type is not None:
                    track_failed_match(puuid, error_type, current_patch, "match_id")
                
                # Find row matching this PUUID
                row_data = df_summoners[df_summoners["puuid"] == puuid].iloc[0]

                # Write each match ID to file
                for match_id in match_history:
                    writer.writerow([
                        row_data["summonerId"],
                        row_data["puuid"],
                        match_id
                    ])

                row_counter += 1  

                # Periodic flush to ensure memory usage efficiency
                if row_counter % 100 == 0:
                    f_output.flush()
                    gc.collect()

        except KeyboardInterrupt:
            tqdm.write("\nProcess interrupted by user")
            return
        except SystemExit:
            # Propagate SystemExit from API client (authentication errors)
            return

        tqdm.write(f"Completed processing {row_counter} summoners")


def track_failed_request(request_id, error_msg, current_patch, req_type):
    """
    Append one JSON line per failure to failed_<req_type>_<patch>.ndjson
    """
    filename = f"failed_{req_type}_{current_patch}.ndjson"
    try:
        with open(filename, "a", encoding="utf-8") as f:
            record = {req_type: request_id, "error_msg": str(error_msg)}
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        # simple fallback if writing fails
        print(f"Warning: could not track {req_type} {request_id}: {e}")

def match_data_helper(df_match_ids: pd.DataFrame, df_match_data: pd.DataFrame, 
                     offset: int, current_patch: str) -> int:
    """
    Process match IDs and fetch detailed match data from Riot API.
    
    Fetches match data for each match ID, validates patch version, and saves
    data in batches of 1000 matches. Implements resumability through offset
    tracking and logs failed matches for later analysis.
    
    Parameters
    ----------
    df_match_ids : pd.DataFrame
        DataFrame containing match IDs to process with columns:
        'matchId', 'summonerId', 'puuid'.
    df_match_data : pd.DataFrame
        DataFrame to hold in-progress matches not yet saved to file.
        Should be empty DataFrame with correct columns on initial call.
    offset : int
        Number of rows already processed from df_match_ids.
        Used for resuming interrupted processing.
    current_patch : str
        Target patch version (e.g., "14.23") to filter matches.
        
    Returns
    -------
    int
        Updated offset representing total rows processed.
        
    Notes
    -----
    - Saves match data in CSV files every 1000 matches
    - Tracks failed matches in NDJSON format for debugging
    - Updates offset.txt after each batch for resumability
    - Only includes matches from the specified patch version
    """
    BATCH_SIZE = 1000
    current_batch_row_count = 0
    row_counter = 0  # Rows processed in this run
    df_match_data_new = df_match_data.copy()
    failed_match_filename = f"failed_matches_{current_patch}.ndjson"
    current_offset = offset

    # Initialize failed matches file
    if not os.path.exists(failed_match_filename):
        open(failed_match_filename, 'w').close()

    # Count existing failed matches
    with open(failed_match_filename, 'r') as f:
        total_failed_matches = sum(1 for _ in f)
        tqdm.write(f'Total failed matches: {total_failed_matches}')

    # Calculate batch boundaries
    total_matches_added = current_offset - total_failed_matches
    current_batch_start = ((current_offset // BATCH_SIZE) * BATCH_SIZE) + 1
    next_batch_start = current_batch_start + BATCH_SIZE

    # Check if there are rows to process
    total_rows = len(df_match_ids) - current_offset
    if total_rows <= 0:
        tqdm.write("No new rows to process")
        return current_offset

    # Initialize API client
    api_client = RiotAPIClient(HEADERS)

    # Process each match
    for _, row in tqdm(df_match_ids.iloc[offset:].iterrows(), 
                      total=total_rows, 
                      desc='Fetching match data'):
        match_id = row['matchId']
        match_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"

        # Fetch match data with retry logic
        match_data, error_type = api_client.request(
            match_url, 
            context=f"match {match_id}"
        )
        
        if error_type is not None:
            # Request failed after all retries
            track_failed_request(match_id, error_type, current_patch, "match_data")
        else:
            # Process successful response
            if 'info' in match_data and 'gameVersion' in match_data['info']:
                game_patch = '.'.join(match_data['info']['gameVersion'].split('.')[:2])
                
                if game_patch == current_patch:
                    # Add match to batch
                    new_row = {
                        'summonerId': row['summonerId'],
                        'puuid': row['puuid'],
                        'matchId': match_id,
                        'matchData': json.dumps(match_data)
                    }
                    df_match_data_new = pd.concat(
                        [df_match_data_new, pd.DataFrame([new_row])], 
                        ignore_index=True
                    )
                    current_batch_row_count += 1
                else:
                    tqdm.write(f"Wrong patch for match {match_id}: {game_patch}")
                    track_failed_request(match_id, "patch_error", current_patch, "match_data")
            else:
                tqdm.write(f"Unexpected response structure for match {match_id}")
                track_failed_request(match_id, "response_structure_error", current_patch, "match_data")

        row_counter += 1
        current_offset += 1

        # Save batch when reaching BATCH_SIZE
        if current_batch_row_count >= BATCH_SIZE:
            batch_filename = (
                f"patch_{current_patch}_rows_{current_batch_start}_to_{next_batch_start-1}.csv"
            )
            tqdm.write(
                f"Completed batch (rows {current_batch_start}-{next_batch_start-1}), "
                f"saving to {batch_filename}"
            )
            df_match_data_new.to_csv(batch_filename, index=False)

            # Update offset for resumability
            with open('offset.txt', 'w') as f:
                f.write(str(current_offset))

            # Reset for next batch
            df_match_data_new = pd.DataFrame(
                columns=['summonerId', 'puuid', 'matchId', 'matchData']
            )

            # Update failed match statistics
            with open(failed_match_filename, 'r') as f:
                updated_total_failed_matches = sum(1 for _ in f)

            count_new_failed_matches = updated_total_failed_matches - total_failed_matches
            total_matches_added = current_offset - updated_total_failed_matches
            total_failed_matches = updated_total_failed_matches
                
            current_batch_row_count = 0
            current_batch_start = ((current_offset // BATCH_SIZE) * BATCH_SIZE) + 1
            next_batch_start = current_batch_start + BATCH_SIZE
            
            tqdm.write(
                f"{row_counter} rows processed this run, "
                f"{count_new_failed_matches} new failed matches"
            )
            gc.collect()

    # Save final partial batch if exists
    if not df_match_data_new.empty:
        # Recalculate total matches for accurate filename
        with open(failed_match_filename, 'r') as f:
            final_failed_matches = sum(1 for _ in f)
        total_matches_added = current_offset - final_failed_matches
        
        batch_filename = (
            f"patch_{current_patch}_rows_{current_batch_start}_to_{total_matches_added}.csv"
        )
        tqdm.write(f"Saving final batch: rows_{current_batch_start}_to_{total_matches_added}")
        df_match_data_new.to_csv(batch_filename, index=False)
        
    # Update final offset
    with open('offset.txt', 'w') as f:
        f.write(str(current_offset))
        
    tqdm.write(f"Completed all rows, {row_counter} rows processed in this run")
    gc.collect()
    
    return current_offset

def collect_match_data(match_ids_file: str, current_patch: str) -> None:
    """Main function to collect match data."""
    # Load match IDs
    df_match_ids = pd.read_csv(match_ids_file)
    df_match_ids = df_match_ids.drop_duplicates('matchId') # Move this to matchId function 

    # Initialize empty DataFrame for match data
    df_match_data = pd.DataFrame(columns=['summonerId', 'puuid', 'matchId', 'matchData'])

    # Get the current offset
    if os.path.exists('offset.txt'):
        with open('offset.txt', 'r') as f:
            content = f.read().strip()
            offset = int(content) if content else 0
    else:
        offset = 0
    
    tqdm.write(f"Starting from offset {offset}")

    try:
        # Process matches
        current_offset = match_data_helper(df_match_ids, df_match_data, offset, current_patch)
    
    except (KeyboardInterrupt, requests.exceptions.HTTPError):
        tqdm.write(f"Code interrupted, last batch progress was not saved")
          
    tqdm.write(f"Task completed successfully, processed total {current_offset} rows")
    
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
    main(start_phase="match_data")  # This will create small file and then collect match data 