from typing import Dict, List, Set, Tuple, Optional
from data_processing.user_data_compiling.pandas_user_data_aggregation import InsufficientSampleError
import pandas as pd
import numpy as np

queue_map = {"draft":[400], "ranked_solo_queue":[420], "ranked_including_flex":[420,440]}

def find_available_champions(df: pd.DataFrame, queue_type: str):
    filtered_df = df[df["queue_id"].isin(queue_map[queue_type])].copy() if queue_type.isin(queue_map) else df.copy()

def extract_vector(df: pd.DataFrame, criterion: str, minimum_games: int):
    if criterion not in {"win_rate", "role_play_rate"}:
        filtered_row = df[df["champion_name"] == criterion]
    else:
        filtered_row = df.iloc[df[criterion].idxmax()]

    if filtered_row.empty or int(filtered_row["total_games_played_in_role"]) < minimum_games:
        raise InsufficientSampleError("champion games for your desired criterion or champion")
    
    if len(filtered_row) > 1:
        # This shouldn't happen if champions per role are unique
        raise ValueError(f"Data integrity issue: Multiple rows with champion_name: {criterion}")
    
    return filtered_row.iloc[0] # Convert to dict in chatbot code

