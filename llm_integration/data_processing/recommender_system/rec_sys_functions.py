from typing import Dict, List, Set, Tuple, Optional
from data_processing.user_data_compiling.pandas_user_data_aggregation import InsufficientSampleError
import pandas as pd
import numpy as np

MINIMUM_GAMES = 10

def filter_by_champion(user_df: pd.DataFrame, champion_name: str):
    filtered_row =  user_df[user_df["champion_name"] == champion_name]
    if filtered_row.empty or int(filtered_row["total_games_played_in_role"]) < MINIMUM_GAMES:
        raise InsufficientSampleError("champion games")
    
    if len(filtered_row) > 1:
        # This shouldn't happen if champions per role are unique
        raise ValueError(f"Data integrity issue: Multiple rows with champion_name {champion_name}")
    
    return filtered_row.iloc[0] # Convert to dict in chatbot code


def filter_by_criterion(df: pd.DataFrame, criterion: str):
    if df.empty or df[criterion].dropna().empty:
        raise ValueError(f"No valid values in column '{criterion}'")
    
    index = df[criterion].idxmax()
    return df.loc[index]