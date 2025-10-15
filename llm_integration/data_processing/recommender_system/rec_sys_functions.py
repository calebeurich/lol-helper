from typing import Dict, List, Set, Tuple, Optional
from llm_integration.data_processing.user_data_compiling.pandas_user_data_aggregation import InsufficientSampleError
import pandas as pd
import numpy as np


def extract_vector(df, criterion, chosen_champion, minimum_games):

    if criterion == "user_choice":
        filtered_row = df.loc[df["champion_name"] == chosen_champion]
        if filtered_row.empty:
            raise KeyError(f"No rows found for champion name {chosen_champion}")
        if len(filtered_row) > 1:
            raise ValueError(f"Data validity issue, more than one row for champion {chosen_champion}")
        return filtered_row, 1

    elif criterion in {"win_rate", "role_play_rate"}:
        candidates = df.loc[df["total_games_per_champion"] >= minimum_games]
        if candidates.empty:
            raise InsufficientSampleError("champion games for your desired criterion or champion")
        # include ties
        try:
            filtered_row = candidates.nlargest(1, criterion, keep="all")
        except TypeError:
            max_val = candidates[criterion].max()
            filtered_row = candidates.loc[candidates[criterion] == max_val]
    else:
        raise ValueError("Invalid criterion")

    n = len(filtered_row)
    if n > 1:
        # return names for disambiguation
        return filtered_row["champion_name"].tolist(), n

    return filtered_row, n

