import argparse, os, glob
import pandas as pd
import json

PATCH = os.getenv("PATCH", "15_6")

def concat_raw_data(local_dir: str) -> pd.DataFrame:
    paths = glob.glob(os.path.join(local_dir, "**", "*.csv"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No CSVs found under: {local_dir}")
    dfs = [pd.read_csv(p) for p in paths]
    print(f"Concatenating {len(dfs)} batches")
    return pd.concat(dfs, ignore_index=True)

def create_df_participants(df_all_matches: pd.DataFrame) -> pd.DataFrame:
    # Parse match_data to dicts (handles strings safely)
    def _to_obj(m):
        if isinstance(m, dict):
            return m
        if isinstance(m, str):
            try:
                return json.loads(m)
            except Exception:
                return {}
        return {}

    df_all_matches = df_all_matches.copy()
    df_all_matches["match_data"] = df_all_matches["match_data"].apply(_to_obj)

    # Filter ranked games only (queueId = 420)
    df_all_matches["queue_id"] = df_all_matches["match_data"].apply(
        lambda m: m.get("info", {}).get("queue_id")
    )
    df_all_matches = df_all_matches[df_all_matches["queue_id"] == 420].copy()

    # Extract participants and explode
    df_all_matches["participants"] = df_all_matches["match_data"].apply(
        lambda m: m.get("info", {}).get("participants", [])
    )
    df_exploded = df_all_matches.explode("participants").reset_index(drop=True)

    # Flatten participant dicts and attach match_id
    df_participants = pd.json_normalize(df_exploded["participants"], sep="_")
    df_participants["match_id"] = df_exploded["match_data"].apply(
        lambda m: m.get("metadata", {}).get("match_id")
    )
    return df_participants

def assign_partner(raw_df):
    raw_df["team_position"] = raw_df["match_data"][""]
    # Group by matchId and teamPosition
    grouped = raw_df.groupby(['match_id', 'team_position'], group_keys=False)
    
    # Create a helper function to find opponent champion
    def find_opponent_champion(group):
        if len(group) != 2:
            return pd.Series(['Opp Champ Error'] * len(group), index=group.index)
        
        # Get champion names in the group
        champs = group['champion_name'].tolist()
        
        # Create a Series with opponent champions
        opponent_champs = pd.Series(
            [champs[1] if champ == champs[0] else champs[0] 
             for champ in group['champion_name']],
            index=group.index
        )
        
        return opponent_champs
    
    # Apply the function to each group with include_groups=False
    return grouped.apply(find_opponent_champion, include_groups=False)

def create_counter_stats_df(raw_df):

    raw_df['opponent_champion'] = assign_partner(raw_df)
    raw_df['champion_and_role'] = raw_df['champion_name'] + '_' + raw_df['team_position']

    agg_counters_df = raw_df.groupby(['champion_and_role', 'opponent_champion']).agg(
        win_rate = ('win_x', 'mean'),
        game_count = ('match_id', 'count')
    ).reset_index()

    win_rate_pivot_df = agg_counters_df.pivot(index='champion_and_role', columns='opponent_champion', values='win_rate')
    game_count_pivot_df = agg_counters_df.pivot(index='champion_and_role', columns='opponent_champion', values='game_count')

    #testing_df = filtered_df.filter(['match_id', 'champion_name', 'opponent_champion'])

    return win_rate_pivot_df, game_count_pivot_df

if __name__ == "__main__":
    default_in  = os.getenv("SM_CHANNEL_raw", "/opt/ml/processing/input/raw")
    default_out = os.getenv("SM_OUTPUT_DIR", "/opt/ml/processing/output")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",  default=default_in)
    parser.add_argument("--output-path", default=default_out)
    args = parser.parse_args()

    print(f"Reading CSVs from: {args.input_path}")
    raw_df = concat_raw_data(args.input_path)
    print(f"Loaded rows: {len(raw_df):,}")
    filtered_df = create_df_participants(raw_df)

    win_rate_pivot_df, game_count_pivot_df = create_counter_stats_df(filtered_df)

    out_dir = os.path.join(args.output_path, f"counter_stats/patch_{PATCH}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing outputs to: {out_dir}")

    win_rate_pivot_df.to_csv(os.path.join(out_dir, "win_rate_pivot_df.csv"), index=False)
    game_count_pivot_df.to_csv(os.path.join(out_dir, "game_count_pivot_df.csv"), index=False)

    print("Job completed.")