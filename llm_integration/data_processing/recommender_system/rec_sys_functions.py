from typing import Dict, List, Set, Tuple, Optional
from llm_integration.data_processing.user_data_compiling.pandas_user_data_aggregation import InsufficientSampleError
from clustering.champion_x_role_clustering_script import ROLE_CONFIG, labels, raw_clustering_features
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import StandardScaler
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


def filter_user_and_global_dfs(role, user_vector, global_users_df, minimum_games):
    # Drop unwanted columns from user_vector
    cols_to_drop = ROLE_CONFIG[role.upper()]["exclude_features"]

    all_feats = set(labels) | set(raw_clustering_features) 
    keep_cols = [
        col for col in all_feats
        if col in user_vector.columns and col not in cols_to_drop
    ]

    # Filter user_vector
    filtered_user_vector = user_vector[keep_cols].fillna(0).copy()
    
    # Insert "puuid" as the third column (index 2)
    if "puuid" in filtered_user_vector.columns:
        filtered_user_vector = filtered_user_vector.drop(columns=["puuid"])
    filtered_user_vector.insert(2, "puuid", "current_user")
    
    # Drop unwanted columns from global_users_df
    cols_to_drop_from_champ = [
        "team_position",
        "mode_individual_position",
        "mode_lane",
        "mode_role"
    ]
    
    meta_cols = ["total_games_played_in_role", "champion_name", "puuid"]
    
    # Combine feature columns with meta columns
    keep_cols_global = list(set(keep_cols) | set(meta_cols))
    keep_cols_global = [col for col in keep_cols_global if col in global_users_df.columns]
    
    filtered_global_user_df = global_users_df[keep_cols_global].fillna(0).copy()
    
    # Drop the columns that exist in the filtered df
    cols_to_actually_drop = [col for col in cols_to_drop_from_champ if col in filtered_global_user_df.columns]
    if cols_to_actually_drop:
        filtered_global_user_df = filtered_global_user_df.drop(columns=cols_to_actually_drop)
    
    # Filter by minimum games
    if "total_games_played_in_role" in filtered_global_user_df.columns:
        filtered_global_user_df = filtered_global_user_df[
            filtered_global_user_df["total_games_played_in_role"] >= minimum_games
        ]
    else:
        print(f"Warning: 'total_games_played_in_role' not found in global_df, skipping minimum games filter")
    
    # Verify all required columns exist in filtered_user_vector but exclude meta columns that are only needed for filtering
    global_feature_cols = [col for col in filtered_global_user_df.columns if col not in ["total_games_played_in_role", "champion_name", "user_name"]]
    user_cols = filtered_user_vector.columns
    missing_cols = set(global_feature_cols) - set(user_cols)
    
    if len(missing_cols) > 0:
        missing_list = list(missing_cols)
        raise ValueError(
            f"The following required columns are missing in user_vector: {missing_list}"
        )
    
    # Reorder filtered_user_vector to match the feature columns of filtered_global_user_df
    # Only align on feature columns, not meta columns
    common_cols = [col for col in filtered_global_user_df.columns if col in filtered_user_vector.columns]
    filtered_user_vector = filtered_user_vector[common_cols]
    
    return filtered_user_vector, filtered_global_user_df


def similar_playstyle_users(user_vector, global_users_df, top_k=3):
    """Simplest effective approach - top N similar players."""
    # Separate meta columns
    meta_cols = ["champion_name", "puuid", "total_games_played_in_role"]
    feature_cols = [col for col in global_users_df.columns if col not in meta_cols]
    
    # Debug: Check which columns have NaN values
    user_nan_cols = user_vector[feature_cols].columns[user_vector[feature_cols].isna().any()].tolist()
    global_nan_cols = global_users_df[feature_cols].columns[global_users_df[feature_cols].isna().any()].tolist()
    
    if user_nan_cols:
        print(f"Columns with NaN in user_vector: {user_nan_cols}")
        print(f"NaN count per column in user_vector:")
        print(user_vector[user_nan_cols].isna().sum())
    
    if global_nan_cols:
        print(f"Columns with NaN in global_df: {global_nan_cols}")
        print(f"NaN count per column in global_df:")
        print(global_users_df[global_nan_cols].isna().sum())

    # Standardize and calculate similarities
    scaler = StandardScaler()
    global_features = scaler.fit_transform(global_users_df[feature_cols])
    user_features = scaler.transform(user_vector[feature_cols].values.reshape(1, -1))
    similarities = cosine_similarity(user_features, global_features)[0]
    
    # Get top 100 most similar players and aggregate by champion
    top_100_idx = np.argsort(similarities)[-100:]
    
    return (
        global_users_df.iloc[top_100_idx]
        .assign(similarity=similarities[top_100_idx])
        .groupby("champion_name")["similarity"]
        .agg(["mean", "count"])
        .query("count >= 3")
        .nlargest(top_k, "mean")
        .index
        .tolist()
    )


def recommend_champions_from_main(user_champion, global_users_df, top_k=3):

    # Find all players who play the user's champion
    main_champion_players = global_users_df[
        global_users_df["champion_name"] == user_champion
    ].copy()
    
    # Get list of these players
    relevant_players = main_champion_players["puuid"].unique()
    
    # Find what other champions these players play
    other_champions = global_users_df[
        (global_users_df["puuid"].isin(relevant_players)) &
        (global_users_df["champion_name"] != user_champion)
    ].copy()
    
    # Create player weights based on their games with the main champion
    player_weights = main_champion_players.groupby("puuid")["total_games_played_in_role"].max()
    
    # Normalize weights (sum to 1 for each player)
    player_weights = np.log1p(player_weights)  # Log scale to prevent extreme weights
    player_weights = player_weights / player_weights.sum()
    
    # Add weights to other champions dataframe
    other_champions["player_weight"] = other_champions["puuid"].map(player_weights)
    
    # Calculate weighted recommendation score
    champion_scores = (
        other_champions
        .groupby("champion_name")
        .agg({
            "player_weight": "sum",  # Weighted support
            "puuid": "nunique",  # Unique player count
            "total_games_played_in_role": "mean" if "total_games_played_in_role" in other_champions.columns else "size"
        })
        .rename(columns={
            "player_weight": "weighted_score",
            "user_name": "player_count",
            "games_played": "avg_games"
        })
    )
    
    return (
        champion_scores
        .nlargest(top_k, 'weighted_score')
        .index
        .tolist()
    )


def find_best_alternative(df, user_champion, minimum_games=10):
    
    filtered_df = df[df["number_of_games"] >= minimum_games].copy()
    wr_matrix = filtered_df.pivot(index="champion_name", columns="opp_champion_name", values="win_rate")
    # Normalize with z-scores to remove outlier bias and meta strength bias
    wr_matrix = (wr_matrix - wr_matrix.mean(axis=1).values[:, None]) / wr_matrix.std(axis=1).values[:, None]
    # Ensure all champions appear in rows and columns
    champions = sorted(set(wr_matrix.index) | set(wr_matrix.columns))
    wr_matrix = wr_matrix.reindex(index=champions, columns=champions)
    # Make it symmetric using reverse values when missing
    wr_matrix = wr_matrix.combine_first(wr_matrix.T)
    # Compute cosine distance
    user_vector = wr_matrix.loc[user_champion].values.reshape(1, -1) # Convert series into 2D row vector
    cosine_dist_scores = cosine_distances(user_vector, wr_matrix.values)[0]

    alternatives = pd.Series(cosine_dist_scores, index=wr_matrix.index)
    alternatives = alternatives.drop(user_champion).sort_values(ascending=False)

    return alternatives.head(3).tolist()


def find_recs_within_cluster(df, user_champion):

    user_vector = df.loc[user_champion].values.reshape(1, -1)
    
    sim_scores = cosine_similarity(user_vector, df.fillna(0).values)[0]
    similar_champs = pd.Series(sim_scores, index=df.index)
    similar_champs = similar_champs.drop(user_champion).sort_values(ascending=False).head(3).tolist()

    cosine_dist_scores = cosine_distances(user_vector, df.values)[0]
    different_champs = pd.Series(cosine_dist_scores, index=df.index)
    different_champs = different_champs.drop(user_champion).sort_values(ascending=False).head(3).tolist()

    return similar_champs, different_champs
    


