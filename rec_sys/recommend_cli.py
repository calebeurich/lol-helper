import argparse
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from recommendation_functions import (
    DataPaths,
    DataRepository,
    normalize_schema,
    recommend_for_user,
)


def infer_role_and_user(repo: DataRepository) -> (str, str):
    # Prefer single user dataset for speed
    user_df_raw = repo.get_dataframe("single_user")
    user_df, _ = normalize_schema(user_df_raw, want_role=True, want_champion=True, want_games=True, want_user_id=True)
    if user_df.empty:
        raise RuntimeError("single_user dataset is empty; cannot infer role/user")

    if "user_id" in user_df.columns:
        agg = (
            user_df.groupby(["user_id", "role"], as_index=False)["games_played"].sum()
            .sort_values("games_played", ascending=False)
        )
        return str(agg.iloc[0]["role"]), str(agg.iloc[0]["user_id"])
    else:
        agg = (
            user_df.groupby(["role"], as_index=False)["games_played"].sum()
            .sort_values("games_played", ascending=False)
        )
        return str(agg.iloc[0]["role"]), "single_user"


def infer_feature_columns(repo: DataRepository, role: str, exclude_cols: Optional[List[str]] = None) -> Optional[List[str]]:
    champ_raw = repo.get_dataframe("champion_x_role")
    champ_df, _ = normalize_schema(champ_raw, want_role=True, want_champion=True, want_games=True, want_winrate=True, want_cluster=True)
    if "cluster" in champ_df.columns:
        return None  # clusters present; no need to infer features
    role_df = champ_df.loc[champ_df["role"] == role]
    if role_df.empty:
        return None
    num_df = role_df.select_dtypes(include=[np.number])
    exclude = set(exclude_cols or []) | {"games_played"}
    candidate_cols = [c for c in num_df.columns if c not in exclude]
    if not candidate_cols:
        return None
    variances = num_df[candidate_cols].var().sort_values(ascending=False)
    return [c for c in variances.head(12).index]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="LoL recommendation CLI")
    parser.add_argument("--role", type=str, help="Role to recommend for (e.g., TOP, JUNGLE)")
    parser.add_argument("--user-id", type=str, help="User identifier (puuid/user_id). Optional for single user dataset")
    parser.add_argument("--consider-top-n", type=int, default=3, help="How many of the user's top played champions to seed from")
    parser.add_argument("--within-cluster", action="store_true", help="Use same-cluster candidates when available")
    parser.add_argument("--k-per-seed", type=int, default=5, help="Candidates per seed champion")
    parser.add_argument("--exclude-played", action="store_true", help="Exclude user's seed champions from final candidates")
    parser.add_argument("--min-games", type=int, default=5, help="Minimum games to count as a top champion seed")
    parser.add_argument("--prefer-single-user", action="store_true", help="Prefer single_user dataset for performance")
    parser.add_argument("--top-n-output", type=int, default=10, help="Number of recommendations to return")
    parser.add_argument("--feature-cols", type=str, help="Comma-separated feature columns for similarity mode; skip to auto-infer")
    parser.add_argument("--sort-by", type=str, help="Optional champion_x_role metric to sort cluster candidates by")

    args = parser.parse_args()

    bucket = os.getenv("BUCKET")
    processed = os.getenv("PROCESSED_DATA_FOLDER")
    patch = os.getenv("PATCH", "patch_15_6")
    if not bucket or not processed:
        raise RuntimeError("Environment variables BUCKET and PROCESSED_DATA_FOLDER are required")

    repo = DataRepository(DataPaths(bucket=bucket, processed_data_folder=processed, patch=patch))

    role = args.role
    user_id = args.user_id
    if not role or not user_id:
        inferred_role, inferred_user = infer_role_and_user(repo)
        role = role or inferred_role
        user_id = user_id or inferred_user

    # Decide whether to use clusters or similarity; if clusters are unavailable and
    # user did not provide features, attempt inference.
    feature_columns_by_role = None
    if args.feature_cols:
        feature_columns_by_role = {role: [c.strip() for c in args.feature_cols.split(",") if c.strip()]}
    else:
        feature_columns = infer_feature_columns(repo, role)
        if feature_columns:
            feature_columns_by_role = {role: feature_columns}

    payload = recommend_for_user(
        repo=repo,
        user_id=user_id,
        role=role,
        consider_top_n_champions=args.consider_top_n,
        within_cluster=args.within_cluster,
        k_per_seed=args.k_per_seed,
        feature_columns_by_role=feature_columns_by_role,
        exclude_played=args.exclude_played,
        min_games=args.min_games,
        prefer_single_user=args.prefer_single_user or True,
        sort_by=args.sort_by,
        attach_semantics=True,
        top_n_output=args.top_n_output,
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


