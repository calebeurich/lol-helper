import os
import json

from dotenv import load_dotenv

import numpy as np
import pandas as pd
from recommendation_functions import (
    DataPaths,
    DataRepository,
    recommend_for_user,
    normalize_schema,
    compute_similar_champions,
    compute_user_champion_winrate_deltas,
    extract_residuals_for_champion,
    attach_semantics_to_candidates,
    attach_cluster_semantics,
)


def main() -> None:
    load_dotenv()

    bucket = os.getenv("BUCKET")
    processed = os.getenv("PROCESSED_DATA_FOLDER")
    patch = os.getenv("PATCH", "patch_15_6")

    if not bucket or not processed:
        raise RuntimeError("BUCKET and PROCESSED_DATA_FOLDER env vars are required (check your .env)")

    repo = DataRepository(DataPaths(bucket=bucket, processed_data_folder=processed, patch=patch))

    # Probe single user dataset for a sample user and role
    single_user_df = repo.get_dataframe("single_user")
    # Normalize to tolerate different column names
    single_user_df, _ = normalize_schema(single_user_df, want_role=True, want_champion=True, want_games=True, want_user_id=True)
    for required in ["role", "champion", "games_played"]:
        if required not in single_user_df.columns:
            raise RuntimeError(f"single_user dataset missing required column after normalization: {required}")

    if "user_id" in single_user_df.columns:
        agg = (
            single_user_df.groupby(["user_id", "role"], as_index=False)["games_played"].sum()
            .sort_values("games_played", ascending=False)
        )
        if agg.empty:
            raise RuntimeError("single_user dataset is empty")
        user_id = str(agg.iloc[0]["user_id"])  # use top user by games
        role = str(agg.iloc[0]["role"])        # corresponding role
    else:
        # No user_id, select most active role
        agg = (
            single_user_df.groupby(["role"], as_index=False)["games_played"].sum()
            .sort_values("games_played", ascending=False)
        )
        if agg.empty:
            raise RuntimeError("single_user dataset is empty")
        user_id = "single_user"
        role = str(agg.iloc[0]["role"])        

    print(f"Testing recommendations for user_id={user_id} role={role}")

    # Try cluster mode first; if clusters are not present, infer similarity features
    champ_role_df_raw = repo.get_dataframe("champion_x_role")
    champ_role_df, _ = normalize_schema(champ_role_df_raw, want_role=True, want_champion=True, want_games=True, want_winrate=True, want_cluster=True)
    within_cluster = "cluster" in champ_role_df.columns

    feature_columns_by_role = None
    if not within_cluster:
        role_df = champ_role_df.loc[champ_role_df["role"] == role]
        num_df = role_df.select_dtypes(include=[np.number])
        exclude_cols = {"games_played"}
        candidate_cols = [c for c in num_df.columns if c not in exclude_cols]
        if candidate_cols:
            variances = num_df[candidate_cols].var().sort_values(ascending=False)
            inferred_features = [c for c in variances.head(12).index]
            feature_columns_by_role = {role: inferred_features}

    payload = recommend_for_user(
        repo=repo,
        user_id=user_id,
        role=role,
        consider_top_n_champions=3,
        within_cluster=within_cluster,
        k_per_seed=5,
        feature_columns_by_role=feature_columns_by_role,
        exclude_played=True,
        min_games=3,
        prefer_single_user=True,
        sort_by=None,
        attach_semantics=True,
        top_n_output=10,
    )

    # Print compact summary
    summary = {
        "user_id": payload.get("user_id"),
        "role": payload.get("role"),
        "seed_champions": payload.get("seed_champions"),
        "recommendations": [
            {
                "champion": r.get("champion"),
                "score": r.get("score"),
                "source": r.get("source"),
                "method": r.get("method"),
            }
            for r in payload.get("recommendations", [])
        ],
    }

    print(json.dumps(summary, indent=2))

    # --- Additional checks ---
    try:
        seeds = summary["seed_champions"]
        first_seed = seeds[0] if seeds else None
        if first_seed:
            print(f"\nTop-5 similar to {first_seed} (inferred features if clusters absent):")
            if feature_columns_by_role:
                sim_df = compute_similar_champions(
                    champ_role_df=champ_role_df,
                    role=role,
                    target_champion=first_seed,
                    feature_columns=feature_columns_by_role.get(role, []),
                    champion_col="champion",
                    role_col="role",
                    k=5,
                )
                print(sim_df.to_string(index=False))
            else:
                print("Skipped: no inferred features (clusters likely present)")
    except Exception as e:
        print(f"Similarity check skipped due to error: {e}")

    # Winrate deltas vs global (requires all-users data with user_id)
    try:
        # Use single-user dataset if user_id is "single_user", otherwise try all-users
        if user_id == "single_user":
            from recommendation_functions import compute_single_user_winrate_deltas
            deltas = compute_single_user_winrate_deltas(
                user_df=single_user_df,
                champ_role_df=champ_role_df,
                role=role,
                champion_col="champion",
                role_col="role",
                user_winrate_col="win_rate",
                global_winrate_col="win_rate",
            )
        else:
            try:
                all_users_raw = repo.get_dataframe("champion_x_role_x_user")
                user_df_for_deltas, _ = normalize_schema(all_users_raw, want_role=True, want_champion=True, want_games=True, want_winrate=True, want_user_id=True)
                deltas = compute_user_champion_winrate_deltas(
                    user_df=user_df_for_deltas,
                    champ_role_df=champ_role_df,
                    user_id=user_id,
                    role=role,
                    champion_col="champion",
                    role_col="role",
                    user_winrate_col="win_rate",
                    global_winrate_col="win_rate",
                )
            except Exception:
                deltas = pd.DataFrame()

        print("\nUser vs global winrate deltas (top 5):")
        print(deltas.head(5).to_string(index=False))
    except Exception as e:
        print(f"Winrate deltas skipped: {e}")

    # Residuals (strengths/weaknesses) if available
    try:
        role_lower = role.lower()
        res_df = repo.get_dataframe("residuals", role=role_lower)
        res_df_norm, _ = normalize_schema(res_df, want_role=True, want_champion=True, want_games=False)
        if first_seed:
            res = extract_residuals_for_champion(
                residuals_df=res_df_norm,
                champion=first_seed,
                role=role,
                role_col="role",
                champion_col="champion",
            )
            print("\nResiduals (strengths/weaknesses) for first seed:")
            print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Residuals skipped: {e}")

    # Attach semantics to first 5 recommendations if available
    try:
        sem_df = repo.get_dataframe("semantic_champion", role=role_lower)
        sem_df_norm, _ = normalize_schema(sem_df, want_role=True, want_champion=True, want_games=False)
        recs_df = pd.DataFrame(payload.get("recommendations", []))
        if not recs_df.empty and "champion" in recs_df.columns:
            enriched = attach_semantics_to_candidates(
                candidates=recs_df,
                semantic_champion_df=sem_df_norm,
                role=role,
                role_col="role",
                champion_col="champion",
            )
            # Attach cluster semantics as well if cluster semantics exist
            try:
                sem_cluster_df = repo.get_dataframe("semantic_cluster", role=role_lower)
                enriched = attach_cluster_semantics(
                    df_with_cluster=enriched,
                    cluster_semantic_df=sem_cluster_df,
                    cluster_col="cluster" if "cluster" in enriched.columns else "cluster",
                )
            except Exception:
                pass
            print("\nRecommendations with semantics (first 5):")
            cols = [c for c in ["champion", "semantic_tags", "semantic_description", "cluster_semantic_tags", "cluster_semantic_description"] if c in enriched.columns]
            print(enriched[cols].head(5).to_string(index=False))
    except Exception as e:
        print(f"Semantics enrichment skipped: {e}")


if __name__ == "__main__":
    main()


