from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import io
import os

import boto3
import numpy as np
import pandas as pd


# -------- Data Access --------


@dataclass
class DataPaths:
    bucket: str
    processed_data_folder: str
    patch: str
    region_name: str = "us-east-2"


class DataRepository:
    """
    Minimal data access layer with simple caching.

    Expected granularities:
      - "single_user": single user's champion x role data
      - "champion_x_role": global champion x role aggregated data
      - "champion_x_role_x_user": all users' champion x role data (heavy)
      - "cluster": per-role cluster centroids/metadata
      - "residuals": per-role champion residuals to clusters
      - "semantic_champion": per-role champion semantic tags and descriptions
      - "semantic_cluster": per-role cluster semantic tags and descriptions

    Note: This class only fetches CSVs from S3 based on naming convention.
    It does not enforce schema; downstream functions validate needed columns.
    """

    def __init__(
        self,
        paths: DataPaths,
        aws_key: Optional[str] = None,
        aws_secret: Optional[str] = None,
    ) -> None:
        self.paths = paths
        self._aws_key = aws_key or os.getenv("AWS_KEY")
        self._aws_secret = aws_secret or os.getenv("AWS_SECRET")
        self._cache: Dict[Tuple[str, Optional[str]], pd.DataFrame] = {}
        self._s3 = boto3.client(
            "s3",
            aws_access_key_id=self._aws_key,
            aws_secret_access_key=self._aws_secret,
            region_name=self.paths.region_name,
        )

    def _key_for(self, granularity: str, role: Optional[str]) -> str:
        p = self.paths
        role_key = role.lower() if isinstance(role, str) else role
        if granularity == "single_user":
            return f"{p.processed_data_folder}/single_user_data/{p.patch}/single_user_aggregated_data.csv"
        if granularity == "champion_x_role":
            return f"{p.processed_data_folder}/champion_x_role/{p.patch}/champion_x_role_aggregated_data.csv"
        if granularity == "champion_x_role_x_user":
            return f"{p.processed_data_folder}/champion_x_role_x_user/{p.patch}/champion_x_role_x_user_aggregated_data.csv"
        if granularity == "cluster":
            if not role:
                raise ValueError("role is required for 'cluster'")
            return f"{p.processed_data_folder}/clusters/{p.patch}/{role_key}_clusters_df.csv"
        if granularity == "residuals":
            if not role:
                raise ValueError("role is required for 'residuals'")
            return f"{p.processed_data_folder}/clusters/{p.patch}/{role_key}_champion_residuals_df.csv"
        if granularity == "semantic_champion":
            if not role:
                raise ValueError("role is required for 'semantic_champion'")
            return f"{p.processed_data_folder}/bedrock_output/{p.patch}/{role_key}_champion_semantic_tags_and_descriptions.csv"
        if granularity == "semantic_cluster":
            if not role:
                raise ValueError("role is required for 'semantic_cluster'")
            return f"{p.processed_data_folder}/bedrock_output/{p.patch}/{role_key}_cluster_semantic_tags_and_descriptions.csv"
        raise ValueError(
            "granularity must be one of: 'single_user', 'champion_x_role', 'champion_x_role_x_user', 'cluster', 'residuals', 'semantic_champion', 'semantic_cluster'"
        )

    def get_dataframe(self, granularity: str, role: Optional[str] = None, use_cache: bool = True) -> pd.DataFrame:
        cache_key = (granularity, role)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        key = self._key_for(granularity, role)
        obj = self._s3.get_object(Bucket=self.paths.bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        if use_cache:
            self._cache[cache_key] = df
        return df


# -------- Core Utilities --------


ROLE_CANDIDATES: List[str] = ["role", "teamPosition", "team_position", "mode_team_position", "mode_role"]
CHAMPION_CANDIDATES: List[str] = ["champion", "championName", "champion_name", "championId", "champion_id", "champion_name"]
GAMES_CANDIDATES: List[str] = ["games_played", "total_games_played_in_role", "games", "match_count", "matches"]
WINRATE_CANDIDATES: List[str] = ["win_rate", "winrate", "win_rate_pct", "win_percent"]
CLUSTER_CANDIDATES: List[str] = ["cluster", "cluster_id", "cluster_label"]
USER_ID_CANDIDATES: List[str] = ["user_id", "puuid", "summoner_id", "account_id"]


def _select_first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_schema(
    df: pd.DataFrame,
    want_role: bool = True,
    want_champion: bool = True,
    want_games: bool = True,
    want_winrate: bool = False,
    want_cluster: bool = False,
    want_user_id: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    Create a shallow copy with standardized columns if present.
    Standard names: role, champion, games_played, win_rate, cluster_id
    Returns (df_renamed, mapping_of_selected_source_columns)
    """
    if df is None or df.empty:
        return df, {}
    rename_map: Dict[str, str] = {}
    mapping: Dict[str, Optional[str]] = {
        "role": None,
        "champion": None,
        "games_played": None,
        "win_rate": None,
        "cluster_id": None,
        "user_id": None,
    }
    if want_role:
        src = _select_first_present(df, ROLE_CANDIDATES)
        if src:
            rename_map[src] = "role"
            mapping["role"] = src
    if want_champion:
        src = _select_first_present(df, CHAMPION_CANDIDATES)
        if src:
            rename_map[src] = "champion"
            mapping["champion"] = src
    if want_games:
        src = _select_first_present(df, GAMES_CANDIDATES)
        if src:
            rename_map[src] = "games_played"
            mapping["games_played"] = src
    if want_winrate:
        src = _select_first_present(df, WINRATE_CANDIDATES)
        if src:
            rename_map[src] = "win_rate"
            mapping["win_rate"] = src
    if want_cluster:
        src = _select_first_present(df, CLUSTER_CANDIDATES)
        if src:
            rename_map[src] = "cluster_id"
            mapping["cluster_id"] = src
    if want_user_id:
        src = _select_first_present(df, USER_ID_CANDIDATES)
        if src:
            rename_map[src] = "user_id"
            mapping["user_id"] = src

    if not rename_map:
        return df, mapping
    renamed = df.rename(columns=rename_map).copy()
    return renamed, mapping


def require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}")


def normalize_win_rate_series(s: pd.Series) -> pd.Series:
    try:
        s_clean = pd.to_numeric(s, errors="coerce")
    except Exception:
        return s
    if s_clean.dropna().max() is not np.nan and s_clean.dropna().max() > 100:
        # Observed upstream issue: 100 -> 10000; scale down by 100
        return s_clean / 100.0
    return s_clean


def get_user_top_champions(
    user_df: pd.DataFrame,
    user_id: str,
    role: str,
    how_many: int = 3,
    min_games: int = 5,
    user_id_col: str = "user_id",
    role_col: str = "role",
    champion_col: str = "champion",
    games_col: str = "games_played",
) -> pd.DataFrame:
    """
    Returns top-N played champions for a user in a role.
    """
    require_columns(user_df, [user_id_col, role_col, champion_col, games_col], "get_user_top_champions")
    df = user_df.loc[(user_df[user_id_col] == user_id) & (user_df[role_col] == role)].copy()
    df = df.loc[df[games_col] >= min_games]
    df = df.sort_values(by=[games_col], ascending=False)
    return df.head(how_many)


def get_top_champions_for_role(
    df: pd.DataFrame,
    role: str,
    how_many: int = 3,
    min_games: int = 5,
    role_col: str = "role",
    champion_col: str = "champion",
    games_col: str = "games_played",
) -> pd.DataFrame:
    """
    Returns top-N played champions for a role when no user_id column exists
    (e.g., a single-user aggregated dataset without explicit user id).
    """
    require_columns(df, [role_col, champion_col, games_col], "get_top_champions_for_role")
    data = df.loc[df[role_col] == role].copy()
    data = data.loc[data[games_col] >= min_games]
    data = data.sort_values(by=[games_col], ascending=False)
    return data.head(how_many)


def get_user_winrate_table(
    user_df: pd.DataFrame,
    user_id: str,
    role: str,
    user_id_col: str = "user_id",
    role_col: str = "role",
    champion_col: str = "champion",
    winrate_col: str = "win_rate",
    games_col: str = "games_played",
) -> pd.DataFrame:
    require_columns(user_df, [user_id_col, role_col, champion_col, winrate_col, games_col], "get_user_winrate_table")
    out = user_df.loc[(user_df[user_id_col] == user_id) & (user_df[role_col] == role)].copy()
    if not out.empty:
        out[winrate_col] = normalize_win_rate_series(out[winrate_col])
    return out


def filter_candidates_by_cluster(
    champ_role_df: pd.DataFrame,
    role: str,
    cluster_id: int,
    role_col: str = "role",
    champion_col: str = "champion",
    cluster_col: str = "cluster",
    exclude: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Returns champions in the same cluster and role, optionally sorted and limited.
    """
    require_columns(champ_role_df, [role_col, champion_col, cluster_col], "filter_candidates_by_cluster")
    df = champ_role_df.loc[(champ_role_df[role_col] == role) & (champ_role_df[cluster_col] == cluster_id)].copy()
    if exclude:
        df = df.loc[~df[champion_col].isin(set(exclude))]
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=[sort_by], ascending=ascending)
    if limit is not None:
        df = df.head(limit)
    return df


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0, 0))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    normalized = matrix / norms
    sim = normalized @ normalized.T
    return sim


def compute_similar_champions(
    champ_role_df: pd.DataFrame,
    role: str,
    target_champion: str,
    feature_columns: Sequence[str],
    role_col: str = "role",
    champion_col: str = "champion",
    k: int = 5,
) -> pd.DataFrame:
    """
    Returns top-k most similar champions to a target within a role using cosine similarity.
    Falls back to empty DataFrame if features are missing.
    """
    needed = [role_col, champion_col, *feature_columns]
    require_columns(champ_role_df, needed, "compute_similar_champions")

    df = champ_role_df.loc[champ_role_df[role_col] == role, [champion_col, *feature_columns]].copy()
    if target_champion not in set(df[champion_col].tolist()):
        return pd.DataFrame(columns=[champion_col, "similarity"])  # target not present

    features = df[feature_columns].to_numpy(dtype=float)
    sim = cosine_similarity_matrix(features)
    idx = df.index.get_indexer(df.loc[df[champion_col] == target_champion].index)[0]

    scores = sim[idx]
    df_out = df[[champion_col]].copy()
    df_out["similarity"] = scores
    df_out = df_out.loc[df_out[champion_col] != target_champion]
    df_out = df_out.sort_values(by=["similarity"], ascending=False).head(k)
    return df_out.reset_index(drop=True)


def get_champion_cluster_id(
    champ_role_df: pd.DataFrame,
    champion: str,
    role: str,
    cluster_col: str = "cluster",
    role_col: str = "role",
    champion_col: str = "champion",
) -> Optional[int]:
    require_columns(champ_role_df, [role_col, champion_col, cluster_col], "get_champion_cluster_id")
    rows = champ_role_df.loc[(champ_role_df[role_col] == role) & (champ_role_df[champion_col] == champion)]
    if rows.empty:
        return None
    return int(rows.iloc[0][cluster_col])


# -------- Recommendation Assembly --------


def assemble_candidate_set(
    champ_role_df: pd.DataFrame,
    role: str,
    seed_champions: Sequence[str],
    within_cluster: bool,
    feature_columns_by_role: Optional[Dict[str, List[str]]] = None,
    k_per_seed: int = 5,
    exclude: Optional[Sequence[str]] = None,
    cluster_col: str = "cluster",
    role_col: str = "role",
    champion_col: str = "champion",
    sort_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a candidate set from user's seed champions by either:
      - taking same-cluster champions, or
      - using cosine similarity on role-specific feature columns.
    Returns a DataFrame with unique champions and optional scores.
    """
    exclude_set = set(exclude or [])
    all_rows: List[pd.DataFrame] = []
    if within_cluster:
        require_columns(champ_role_df, [role_col, champion_col, cluster_col], "assemble_candidate_set within_cluster")
        for champ in seed_champions:
            cluster_id = get_champion_cluster_id(
                champ_role_df=champ_role_df,
                champion=champ,
                role=role,
                cluster_col=cluster_col,
                role_col=role_col,
                champion_col=champion_col,
            )
            if cluster_id is None:
                continue
            rows = filter_candidates_by_cluster(
                champ_role_df=champ_role_df,
                role=role,
                cluster_id=cluster_id,
                role_col=role_col,
                champion_col=champion_col,
                cluster_col=cluster_col,
                exclude=[*exclude_set, *seed_champions],
                limit=k_per_seed,
                sort_by=sort_by,
                ascending=False,
            )
            all_rows.append(rows[[champion_col]].assign(source=champ, method="cluster"))
    else:
        if not feature_columns_by_role or role not in feature_columns_by_role:
            return pd.DataFrame(columns=[champion_col, "similarity", "source", "method"]).dropna()
        feature_columns = feature_columns_by_role[role]
        for champ in seed_champions:
            sim_df = compute_similar_champions(
                champ_role_df=champ_role_df,
                role=role,
                target_champion=champ,
                feature_columns=feature_columns,
                role_col=role_col,
                champion_col=champion_col,
                k=k_per_seed,
            )
            if not sim_df.empty:
                sim_df = sim_df.loc[~sim_df[champion_col].isin(exclude_set)]
                all_rows.append(sim_df.assign(source=champ, method="similarity"))

    if not all_rows:
        return pd.DataFrame(columns=[champion_col, "score", "source", "method"]).dropna()

    merged = pd.concat(all_rows, axis=0, ignore_index=True)
    if "similarity" in merged.columns:
        merged = merged.rename(columns={"similarity": "score"})
        merged["score"] = merged["score"].astype(float)
    else:
        merged["score"] = np.nan

    # Deduplicate by champion, keep max score
    merged = merged.sort_values(by=["score"], ascending=False)
    merged = merged.drop_duplicates(subset=[champion_col], keep="first")
    # Attach cluster id if present in champ_role_df
    cluster_source_col = None
    for c in ["cluster", "cluster_id"]:
        if c in champ_role_df.columns:
            cluster_source_col = c
            break
    if cluster_source_col:
        cluster_map = champ_role_df.loc[
            champ_role_df[role_col] == role, [champion_col, cluster_source_col]
        ].drop_duplicates()
        merged = merged.merge(cluster_map, on=champion_col, how="left")
    return merged.reset_index(drop=True)


def attach_semantics_to_candidates(
    candidates: pd.DataFrame,
    semantic_champion_df: Optional[pd.DataFrame],
    role: str,
    role_col: str = "role",
    champion_col: str = "champion",
    tag_col: str = "semantic_tags",
    desc_col: str = "semantic_description",
) -> pd.DataFrame:
    if semantic_champion_df is None or semantic_champion_df.empty or candidates is None or candidates.empty:
        return candidates
    # Two possible schemas:
    # 1) semantic df has separate champion and role columns -> merge on champion within role
    # 2) semantic df has an 'id' column like "Yone_TOP" -> create the same id in candidates and merge
    if "id" in semantic_champion_df.columns:
        sem = semantic_champion_df.copy()
        # Auto-detect tag/description columns if not present as provided
        if tag_col not in sem.columns:
            cand = [c for c in sem.columns if "tag" in c.lower()]
            if cand:
                tag_col = cand[0]
        if desc_col not in sem.columns:
            cand = [c for c in sem.columns if "desc" in c.lower()]
            if cand:
                desc_col = cand[0]
        keep_cols = ["id"] + [c for c in [tag_col, desc_col] if c in sem.columns]
        sem = sem[keep_cols]
        out = candidates.copy()
        out["id"] = out[champion_col].astype(str) + "_" + str(role)
        out = out.merge(sem, on="id", how="left").drop(columns=["id"], errors="ignore")
        # Standardize names if detected
        rename_map: Dict[str, str] = {}
        if tag_col in out.columns and tag_col != "semantic_tags":
            rename_map[tag_col] = "semantic_tags"
        if desc_col in out.columns and desc_col != "semantic_description":
            rename_map[desc_col] = "semantic_description"
        if rename_map:
            out = out.rename(columns=rename_map)
        return out
    # Fallback to champion/role merge if both present
    require_columns(semantic_champion_df, [role_col, champion_col], "attach_semantics_to_candidates")
    sem = semantic_champion_df.loc[semantic_champion_df[role_col] == role, [champion_col, tag_col, desc_col]].copy()
    return candidates.merge(sem, on=champion_col, how="left")


def attach_cluster_semantics(
    df_with_cluster: pd.DataFrame,
    cluster_semantic_df: Optional[pd.DataFrame],
    cluster_col: str = "cluster",
    id_col: str = "id",
    tag_col: str = "semantic_tags",
    desc_col: str = "semantic_description",
) -> pd.DataFrame:
    """
    Attach cluster-level semantics using the cluster id mapping.
    Expects df_with_cluster to have a numeric/categorical cluster column that matches
    the 'id' field in the cluster_semantic_df.
    """
    if cluster_semantic_df is None or cluster_semantic_df.empty or df_with_cluster is None or df_with_cluster.empty:
        return df_with_cluster

    sem = cluster_semantic_df.copy()
    # Auto-detect semantics columns if differently named
    if tag_col not in sem.columns:
        cand = [c for c in sem.columns if "tag" in c.lower()]
        if cand:
            tag_col = cand[0]
    if desc_col not in sem.columns:
        cand = [c for c in sem.columns if "desc" in c.lower()]
        if cand:
            desc_col = cand[0]

    keep_cols = [id_col] + [c for c in [tag_col, desc_col] if c in sem.columns]
    sem = sem[keep_cols]
    out = df_with_cluster.merge(sem, left_on=cluster_col, right_on=id_col, how="left")
    out = out.drop(columns=[id_col], errors="ignore")

    rename_map: Dict[str, str] = {}
    if tag_col in out.columns and tag_col != "cluster_semantic_tags":
        rename_map[tag_col] = "cluster_semantic_tags"
    if desc_col in out.columns and desc_col != "cluster_semantic_description":
        rename_map[desc_col] = "cluster_semantic_description"
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def build_recommendation_payload(
    user_id: str,
    role: str,
    seed_champions_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    user_winrate_df: Optional[pd.DataFrame] = None,
    top_n: int = 10,
    champion_col: str = "champion",
) -> Dict:
    seeds = seed_champions_df[champion_col].tolist()
    recs = candidates_df.head(top_n)
    rec_list: List[Dict] = []
    for _, row in recs.iterrows():
        rec = {
            "champion": row.get(champion_col),
            "score": float(row.get("score", np.nan)) if pd.notna(row.get("score", np.nan)) else None,
            "source": row.get("source"),
            "method": row.get("method"),
        }
        if "cluster" in recs.columns:
            rec["cluster"] = row.get("cluster")
        if "semantic_tags" in row:
            rec["semantic_tags"] = row.get("semantic_tags")
        if "semantic_description" in row:
            rec["semantic_description"] = row.get("semantic_description")
        if "cluster_semantic_tags" in row:
            rec["cluster_semantic_tags"] = row.get("cluster_semantic_tags")
        if "cluster_semantic_description" in row:
            rec["cluster_semantic_description"] = row.get("cluster_semantic_description")
        rec_list.append(rec)

    payload: Dict = {
        "user_id": user_id,
        "role": role,
        "seed_champions": seeds,
        "recommendations": rec_list,
    }
    if user_winrate_df is not None and not user_winrate_df.empty:
        payload["user_winrates"] = user_winrate_df.to_dict(orient="records")
    return payload


# -------- Orchestrator --------


def recommend_for_user(
    repo: DataRepository,
    user_id: str,
    role: str,
    consider_top_n_champions: int = 3,
    within_cluster: bool = True,
    k_per_seed: int = 5,
    feature_columns_by_role: Optional[Dict[str, List[str]]] = None,
    exclude_played: bool = True,
    min_games: int = 5,
    prefer_single_user: bool = True,
    # Column names
    user_id_col: str = "user_id",
    role_col: str = "role",
    champion_col: str = "champion",
    games_col: str = "games_played",
    winrate_col: str = "win_rate",
    cluster_col: str = "cluster_id",
    sort_by: Optional[str] = None,
    attach_semantics: bool = True,
    top_n_output: int = 10,
) -> Dict:
    """
    High-level API to assemble a recommendation payload consumable by an LLM agent.
    Keeps logic minimal and transparent for iterative extension.
    """

    # Load required dataframes
    champ_role_df_raw = repo.get_dataframe("champion_x_role")
    champ_role_df, champ_role_map = normalize_schema(
        champ_role_df_raw, want_role=True, want_champion=True, want_games=True, want_winrate=True, want_cluster=True
    )

    # Choose user table: prefer single user for local speed if requested
    if prefer_single_user:
        user_df_raw = repo.get_dataframe("single_user")
    else:
        try:
            user_df_raw = repo.get_dataframe("champion_x_role_x_user")
        except Exception:
            user_df_raw = repo.get_dataframe("single_user")

    # Normalize user df (may or may not have user_id)
    user_df, user_map = normalize_schema(
        user_df_raw, want_role=True, want_champion=True, want_games=True, want_winrate=True, want_cluster=False, want_user_id=True
    )

    # Seeds: user's top-N champions in role
    if user_id_col in user_df.columns:
        seeds_df = get_user_top_champions(
            user_df=user_df,
            user_id=user_id,
            role=role,
            how_many=consider_top_n_champions,
            min_games=min_games,
            user_id_col=user_id_col,
            role_col="role",
            champion_col="champion",
            games_col="games_played",
        )
    else:
        # single_user dataset without explicit user id
        seeds_df = get_top_champions_for_role(
            df=user_df,
            role=role,
            how_many=consider_top_n_champions,
            min_games=min_games,
            role_col="role",
            champion_col="champion",
            games_col="games_played",
        )
    seed_champions = seeds_df[champion_col].tolist()

    # Candidate set
    exclude_list = seed_champions if exclude_played else []
    # If within_cluster is requested but cluster column is missing, fallback to similarity if possible
    effective_within_cluster = within_cluster and ("cluster" in champ_role_df.columns)
    candidates = assemble_candidate_set(
        champ_role_df=champ_role_df,
        role=role,
        seed_champions=seed_champions,
        within_cluster=effective_within_cluster,
        feature_columns_by_role=feature_columns_by_role,
        k_per_seed=k_per_seed,
        exclude=exclude_list,
        cluster_col="cluster",
        role_col="role",
        champion_col="champion",
        sort_by=sort_by,
    )

    # Optional semantics
    if attach_semantics:
        try:
            sem_raw = repo.get_dataframe("semantic_champion", role=role)
            sem_df, _ = normalize_schema(sem_raw, want_role=True, want_champion=True, want_games=False, want_winrate=False, want_cluster=False)
        except Exception:
            sem_df = None
        candidates = attach_semantics_to_candidates(
            candidates=candidates,
            semantic_champion_df=sem_df,
            role=role,
            role_col="role",
            champion_col="champion",
        )

        # Attach cluster semantics if candidates include a cluster mapping column
        try:
            cluster_sem_raw = repo.get_dataframe("semantic_cluster", role=role)
        except Exception:
            cluster_sem_raw = None
        # Ensure candidates have cluster identifier
        cluster_col_present = "cluster_id" if "cluster_id" in candidates.columns else ("cluster" if "cluster" in candidates.columns else None)
        if cluster_sem_raw is not None:
            if cluster_col_present is None:
                # Try to map cluster via residuals if champ_role_df lacks it
                try:
                    res_raw = repo.get_dataframe("residuals", role=role)
                    res_df, _ = normalize_schema(res_raw, want_role=True, want_champion=True, want_games=False)
                    # Expect residuals to have a cluster column; preserve its name
                    res_cluster_col = "cluster" if "cluster" in res_df.columns else None
                    if res_cluster_col:
                        mapping = res_df[["champion", res_cluster_col]].drop_duplicates()
                        candidates = candidates.merge(mapping, on="champion", how="left")
                        cluster_col_present = res_cluster_col
                except Exception:
                    pass
            if cluster_col_present is not None:
                candidates = attach_cluster_semantics(
                    df_with_cluster=candidates,
                    cluster_semantic_df=cluster_sem_raw,
                    cluster_col=cluster_col_present,
                    id_col="id",
                )

    # User winrates for context
    if user_id_col in user_df.columns and _select_first_present(user_df, WINRATE_CANDIDATES):
        # Ensure win_rate is normalized in user df
        user_df_wr, _ = normalize_schema(user_df, want_role=True, want_champion=True, want_games=True, want_winrate=True, want_cluster=False)
        user_winrates = get_user_winrate_table(
            user_df=user_df_wr,
            user_id=user_id,
            role=role,
            user_id_col=user_id_col,
            role_col="role",
            champion_col="champion",
            winrate_col="win_rate",
            games_col="games_played",
        )
    else:
        user_winrates = pd.DataFrame()

    return build_recommendation_payload(
        user_id=user_id,
        role=role,
        seed_champions_df=seeds_df,
        candidates_df=candidates,
        user_winrate_df=user_winrates,
        top_n=top_n_output,
        champion_col=champion_col,
    )


__all__ = [
    "DataPaths",
    "DataRepository",
    "get_user_top_champions",
    "get_user_winrate_table",
    "compute_similar_champions",
    "get_champion_cluster_id",
    "filter_candidates_by_cluster",
    "assemble_candidate_set",
    "attach_semantics_to_candidates",
    "build_recommendation_payload",
    "recommend_for_user",
]


# -------- Optional: Analysis Helpers for the Agent --------


def compute_user_champion_winrate_deltas(
    user_df: pd.DataFrame,
    champ_role_df: pd.DataFrame,
    user_id: str,
    role: str,
    user_id_col: str = "user_id",
    role_col: str = "role",
    champion_col: str = "champion",
    user_winrate_col: str = "win_rate",
    global_winrate_col: str = "win_rate",
) -> pd.DataFrame:
    """
    For a user's champions in a role, compare their win rate to global win rate for that champion+role.
    Returns columns: champion, user_win_rate, global_win_rate, delta_win_rate.
    """
    require_columns(user_df, [user_id_col, role_col, champion_col, user_winrate_col], "compute_user_champion_winrate_deltas user")
    require_columns(champ_role_df, [role_col, champion_col, global_winrate_col], "compute_user_champion_winrate_deltas global")

    u = user_df.loc[(user_df[user_id_col] == user_id) & (user_df[role_col] == role), [champion_col, user_winrate_col]].copy()
    g = champ_role_df.loc[champ_role_df[role_col] == role, [champion_col, global_winrate_col]].copy()
    # Normalize possible upstream scaling issues
    u[user_winrate_col] = normalize_win_rate_series(u[user_winrate_col])
    g[global_winrate_col] = normalize_win_rate_series(g[global_winrate_col])
    u = u.rename(columns={user_winrate_col: "user_win_rate"})
    g = g.rename(columns={global_winrate_col: "global_win_rate"})
    m = u.merge(g, on=champion_col, how="left")
    m["delta_win_rate"] = m["user_win_rate"] - m["global_win_rate"]
    return m.sort_values(by=["delta_win_rate"], ascending=False).reset_index(drop=True)


def compute_single_user_winrate_deltas(
    user_df: pd.DataFrame,
    champ_role_df: pd.DataFrame,
    role: str,
    role_col: str = "role",
    champion_col: str = "champion",
    user_winrate_col: str = "win_rate",
    global_winrate_col: str = "win_rate",
) -> pd.DataFrame:
    """
    For a single-user dataset (no user_id column), compare all champions in the role against global averages.
    Returns columns: champion, user_win_rate, global_win_rate, delta_win_rate.
    """
    require_columns(user_df, [role_col, champion_col, user_winrate_col], "compute_single_user_winrate_deltas user")
    require_columns(champ_role_df, [role_col, champion_col, global_winrate_col], "compute_single_user_winrate_deltas global")

    u = user_df.loc[user_df[role_col] == role, [champion_col, user_winrate_col]].copy()
    g = champ_role_df.loc[champ_role_df[role_col] == role, [champion_col, global_winrate_col]].copy()
    # Normalize possible upstream scaling issues
    u[user_winrate_col] = normalize_win_rate_series(u[user_winrate_col])
    g[global_winrate_col] = normalize_win_rate_series(g[global_winrate_col])
    u = u.rename(columns={user_winrate_col: "user_win_rate"})
    g = g.rename(columns={global_winrate_col: "global_win_rate"})
    m = u.merge(g, on=champion_col, how="left")
    m["delta_win_rate"] = m["user_win_rate"] - m["global_win_rate"]
    return m.sort_values(by=["delta_win_rate"], ascending=False).reset_index(drop=True)


def extract_residuals_for_champion(
    residuals_df: pd.DataFrame,
    champion: str,
    role: str,
    role_col: str = "role",
    champion_col: str = "champion",
    strength_prefix: str = "strength_",
    weakness_prefix: str = "weakness_",
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extract top strengths/weaknesses for a champion from residuals table.
    Looks for columns like strength_1_name, strength_1_value, weakness_1_name, weakness_1_value.
    """
    require_columns(residuals_df, [role_col, champion_col], "extract_residuals_for_champion base")
    row = residuals_df.loc[(residuals_df[role_col] == role) & (residuals_df[champion_col] == champion)]
    if row.empty:
        return {"strengths": [], "weaknesses": []}
    row = row.iloc[0].to_dict()

    def collect(prefix: str) -> List[Tuple[str, float]]:
        pairs: List[Tuple[str, float]] = []
        # Scan for numbered fields
        i = 1
        while True:
            name_key = f"{prefix}{i}_name"
            value_key = f"{prefix}{i}_value"
            if name_key not in row or value_key not in row:
                break
            name = row.get(name_key)
            value = row.get(value_key)
            if pd.notna(name) and pd.notna(value):
                try:
                    pairs.append((str(name), float(value)))
                except Exception:
                    pass
            i += 1
        return pairs

    return {
        "strengths": collect(strength_prefix),
        "weaknesses": collect(weakness_prefix),
    }



