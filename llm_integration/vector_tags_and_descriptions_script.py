import os, io, json
import numpy as np
import pandas as pd
import boto3
from dotenv import load_dotenv
from botocore.config import Config

# ---- env / config ----
load_dotenv()
BUCKET  = os.getenv("BUCKET")
PREFIX  = os.getenv("PROCESSED_DATA_FOLDER")
PATCH   = "patch_15_6"
LLM_ID  = os.getenv("LLM_ID")                   # model id or inference profile id (Converse-capable)
REGION  = os.getenv("REGION", "us-east-2")
RUN_TOKEN_CAP = int(os.getenv("RUN_TOKEN_CAP", "50000"))  # hard stop to avoid spiralling costs
IN_COST_PER_K  = float(os.getenv("IN_COST_PER_K",  "0.003"))  # $ per 1k input tokens
OUT_COST_PER_K = float(os.getenv("OUT_COST_PER_K", "0.015"))  # $ per 1k output tokens

cfg = Config(read_timeout=120, retries={"max_attempts": 3, "mode": "adaptive"})
rt  = boto3.client("bedrock-runtime", region_name=REGION, config=cfg)

# ---- S3 load ----
def get_processed_dataframe(role: str) -> pd.DataFrame:
    key = f"{PREFIX}/clusters/{PATCH}/{role.lower()}_vectors_df.csv"
    s3  = boto3.client("s3", region_name=REGION, config=cfg)
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

# ---- tiny utilities ----
def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"champion_name","team_position","cluster","distance_to_centroid",
               "cluster_description","cluster_tags","champion_description","champion_tags"}
    return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

def _top_dir_pairs(series: pd.Series, k: int) -> list[str]:
    # Return like ["ap_burst:high","objective_control:low",...]
    order = series.abs().sort_values(ascending=False).head(k).index
    out = []
    for f in order:
        out.append(f"{f}:{'high' if series[f] >= 0 else 'low'}")
    return out

def _converse_json(prompt: str, max_tokens=90, temperature=0.2):
    """Ask for STRICT JSON {description:str, tags:list[str]} and parse safely."""
    resp = rt.converse(
        modelId=LLM_ID,
        system=[{"text":"Return STRICT JSON only as {\"description\":\"...\",\"tags\":[\"...\"]}. No markdown, no extra text."}],
        messages=[{"role":"user","content":[{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": 0.95, "stopSequences": ["\n\n"]},
    )
    parts = resp["output"]["message"]["content"]
    text  = "".join(p.get("text","") for p in parts if "text" in p).strip()
    try:
        data = json.loads(text)
        desc = str(data.get("description","")).strip()
        tags = [str(t).strip() for t in data.get("tags", []) if str(t).strip()]
    except Exception:
        desc, tags = text, []
    usage = resp.get("usage", {"inputTokens":0,"outputTokens":0})
    return desc, tags, usage

# ---- main: cluster + champion descriptions + tags ----
def describe_clusters_and_champions(df: pd.DataFrame,
                                    top_champs_per_cluster: int = 5,
                                    top_feats_cluster: int = 6,
                                    top_feats_champion: int = 5) -> pd.DataFrame:
    df = df.copy()
    feat_cols = _numeric_feature_cols(df)
    if not feat_cols:
        raise ValueError("No numeric feature columns found to derive semantic tags.")

    # Precompute global mean and per-cluster means for quick 'direction' summaries
    global_mean = df[feat_cols].mean()
    cluster_means = df.groupby("cluster")[feat_cols].mean()

    run_in = run_out = 0
    cluster_desc, cluster_tags = {}, {}

    # ---- pass 1: clusters ----
    for cid, g in df.groupby("cluster"):
        # short champion list for context (closest to centroid if available)
        champs = (g.nsmallest(top_champs_per_cluster, "distance_to_centroid")["champion_name"].tolist()
                  if "distance_to_centroid" in g.columns else g["champion_name"].tolist()[:top_champs_per_cluster])

        # feature directions for this cluster: cluster mean vs global mean
        diffs = (cluster_means.loc[cid] - global_mean)
        feat_summary = "; ".join(_top_dir_pairs(diffs, top_feats_cluster))

        prompt = (
            "Given these signals, produce a concise English description AND semantic tags.\n"
            f"Champions: {', '.join(champs)}\n"
            f"Feature directions (cluster vs global): {feat_summary}\n\n"
            'Return JSON: {"description":"one sentence (<= 20 words)","tags":["tag1","tag2",...]}'
        )

        desc, tags, usage = _converse_json(prompt, max_tokens=90, temperature=0.25)
        cluster_desc[int(cid)] = desc
        cluster_tags[int(cid)] = tags
        run_in  += usage.get("inputTokens", 0) or 0
        run_out += usage.get("outputTokens", 0) or 0
        if (run_in + run_out) > RUN_TOKEN_CAP:
            print("Token cap reached during cluster pass; stopping.")
            break

    df["cluster_description"] = df["cluster"].map(cluster_desc).fillna("")
    df["cluster_tags"]        = df["cluster"].map(cluster_tags).apply(lambda x: "|".join(x) if isinstance(x,list) else "")

    # ---- pass 2: champions ----
    # Precompute: per-cluster mean for directions
    out_desc, out_tags = {}, {}
    for _, row in df[["champion_name","team_position","cluster"] + feat_cols].iterrows():
        cid = row["cluster"]
        name, role = row["champion_name"], row["team_position"]

        # champion directions vs its own cluster mean → what makes THIS champ stand out
        diffs = (row[feat_cols] - cluster_means.loc[cid]).astype(float)
        feat_summary = "; ".join(_top_dir_pairs(diffs, top_feats_champion))

        prompt = (
            "Produce a concise English description and semantic tags for this champion in role context.\n"
            f"Champion: {name} ({role})\n"
            f"Feature directions (champion vs its cluster mean): {feat_summary}\n"
            'Return JSON: {"description":"one sentence (<= 15 words)","tags":["tag1","tag2",...]}'
        )

        desc, tags, usage = _converse_json(prompt, max_tokens=80, temperature=0.25)
        out_desc[(name, role)] = desc
        out_tags[(name, role)] = tags
        run_in  += usage.get("inputTokens", 0) or 0
        run_out += usage.get("outputTokens", 0) or 0
        if (run_in + run_out) > RUN_TOKEN_CAP:
            print("Token cap reached during champion pass; stopping.")
            break

    df["champion_description"] = df.apply(
        lambda r: out_desc.get((r["champion_name"], r["team_position"]), ""), axis=1
    )
    df["champion_tags"] = df.apply(
        lambda r: "|".join(out_tags.get((r["champion_name"], r["team_position"]), [])), axis=1
    )

    # ---- summary / estimated cost ----
    total_in, total_out = run_in, run_out
    est_cost = (total_in/1000.0)*IN_COST_PER_K + (total_out/1000.0)*OUT_COST_PER_K
    print(f"Token usage — in:{total_in} out:{total_out} total:{total_in+total_out}  ~ est. cost: ${est_cost:.4f}")

    return df