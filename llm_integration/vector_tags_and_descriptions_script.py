import os, io, json
import numpy as np
from dotenv import load_dotenv
from botocore.config import Config
import math
from typing import Dict, List, Tuple
import pandas as pd
import boto3

# ---- env / config ----
load_dotenv()
REGION = os.getenv("REGION", "us-east-2")
MODEL_ID = os.getenv("PROFILE_ARN")  # <-- set this to your inference profile ARN
ANTHROPIC_VERSION = "bedrock-2023-05-31"
BUCKET  = os.getenv("BUCKET")
PREFIX  = os.getenv("PROCESSED_DATA_FOLDER")
PATCH   = "patch_15_6"

cfg = Config(read_timeout=120, retries={"max_attempts": 3, "mode": "adaptive"})
rt  = boto3.client("bedrock-runtime", region_name=REGION, config=cfg)

# Cost controls
MAX_TOTAL_TOKENS_APPROX = 100_000        # rough budget; ~chars/4
MAX_TOKENS_PER_CALL     = 1500           # limit per batch response
BATCH_SIZE              = 12            # small batch to keep prompts compact
MAX_TAGS                = 7
MIN_TAGS                = 3
DESC_WORD_LIMIT         = 35            # short, 1 sentence

# === Bedrock client ===
def bedrock_client():
    if not MODEL_ID or not MODEL_ID.startswith("arn:aws:bedrock:"):
        raise RuntimeError("PROFILE_ARN env var must be set to your inference profile ARN.")
    return boto3.client("bedrock-runtime", region_name=REGION)

# === Token estimate: extremely rough but simple ===
def approx_tokens_from_text(s: str) -> int:
    # crude: 1 token ~= 4 chars
    return math.ceil(len(s) / 4)

# === Prompt builder ===
SYSTEM_RULES = (
    "You write ultra-compact, gameplay-relevant outputs for League of Legends. "
    "Only use the provided features. Do not invent lore or facts beyond them. "
    "Prefer established LoL terms when obvious (e.g., split pusher, diver, frontliner, enchanter, engage, peel, burst, poke, skirmisher, teamfight initiator, objective control). "
    f"Return exactly one JSON object per line (JSONL). Each object must include keys: id, kind, role, description, tags. "
    f"description: <= {DESC_WORD_LIMIT} words, single sentence. tags: {MIN_TAGS}-{MAX_TAGS} concise tags (1-3 words). "
    "Keep tags semantically meaningful; combine related signals under common gameplay concepts. "
    "If signals are weak or mixed, still choose the best concise tags rather than echoing raw feature names."
)

def build_batch_prompt(items: List[dict]) -> str:
    header = (
        "TASK: For each item, produce JSON: "
        '{"id": "...", "kind":"cluster|champion", "role":"...", '
        '"description":"...", "tags":["...", "..."]}\n'
        f"Constraints: description <= {DESC_WORD_LIMIT} words; {MIN_TAGS}-{MAX_TAGS} tags; use only given features.\n"
        "Items (JSONL input):\n"
    )
    lines = []
    for it in items:
        strengths = [[n, round(float(v), 2)] for n, v in it["strengths"]]
        weaknesses = [[n, round(float(v), 2)] for n, v in it["weaknesses"]]
        j = {
            "id": it["id"],
            "kind": it["kind"],
            "role": it["role"],
            "strengths": strengths,
            "weaknesses": weaknesses,
        }
        lines.append(json.dumps(j, ensure_ascii=False))
    return header + "\n".join(lines)

# === Bedrock call ===
def call_bedrock(messages_text: str, client) -> str:
    body = {
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": MAX_TOKENS_PER_CALL,
        "temperature": 0.2,
        # <-- system prompt goes here, not as a message
        "system": [{"type": "text", "text": SYSTEM_RULES}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": messages_text}]}
        ],
    }

    # MODEL_ID should be your inference profile ARN (PROFILE_ARN)
    resp = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    payload = json.loads(resp["body"].read())

    # Robust text extraction for Anthropic on Bedrock
    text = (
        payload.get("output", {})
               .get("message", {})
               .get("content", [{}])[0]
               .get("text")
        or payload.get("content", [{}])[0].get("text")
        or ""
    )
    return text

# === Helpers to read your DF schema ===
def extract_top_pairs(row: pd.Series, prefix: str, k: int = 10) -> List[Tuple[str, float]]:
    pairs = []
    for i in range(1, k + 1):
        name_col = f"{prefix}_{i}_name"
        val_col  = f"{prefix}_{i}_value"
        if name_col in row and val_col in row:
            n = str(row[name_col])
            try:
                v = float(row[val_col])
            except Exception:
                v = 0.0
            pairs.append((n, v))
    return pairs

def make_items_for_role(role: str, cluster_df: pd.DataFrame, champ_df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    cluster_items, champ_items = [], []
    for _, row in cluster_df.iterrows():
        idx = str(row["cluster"])
        strengths = extract_top_pairs(row, "strength", 10)
        weaknesses = extract_top_pairs(row, "weakness", 10)
        cluster_items.append({
            "id": str(idx),
            "kind": "cluster",
            "role": role,
            "strengths": strengths,
            "weaknesses": weaknesses,
        })
    for _, row in champ_df.iterrows():
        cid = str(row["champion_name"]) + "__" + str(row["team_position"])
        strengths = extract_top_pairs(row, "strength", 10)
        weaknesses = extract_top_pairs(row, "weakness", 10)
        champ_items.append({
            "id": cid,
            "kind": "champion",
            "role": role,
            "strengths": strengths,
            "weaknesses": weaknesses,
        })
    return cluster_items, champ_items

# === Main runner ===
def run_labelling(
    cluster_dfs: Dict[str, pd.DataFrame],
    champion_dfs: Dict[str, pd.DataFrame],
    max_total_tokens_approx: int = MAX_TOTAL_TOKENS_APPROX,
    batch_size: int = BATCH_SIZE,
    desired_roles = "all" # If not all, input as list of roles desired to run
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    client = bedrock_client()

    if desired_roles == "all":
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    else:
        roles = desired_roles

    total_token_estimate = 0
    cluster_outputs: Dict[str, List[dict]] = {r: [] for r in roles}
    champ_outputs: Dict[str, List[dict]]   = {r: [] for r in roles}

    for role in roles:
        cdf = cluster_dfs[role]
        hdf = champion_dfs[role]
        cluster_items, champ_items = make_items_for_role(role, cdf, hdf)

        for kind, items in (("cluster", cluster_items), ("champion", champ_items)):
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                prompt = build_batch_prompt(batch)

                est = approx_tokens_from_text(prompt)
                if total_token_estimate + est > max_total_tokens_approx:
                    break

                text = call_bedrock(prompt, client)
                total_token_estimate += est + MAX_TOKENS_PER_CALL

                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        rid  = str(obj.get("id", ""))
                        desc = str(obj.get("description", "")).strip()
                        tags = obj.get("tags", [])
                        if isinstance(tags, list):
                            tags = [str(t).strip() for t in tags][:MAX_TAGS]
                        else:
                            tags = []
                        out = {"id": rid, "role": role, "description": desc, "tags": tags}
                        if kind == "cluster":
                            cluster_outputs[role].append(out)
                        else:
                            champ_outputs[role].append(out)
                    except Exception:
                        pass

    cluster_results = {
        role: pd.DataFrame(cluster_outputs[role], columns=["id", "role", "description", "tags"])
        for role in roles
    }
    champion_results = {
        role: pd.DataFrame(champ_outputs[role], columns=["id", "role", "description", "tags"])
        for role in roles
    }

    return cluster_results, champion_results


# ---- S3 load ----
def get_processed_dataframe(role: str, req_type: str) -> pd.DataFrame:
    """req_type = champion_residuals or clusters"""
    key = f"{PREFIX}/clusters/{PATCH}/{role.lower()}_{req_type}_df.csv"
    s3  = boto3.client("s3", region_name=REGION, config=cfg)
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def main(roles: list):

    for role in roles:
    # Saving locally for now, to be streamlined with S3 uploading later
        role_df = get_processed_dataframe(role.lower(), "clusters")
        cluster_dfs = {role.upper() : role_df}
        
        champion_df = get_processed_dataframe(role.lower(), "champion_residuals")
        champion_dfs = {role.upper() : champion_df}

        # Just for S3 file integrity checks and debugging
        cluster_dfs[role.upper()].to_csv("s3_cluster_test.csv")
        champion_dfs[role.upper()].to_csv("s3_champion_test.csv")

        cluster_results, champion_results = run_labelling(cluster_dfs, champion_dfs, max_total_tokens_approx = MAX_TOTAL_TOKENS_APPROX, batch_size= BATCH_SIZE, desired_roles = [role])
        
        cluster_results[role.upper()].to_csv(f"{role.lower()}_cluster_semantic_tags_and_descriptions.csv")
        champion_results[role.upper()].to_csv(f"{role.lower()}_champion_semantic_tags_and_descriptions.csv")

if __name__ == "__main__":
    main(["UTILITY"])