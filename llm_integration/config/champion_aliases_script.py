import re
import json
import unicodedata
from typing import Dict, Set, Optional
import requests

DD_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DD_CHAMPS_URL_TPL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"

def fetch_ddragon_version(version: Optional[str] = None, timeout: int = 20) -> str:
    """Return a Data Dragon version. If None, fetch the latest."""
    if version:
        return version
    resp = requests.get(DD_VERSIONS_URL, timeout=timeout)
    resp.raise_for_status()
    versions = resp.json()
    if not isinstance(versions, list) or not versions:
        raise RuntimeError("Unexpected versions payload from Data Dragon.")
    return versions[0]  # latest

def fetch_champion_names_from_ddragon(version: Optional[str] = None, timeout: int = 20) -> Set[str]:
    """Fetch canonical champion display names (exact strings) from Data Dragon."""
    v = fetch_ddragon_version(version, timeout=timeout)
    url = DD_CHAMPS_URL_TPL.format(version=v)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", {})
    names = {entry.get("name", "").strip() for entry in data.values()}
    return {n for n in names if n and 1 < len(n) <= 40}

def ascii_fold(s: str) -> str:
    """Remove accents but keep original casing."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def dd_name_to_pascal_compact(name: str) -> str:
    """
    PascalCase, no special characters:
      - Split on any non-alphanumeric boundary
      - Keep original token capitalization (so 'LeBlanc' stays 'LeBlanc')
      - Concatenate tokens
      Examples:
        "K'Sante"            -> "KSante"
        "Jarvan IV"          -> "JarvanIV"
        "Nunu & Willump"     -> "NunuWillump"
        "Cho'Gath"           -> "ChoGath"
        "Kha'Zix"            -> "KhaZix"
    """
    s = ascii_fold(name)
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", s) if t]
    return "".join(tokens)

def build_formatted_to_exact_map(version: Optional[str] = None) -> Dict[str, str]:
    """Return {FormattedPascalNoSpecial: ExactDDName}."""
    exact_names = sorted(fetch_champion_names_from_ddragon(version=version))
    return {dd_name_to_pascal_compact(n): n for n in exact_names}

if __name__ == "__main__":
    # Pin a version (e.g., "15.20.1") or use None for latest
    mapping = build_formatted_to_exact_map(version="15.20.1")
    mapping["MonkeyKing"] = "Wukong"
    print(f"champions: {len(mapping)} entries\n")

    # Spot checks
    for sample in ["K'Sante", "Jarvan IV", "Nunu & Willump", "Cho'Gath", "Kha'Zix", "LeBlanc", "Dr. Mundo"]:
        key = dd_name_to_pascal_compact(sample)
        print(f"{sample:16s} -> key='{key}' -> value='{mapping.get(key)}'")

    with open("champion_aliases.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print("\nWrote champion_aliases.json")