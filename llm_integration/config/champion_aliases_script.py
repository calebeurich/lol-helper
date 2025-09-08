import re
import json
import unicodedata
from typing import Dict, Set, Any
import requests
from bs4 import BeautifulSoup

# Switch into main aggregation pipeline, so champion aliases are compiled with it

# -------- Fetch champion names by parsing League of Legends official website --------
def fetch_from_official():
    url = "https://www.leagueoflegends.com/en-us/champions/"
    html = requests.get(url, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")

    names = {
        a.get_text(strip=True)
        for a in soup.select("a")
        if isinstance(a.get("href"), str) and a.get("href", "").startswith("/en-us/champions/")
    }
    # Clean
    return {n.strip() for n in names if n and len(n) <= 40}


def fetch_from_universe():
    url = "https://universe.leagueoflegends.com/en_US/champions/"
    html = requests.get(url, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    texts = {text.get_text(strip=True) for text in soup.find_all(True)}
    keep = set()
    for text in texts:
        if 2 <= len(text) <= 40 and re.search(r"[A-Za-z]", t):
            # crude filter; add more to this denylist if needed
            if text.lower() not in {"explore", "runeterra", "the void", "freljord", "noxus", "zaun", "piltover"}:
                keep.add(text)
    return keep


def get_og_names():
    names = fetch_from_official()
    if len(names) < 50: 
        names |= fetch_from_universe()
    return {name.strip() for name in names if name.strip()}


# -------- Helpers to Clean Text --------
def ascii_fold(string):
    return unicodedata.normalize("NFKD", string).encode("ascii", "ignore").decode("ascii")


def keep_apostrophes(string):
    # keep apostrophes, remove spaces
    string = ascii_fold(string).lower().strip()
    string = re.sub(r"\s+", " ", string)
    return string


def remove_apostrophes(string):
    # remove apostrophes/dots/dashes/spaces
    string = keep_apostrophes(string)
    string = re.sub(r"[’'`]", "", string) # drop apostrophes
    string = re.sub(r"[.\-]", "", string) # drop dots/dashes
    return string


def remove_space(string):
    return remove_apostrophes(string).replace(" ", "")


# -------- Variant generator (clean) --------
ROMAN_MAP = {" i": " 1", " ii": " 2", " iii": " 3", " iv": " 4", " v": " 5"}

def variants_for(name):
    # Account for spaces, apostrophes and roman numerals
    variants = set()

    cleaned_with_apostrophes = keep_apostrophes(name)   # e.g. kha'zix
    cleaned_without_apostrophes = remove_apostrophes(name)     # e.g. khazixs

    # Include cleaned variant without apostrophes or spaces
    variants.add(cleaned_without_apostrophes)                   # "khazix"
    variants.add(remove_space(name))                # "khazix", "missfortune", "jarvaniv"

    # If there are spaces (after punctuation removal), add hyphen/underscore
    if " " in cleaned_without_apostrophes:
        variants.add(cleaned_without_apostrophes.replace(" ", "-"))
        variants.add(cleaned_without_apostrophes.replace(" ", "_"))

    # Apostrophe-as-space (kha zix, vel koz, cho gath, kog maw)
    if "'" in cleaned_with_apostrophes:
        variants.add(cleaned_with_apostrophes)
        replace_apostrophe_w_space = re.sub(r"\s+", " ", cleaned_with_apostrophes.replace("'", " ")).strip()
        if replace_apostrophe_w_space:
            variants.add(replace_apostrophe_w_space)
            if " " in replace_apostrophe_w_space:
                variants.add(replace_apostrophe_w_space.replace(" ", "-"))
                variants.add(replace_apostrophe_w_space.replace(" ", "_"))

    # Roman numerals → numbers with and without space (jarvan iv: jarvan 4, jarvan4)
    for roman, number in ROMAN_MAP.items():
        if cleaned_without_apostrophes.endswith(roman):
            number_spaced = cleaned_without_apostrophes.replace(roman, number)   # "jarvan 4"
            variants.add(number_spaced)
            variants.add(number_spaced.replace(" ", ""))          # "jarvan4"

    return {text.strip() for text in variants if text.strip()}


# -------- Hand-picked cases --------
EXTRA_ALIASES: Dict[str, str] = {
    "mf": "Miss Fortune",
    "j4": "Jarvan IV",
    "mundo": "Dr. Mundo",
    "lb": "LeBlanc",
    "kass": "Kassadin",
    "cho": "Cho'Gath",
    "kata": "Katarina",
    "yi": "Master Yi",
    "ww": "Warwick",
    "voli": "Volibear",
    "vlad": "Vladimir",
    "morde": "Mordekaiser",
    # apostrophe names common misspellings
    "khazix": "Kha'Zix",
    "velkoz": "Vel'Koz",
    "kogmaw": "Kog'Maw",
    "chogath": "Cho'Gath",
}


# -------- Build alias into og map --------
def build_alias_map() -> Dict[str, str]:
    canon = sorted(get_og_names())

    # Filter obvious non-champion noise
    DENY = {"Explore", "Runeterra", "The Void", "Freljord", "Noxus", "Zaun", "Piltover"}
    canon = [c for c in canon if c not in DENY]

    alias_to_og: Dict[str, str] = {}

    # Always map cleaned keys back to the exact canonical name
    for name in canon:
        # og cleaned keys
        alias_to_og[remove_apostrophes(name)] = name
        alias_to_og[remove_space(name)] = name

        # apostrophe-as-space variants
        cleaned_with_apostrophes = keep_apostrophes(name)
        if "'" in cleaned_with_apostrophes:
            replace_apostrophe_w_space = re.sub(r"\s+", " ", cleaned_with_apostrophes.replace("'", " ")).strip()
            if replace_apostrophe_w_space:
                alias_to_og[replace_apostrophe_w_space] = name
                if " " in replace_apostrophe_w_space:
                    alias_to_og[replace_apostrophe_w_space.replace(" ", "-")] = name
                    alias_to_og[replace_apostrophe_w_space.replace(" ", "_")] = name

        # generated variants
        for alias in variants_for(name):
            alias_to_og[alias] = name

    # Overlay extras
    for alias, og_name in EXTRA_ALIASES.items():
        alias_to_og[alias.strip().lower()] = og_name

    return alias_to_og


if __name__ == "__main__":
    alias_map = build_alias_map()
    print(f"aliases: {len(alias_map)} entries")

    # quick spot checks
    samples = [
        "velkoz", "vel koz", "vel-koz", "vel_koz",
        "khazix", "kha zix", "jarvan 4", "jarvan4",
        "missfortune", "dr mundo", "mf", "j4"
    ]
    for sample in samples:
        print(f"{sample:12s} -> {alias_map.get(sample)}")

    with open("champion_aliases.json", "w", encoding="utf-8") as f:
        json.dump(alias_map, f, ensure_ascii=False, indent=2)
    print("Wrote champion_aliases.json")