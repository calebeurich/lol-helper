import pandas as pd
import os
import requests
from dotenv import load_dotenv

#.env variables; API key and my own IGN + tag
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
my_game_name = os.getenv("my_game_name")
my_tag_line = os.getenv("my_tag_line")

#variable to input API key in the header of request instead of query
headers = {
    "X-Riot-Token" : RIOT_API_KEY
}

#function to pull puuid using summoner_id
def get_puuid(summoner_id):
    api_url = f"https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
    try:
        resp = requests.get(api_url, headers=headers )  #Insert API link here
        resp.raise_for_status()
    except requests.HTTPError:
        print("Couldn't complete request")   
    player_info = resp.json()
    return player_info["puuid"]

#need to create logic to get next 100s of match ids
def get_match_id(puuid):
    api_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count=100"
    try:
        resp = requests.get(api_url, headers=headers )  #Insert API link here
        resp.raise_for_status()
    except requests.HTTPError:
        print("Couldn't complete request")
    match_ids = resp.json()
    return match_ids



#need to create function for match data and filter by ranked and patch