import pandas as pd
import json
from typing import Tuple

def create_match_df(match_data_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create separate dataframes for participants and teams from match data"""
    df_matches = pd.read_csv(match_data_file)
    df_matches["matchData"] = df_matches["matchData"].apply(json.loads)
    
    return (
        create_df_participants(df_matches),
        create_df_teamdata(df_matches)
    )

def create_df_participants(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Extract participant data from matches"""
    df_matches["participants"] = df_matches["matchData"].apply(lambda m: m["info"]["participants"])
    df_exploded = df_matches.explode("participants").reset_index(drop=True)
    df_participants = pd.json_normalize(df_exploded["participants"], sep="_")
    df_participants["matchId"] = df_exploded["matchData"].apply(lambda m: m["metadata"]["matchId"])
    return df_participants

def create_df_teamdata(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Extract team data from matches"""
    df_matches["teams"] = df_matches["matchData"].apply(lambda m: m["info"]["teams"])
    df_exploded = df_matches.explode("teams").reset_index(drop=True)
    df_teams = pd.json_normalize(df_exploded["teams"], sep="_")
    df_teams["matchId"] = df_exploded["matchData"].apply(lambda m: m["metadata"]["matchId"])
    return df_teams

def aggregate_champion_data(df_participants: pd.DataFrame, df_teams: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate match data to create champion statistics.
    
    Args:
        df_participants: DataFrame containing participant-level match data
        df_teams: DataFrame containing team-level match data
    
    Returns:
        DataFrame containing aggregated champion statistics
    """
    # Create a teamId column in participants based on participantId
    df_participants['teamId'] = df_participants['participantId'].apply(lambda x: 100 if x <= 5 else 200)
    
    # Merge participants with their team data
    merged_df = df_participants.merge(
        df_teams,
        on=['matchId', 'teamId'],
        how='left'
    )
    
    # Define aggregation metrics
    agg_dict = {
        # Core stats
        'kills': 'mean',
        'deaths': 'mean',
        'assists': 'mean',
        'matchId': 'count',
        
        # Combat stats
        'totalDamageDealtToChampions': 'mean',
        'totalDamageTaken': 'mean',
        'magicDamageDealtToChampions': 'mean',
        'physicalDamageDealtToChampions': 'mean',
        'trueDamageDealtToChampions': 'mean',
        'largestCriticalStrike': 'mean',
        'timeCCingOthers': 'mean',
        'totalHealsOnTeammates': 'mean',
        'totalDamageShieldedOnTeammates': 'mean',
        
        # Economy stats
        'goldEarned': 'mean',
        'goldSpent': 'mean',
        'totalMinionsKilled': 'mean',
        'neutralMinionsKilled': 'mean',
        
        # Vision stats
        'visionScore': 'mean',
        'wardsPlaced': 'mean',
        'wardsKilled': 'mean',
        'visionWardsBoughtInGame': 'mean',
        
        # Team objectives
        'objectives_baron_kills': 'mean',
        'objectives_champion_kills': 'mean',
        'objectives_dragon_kills': 'mean',
        'objectives_inhibitor_kills': 'mean',
        'objectives_riftHerald_kills': 'mean',
        'objectives_tower_kills': 'mean',
        
        # First objective rates
        'objectives_baron_first': 'mean',
        'objectives_dragon_first': 'mean',
        'objectives_inhibitor_first': 'mean',
        'objectives_riftHerald_first': 'mean',
        'objectives_tower_first': 'mean',
        
        # Game length related
        'timePlayed': 'mean',
        'longestTimeSpentLiving': 'mean',
        
        # Multikill stats
        'doubleKills': 'mean',
        'tripleKills': 'mean',
        'quadraKills': 'mean',
        'pentaKills': 'mean',
        
        # Combat achievements
        'firstBloodKill': 'mean',
        'firstTowerKill': 'mean',
        'turretKills': 'mean',
        'inhibitorKills': 'mean',
    }
    
    # Aggregate by champion
    champion_stats = merged_df.groupby(['championId', 'championName']).agg(agg_dict).round(2)
    
    # Rename columns for clarity
    column_renames = {
        'matchId': 'games_played',
        'objectives_baron_first': 'first_baron_rate',
        'objectives_dragon_first': 'first_dragon_rate',
        'objectives_inhibitor_first': 'first_inhibitor_rate',
        'objectives_riftHerald_first': 'first_herald_rate',
        'objectives_tower_first': 'first_tower_rate',
        'firstBloodKill': 'first_blood_rate',
        'firstTowerKill': 'first_tower_kill_rate'
    }
    champion_stats = champion_stats.rename(columns=column_renames)
    
    # Calculate derived stats
    champion_stats['kda'] = ((champion_stats['kills'] + champion_stats['assists']) / 
                            champion_stats['deaths']).round(2)
    
    champion_stats['cs_per_minute'] = ((champion_stats['totalMinionsKilled'] + 
                                       champion_stats['neutralMinionsKilled']) / 
                                      (champion_stats['timePlayed'] / 60)).round(2)
    
    champion_stats['gold_per_minute'] = (champion_stats['goldEarned'] / 
                                        (champion_stats['timePlayed'] / 60)).round(2)
    
    champion_stats['damage_per_minute'] = (champion_stats['totalDamageDealtToChampions'] / 
                                          (champion_stats['timePlayed'] / 60)).round(2)
    
    return champion_stats.reset_index()

def main():
    """Main analysis function"""
    df_participants, df_teams = create_match_df("all_match_data.csv")
    champion_stats = aggregate_champion_data(df_participants, df_teams)
    champion_stats.to_csv("champion_statistics.csv")
    
    print("\nFinal champion statistics:")
    print(f"Total champions analyzed: {len(champion_stats)}")
    print("\nTop 5 most played champions:")
    print(champion_stats.sort_values('games_played', ascending=False).head())

if __name__ == "__main__":
    main() 