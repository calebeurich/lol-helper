from IPython.display import display
import pandas as pd
import os
import requests
import json
from dotenv import load_dotenv
import time
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Function to create a list with all match data
def create_match_df(dataframe_csv: str) -> list[dict]:
    df_all_matches = pd.read_csv(dataframe_csv)
    df_all_matches["matchData"] = df_all_matches["matchData"].apply(json.loads)
    df_participants = create_df_participants(df_all_matches)
    df_teams = create_df_teamdata(df_all_matches)
    return df_participants, df_teams  
    
# Create a function to create a DataFrame from the participants column (i.e. a list of dictionaries)
def create_df_participants(df_all_matches: pd.DataFrame) -> pd.DataFrame:
    # Temporary code to filter by ranked games only (queuId = 420), will be removed once we include this logic where we pull match ids
    df_all_matches["queueId"] = df_all_matches["matchData"].apply(lambda m: m["info"]["queueId"])
    df_all_matches = df_all_matches[df_all_matches["queueId"] == 420].copy(deep = True)
    # Actual function logic below remains unchanged
    df_all_matches["participants"] = df_all_matches["matchData"].apply(lambda m: m["info"]["participants"])
    df_exploded = df_all_matches.explode("participants").reset_index(drop=True)
    df_participants = pd.json_normalize(df_exploded["participants"], sep = "_")
    df_participants["matchId"] = df_exploded["matchData"].apply(lambda m: m["metadata"]["matchId"])
    return df_participants

# Create a function to create a DataFrame with data from each team (each row is a team), like bans and objectives
def create_df_teamdata(df_all_matches: pd.DataFrame) -> pd.DataFrame:
    # Temporary code to filter by ranked games only (queuId = 420), will be removed once we include this logic where we pull match ids
    df_all_matches["queueId"] = df_all_matches["matchData"].apply(lambda m: m["info"]["queueId"])
    df_all_matches = df_all_matches[df_all_matches["queueId"] == 420].copy(deep = True)
    # Actual function logic below remains unchanged
    df_all_matches["teams"] = df_all_matches["matchData"].apply(lambda m: m["info"]["teams"])
    df_exploded = df_all_matches.explode("teams").reset_index(drop=True)
    df_teams = pd.json_normalize(df_exploded["teams"], sep = "_")
    df_teams["matchId"] = df_exploded["matchData"].apply(lambda m: m["metadata"]["matchId"])
    return df_teams

df_participants, df_teams = create_match_df("match_data.csv")

def aggregate_champion_data(df_participants: pd.DataFrame, df_teams: pd.DataFrame) -> pd.DataFrame:
    # Create a teamId column in participants based on participantId (1-5 = 100, 6-10 = 200)
    df_participants['teamId'] = df_participants['participantId'].apply(lambda x: 100 if x <= 5 else 200)
    # Create an indicator column for games where champion had a dragon takedown, and subsequent columns with the timing of first dragon takedown
    df_participants['had_dragon_takedown'] = df_participants['challenges_earliestDragonTakedown'].apply(lambda x: 1 if pd.notnull(x) else 0)
    df_participants['sum_first_drag_tkd_min_5to7'] = df_participants['challenges_earliestDragonTakedown'].apply(
        lambda x: 1 if pd.notnull(x) and 300 < x <= 420 else 0)
    df_participants['sum_first_drag_tkd_min_7to11'] = df_participants['challenges_earliestDragonTakedown'].apply(
        lambda x: 1 if pd.notnull(x) and 420 < x <= 660 else 0)
    df_participants['sum_first_drag_tkd_min_11to15'] = df_participants['challenges_earliestDragonTakedown'].apply(
        lambda x: 1 if pd.notnull(x) and 660 < x <= 900 else 0)
    df_participants['sum_first_drag_tkd_min_15+'] = df_participants['challenges_earliestDragonTakedown'].apply(
        lambda x: 1 if pd.notnull(x) and x > 900 else 0)
    # Ensure that junglerkillsEarlyJungle only returns values when jungler did play position (key value pair only appear for junglers)
    # Make sure to check if this performs well
    check_if_jungler = (
        (df_participants['teamPosition'] == 'JUNGLE') & 
        (df_participants['challenges_playedChampSelectPosition'] == 1)
    )
    df_participants.loc[~check_if_jungler, 'challenges_junglerKillsEarlyJungle'] = pd.NA
    df_participants.loc[~check_if_jungler, 'challenges_killsOnLanersEarlyJungleAsJungler'] = pd.NA
    

    
    # Merge participants with their team data
    merged_df = df_participants.merge(
        df_teams,
        on=['matchId', 'teamId'],
        how='left'
    )
    
    # Aggregate by champion
    champion_stats = merged_df.groupby(['championId', 'championName']).agg({
        # Core stats
        'kills': 'mean',
        'deaths': 'mean',
        'challenges_deathsByEnemyChamps': 'mean',
        'assists': 'mean',
        'challenges_killParticipation': 'mean',
        'challenges_takedowns': 'mean',
        'matchId': 'count', # Count number of games played per champion
        'win_x': 'count',
        
        # Offensive combat stats
        'challenges_highestChampionDamage': 'sum', # Gives value of 1 when champ dealt highest dmg of the game, will use to derive %games as highest dmg dealer
        'challenges_damagePerMinute': 'mean',
        'totalDamageDealtToChampions': 'mean',
        'totalDamageDealt': 'mean',
        'magicDamageDealtToChampions': 'mean',
        'magicDamageDealt': 'mean',
        'physicalDamageDealtToChampions': 'mean',
        'physicalDamageDealt': 'mean',
        'trueDamageDealtToChampions': 'mean',
        'trueDamageDealt': 'mean',
        'challenges_teamDamagePercentage': 'mean',
        'largestCriticalStrike': 'mean',
        'challenges_highestCrowdControlScore': 'sum', # Gives value of 1 when champ had highet cc score of game, will use to derive %games as highest cc dealer
        'timeCCingOthers': 'mean',
        'totalTimeCCDealt': 'mean',
        'challenges_enemyChampionImmobilizations': 'mean',
        
        # Defensive combat stats
        'totalDamageTaken': 'mean',
        'challenges_damageTakenOnTeamPercentage': 'mean',
        'longestTimeSpentLiving': 'mean',
        'totalTimeSpentDead': 'mean',
        'magicDamageTaken': 'mean',
        'physicalDamageTaken': 'mean',
        'trueDamageTaken': 'mean',
        'totalHealsOnTeammates': 'mean',
        'totalUnitsHealed': 'mean',
        'totalHeal': 'mean',
        'totalDamageShieldedOnTeammates': 'mean',
        'challenges_effectiveHealAndShielding': 'mean',
        'damageSelfMitigated': 'mean',

        # Support (playstyle, not role)
        'challenges_fasterSupportQuestCompletion': 'sum', # Gives value of 1 for support who completed supp quest firts in the match

        # Spell cast and mechanics
        'spell1Casts': 'mean',
        'spell2Casts': 'mean',
        'spell3Casts': 'mean',
        'spell4Casts': 'mean',
        'challenges_abilityUses': 'mean',
        'challenges_dodgeSkillShotsSmallWindow': 'mean',
        'challenges_skillshotsDodged': 'mean',
        'challenges_landSkillShotsEarlyGame': 'mean',
        'challenges_skillshotsHit': 'mean',
        'challenges_immobilizeAndKillWithAlly': 'mean',
        'challenges_killAfterHiddenWithAlly': 'mean',
        'challenges_pickKillWithAlly': 'mean',
        'challenges_killedChampTookFullTeamDamageSurvived': 'mean',
        'challenges_survivedSingleDigitHpCount': 'mean',
        'challenges_survivedThreeImmobilizesInFight': 'mean',
        'challenges_tookLargeDamageSurvived': 'mean',
        'challenges_killsNearEnemyTurret': 'mean',
        'challenges_killsUnderOwnTurret': 'mean',
        'challenges_killsOnOtherLanesEarlyJungleAsLaner': 'mean',
        'challenges_knockEnemyIntoTeamAndKill': 'mean',
        'challenges_multikillsAfterAggressiveFlash': 'mean',
        'challenges_outnumberedKills': 'mean',
        'challenges_outnumberedNexusKill': 'mean',
        'challenges_quickCleanse': 'mean',
        'challenges_quickSoloKills': 'mean',
        'challenges_soloKills': 'mean',
        'challenges_takedownsAfterGainingLevelAdvantage': 'mean',
        'challenges_saveAllyFromDeath': 'mean',
        'challenges_takedownsInAlcove': 'mean',

        # Experience stats
        'champExperience': 'mean',
        'champLevel': 'mean', 
        
        # Economy stats
        'goldEarned': 'mean',
        'challenges_goldPerMinute': 'mean',
        'goldSpent': 'mean',
        'totalMinionsKilled': 'mean',
        'challenges_laneMinionsFirst10Minutes': 'mean',
        'bountyLevel': 'mean',
        'challenges_bountyGold': 'mean',
        'consumablesPurchased': 'mean',
        'itemsPurchased': 'mean', # int of number of items purchased in game, might be useless as it seems to count items and components as well
        'challenges_earlyLaningPhaseGoldExpAdvantage': 'mean',
        'challenges_laningPhaseGoldExpAdvantage': 'mean',
        'challenges_maxLevelLeadLaneOpponent': 'mean',
        'challenges_maxCsAdvantageOnLaneOpponent': 'mean',
        'challenges_fastestLegendary': 'count', # Counts how many games the champion was first to acquire an item
        
        # Jungle related
        'neutralMinionsKilled': 'mean', # Jungle monsters/farm
        'challenges_buffsStolen': 'mean',
        'challenges_epicMonsterKillsWithin30SecondsOfSpawn': 'mean',
        'challenges_junglerKillsEarlyJungle': 'mean', # Note: only 2 players per game will have this stat, the junglers, might be an issue when champs don't play assigned role, might be worth writing logic with 'challenges_playedChampSelectPosition' 
        'challenges_killsOnLanersEarlyJungleAsJungler': 'mean',
        'challenges_getTakedownsInAllLanesEarlyJungleAsLaner': 'mean',
        'challenges_initialBuffCount': 'mean',
        'challenges_initialCrabCount': 'mean',
        'challenges_scuttleCrabKills': 'mean',
        'challenges_jungleCsBefore10Minutes': 'mean',
        'challenges_junglerTakedownsNearDamagedEpicMonster': 'mean',
        'challenges_killsWithHelpFromEpicMonster': 'mean',

        # Vision stats
        'challenges_highestWardKills': 'sum',
        'visionScore': 'mean',
        'challenges_visionScorePerMinute': 'mean',
        'challenges_visionScoreAdvantageLaneOpponent': 'mean',
        'challenges_stealthWardsPlaced': 'mean',
        'wardsPlaced': 'mean',
        'challenges_wardsGuarded': 'mean',
        'wardsKilled': 'mean',
        'challenges_wardTakedowns': 'mean',
        'challenges_wardTakedownsBefore20M': 'mean',
        'challenges_twoWardsOneSweeperCount': 'mean',
        'visionWardsBoughtInGame': 'mean',
        'detectorWardsPlaced': 'mean', # Same as control wards
        'challenges_controlWardTimeCoverageInRiverOrEnemyHalf': 'mean',
        'challenges_unseenRecalls': 'mean',
        
        # Team objectives/stats - from df_teams - need to rename column headers for clarity 
        'objectives_baron_kills': 'mean',
        'objectives_champion_kills': 'mean',
        'challenges_teamRiftHeraldKills': 'mean',
        'objectives_dragon_kills': 'mean',
        'challenges_perfectDragonSoulsTaken': 'mean', # Data is from df_participants but is on a per team basis
        'challenges_teamElderDragonKills': 'mean',
        'objectives_inhibitor_kills': 'mean',
        'objectives_riftHerald_kills': 'mean',
        'objectives_tower_kills': 'mean',
        'inhibitorsLost': 'mean', # Data is from df_participants but is on a per team basis
        'nexusLost': 'mean', # Data is from df_participants but is on a per team basis
        'turretsLost': 'mean', # Data is from df_participants but is on a per team basis
        'challenges_acesBefore15Minutes': 'mean', # Data is from df_participants but is on a per team basis
        'challenges_flawlessAces': 'mean', # Data is from df_participants but is on a per team basis
        'challenges_shortestTimeToAceFromFirstTakedown': 'mean', # Data is from df_participants but am not sure if it is on a per team basis
        'challenges_elderDragonKillsWithOpposingSoul': 'mean', # Data is from df_participants but idk if this is on a per team or individual who killed elder basis
        'challenges_maxKillDeficit': 'mean', # Data is from df_participants but is on a per team basis
        'challenges_perfectGame': 'mean', # Data is from df_participants but is on a per team basis
        
        # Individual participant damage to structures
        'damageDealtToBuildings': 'mean',
        'damageDealtToTurrets': 'mean',
        'challenges_turretPlatesTaken': 'mean',
        'damageDealtToObjectives': 'mean',
        'firstTowerKill': 'mean', # Boolean for champ who took first tower
        'challenges_takedownOnFirstTurret': 'mean', # Value of 1 or 0, states whether participant had takedown on first turret 
        'challenges_quickFirstTurret': 'mean',
        'challenges_firstTurretKilled': 'mean', # Value of 1 or 0, states whether team took first turret or not
        'challenges_firstTurretKilledTime': 'mean',
        'firstTowerAssist': 'mean',
        'challenges_kTurretsDestroyedBeforePlatesFall': 'mean', # Seems to be on a per lane basis
        'turretKills': 'mean',
        'turretTakedowns': 'mean',
        'challenges_turretTakedowns': 'mean', # Compare with above 
        'challenges_soloTurretsLategame': 'mean',
        'inhibitorKills': 'mean',
        'inhibitorTakedowns': 'mean', # Takedown = kill or assist
        'nexusKills': 'mean', # Who last hit the nexus
        'nexusTakedowns': 'mean',
        'challenges_hadOpenNexus': 'mean',
        'challenges_turretsTakenWithRiftHerald': 'mean',
        'challenges_multiTurretRiftHeraldCount': 'mean',

        # Individual participant objectives
        'baronKills': 'mean',
        'challenges_soloBaronKills': 'mean',
        'challenges_baronTakedowns': 'mean',
        'dragonKills': 'mean',
        'challenges_dragonTakedowns': 'mean',
        'objectivesStolen': 'mean',
        'challenges_epicMonsterSteals': 'mean',
        'challenges_epicMonsterStolenWithoutSmite': 'mean',
        'challenges_epicMonsterKillsNearEnemyJungler': 'mean',
        'objectivesStolenAssists': 'mean',
        'challenges_riftHeraldTakedowns': 'mean',
        'challenges_voidMonsterKill': 'mean', # Void grubs? I think this counts herald as well, might want to derive grubs stats from here
        
        # First objective rates - from df_teams
        'objectives_baron_first': 'mean',
        'objectives_dragon_first': 'mean',
        'objectives_inhibitor_first': 'mean',
        'objectives_riftHerald_first': 'mean',
        'objectives_tower_first': 'mean',
   
        # Game length related
        'timePlayed': 'mean',
        'challenges_gameLength': 'mean', # Compare with above and delete one
        'gameEndedInEarlySurrender': 'mean', # Probably needs to be a derived stat, might need count first (might work with mean)
        'gameEndedInSurrender': 'mean', # Probably needs to be a derived stat, might need count first
        'teamEarlySurrendered': 'mean',
        
        # Multikill stats
        'doubleKills': 'mean',
        'tripleKills': 'mean',
        'quadraKills': 'mean',
        'pentaKills': 'mean',
        'largestMultiKill': 'mean',
        'challenges_multikills': 'mean',
        'challenges_multiKillOneSpell': 'mean',
        'killingSprees': 'mean',
        'challenges_killingSprees': 'mean', # Check if equal and delete one
        'challenges_legendaryCount':'mean', # Not too sure what this is, my current guess is how many times champ was legendary
        'largestKillingSpree': 'mean',
        'unrealKills' : 'mean', # Might be a zero stat, need to check
        'challenges_12AssistStreakCount': 'mean',
        'challenges_elderDragonMultikills': 'mean',
        'challenges_fullTeamTakedown': 'mean',
        
        # Earliest baron taken by team in games where they took baron (still need to decide what to do with this)
        'challenges_earliestBaron': 'mean',
        
        # Earliest dragon takedown stats:
        'challenges_earliestDragonTakedown': 'mean',
        'had_dragon_takedown': 'sum',
        'sum_first_drag_tkd_min_5to7': 'sum',
        'sum_first_drag_tkd_min_7to11': 'sum',
        'sum_first_drag_tkd_min_11to15': 'sum',
        'sum_first_drag_tkd_min_15+': 'sum',

        # Items - decide whetheh to use data from match or from timeline and make sure to indluce item tags in df 

        # Individual participant combat and obj achievements
        'firstBloodKill': 'mean',
        'firstBloodAssist': 'mean',     
        'challenges_takedownsBeforeJungleMinionSpawn': 'mean',
        'challenges_takedownsFirstXMinutes': 'mean', # Minute of first takedown
        
        # Misc
        'challenges_blastConeOppositeOpponentCount': 'mean',
        'challenges_completeSupportQuestInTime': 'mean',
        'challenges_dancedWithRiftHerald': 'mean',
        'challenges_doubleAces': 'mean',
        'challenges_fistBumpParticipation': 'mean',
        'challenges_mejaisFullStackInTime': 'mean',
        'challenges_outerTurretExecutesBefore10Minutes': 'mean',
        'challenges_takedownsInEnemyFountain': 'mean',
        
        # Position/Role - find aggregation key word for most common string
        'individualPosition': lambda x: x.mode()[0] if not x.mode().empty else None, # Best guess for which position the player actually played in isolation of anything else
        'lane': lambda x: x.mode()[0] if not x.mode().empty else None, # Gives slightly different string than above, might have something to do with where champ spent most of the time
        'role': lambda x: x.mode()[0] if not x.mode().empty else None,
        'teamPosition': lambda x: x.mode()[0] if not x.mode().empty else None, # The teamPosition is the best guess for which position the player actually played if we add the constraint that each team must have one top player, one jungle, one middle, etc
        'challenges_playedChampSelectPosition': 'mean'

    }).round(2)
    
    # Rename columns for clarity
    champion_stats = champion_stats.rename(columns={
        'matchId': 'games_played', # Rename the count column to a more descriptive name
        'objectives_champion_kills': 'team_kills',
        'objectives_baron_first': 'team_first_baron_rate',
        'objectives_baron_kills': 'team_barons',
        'objectives_dragon_first': 'team_first_dragon_rate',
        'objectives_dragon_kills': 'team_dragons',
        'objectives_inhibitor_first': 'team_first_inhibitor_rate',
        'objectives_inhibitor_kills': 'team_inhibs_taken',
        'objectives_riftHerald_first': 'team_first_herald_rate',
        'objectives_riftHerald_kills': 'team_heralds_taken',
        'objectives_tower_first': 'team_first_tower_rate',
        'objectives_tower_kills': 'team_towers_taken',
        'firstBloodKill': 'first_blood_rate',
        'firstTowerKill': 'first_tower_kill_rate'
    })
    
    # Calculate derived stats
    champion_stats['kda'] = ((champion_stats['kills'] + champion_stats['assists']) / 
                            champion_stats['deaths']).round(2)
    champion_stats['winrate'] = ((champion_stats['win_x'] ) / 
                            champion_stats['games_played']).round(2)
    champion_stats['cs_per_minute'] = ((champion_stats['totalMinionsKilled'] + champion_stats['neutralMinionsKilled']) / 
                                      (champion_stats['timePlayed'] / 60)).round(2)
    champion_stats['gold_per_minute'] = (champion_stats['goldEarned'] / 
                                        (champion_stats['timePlayed'] / 60)).round(2)
    champion_stats['damage_per_minute'] = (champion_stats['totalDamageDealtToChampions'] / 
                                          (champion_stats['timePlayed'] / 60)).round(2)
    champion_stats['pct_first_drag_tkd_min_5to7'] = (((champion_stats['sum_first_drag_tkd_min_5to7']) / 
                                                      champion_stats['had_dragon_takedown']) * 100).round(2)
    champion_stats['pct_first_drag_tkd_min_7to11'] = (((champion_stats['sum_first_drag_tkd_min_7to11']) / 
                                                      champion_stats['had_dragon_takedown']) * 100).round(2)
    champion_stats['pct_first_drag_tkd_min_11to15'] = (((champion_stats['sum_first_drag_tkd_min_11to15']) / 
                                                      champion_stats['had_dragon_takedown']) * 100).round(2)
    champion_stats['pct_first_drag_tkd_min_15+'] = (((champion_stats['sum_first_drag_tkd_min_15+']) / 
                                                      champion_stats['had_dragon_takedown']) * 100).round(2)
    # Statistic measuring what % of games played this champion had the highest damage dealth of the game (same logic for other ones below)
    champion_stats['pct_highest_dmg_in_match'] = (((champion_stats['challenges_highestChampionDamage']) / 
                                                      champion_stats['games_played']) * 100).round(2)
    champion_stats['pct_highest_cc_in_match'] = (((champion_stats['challenges_highestCrowdControlScore']) / 
                                                      champion_stats['games_played']) * 100).round(2)
    champion_stats['pct_highest_ward_kills_in_match'] = (((champion_stats['challenges_highestWardKills']) / 
                                                      champion_stats['games_played']) * 100).round(2)
    champion_stats['pct_fastest_supp_quest_in_match'] = (((champion_stats['challenges_fasterSupportQuestCompletion']) / 
                                                      champion_stats['games_played']) * 100).round(2)
    champion_stats['pct_first_to_buy_item_in_match'] = (((champion_stats['challenges_fastestLegendary']) / 
                                                      champion_stats['games_played']) * 100).round(2)
    
    

    champion_stats = champion_stats.drop([
        'sum_first_drag_tkd_min_5to7', 'sum_first_drag_tkd_min_7to11', 'sum_first_drag_tkd_min_11to15', 'sum_first_drag_tkd_min_15+',
        'challenges_earliestDragonTakedown', 'challenges_highestChampionDamage', 'challenges_highestCrowdControlScore',
        'challenges_fasterSupportQuestCompletion'
    ], axis = 1)


    return champion_stats

# Create and display the aggregated data
champion_stats = aggregate_champion_data(df_participants, df_teams)
# display(champion_stats.sort_values('challenges_earliestDragonTakedown', ascending=False))
champion_stats.to_csv("champion_stats.csv", index = True)
total_games = champion_stats['games_played'].sum()


print(total_games)