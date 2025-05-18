from IPython.display import display
import pandas as pd
import json
from dotenv import load_dotenv
from items_and_summs_module import tag_finder
from items_and_summs_module import item_filter
from items_and_summs_module import get_summ_spell_name

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Open item_dict data
with open('item_id_tags.json', 'r') as data:
    items_dict = json.load(data)
# Function to convert item ids into list of tags
def item_ids_to_tags(row):
    return (tag_finder(str(row['item0'])) + tag_finder(str(row['item1'])) + tag_finder(str(row['item2'])) + 
            tag_finder(str(row['item3'])) + tag_finder(str(row['item4'])) + tag_finder(str(row['item5'])))
# Function to count total items purchased to derive pct of tags statistic
def completed_items_count(row):
    return (item_filter(str(row['item0'])) + item_filter(str(row['item1'])) + item_filter(str(row['item2'])) + 
            item_filter(str(row['item3'])) + item_filter(str(row['item4'])) + item_filter(str(row['item5'])))

def summ_spells_per_game(row):
    return [get_summ_spell_name(row['summoner1Id']), get_summ_spell_name(row['summoner2Id'])]

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
    # Early filtering: Remove entries with null or invalid positions
    df_participants = df_participants[df_participants['teamPosition'].notna()]
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
    # Item tags column
    df_participants['item_tags'] = df_participants.apply(item_ids_to_tags, axis=1)
    df_participants['completed_items'] = df_participants.apply(completed_items_count, axis=1)
    # Summoner spell columns
    df_participants['summspells_per_game'] = df_participants.apply(summ_spells_per_game, axis=1)
    # Ensure that junglerkllsEarlyJungle only returns values when jungler did play position (key value pair only appear for junglers)
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
    
    # Filtering out champion+role that have less than 5% of total games played by champion
    champ_games = merged_df.groupby('championId')['matchId'].nunique()
    champ_role_games = merged_df.groupby(['championId', 'teamPosition'])['matchId'].nunique()
    percentages = champ_role_games / champ_games[champ_role_games.index.get_level_values(0)].values * 100
    valid_champ_roles = percentages[percentages >= 5].index.tolist()
    filtered_df = merged_df[merged_df.set_index(['championId', 'teamPosition']).index.isin(valid_champ_roles)]

    # Aggregate by champion
    champion_stats = filtered_df.groupby(['championId', 'championName', 'teamPosition']).agg(
        # Core stats (KDA and number of games)
        total_games_played_in_role = ('matchId', 'count'), # Count number of games played per champion
        avg_kills = ('kills', 'mean'),
        avg_deaths = ('deaths', 'mean'),
        avg_deaths_by_enemy_champs = ('challenges_deathsByEnemyChamps', 'mean'), # Need to find out what this is
        avg_assists = ('assists', 'mean'),
        avg_kill_participation = ('challenges_killParticipation', 'mean'),
        avg_takedowns = ('challenges_takedowns', 'mean'),
        total_wins = ('win_x', 'sum'),
        
        # Damage Dealt 
        total_games_with_highest_damage = ('challenges_highestChampionDamage', 'sum'), # Gives value of 1 when champ dealt highest dmg of the game, used to derive %games as highest dmg dealer
        avg_pct_damage_in_team = ('challenges_teamDamagePercentage', 'mean'),
        avg_dmg_per_minute = ('challenges_damagePerMinute', 'mean'),
        avg_dmg_dealt_to_champions = ('totalDamageDealtToChampions', 'mean'),
        avg_dmg_dealt = ('totalDamageDealt', 'mean'), # Maybe remove this? Do we care about damage dealt to non champions?
        avg_magic_dmg_to_champs = ('magicDamageDealtToChampions', 'mean'),
        avg_magic_dmg = ('magicDamageDealt', 'mean'), # Maybe remove this? Do we care about damage dealt to non champions?
        avg_physical_dmg_to_champs = ('physicalDamageDealtToChampions', 'mean'),
        avg_physical_dmg = ('physicalDamageDealt', 'mean'),
        avg_true_dmg_to_champs = ('trueDamageDealtToChampions', 'mean'),
        avg_true_dmg = ('trueDamageDealt', 'mean'),
        avg_largest_crit = ('largestCriticalStrike', 'mean'),
        
        # Crowd Control         
        total_games_with_highest_cc_score = ('challenges_highestCrowdControlScore', 'sum'), # Gives value of 1 when champ had highet cc score of game
        avg_time_ccing_champs = ('timeCCingOthers', 'mean'),
        avg_total_time_cc_dealt = ('totalTimeCCDealt', 'mean'),
        avg_champ_immobilizations = ('challenges_enemyChampionImmobilizations', 'mean'),
        
        # Damage Taken
        avg_dmg_taken_team_pct = ('challenges_damageTakenOnTeamPercentage', 'mean'),
        avg_dmg_taken = ('totalDamageTaken', 'mean'),
        avg_magic_dmg_taken = ('magicDamageTaken', 'mean'),
        avg_physical_dmg_taken = ('physicalDamageTaken', 'mean'),
        avg_tru_dmg_taken = ('trueDamageTaken', 'mean'),
        avg_dmg_self_mitigated = ('damageSelfMitigated', 'mean'),
            # Tanking in teamfights
        avg_times_killed_champ_took_full_team_dmg_and_survived = ('challenges_killedChampTookFullTeamDamageSurvived', 'mean'), # Need to check what this is
        avg_times_survived_single_digit_hp = ('challenges_survivedSingleDigitHpCount', 'mean'),
        avg_times_survived_3_immobilizes_in_fight = ('challenges_survivedThreeImmobilizesInFight', 'mean'),
        avg_times_took_large_dmg_survived = ('challenges_tookLargeDamageSurvived', 'mean'),
        
        # Healing and Shielding + Support
        avg_total_healing = ('totalHeal', 'mean'),
        avg_heals_on_teammate = ('totalHealsOnTeammates', 'mean'),
        avg_total_units_healed = ('totalUnitsHealed', 'mean'),
        avg_dmg_shielded_on_team = ('totalDamageShieldedOnTeammates', 'mean'),
        avg_effective_heal_and_shield = ('challenges_effectiveHealAndShielding', 'mean'),
        total_games_first_supp_quest = ('challenges_fasterSupportQuestCompletion', 'sum'), # Gives value of 1 for support who completed supp quest firts in the match
        avg_supp_quest_completion_time = ('challenges_completeSupportQuestInTime', 'mean'),

        avg_longest_time_alive = ('longestTimeSpentLiving', 'mean'),
        avg_time_spent_dead = ('totalTimeSpentDead', 'mean'),
            
        # Spell casts
        avg_spell1_casts = ('spell1Casts', 'mean'),
        avg_spell2_casts = ('spell2Casts', 'mean'),
        avg_spell3_casts = ('spell3Casts', 'mean'),
        avg_spell4_casts = ('spell4Casts', 'mean'),
        avg_ability_uses = ('challenges_abilityUses', 'mean'),

        # Skill shot related (dodging and hitting)
        avg_times_dodged_skill_small_window = ('challenges_dodgeSkillShotsSmallWindow', 'mean'),
        avg_skillshots_dodged = ('challenges_skillshotsDodged', 'mean'),
        avg_skillshots_landed_early_game = ('challenges_landSkillShotsEarlyGame', 'mean'),
        avg_skillshots_hit = ('challenges_skillshotsHit', 'mean'),

        # Picks
        avg_times_immobilize_and_kill_with_ally = ('challenges_immobilizeAndKillWithAlly', 'mean'),
        avg_times_got_kill_after_hidden_with_ally = ('challenges_killAfterHiddenWithAlly', 'mean'),
        avg_times_pick_kill_with_ally = ('challenges_pickKillWithAlly', 'mean'),
        avg_times_knock_enemy_into_team_and_kill = ('challenges_knockEnemyIntoTeamAndKill', 'mean'),

        # Kills under or near turret
        avg_kills_near_enemy_turret = ('challenges_killsNearEnemyTurret', 'mean'),
        avg_kills_under_own_turret = ('challenges_killsUnderOwnTurret', 'mean'),
        
        # Misc mechanics
        avg_multikills_after_aggressive_flash = ('challenges_multikillsAfterAggressiveFlash', 'mean'),
        avg_outnumbered_kills = ('challenges_outnumberedKills', 'mean'),
        avg_outnumbered_nexus_kill = ('challenges_outnumberedNexusKill', 'mean'),
        avg_quick_cleanse = ('challenges_quickCleanse', 'mean'),

        # Misc laning
            # Kills, takedowns and plays 
        avg_quick_solo_kills = ('challenges_quickSoloKills', 'mean'),
        avg_solo_kills = ('challenges_soloKills', 'mean'),
        avg_takedowns_after_gaining_lvl_adv = ('challenges_takedownsAfterGainingLevelAdvantage', 'mean'),
        avg_kills_on_other_lanes_early_as_jgler = ('challenges_killsOnOtherLanesEarlyJungleAsLaner', 'mean'),
        avg_times_save_ally_from_death = ('challenges_saveAllyFromDeath', 'mean'),
        avg_takedowns_in_alcove = ('challenges_takedownsInAlcove', 'mean'),
            # First blood and early kills
        total_games_first_blood_kill = ('firstBloodKill', 'sum'), # Boolean - consider multiplying by 100
        total_games_first_blood_assist = ('firstBloodAssist', 'mean'), # Boolean    
        avg_takedowns_before_jg_camps_spawn = ('challenges_takedownsBeforeJungleMinionSpawn', 'mean'),
        avg_first_takedown_time = ('challenges_takedownsFirstXMinutes', 'mean'), # Minute of first takedown

        # Summoner spells
        summspells_per_game = ('item_tags', 'sum'),

        # Experience stats
        avg_champ_exp_at_game_end = ('champExperience', 'mean'),
        avg_champ_lvl_at_game_end = ('champLevel', 'mean'), 
        
        # Economy stats
            # Gold and EXP stats
        avg_gold_earned_per_game = ('goldEarned', 'mean'),
        avg_gold_per_minute = ('challenges_goldPerMinute', 'mean'),
        avg_gold_spent = ('goldSpent', 'mean'),
        avg_bounty_lvl = ('bountyLevel', 'mean'), # Need to make sure what this is
        avg_bounty_gold = ('challenges_bountyGold', 'mean'),
        total_games_with_early_lanephase_gold_exp_adv = ('challenges_earlyLaningPhaseGoldExpAdvantage', 'mean'), # Need to make sure what this is, is a boolean
        total_games_with_lanephase_gold_exp_adv = ('challenges_laningPhaseGoldExpAdvantage', 'mean'),
        avg_max_lvl_lead_lane_opp = ('challenges_maxLevelLeadLaneOpponent', 'mean'), # I believe this is the lvl lead ahead of the opponent, need to make sure
            # Farm stats
        avg_cs = ('totalMinionsKilled', 'mean'),
        avg_cs_10_mins = ('challenges_laneMinionsFirst10Minutes', 'mean'),
        avg_max_cs_over_lane_opp = ('challenges_maxCsAdvantageOnLaneOpponent', 'mean'),
            # Item purchase stats
        avg_consumables_purchased = ('consumablesPurchased', 'mean'),
        avg_number_of_items_purchased = ('itemsPurchased', 'mean'), # int of number of items purchased in game, might be useless as it seems to count items and components as well
        total_games_fastest_item_completion = ('challenges_fastestLegendary', 'sum'), # Counts how many games the champion was first to acquire an item

        # Item tags
        item_tags = ('item_tags', 'sum'),
        completed_items = ('completed_items', 'sum'),
        
        # Jungle related stats
            # Jungle farm
        avg_neutral_monsters_cs = ('neutralMinionsKilled', 'mean'), # Jungle monsters/farm - note that this shows for all players
        avg_buffs_stolen = ('challenges_buffsStolen', 'mean'),
        avg_initial_buff_count = ('challenges_initialBuffCount', 'mean'),
        avg_epic_monster_kills_within_30s_of_spawn = ('challenges_epicMonsterKillsWithin30SecondsOfSpawn', 'mean'),
        avg_initial_crab_count = ('challenges_initialCrabCount', 'mean'),
        avg_crabs_per_game = ('challenges_scuttleCrabKills', 'mean'),
        avg_jg_cs_before_10m = ('challenges_jungleCsBefore10Minutes', 'mean'),
            # Jungle combat
        avg_jgler_kills_early_jungle = ('challenges_junglerKillsEarlyJungle', 'mean'), # Note: only 2 players per game will have this stat, the junglers, might be an issue when champs don't play assigned role, might be worth writing logic with 'challenges_playedChampSelectPosition' 
        avg_jgler_early_kills_on_laners = ('challenges_killsOnLanersEarlyJungleAsJungler', 'mean'),
        total_games_jgler_had_early_tkdowns_in_all_lanes = ('challenges_getTakedownsInAllLanesEarlyJungleAsLaner', 'mean'), # Need to double check what this gives, dont know if it is a boolean or count of early kills
        avg_jgler_tkdowns_near_damaged_epic_monsters = ('challenges_junglerTakedownsNearDamagedEpicMonster', 'mean'),
        avg_kills_with_help_from_epic_monster = ('challenges_killsWithHelpFromEpicMonster', 'mean'),

        # Vision stats
            # Vision score and wards placed + unseen recalls
        avg_vision_score = ('visionScore', 'mean'),
        avg_vision_score_per_min = ('challenges_visionScorePerMinute', 'mean'),
        avg_vision_score_adv_over_lane_opp = ('challenges_visionScoreAdvantageLaneOpponent', 'mean'),
        avg_stealth_wards_placed = ('challenges_stealthWardsPlaced', 'mean'),
        avg_wards_placed = ('wardsPlaced', 'mean'), # Need to check if duplicate of above
        avg_wards_guarded = ('challenges_wardsGuarded', 'mean'),
        avg_ctrol_wards_placed = ('detectorWardsPlaced', 'mean'), # Same as control wards
        avg_ctrl_ward_time_coverage_in_river_or_enemy_half = ('challenges_controlWardTimeCoverageInRiverOrEnemyHalf', 'mean'),
        avg_unseen_recalls = ('challenges_unseenRecalls', 'mean'),
            # Wards killed
        total_games_with_highest_wards_killed = ('challenges_highestWardKills', 'sum'),
        avg_wards_killed = ('wardsKilled', 'mean'),
        avg_ward_tkdowns = ('challenges_wardTakedowns', 'mean'),
        avg_ward_tkdowns_beforem_20m = ('challenges_wardTakedownsBefore20M', 'mean'),
        avg_2_wards_killed_w_1_sweeper = ('challenges_twoWardsOneSweeperCount', 'mean'),
        avg_ctrl_wards_bought = ('visionWardsBoughtInGame', 'mean'),
        
        # Teamwide stats (mostly from df_teams with some from df_participants)
            # First objective rates
        total_games_team_took_first_baron = ('objectives_baron_first', 'mean'),
        avg_earliest_baron_by_team_time = ('challenges_earliestBaron', 'mean'), # Earliest baron taken by team in games where they took baron
        total_games_team_took_first_drag = ('objectives_dragon_first', 'mean'),
        total_games_team_took_first_inhib = ('objectives_inhibitor_first', 'mean'),
        total_games_team_took_first_herald = ('objectives_riftHerald_first', 'mean'),
        total_games_team_took_first_turret = ('objectives_tower_first', 'mean'),
            # Team objectives
        avg_baron_kills_by_team = ('objectives_baron_kills', 'mean'),
        avg_herald_kills_by_team = ('objectives_riftHerald_kills', 'mean'),
        avg_dragon_kills_by_team = ('objectives_dragon_kills', 'mean'),
        total_games_with_perfect_drag_soul_taken = ('challenges_perfectDragonSoulsTaken', 'mean'), # Data is from df_participants but is on a per team basis
        avg_elder_dragon_kills_by_team = ('challenges_teamElderDragonKills', 'mean'),
        avg_elder_dragon_kills_w_opposing_soul = ('challenges_elderDragonKillsWithOpposingSoul', 'mean'), # Data is from df_participants but idk if this is on a per team or individual who killed elder basis
            # Team structures
        avg_inhib_kills_by_team = ('objectives_inhibitor_kills', 'mean'),
        avg_tower_kills_by_team = ('objectives_tower_kills', 'mean'),
        avg_inhibs_lost_by_team = ('inhibitorsLost', 'mean'), # Data is from df_participants but is on a per team basis
        avg_nexus_lost_by_team = ('nexusLost', 'mean'), # Data is from df_participants but is on a per team basis
        avg_turrets_lost_by_team = ('turretsLost', 'mean'), # Data is from df_participants but is on a per team basis
        total_games_first_turret_taken_by_team = ('challenges_firstTurretKilled', 'sum'), # Value of 1 or 0, states whether team took first turret or not
        avg_first_turret_kill_time_by_team = ('challenges_firstTurretKilledTime', 'mean'), # On a TEAM basis, not individual champion
            # Team kills
        avg_total_team_champ_kills = ('objectives_champion_kills', 'mean'),
        avg_team_aces_before_15_by_team = ('challenges_acesBefore15Minutes', 'mean'), # Data is from df_participants but is on a per team basis
        avg_flawless_aces_by_team = ('challenges_flawlessAces', 'mean'), # Data is from df_participants but is on a per team basis (make sure it indeed is on team basis)
        avg_shortes_time_to_ace_from_1st_tkdown = ('challenges_shortestTimeToAceFromFirstTakedown', 'mean'), # Data is from df_participants but am not sure if it is on a per team basis
        avg_max_kill_deficit = ('challenges_maxKillDeficit', 'mean'), # Data is from df_participants but is on a per team basis
        total_games_that_are_perfect_games = ('challenges_perfectGame', 'sum'), # Data is from df_participants but is on a per team basis
        
        # Individual participant damage to structures
            # Damage dealt to structures
        avg_indiv_dmg_dealt_to_buildings = ('damageDealtToBuildings', 'mean'),
        avg_indiv_dmg_dealth_to_turrets = ('damageDealtToTurrets', 'mean'),
        avg_indiv_turret_plates_taken = ('challenges_turretPlatesTaken', 'mean'),
            # First tower
        total_games_indiv_killed_1st_tower = ('firstTowerKill', 'sum'), # Boolean for champ who took first tower
        total_games_indiv_tkdown_1st_tower = ('challenges_takedownOnFirstTurret', 'sum'), # Value of 1 or 0, states whether participant had takedown on first turret 
        total_games_indiv_took_1st_tower_quick = ('challenges_quickFirstTurret', 'sum'), # Boolean value, we don't know what quick means here in terms of time
        total_games_indiv_had_1st_turret_assist = ('firstTowerAssist', 'sum'),
            # Turrets kills/takedowns
        avg_turrets_killed_before_plates_fell = ('challenges_kTurretsDestroyedBeforePlatesFall', 'mean'), # Seems to be on a per lane basis
        avg_indiv_tower_kills = ('turretKills', 'mean'),
        avg_indiv_tower_tkdowns = ('turretTakedowns', 'mean'),
        avg_indiv_tower_tkdowns2 = ('challenges_turretTakedowns', 'mean'), # Compare with above 
        avg_indiv_solo_towers_kills_late_game = ('challenges_soloTurretsLategame', 'mean'),
        avg_indiv_towers_taken_w_rift_herald = ('challenges_turretsTakenWithRiftHerald', 'mean'),
        avg_indiv_multi_towers_taken_w_rift_herald = ('challenges_multiTurretRiftHeraldCount', 'mean'),
            # Inhibitor and nexus kills/takedowns + misc
        avg_indiv_inhib_kills = ('inhibitorKills', 'mean'),
        avg_indiv_inhib_tkdowns = ('inhibitorTakedowns', 'mean'), # Takedown = kill or assist
        avg_indiv_nexus_kills = ('nexusKills', 'mean'), # Who last hit the nexus
        avg_indiv_nexus_tkdowns = ('nexusTakedowns', 'mean'),
        total_games_with_open_nexus = ('challenges_hadOpenNexus', 'mean'),
        
        # Individual participant objectives
            # Objective kills/takedowns
        avg_indiv_dmg_dealt_to_objs = ('damageDealtToObjectives', 'mean'),
        avg_indiv_baron_kills = ('baronKills', 'mean'),
        avg_indiv_solo_baron_kills = ('challenges_soloBaronKills', 'mean'),
        avg_indiv_baron_tkdowns = ('challenges_baronTakedowns', 'mean'),
        avg_indiv_dragon_kills = ('dragonKills', 'mean'),
        avg_indiv_dragon_tkdowns = ('challenges_dragonTakedowns', 'mean'),
        avg_indiv_rift_herald_tkdowns = ('challenges_riftHeraldTakedowns', 'mean'),
        avg_indiv_void_monster_kills = ('challenges_voidMonsterKill', 'mean'), # Void grubs? I have seen this have a valye of 7 when the champion did NOT have a rift herald takedown
            # Objective steals
        avg_objs_stolen = ('objectivesStolen', 'mean'),
        avg_objs_stolen_assists = ('objectivesStolenAssists', 'mean'),
        avg_epic_monster_steals = ('challenges_epicMonsterSteals', 'mean'), # Check if duplicate of the one above
        avg_epic_monster_steals_without_smite = ('challenges_epicMonsterStolenWithoutSmite', 'mean'),
        avg_epic_monsters_killed_near_enemy_jgler = ('challenges_epicMonsterKillsNearEnemyJungler', 'mean'),
            # Earliest dragon takedown stats (used for derived stats)
        avg_earliest_drag_tkd = ('challenges_earliestDragonTakedown', 'mean'),
        number_of_games_had_drag_tkd = ('had_dragon_takedown', 'sum'),
        number_of_games_had_drag_tkd_min_5to7 = ('sum_first_drag_tkd_min_5to7', 'sum'),
        number_of_games_had_drag_tkd_min_7to11 = ('sum_first_drag_tkd_min_7to11', 'sum'),
        number_of_games_had_drag_tkd_min_11to15 = ('sum_first_drag_tkd_min_11to15', 'sum'),
        number_of_games_had_drag_tkd_min_15plus = ('sum_first_drag_tkd_min_15+', 'sum'),
   
        # Game length related
        avg_time_played_per_game = ('timePlayed', 'mean'),
        avg_game_length = ('challenges_gameLength', 'mean'), # Compare with above and delete one
        total_games_ended_in_early_ff = ('gameEndedInEarlySurrender', 'mean'), # Probably needs to be a derived stat, might need count first (might work with mean)
        total_games_ended_in_ff = ('gameEndedInSurrender', 'mean'), # Probably needs to be a derived stat, might need count first
        total_games_team_ffd = ('teamEarlySurrendered', 'mean'),
        
        # Multikill and killing spree stats
            # Multikills
        avg_doublekills = ('doubleKills', 'mean'),
        avg_triplekills = ('tripleKills', 'mean'),
        avg_quadrakills = ('quadraKills', 'mean'),
        avg_pentakills = ('pentaKills', 'mean'),
        avg_largest_multikill = ('largestMultiKill', 'mean'),
        avg_number_of_multikills = ('challenges_multikills', 'mean'), # Number of multikills, probably counting how often a champion kills 2+ champs
        avg_multikills_with_one_spell = ('challenges_multiKillOneSpell', 'mean'),
            # Killing sprees stats
        avg_killing_sprees = ('killingSprees', 'mean'),
        avg_killing_sprees2 = ('challenges_killingSprees', 'mean'), # Check if equal and delete one
        avg_legendary_count = ('challenges_legendaryCount', 'mean'), # Not too sure what this is, my current guess is how many times champ was legendary (do they need to die and get leg again or it counts kills above leg?)
        avg_largest_killing_spee = ('largestKillingSpree', 'mean'),
            # Misc
        avg_unreal_kills = ('unrealKills', 'mean'), # Might be a zero stat, need to check
        avg_12_assist_streaks = ('challenges_12AssistStreakCount', 'mean'),
        avg_elder_drag_multikills = ('challenges_elderDragonMultikills', 'mean'),
        avg_full_team_tkds = ('challenges_fullTeamTakedown', 'mean'),

        # Items - decide whether to use data from match or from timeline and make sure to indluce item tags in df 

        # Misc
        avg_times_blast_cone_enemy = ('challenges_blastConeOppositeOpponentCount', 'mean'),
        total_games_danced_with_rift_herald = ('challenges_dancedWithRiftHerald', 'mean'),
        avg_double_aces = ('challenges_doubleAces', 'mean'),
        avg_fist_bump_participations = ('challenges_fistBumpParticipation', 'mean'),
        avg_mejai_full_stack_time = ('challenges_mejaisFullStackInTime', 'mean'), # need to exclude zero values as it gives zero when they didn't have mejais
        avg_outer_turret_executes_before_10m = ('challenges_outerTurretExecutesBefore10Minutes', 'mean'),
        avg_tkds_in_enemy_fountain = ('challenges_takedownsInEnemyFountain', 'mean'),
        
        # Position/Role - find aggregation key word for most common string
        mode_individual_position = ('individualPosition', lambda x: x.mode()[0] if not x.mode().empty else None), # Best guess for which position the player actually played in isolation of anything else
        mode_lane = ('lane', lambda x: x.mode()[0] if not x.mode().empty else None), # Gives slightly different string than above, might have something to do with where champ spent most of the time
        mode_role = ('role', lambda x: x.mode()[0] if not x.mode().empty else None),
        mode_team_position = ('teamPosition', lambda x: x.mode()[0] if not x.mode().empty else None), # The teamPosition is the best guess for which position the player actually played if we add the constraint that each team must have one top player, one jungle, one middle, etc
        total_games_played_champ_select_position = ('challenges_playedChampSelectPosition', 'mean')

    ).round(2)
    
    # Calculate derived stats
    champion_stats['kda'] = ((champion_stats['avg_kills'] + champion_stats['avg_assists']) / 
                            champion_stats['avg_deaths']).round(2)
    champion_stats['winrate'] = ((champion_stats['total_wins'] ) / 
                            champion_stats['total_games_played_in_role']).round(2)
    champion_stats['cs_per_minute'] = ((champion_stats['avg_cs'] + champion_stats['avg_neutral_monsters_cs']) / 
                                      (champion_stats['avg_time_played_per_game'] / 60)).round(2)
    champion_stats['gold_per_minute'] = (champion_stats['avg_gold_earned_per_game'] / 
                                        (champion_stats['avg_time_played_per_game'] / 60)).round(2)
    champion_stats['damage_per_minute'] = (champion_stats['avg_dmg_dealt_to_champions'] / 
                                          (champion_stats['avg_time_played_per_game'] / 60)).round(2)
    # Derived statistics related to avg first dragon tkdown timing
    champion_stats['pct_first_drag_tkd_min_5to7'] = (((champion_stats['number_of_games_had_drag_tkd_min_5to7']) / 
                                                      champion_stats['number_of_games_had_drag_tkd']) * 100).round(2)
    champion_stats['pct_first_drag_tkd_min_7to11'] = (((champion_stats['number_of_games_had_drag_tkd_min_7to11']) / 
                                                      champion_stats['number_of_games_had_drag_tkd']) * 100).round(2)
    champion_stats['pct_first_drag_tkd_min_11to15'] = (((champion_stats['number_of_games_had_drag_tkd_min_11to15']) / 
                                                      champion_stats['number_of_games_had_drag_tkd']) * 100).round(2)
    champion_stats['pct_first_drag_tkd_min_15+'] = (((champion_stats['number_of_games_had_drag_tkd_min_15plus']) / 
                                                      champion_stats['number_of_games_had_drag_tkd']) * 100).round(2)
    # Percentage of games where X - derived statistics
    champion_stats['pct_highest_dmg_in_match'] = (((champion_stats['total_games_with_highest_damage']) / 
                                                      champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_highest_cc_in_match'] = (((champion_stats['total_games_with_highest_cc_score']) / 
                                                      champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_highest_ward_kills_in_match'] = (((champion_stats['total_games_with_highest_wards_killed']) / 
                                                      champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_fastest_supp_quest_in_match'] = (((champion_stats['total_games_first_supp_quest']) / 
                                                      champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_first_to_buy_item_in_match'] = (((champion_stats['total_games_fastest_item_completion']) / 
                                                      champion_stats['total_games_played_in_role']) * 100).round(2)
    # Derived stats for early lane, kills
    champion_stats['pct_games_first_blood_kill'] = (((champion_stats['total_games_first_blood_kill']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_games_first_blood_assist'] = (((champion_stats['total_games_first_blood_assist']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_with_early_lanephase_gold_exp_adv'] = (((champion_stats['total_games_with_early_lanephase_gold_exp_adv']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_with_lanephase_gold_exp_adv'] = (((champion_stats['total_games_with_lanephase_gold_exp_adv']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_games_jgler_had_early_tkdowns_in_all_lanes'] = (((champion_stats['total_games_jgler_had_early_tkdowns_in_all_lanes']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    # Derived stats related to early objs and misc
    champion_stats['pct_of_games_team_took_first_baron'] = (((champion_stats['total_games_team_took_first_baron']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_team_took_first_drag'] = (((champion_stats['total_games_team_took_first_drag']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_team_took_first_inhib'] = (((champion_stats['total_games_team_took_first_inhib']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_games_team_took_first_herald'] = (((champion_stats['total_games_team_took_first_herald']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_team_took_first_turret'] = (((champion_stats['total_games_team_took_first_turret']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_with_perfect_drag_soul_taken'] = (((champion_stats['total_games_with_perfect_drag_soul_taken']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_first_turret_taken_by_team'] = (((champion_stats['total_games_first_turret_taken_by_team']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_that_are_perfect_games'] = (((champion_stats['total_games_that_are_perfect_games']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_indiv_killed_1st_tower'] = (((champion_stats['total_games_indiv_killed_1st_tower']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_indiv_tkdown_1st_tower'] = (((champion_stats['total_games_indiv_tkdown_1st_tower']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_indiv_had_1st_turret_assist'] = (((champion_stats['total_games_indiv_had_1st_turret_assist']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_indiv_took_1st_tower_quick'] = (((champion_stats['total_games_indiv_took_1st_tower_quick']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_with_open_nexus'] = (((champion_stats['total_games_with_open_nexus']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_ended_in_early_ff'] = (((champion_stats['total_games_ended_in_early_ff']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_ended_in_ff'] = (((champion_stats['total_games_ended_in_ff']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_team_ffd'] = (((champion_stats['total_games_team_ffd']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_danced_with_rift_herald'] = (((champion_stats['total_games_danced_with_rift_herald']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    champion_stats['pct_of_games_played_champ_select_position'] = (((champion_stats['total_games_played_champ_select_position']) / 
                                                   champion_stats['total_games_played_in_role']) * 100).round(2)
    # Item tags stats
    champion_stats['pct_items_abilityhaste_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('AbilityHaste')) / 
                                                    champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_spellblock_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('SpellBlock')) / 
                                                  champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_armor_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('Armor')) / 
                                             champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_criticalstrike_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('CriticalStrike')) / 
                                                      champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_lifesteal_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('LifeSteal')) / 
                                                 champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_nonbootsmovement_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('NonbootsMovement')) / 
                                                        champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_tenacity_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('Tenacity')) / 
                                                champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_armorpenetration_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('ArmorPenetration')) / 
                                                        champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_healthregen_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('HealthRegen')) / 
                                                   champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_aura_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('Aura')) / 
                                            champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_attackspeed_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('AttackSpeed')) / 
                                                   champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_goldper_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('GoldPer')) / 
                                               champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_vision_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('Vision')) / 
                                              champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_jungle_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('Jungle')) / 
                                              champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_armor_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('Armor')) / 
                                             champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_healthregen_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('HealthRegen')) / 
                                                   champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_active_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('Active')) / 
                                              champion_stats['completed_items'] * 100).round(2)
    champion_stats['pct_items_nonbootsmovement_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('NonbootsMovement')) / 
                                                        champion_stats['completed_items'] * 100).round(2)
    # Summoner spell stats
    champion_stats['pct_games_w_barrier'] = (champion_stats['summspells_per_game'].apply(lambda tags: tags.count('barrier')) / 
                                              champion_stats['total_games_played_in_role'] * 100).round(2)
    champion_stats['pct_games_w_cleanse'] = (champion_stats['summspells_per_game'].apply(lambda tags: tags.count('cleanse')) / 
                                              champion_stats['total_games_played_in_role'] * 100).round(2)
    champion_stats['pct_games_w_exhaust'] = (champion_stats['summspells_per_game'].apply(lambda tags: tags.count('exhaust')) / 
                                              champion_stats['total_games_played_in_role'] * 100).round(2)
    champion_stats['pct_games_w_flash'] = (champion_stats['summspells_per_game'].apply(lambda tags: tags.count('flash')) / 
                                              champion_stats['total_games_played_in_role'] * 100).round(2)
    champion_stats['pct_games_w_ghost'] = (champion_stats['summspells_per_game'].apply(lambda tags: tags.count('ghost')) / 
                                              champion_stats['total_games_played_in_role'] * 100).round(2)
    champion_stats['pct_games_w_heal'] = (champion_stats['summspells_per_game'].apply(lambda tags: tags.count('heal')) / 
                                              champion_stats['total_games_played_in_role'] * 100).round(2)
    champion_stats['pct_games_w_ignite'] = (champion_stats['summspells_per_game'].apply(lambda tags: tags.count('ignite')) / 
                                              champion_stats['total_games_played_in_role'] * 100).round(2)
    champion_stats['pct_games_w_smite'] = (champion_stats['summspells_per_game'].apply(lambda tags: tags.count('smite')) / 
                                              champion_stats['total_games_played_in_role'] * 100).round(2)
    champion_stats['pct_games_w_teleport'] = (champion_stats['summspells_per_game'].apply(lambda tags: tags.count('teleport')) / 
                                              champion_stats['total_games_played_in_role'] * 100).round(2)
    

    champion_stats = champion_stats.drop([
        'number_of_games_had_drag_tkd_min_5to7', 'number_of_games_had_drag_tkd_min_7to11', 'number_of_games_had_drag_tkd_min_11to15', 'number_of_games_had_drag_tkd_min_15plus',
        'total_games_with_highest_damage', 'total_games_with_highest_cc_score', 'total_games_with_highest_wards_killed', 'total_games_jgler_had_early_tkdowns_in_all_lanes', 
        'total_games_with_lanephase_gold_exp_adv', 'total_games_with_early_lanephase_gold_exp_adv',

        'total_games_team_took_first_baron', 'total_games_team_took_first_drag', 'total_games_team_took_first_inhib', 'total_games_team_took_first_herald', 'total_games_team_took_first_turret',

        'total_games_first_supp_quest', 'total_games_with_perfect_drag_soul_taken', 'total_games_first_turret_taken_by_team', 'total_games_that_are_perfect_games',

        'total_games_indiv_killed_1st_tower', 'total_games_indiv_tkdown_1st_tower', 'total_games_indiv_had_1st_turret_assist', 'total_games_indiv_took_1st_tower_quick',

        'total_games_with_open_nexus', 'total_games_ended_in_early_ff', 'total_games_ended_in_ff', 'total_games_team_ffd', 'total_games_danced_with_rift_herald', 'total_games_played_champ_select_position'
    ], axis = 1)


    return champion_stats, filtered_df

# Create and display the aggregated data
champion_stats, filtered_df = aggregate_champion_data(df_participants, df_teams)
# display(champion_stats.sort_values('challenges_earliestDragonTakedown', ascending=False))
champion_stats.to_csv("champion_stats.csv", index = True)
filtered_df.to_csv("filtered_df.csv", index = True)
total_games = champion_stats['total_games_played_in_role'].sum()


print(total_games)
