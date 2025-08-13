from dotenv import load_dotenv

import numpy as np
import pandas as pd
import os, boto3, io
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timezone

load_dotenv()
BUCKET = os.getenv("BUCKET")
PROCESSED_DATA_FOLDER = os.getenv("PROCESSED_DATA_FOLDER")
PATCH = "patch_15_6"

labels = ["team_position", "champion_name"]

item_tags =[
    "pct_of_matches_with_magic_resist",
    "pct_of_matches_with_jungle",
    "pct_of_matches_with_tenacity",
    "pct_of_matches_with_lane",
    "pct_of_matches_with_spell_block",
    "pct_of_matches_with_armor_penetration",
    "pct_of_matches_with_life_steal",
    "pct_of_matches_with_health",
    "pct_of_matches_with_attack_speed",
    "pct_of_matches_with_critical_strike",
    "pct_of_matches_with_ability_haste",
    "pct_of_matches_with_on_hit",
    "pct_of_matches_with_damage",
    "pct_of_matches_with_slow",
    "pct_of_matches_with_spell_damage",
    "pct_of_matches_with_spell_vamp",
    "pct_of_matches_with_aura",
    "pct_of_matches_with_magic_penetration",
    "pct_of_matches_with_cooldown_reduction",
    "pct_of_matches_with_armor",
    "pct_of_matches_with_health_regen",
    "pct_of_matches_with_mana",
    "pct_of_matches_with_nonboots_movement"
]

jungle_specific_features = ["pct_of_matches_with_jungle", "avg_kills_early_jungle_as_jungler",
                            "avg_early_kills_on_laners_as_jungler", "avg_enemy_jungle_cs_differential_early",
                            "avg_buffs_stolen", "avg_jungle_monsters_cs"]
## Excluded from jg
laners_specific_features = ["avg_kills_on_other_lanes_early_as_laner", "avg_times_had_early_takedowns_in_all_lanes_as_laner",
                            "pct_of_games_indiv_killed_1st_tower", "pct_of_games_team_took_first_dragon", "pct_of_games_had_drag_takedown",
                            "pct_of_matches_with_mana", "avg_vision_score_advantage_over_lane_opponent",]

top_exclude_features = [
    "pct_of_matches_with_magic_penetration", "percent_of_games_with_fully_stacked_mejais", "pct_of_matches_with_lane", 
    "pct_of_matches_with_critical_strike", "pct_of_games_team_took_first_dragon", "pct_of_matches_with_mana", "pct_of_games_team_took_first_dragon",
    "avg_multikills_with_one_spell", "pct_of_matches_with_life_steal", "avg_initial_crab_count", "avg_epic_monster_kills_within_30s_of_spawn",
    "pct_of_matches_with_spell_damage", "pct_of_matches_with_damage", "avg_quadrakills", "avg_spell4_casts",
    "avg_takedowns_after_gaining_lvl_advantage", "avg_times_had_early_takedowns_in_all_lanes_as_laner", "avg_times_knock_enemy_into_team_and_kill",
    "avg_takedowns_in_alcove", "avg_individual_rift_herald_takedowns", "pct_of_games_had_drag_takedown", "avg_first_turret_kill_time_by_team",
    "avg_max_cs_lead_over_lane_opponent", "avg_enemy_jungle_minions_killed", "avg_champ_exp_at_game_end", "avg_elder_dragon_kills_w_opposing_soul",
    "pct_of_games_first_blood_assist", "pct_magic_damage", "pct_physical_damage", "avg_crabs_per_game", "avg_individual_dragon_takedowns", "popularity_in_role",
    "avg_minions_killed_by_10_mins", "avg_ward_takedowns_before_20m"
    

] + jungle_specific_features + item_tags

mid_exclude_features = [
    "pct_of_matches_with_magic_penetration", "percent_of_games_with_fully_stacked_mejais", "pct_of_matches_with_lane", 
    "pct_of_matches_with_critical_strike", "pct_of_games_team_took_first_dragon", "pct_of_matches_with_mana", "pct_of_games_team_took_first_dragon",
    "avg_multikills_with_one_spell", "pct_of_matches_with_life_steal", "avg_initial_crab_count", "avg_epic_monster_kills_within_30s_of_spawn",
    "pct_of_matches_with_spell_damage", "pct_of_matches_with_damage", "avg_quadrakills", "avg_spell4_casts",
    "avg_takedowns_after_gaining_lvl_advantage", "avg_times_had_early_takedowns_in_all_lanes_as_laner", "avg_times_knock_enemy_into_team_and_kill",
    "avg_takedowns_in_alcove", "avg_individual_rift_herald_takedowns", "pct_of_games_had_drag_takedown", "avg_first_turret_kill_time_by_team",
    "avg_max_cs_lead_over_lane_opponent", "avg_enemy_jungle_minions_killed", "avg_champ_exp_at_game_end", "avg_elder_dragon_kills_w_opposing_soul",
    "pct_of_games_first_blood_assist", "pct_magic_damage", "pct_physical_damage", "avg_crabs_per_game", "avg_individual_dragon_takedowns", "popularity_in_role",
    "avg_minions_killed_by_10_mins", "avg_ward_takedowns_before_20m"
    

] + jungle_specific_features + item_tags

bot_exclude_features = [
    "pct_of_matches_with_magic_penetration", "percent_of_games_with_fully_stacked_mejais", "pct_of_matches_with_lane", 
    "pct_of_matches_with_critical_strike", "pct_of_games_team_took_first_dragon", "pct_of_matches_with_mana", "pct_of_games_team_took_first_dragon",
    "avg_multikills_with_one_spell", "pct_of_matches_with_life_steal", "avg_initial_crab_count", "avg_epic_monster_kills_within_30s_of_spawn",
    "pct_of_matches_with_spell_damage", "pct_of_matches_with_damage", "avg_quadrakills", "avg_spell4_casts",
    "avg_takedowns_after_gaining_lvl_advantage", "avg_times_had_early_takedowns_in_all_lanes_as_laner", "avg_times_knock_enemy_into_team_and_kill",
    "avg_takedowns_in_alcove", "avg_individual_rift_herald_takedowns", "pct_of_games_had_drag_takedown", "avg_first_turret_kill_time_by_team",
    "avg_max_cs_lead_over_lane_opponent", "avg_enemy_jungle_minions_killed", "avg_champ_exp_at_game_end", "avg_elder_dragon_kills_w_opposing_soul",
    "pct_of_games_first_blood_assist", "pct_magic_damage", "pct_physical_damage", "avg_crabs_per_game", "avg_individual_dragon_takedowns", "popularity_in_role",
    "avg_minions_killed_by_10_mins", "avg_ward_takedowns_before_20m", "avg_kills_on_other_lanes_early_as_laner", 'avg_times_pick_kill_with_ally',
    'pct_of_matches_with_ghost', 'pct_of_matches_with_ignite', 'avg_total_healing'
    

] + jungle_specific_features + item_tags

sup_exclude_features = [
    "pct_of_matches_with_magic_penetration", "percent_of_games_with_fully_stacked_mejais", "pct_of_matches_with_lane", 
    "pct_of_matches_with_critical_strike", "pct_of_matches_with_mana",
    "avg_multikills_with_one_spell", "pct_of_matches_with_life_steal", "avg_initial_crab_count", "avg_epic_monster_kills_within_30s_of_spawn",
    "pct_of_matches_with_spell_damage", "pct_of_matches_with_damage", "avg_quadrakills", "avg_spell4_casts",
     "avg_times_had_early_takedowns_in_all_lanes_as_laner",
    "avg_takedowns_in_alcove", "avg_individual_rift_herald_takedowns", "pct_of_games_had_drag_takedown", "avg_first_turret_kill_time_by_team",
    "avg_max_cs_lead_over_lane_opponent", "avg_enemy_jungle_minions_killed", "avg_champ_exp_at_game_end", "avg_elder_dragon_kills_w_opposing_soul",
    "pct_of_games_first_blood_assist", "pct_magic_damage", "pct_physical_damage", "avg_crabs_per_game", "avg_individual_dragon_takedowns",
    "avg_minions_killed_by_10_mins", "avg_initial_buff_count", 'pct_of_matches_with_teleport', 'avg_individual_void_monster_kills', 'pct_of_games_indiv_killed_1st_tower',
    'total_games_fastest_item_completion', 'avg_ward_takedowns_before_20m', 'avg_max_level_lead_over_lane_opp', 'avg_skillshots_hit', 'avg_number_of_items_purchased',
    'avg_individual_solo_towers_kills_late_game', 'avg_ability_uses', 'avg_individual_inhibitor_kills', 'avg_indiv_dmg_dealt_to_buildings', 'pct_of_matches_with_ghost',
    'avg_indiv_turret_plates_taken'
    

] + jungle_specific_features + item_tags



raw_clustering_features = [
    # Core 
    "avg_kills",
    "avg_deaths",
    "avg_assists",
    "avg_kill_participation",
    # Damage dealt
    "average_damage_per_minute",
    "avg_damage_dealt_to_champions",
    # Damage taken
    "avg_damage_self_mitigated",
    "avg_times_survived_three_immobilizes_in_fight",
    # CC
    "avg_time_ccing_others",
    "avg_times_applied_cc_on_others",
    "avg_enemy_champion_immobilizations",
    # Healing and shielding + Protecting
    "avg_total_healing",
    # Ability casts + Skillshots
    "avg_spell4_casts",
    "avg_ability_uses",
    "avg_skillshots_dodged",
    "avg_skillshots_landed_early_game",
    "avg_skillshots_hit",
    # Picks
    "avg_times_immobilize_and_kill_with_ally",
    "avg_times_pick_kill_with_ally",
    "avg_times_knock_enemy_into_team_and_kill",
    # Kill types
    "avg_kills_near_enemy_turret",
    "avg_outnumbered_kills",
    "avg_quick_solo_kills",
    "avg_solo_kills",
    # Laning kills
    "avg_takedowns_after_gaining_lvl_advantage",
    "avg_takedowns_in_alcove",
    "avg_first_takedown_time",
    # Summoner spells
    "pct_of_matches_with_ghost",
    "pct_of_matches_with_teleport",
    "pct_of_matches_with_ignite",
    # Gold and XP
    "avg_champ_exp_at_game_end",
    "avg_gold_earned_per_game",
    "avg_gold_per_minute",
    "pct_of_games_with_early_lane_phase_gold_exp_adv", # End the early laning phase (7 minutes) with 20% more gold and experience than your role opponent on Summoner's Rift
    "pct_of_games_with_lanephase_gold_exp_adv", # End the laning phase (14 minutes) with 20% more gold and experience than your role opponent 
    "avg_max_level_lead_over_lane_opp",
    # CS
    "avg_minions_killed",
    "avg_minions_killed_by_10_mins",
    "avg_cs_per_minute",
    "avg_max_cs_lead_over_lane_opponent",
    # Items
    "avg_number_of_items_purchased",
    "total_games_fastest_item_completion",
    # Jungle CS
    "avg_enemy_jungle_minions_killed",
    "avg_jungle_monsters_cs",
    "avg_buffs_stolen",
    "avg_initial_buff_count",
    "avg_epic_monster_kills_within_30s_of_spawn",
    "avg_initial_crab_count",
    "avg_crabs_per_game",
    "avg_enemy_jungle_cs_differential_early",
    # Vision Score
    "avg_vision_score_advantage_over_lane_opponent",
    "avg_ward_takedowns_before_20m",
    # Objs
    "pct_of_games_team_took_first_dragon",
    "avg_individual_rift_herald_takedowns",
    "avg_individual_dragon_takedowns",
    "avg_elder_dragon_kills_w_opposing_soul",
    "avg_indiv_dmg_dealt_to_buildings",
    "avg_indiv_turret_plates_taken",
    "pct_of_games_indiv_killed_1st_tower",
    "avg_individual_solo_towers_kills_late_game",
    "avg_individual_inhibitor_kills",
    "avg_individual_void_monster_kills",
    "pct_of_games_had_drag_takedown",
    # Game length
    "avg_time_played_per_game_minutes",
    # Multikills + Killing sprees
    "pct_of_games_first_blood_kill",
    "pct_of_games_first_blood_assist",
    "avg_number_of_multikills",
    "avg_quadrakills",
    "avg_multikills_with_one_spell",
    "avg_killing_sprees",
    # Mejais
    "percent_of_games_with_fully_stacked_mejais", ## new
    # Item tags
    "pct_of_matches_with_magic_resist",
    "pct_of_matches_with_jungle",
    "pct_of_matches_with_tenacity",
    "pct_of_matches_with_lane",
    "pct_of_matches_with_spell_block",
    "pct_of_matches_with_armor_penetration",
    "pct_of_matches_with_life_steal",
    "pct_of_matches_with_health",
    "pct_of_matches_with_attack_speed",
    "pct_of_matches_with_critical_strike",
    "pct_of_matches_with_ability_haste",
    "pct_of_matches_with_on_hit",
    "pct_of_matches_with_damage",
    "pct_of_matches_with_slow",
    "pct_of_matches_with_spell_damage",
    "pct_of_matches_with_spell_vamp",
    "pct_of_matches_with_aura",
    "pct_of_matches_with_magic_penetration",
    "pct_of_matches_with_cooldown_reduction",
    "pct_of_matches_with_armor",
    "pct_of_matches_with_health_regen",
    "pct_of_matches_with_mana",
    "pct_of_matches_with_nonboots_movement"
]


derived_clustering_features = [
    "popularity_in_role", 
    "role_popularity_for_champion",
    "pct_magic_damage",
    "pct_physical_damage",
    "pct_true_damage",
    "avg_damage_taken_per_death",
    "avg_effective_heal_and_shield",
    "avg_kills_on_other_lanes_early_as_laner",
    "avg_kills_early_jungle_as_jungler",
    "avg_early_kills_on_laners_as_jungler",
    "avg_times_had_early_takedowns_in_all_lanes_as_laner",
    "avg_control_ward_time_coverage_in_river_or_enemy_half",
    "avg_first_turret_kill_time_by_team",
    "avg_individual_tower_assists"
]


# Define role mappings
ROLE_CONFIG = {
    "JUNGLE": {
        "exclude_features": laners_specific_features,
        "number_of_clusters": 8
    },
    "TOP": {
        "exclude_features": top_exclude_features,
        "number_of_clusters": 9
    },
    "MIDDLE": {
        "exclude_features": mid_exclude_features,
        "number_of_clusters": 9
    },
    "BOTTOM": {
        "exclude_features": bot_exclude_features,
        "number_of_clusters": 6
    },
    "UTILITY": {
        "exclude_features": sup_exclude_features,
        "number_of_clusters": 5
    }
}


def get_processed_dataframe() -> pd.DataFrame:

    key = f"{PROCESSED_DATA_FOLDER}/champion_x_role/{PATCH}/champion_x_role_aggregated_data.csv"

    # Pull the object
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=key)

    # Read it straight into pandas
    champion_x_role_df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    return champion_x_role_df


def additional_feature_engineering(champion_x_role_df: pd.DataFrame) -> pd.DataFrame:
    df = champion_x_role_df
    # Creating features
    df["total_role_games"] = df.groupby("team_position")["total_games_played_in_role"].transform("sum")
    # Popularity metrics
    df["popularity_in_role"] = df["total_games_played_in_role"]/df["total_role_games"]
    df["role_popularity_for_champion"] = df["total_games_played_in_role"]/df["total_games_per_champion"]
    # Damage done metrics
    df["pct_magic_damage"] = df["avg_magic_damage_dealt_to_champions"]/df["avg_damage_dealt_to_champions"]
    df["pct_physical_damage"] = df["avg_physical_damage_dealt_to_champions"]/df["avg_damage_dealt_to_champions"]
    df["pct_true_damage"] = df["avg_true_damage_dealt_to_champions"]/df["avg_damage_dealt_to_champions"]
    # Damage taken metrics
    df["avg_damage_taken_per_death"] = df["avg_damage_taken"]/df["avg_deaths"]
    # JUNGLE (REVISIT UPSTREAM)
    df["avg_kills_on_other_lanes_early_as_laner"] = np.where( 
        df["team_position"] != "JUNGLE", 
        df["avg_kills_on_other_lanes_early_as_laner"], # As a laner, in a single game, get kills before 10 minutes outside your lane (anyone but your lane opponent)
        pd.NA
    )
    df["avg_times_had_early_takedowns_in_all_lanes_as_laner"] = np.where( 
        df["team_position"] != "JUNGLE", 
        df["avg_times_had_early_takedowns_in_all_lanes_as_laner"], # As a laner, in a single game, get kills before 10 minutes outside your lane (anyone but your lane opponent)
        pd.NA
    )
    df["avg_kills_early_jungle_as_jungler"] = np.where(
        df["team_position"] == "JUNGLE", 
        df["avg_jungler_kills_early_jungle"], # As jungler, get kills on the enemy jungler in their own jungle before 10 minutes
        pd.NA
    )
    df["avg_early_kills_on_laners_as_jungler"] = np.where(
        df["team_position"] == "JUNGLE", 
        df["avg_jungler_early_kills_on_laners"], # As jungler, get kills on top lane, mid lane, bot lane, or support players before 10 minutes
        pd.NA
    )
    df["pct_of_matches_with_jungle"] = np.where(
        df["team_position"] == "JUNGLE", 
        df["pct_of_matches_with_jungle"], # As jungler, get kills on top lane, mid lane, bot lane, or support players before 10 minutes
        pd.NA
    )
    df["avg_enemy_jungle_cs_differential_early"] = np.where(
        df["team_position"] == "JUNGLE", 
        df["avg_enemy_jungle_cs_differential_early"], # As jungler, get kills on top lane, mid lane, bot lane, or support players before 10 minutes
        pd.NA
    )
    df["avg_supp_quest_completion_time"] = np.where(
        df["team_position"] == "UTILITY", 
        df["avg_supp_quest_completion_time"], # As a support, average support quest completion time
        pd.NA
    )
    # Structures
    df["avg_individual_tower_assists"] = df["avg_individual_tower_takedowns"] - df["avg_individual_tower_kills"]

    return df


def filter_role_data(
    df: pd.DataFrame,
    role: str,
    labels: list[str],
    raw_clustering_features: list[str],
    derived_clustering_features: list[str],
    exclude_features: list[str],
    min_popularity: float = 0.1
) -> pd.DataFrame:
    """
    1. Select only label + clustering features, minus any excluded.
    2. Filter to the given role.
    3. Filter out low-popularity champions.
    4. Report & fill any missing values.
    """
    # 1) Build the “keep” list, ensuring we never drop the two needed filters
    required = {"team_position", "role_popularity_for_champion"}
    all_feats = set(labels) | set(raw_clustering_features) | set(derived_clustering_features) | required
    keep_cols = [
        col for col in all_feats
        if col in df.columns and col not in exclude_features
    ]
    
    # 2) Subset & filter by role
    role_df = df.loc[df["team_position"] == role, keep_cols].copy()
    
    # 3) Filter by popularity
    role_df = role_df.loc[
        role_df["role_popularity_for_champion"] > min_popularity
    ].copy()
    
    # 4) Check for missing values and fill them
    missing = role_df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        print(f"\n[filter_role_data] {role} — missing values:\n{missing}\n")
    role_df.fillna(0, inplace=True)
    
    return role_df


def cluster_and_vectors(df: pd.DataFrame, n_clusters: int = 8, random_state: int = 42):
    """
    Runs KMeans on all non-label columns, assigns clusters,
    then prints out cluster memberships and:
      - overall feature importance (range across centroids)
      - per-cluster drivers (deviation from global centroid)
    """
    # Make a clean copy and drop any existing cluster column
    df = df.copy()
    df.drop(columns=["cluster"], errors="ignore", inplace=True)

    # 0) Identify feature columns (exclude labels only)
    label_cols   = ["champion_name", "team_position"]
    feature_cols = [c for c in df.columns if c not in label_cols]

    # 1) scale + cluster
    scaler   = StandardScaler()
    X        = df[feature_cols].to_numpy()
    X_scaled = scaler.fit_transform(X)

    km       = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels   = km.fit_predict(X_scaled)

    # 2) champion/member vectors (scaled)
    member_vec_df = pd.DataFrame(X_scaled, columns=feature_cols)
    member_vec_df.insert(0, "cluster", labels)
    member_vec_df.index = pd.MultiIndex.from_frame(df[label_cols])  # champion + role index

    # 3) cluster centroid vectors (scaled)
    centroids_scaled = km.cluster_centers_                             # shape (k, n_features)
    cluster_vec_df   = pd.DataFrame(centroids_scaled, columns=feature_cols)
    cluster_vec_df.index.name = "cluster"

    # 4) residual vector & distance (scaled)
    assigned_centers = centroids_scaled[labels]                        # (n_samples, n_features)
    residuals        = X_scaled - assigned_centers                     # member minus its centroid
    distances        = np.linalg.norm(residuals, axis=1)               # Euclidean distance

    # Attach residual & distance to member table
    member_vec_df.insert(1, "euclidean_distance_to_centroid", distances)
    # Keep the full residual vector per member as a list; round to shrink size
    member_vec_df.insert(2, "residual_vec_scaled", [np.round(r, 4).tolist() for r in residuals])

    # Return scaler to convert back into original units later for LLM context if needed
    return member_vec_df, cluster_vec_df, scaler


def save_latest_s3fs(member_df, cluster_df, role):
    prefix = f"s3://{BUCKET}/{PROCESSED_DATA_FOLDER}/clusters/{PATCH}"
    member_df.to_csv(f"{prefix}/{role.lower()}_member_vectors.csv", index=True)
    cluster_df.to_csv(f"{prefix}/{role.lower()}_cluster_vectors.csv", index=True)


def main():
    
    champion_x_role_df = get_processed_dataframe()
    champion_x_role_df = additional_feature_engineering(champion_x_role_df)

    for role, config in ROLE_CONFIG.items():
        filtered_role_df = filter_role_data(champion_x_role_df, role, labels, raw_clustering_features, derived_clustering_features, config['exclude_features'])
        member_vec_df, cluster_vec_df, scaler = cluster_and_vectors(filtered_role_df, config["number_of_clusters"])
        save_latest_s3fs(member_vec_df, cluster_vec_df, role)

if __name__ == "__main__":
    main()