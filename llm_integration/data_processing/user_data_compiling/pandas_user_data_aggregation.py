import warnings
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np

RANKED_SOLO_DUO_QUEUE_ID = 420
DEFAULT_PARTITIONS = 100
PATCH_ERROR_PATTERN = r"Patch Error: Patch \d+(\.\d+)?"

# Column and field name constants
MATCH_DATA = "match_data"
MATCH_ID = "match_id"
TEAM_POSITION = "team_position"

ALL_TEAM_POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

POSITION_SPECIFIC_METRICS = {
    "JUNGLE": {
        "jungler_kills_early_jungle": "challenges.jungler_kills_early_jungle", # As a jungler, get kills on the enemy jungler in their own jungle before 10 minutes
        "kills_on_laners_early_jungle_as_jungler": "challenges.kills_on_laners_early_jungle_as_jungler", # As a jungler, get kills on top lane, mid lane, bot lane, or support players before 10 minutes
        "more_enemy_jungle_cs_than_opponent_as_jungler": "challenges.more_enemy_jungle_than_opponent" # As a jungler, before 10 minutes, take more of the opponent's jungle than them
    },
    "UTILITY": {
        "complete_support_quest_in_time": "challenges.complete_support_quest_in_time"
    }
}

LANER_SPECIFIC_METRICS = {
    "kills_on_other_lanes_early_as_laner": "challenges.kills_on_other_lanes_early_jungle_as_laner", # As a laner, in a single game, get kills before 10 minutes outside your lane (anyone but your lane opponent)
    "takedowns_in_all_lanes_early_as_laner": "challenges.get_takedowns_in_all_lanes_early_jungle_as_laner" # As a laner, get a takedown in all three lanes within 10 minutes
} 


# Add additional role-specific metrics below if needed
TOP_METRICS = {} 
MID_METRICS = {}
BOT_METRICS = {}

class DragonTimings:
    """Dragon timing thresholds in seconds."""
    SPAWN_TIME = 300  # 5:00 - Dragons first spawn
    EARLY_WINDOW_END = 420  # 7:00
    MID_WINDOW_END = 660  # 11:00
    LATE_WINDOW_END = 900  # 15:00
    
    # Window definitions
    EARLY_WINDOW = (301, 420)  # 5:01-7:00
    MID_WINDOW = (421, 660)    # 7:01-11:00
    LATE_WINDOW = (661, 900)   # 11:01-15:00

class InsufficientSampleError(Exception):
    def __init__(self, sample_type: str):
        self.sample_type = sample_type
        message = f"Insufficient sample of {sample_type}"
        super().__init__(message)


SUMMONER_SPELLS_DICT = {
    "1" : "cleanse",
    "3" : "exhaust",
    "4" : "flash",
    "6" : "ghost", 
    "7" : "heal",
    "11" : "smite", 
    "12" : "teleport",
    "14" : "ignite",
    "21" : "barrier"
}


def create_matches_df(
    raw_master_df: pd.DataFrame,
    queue_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    queue_id_map = {
        "draft": [400],
        "ranked": [420],
        "both": [400, 420]
    }

    if queue_type not in queue_id_map:
        raise ValueError(f"Invalid queue_type: {queue_type}, must be one of {list(queue_id_map)}")

    # Parse all JSON at once
    match_data_col = raw_master_df["match_data"]#.apply(json.loads)

    # Filter for desired matches
    filtered_df = raw_master_df.copy()
    filtered_df["queue_id"] = match_data_col.map(lambda x: x["info"]["queue_id"])
    filtered_df = filtered_df[filtered_df["queue_id"].isin(queue_id_map[queue_type])].copy()

    if filtered_df.empty:
        raise InsufficientSampleError("matches") 

    filtered_df["participants"] = match_data_col.loc[filtered_df.index].map(lambda x: x["info"]["participants"])
    filtered_df["teams"] = match_data_col.loc[filtered_df.index].map(lambda x: x["info"]["teams"])
    filtered_df["match_id"] = match_data_col.loc[filtered_df.index].map(lambda x: x["metadata"]["match_id"])

    # Create participants_df, one row per participant
    exploded_participants = filtered_df[["match_id", "participants"]].explode("participants", ignore_index=True)
    normalized_participants = pd.json_normalize(exploded_participants["participants"], sep="_")
    participants_df = pd.concat([exploded_participants[["match_id"]], normalized_participants], axis=1)

    # Create teams_df, one row per team, two rows per match
    exploded_teams = filtered_df[["match_id", "teams"]].explode("teams", ignore_index=True)
    normalized_teams = pd.json_normalize(exploded_teams["teams"], sep="_")
    teams_df = pd.concat([exploded_teams[["match_id"]], normalized_teams], axis=1)

    return participants_df, teams_df


def map_tags_and_summoner_spells_to_df(
    participants_df: pd.DataFrame,
    items_dict: Dict[str, List[str]],
    summoner_spells_dict: Dict[str, str],
) -> Tuple[pd.DataFrame, Set[str], Set[str]]:

    NUM_ITEM_SLOTS = 6

    # Validation
    if not items_dict:
        raise ValueError("items_dict cannot be empty")
    if not summoner_spells_dict:
        raise ValueError("summoner_spells_dict cannot be empty")

    # Universe sets
    unique_item_tags: Set[str] = {tag for tags in items_dict.values() for tag in tags}
    unique_summoner_spells: Set[str] = set(summoner_spells_dict.values())

    # Map item IDs to tags
    # Ensure Series type matches dict keys (keys are str in items_dict)
    for i in range(NUM_ITEM_SLOTS):
        participants_df[f"tags_{i}"] = participants_df[f"item{i}"].astype(str).map(items_dict)

    tag_cols = [f"tags_{i}" for i in range(NUM_ITEM_SLOTS)]

    # Number of completed items
    participants_df["number_of_items_completed"] = (
        participants_df[tag_cols].notna().sum(axis=1).astype("Int64")
    )

    # Build per-row item_tags and per-tag counts
    # Long form: one row per (original_row, slot), keep only non-null lists
    long = (
        participants_df[tag_cols]
        .melt(ignore_index=False, value_name="tag_list")
        .drop(columns="variable")
        .dropna(subset=["tag_list"])
        .explode("tag_list")
        .rename(columns={"tag_list": "tag"})
    )

    # Per-row list of all tags
    item_tags_series = long.groupby(level=0)["tag"].agg(list)
    participants_df["item_tags"] = (
        item_tags_series.reindex(participants_df.index).apply(lambda x: x if isinstance(x, list) else []).tolist()
    )

    # Per-tag counts per row
    tag_counts = (
        long.groupby(level=0)["tag"]
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=sorted(unique_item_tags), fill_value=0)  # ensure consistent columns
    )
    tag_counts = tag_counts.add_prefix("tag_[").add_suffix("]_count").astype("Int64")

    # Join counts back
    participants_df = participants_df.join(tag_counts)

    # Map summoner spell IDs to names
    s1 = participants_df["summoner1_id"].astype(str).map(summoner_spells_dict)
    s2 = participants_df["summoner2_id"].astype(str).map(summoner_spells_dict)
    participants_df["summoner_spells_per_game"] = np.column_stack([s1, s2]).tolist()

    spells_long = (
        pd.Series(participants_df["summoner_spells_per_game"], index=participants_df.index)
        .explode()
        .dropna()
        .to_frame("spell")
    )
    spell_counts = (
        spells_long.groupby(level=0)["spell"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=sorted(unique_summoner_spells), fill_value=0)
    )
    # Convert counts (0/1/2) to presence (0/1), compact dtype, and add "has_" prefix
    has_spell = spell_counts.gt(0).astype("Int8")
    has_spell.columns = [f"has_{c}" for c in has_spell.columns]
    participants_df = participants_df.join(has_spell)

    # Cleanup
    cols_to_drop = (
        tag_cols +
        [f"item{i}" for i in range(NUM_ITEM_SLOTS)] +
        ["summoner1_id", "summoner2_id", "item_tags"]  # drop item_tags if you don't need to keep it
    )
    participants_df = participants_df.drop(columns=[c for c in cols_to_drop if c in participants_df.columns], errors="ignore")

    return participants_df, unique_item_tags, unique_summoner_spells


def derive_participant_dragon_stats(participants_df: pd.DataFrame):
    """Helper function to create dragon statistics columns"""
    # Note: for participants with no dragon takedown in match, the "earliest_dragon_takedown" has a NULL value
    ## For now we will keep it as is because PySpark ignore nulls and since these columns are used for summation statistics it doesn't matter
    drag_takedown = participants_df["challenges_earliest_dragon_takedown"]
    
    participants_df_with_dragon_stats = participants_df.assign(
        had_dragon_takedown = drag_takedown.notna().astype("int8"),
        # 5-7 min => 301-420 seconds
        first_drag_takedown_min_5_to_7 = drag_takedown.between(301, 420, inclusive="both").astype("int8"),
        # 7–11 min => 421–660
        first_drag_takedown_min_7_to_11 = drag_takedown.between(421, 660, inclusive="both").astype("int8"),
        # 11–15 min => 661–900
        first_drag_takedown_min_11_to_15= drag_takedown.between(661, 900, inclusive="both").astype("int8"),
        # 15+ => > 900
        first_drag_takedown_min_15_plus = drag_takedown.gt(900).astype("int8"),
    )

    return participants_df_with_dragon_stats


def extract_fields_with_exclusions(
    participants_df: pd.DataFrame,
    position_specific_fields: Dict[str, Dict[str, str]] = POSITION_SPECIFIC_METRICS,
    laner_specific_fields: Dict[str, str] = LANER_SPECIFIC_METRICS,
    all_positions: Optional[List[str]] = None,
    position_column: str = "team_position",
    match_data_column: str = "match_data",   # prefix for flattened cols
) -> pd.DataFrame:
    """
    Same logic as the Spark function, but:
    - rows where position doesn't match -> <NA>
    - rows where position matches but field missing/NaN -> <NA>
    Expects flattened columns like f"{match_data_column}_{field_path.replace('.', '_')}".
    """
    if all_positions is None:
        all_positions = ["JUNGLE", "MID", "BOTTOM", "UTILITY", "TOP"]

    # Build operations: (position, new_col, field_path)
    operations = []
    for position, mapping in position_specific_fields.items():
        for new_col, field_path in mapping.items():
            operations.append((position, new_col, field_path))
    for new_col, field_path in laner_specific_fields.items():
        for position in (p for p in all_positions if p != "JUNGLE"):
            operations.append((position, new_col, field_path))

    pos_ser = participants_df[position_column]
    new_cols: Dict[str, pd.Series] = {}

    for position, new_col, field_path in operations:
        src_col = f"{match_data_column}_{field_path.replace('.', '_')}"
        if src_col in participants_df.columns:
            # numeric with NA preserved (no fill with 0)
            src_num = pd.to_numeric(participants_df[src_col], errors="coerce").astype("Int64")
        else:
            # whole-column NA when field absent
            src_num = pd.Series(pd.NA, index=participants_df.index, dtype="Int64")

        # keep value only where position matches; else NA
        val = src_num.where(pos_ser == position, other=pd.NA)

        # If multiple ops write the same new_col (different positions), use first non-NA.
        # Positions are mutually exclusive, so combine_first yields the single matching value or NA.
        new_cols[new_col] = new_cols.get(new_col, pd.Series(pd.NA, index=participants_df.index, dtype="Int64")).combine_first(val)

    return participants_df.assign(**new_cols)


def aggregate_champion_data(merged_df, all_item_tags, all_summoner_spells, user_puuid): # Add granularity input
    
    merged_df = merged_df[merged_df["puuid"] == user_puuid].copy()

    mejais_check = pd.to_numeric(merged_df["challenges_mejais_full_stack_in_time"], errors="coerce")
    merged_df["fully_stacked_mejais"] = pd.Series(pd.NA, index=merged_df.index, dtype="Int64")
    merged_df.loc[mejais_check.gt(0), "fully_stacked_mejais"] = 1

    grouped_df = merged_df.groupby(["champion_id", "champion_name","team_position"], dropna=False, as_index=False)

    tag_aggs = {
        f"avg_{tag}_count": (f"tag_[{tag}]_count", "mean")
        for tag in all_item_tags
    }   

    summ_spells_aggs = {
        f"pct_of_matches_with_{summoner_spell}": (f"has_{summoner_spell}", lambda x: 100*x.mean())
        for summoner_spell in all_summoner_spells    
    }

    agg_map = dict(
        # counts / first
        total_games_played_in_role = ("match_id", "count"),        
        total_games_per_champion   = ("champion_name", "count"),

        # core stats
        avg_kills                  = ("kills", "mean"),
        avg_deaths                 = ("deaths", "mean"),
        avg_deaths_by_enemy_champs = ("challenges_deaths_by_enemy_champs", "mean"),
        avg_assists                = ("assists", "mean"),
        avg_kill_participation     = ("challenges_kill_participation", "mean"),
        avg_takedowns              = ("challenges_takedowns", "mean"),

        total_wins = ("win", lambda s: (s.astype("Int64").fillna(0) * 100).sum()),

        # damage dealt stats
        # keep as sum (Spark comment: derive % later because of NULLs)
        sum_of_games_with_highest_damage_dealt = ("challenges_highest_champion_damage", "sum"),
        avg_pct_damage_dealt_in_team           = ("challenges_team_damage_percentage", "mean"),
        average_damage_per_minute              = ("challenges_damage_per_minute", "mean"),
        avg_damage_dealt_to_champions          = ("total_damage_dealt_to_champions", "mean"),
        avg_total_damage_dealt                 = ("total_damage_dealt", "mean"),
        avg_magic_damage_dealt_to_champions    = ("magic_damage_dealt_to_champions", "mean"),
        avg_total_magic_damage_dealt           = ("magic_damage_dealt", "mean"),
        avg_physical_damage_dealt_to_champions = ("physical_damage_dealt_to_champions", "mean"),
        avg_total_physical_damage_dealt        = ("physical_damage_dealt", "mean"),
        avg_true_damage_dealt_to_champions     = ("true_damage_dealt_to_champions", "mean"),
        avg_total_true_damage_dealt            = ("true_damage_dealt", "mean"),
        avg_largest_critical_strike            = ("largest_critical_strike", "mean"),

        # Damage Taken
        avg_pct_damage_taken_in_team            = ("challenges_damage_taken_on_team_percentage", "mean"),
        avg_damage_taken                        = ("total_damage_taken", "mean"),
        avg_magic_damage_taken                  = ("magic_damage_taken", "mean"),
        avg_physical_damage_taken               = ("physical_damage_taken", "mean"),
        avg_true_damage_taken                   = ("true_damage_taken", "mean"),
        avg_damage_self_mitigated               = ("damage_self_mitigated", "mean"),

        # Situational damage taken
        avg_times_killed_champ_took_full_team_damage_and_survived = ("challenges_killed_champ_took_full_team_damage_survived", "mean"),
        avg_times_survived_single_digit_hp                        = ("challenges_survived_single_digit_hp_count", "mean"),
        avg_times_survived_three_immobilizes_in_fight             = ("challenges_survived_three_immobilizes_in_fight", "mean"),
        avg_times_took_large_damage_survived                      = ("challenges_took_large_damage_survived", "mean"),

        # Crowd control
        sum_of_games_with_highest_crowd_control_score = ("challenges_highest_crowd_control_score", "sum"),  # keep sum; derive % later
        avg_time_ccing_others                         = ("time_c_cing_others", "mean"),
        avg_times_applied_cc_on_others                = ("total_time_cc_dealt", "mean"),
        avg_enemy_champion_immobilizations            = ("challenges_enemy_champion_immobilizations", "mean"),

        # Healing / shielding / support
        avg_total_healing                 = ("total_heal", "mean"),
        avg_heals_on_teammate             = ("total_heals_on_teammates", "mean"),
        avg_total_units_healed            = ("total_units_healed", "mean"),
        avg_dmg_shielded_on_team          = ("total_damage_shielded_on_teammates", "mean"),
        avg_effective_heal_and_shield     = ("challenges_effective_heal_and_shielding", "mean"),
        sum_of_games_completed_supp_quest_first = ("challenges_faster_support_quest_completion",
                                                lambda s: s.astype("Int64").fillna(0).sum()),
        avg_supp_quest_completion_time    = ("complete_support_quest_in_time", "mean"),

        # Misc
        avg_longest_time_spent_alive = ("longest_time_spent_living", "mean"),
        avg_time_spent_dead          = ("total_time_spent_dead", "mean"),

        # Spell casts
        avg_spell1_casts  = ("spell1_casts", "mean"),
        avg_spell2_casts  = ("spell2_casts", "mean"),
        avg_spell3_casts  = ("spell3_casts", "mean"),
        avg_spell4_casts  = ("spell4_casts", "mean"),
        avg_ability_uses  = ("challenges_ability_uses", "mean"),

        # Skillshot related (dodging and hitting)
        avg_times_dodged_skillshot_in_small_window = ("challenges_dodge_skill_shots_small_window", "mean"),
        avg_skillshots_dodged                      = ("challenges_skillshots_dodged", "mean"),
        avg_skillshots_landed_early_game           = ("challenges_land_skill_shots_early_game", "mean"),
        avg_skillshots_hit                         = ("challenges_skillshots_hit", "mean"),

        # Picks
        avg_times_immobilize_and_kill_with_ally   = ("challenges_immobilize_and_kill_with_ally", "mean"),
        avg_times_got_kill_after_hidden_with_ally = ("challenges_kill_after_hidden_with_ally", "mean"),
        avg_times_pick_kill_with_ally             = ("challenges_pick_kill_with_ally", "mean"),
        avg_times_knock_enemy_into_team_and_kill  = ("challenges_knock_enemy_into_team_and_kill", "mean"),

        # Kills under/near turret
        avg_kills_near_enemy_turret               = ("challenges_kills_near_enemy_turret", "mean"),
        avg_kills_under_own_turret                = ("challenges_kills_under_own_turret", "mean"),

        # Misc mechanics
        avg_multikills_after_aggressive_flash     = ("challenges_multikills_after_aggressive_flash", "mean"),
        avg_outnumbered_kills                     = ("challenges_outnumbered_kills", "mean"),
        avg_times_outnumbered_nexus_kill          = ("challenges_outnumbered_nexus_kill", "mean"),
        avg_times_quick_cleanse                   = ("challenges_quick_cleanse", "mean"),

        # Misc laning — kills, takedowns and plays
        avg_quick_solo_kills                      = ("challenges_quick_solo_kills", "mean"),
        avg_solo_kills                            = ("challenges_solo_kills", "mean"),
        avg_takedowns_after_gaining_lvl_advantage = ("challenges_takedowns_after_gaining_level_advantage", "mean"),
        avg_kills_on_other_lanes_early_as_laner   = ("kills_on_other_lanes_early_as_laner", "mean"),
        avg_times_save_ally_from_death            = ("challenges_save_ally_from_death", "mean"),
        avg_takedowns_in_alcove                   = ("challenges_takedowns_in_alcove", "mean"),

        # First blood / early kills (Spark: avg(int * 100))
        pct_of_games_first_blood_kill    = ("first_blood_kill",
                                            lambda s: (s.astype("Int64") * 100).mean()),
        pct_of_games_first_blood_assist  = ("first_blood_assist",
                                            lambda s: (s.astype("Int64") * 100).mean()),
        avg_takedowns_before_jungle_camps_spawn = ("challenges_takedowns_before_jungle_minion_spawn", "mean"),
        avg_first_takedown_time          = ("challenges_takedowns_first_x_minutes", "mean"),

        # Summoner spell cast counts
        avg_summoner_spell1_casts_per_game = ("summoner1_casts", "mean"),
        avg_summoner_spell2_casts_per_game = ("summoner2_casts", "mean"),

        # Experience / Gold
        avg_champ_exp_at_game_end         = ("champ_experience", "mean"),
        avg_champ_level_at_game_end       = ("champ_level", "mean"),
        avg_gold_earned_per_game          = ("gold_earned", "mean"),
        avg_gold_per_minute               = ("challenges_gold_per_minute", "mean"),
        avg_gold_spent                    = ("gold_spent", "mean"),
        avg_bounty_lvl                    = ("bounty_level", "mean"),
        avg_bounty_gold                   = ("challenges_bounty_gold", "mean"),
        # Early/laning phase gold+exp advantage (Spark: avg(col * 100))
        pct_of_games_with_early_lane_phase_gold_exp_adv = (
            "challenges_early_laning_phase_gold_exp_advantage",
            lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()
        ),
        pct_of_games_with_lanephase_gold_exp_adv = (
            "challenges_laning_phase_gold_exp_advantage",
            lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()
        ),
        avg_max_level_lead_over_lane_opp = ("challenges_max_level_lead_lane_opponent", "mean"),

        # Minions
        avg_minions_killed                 = ("total_minions_killed", "mean"),
        avg_minions_killed_by_10_mins      = ("challenges_lane_minions_first10_minutes", "mean"),
        avg_max_cs_lead_over_lane_opponent = ("challenges_max_cs_advantage_on_lane_opponent", "mean"),

        # Item purchases
        avg_consumables_purchased          = ("consumables_purchased", "mean"),
        avg_number_of_items_purchased      = ("items_purchased", "mean"),
        # Spark: sum(isNotNull(fastest_legendary).cast("int"))
        total_games_fastest_item_completion = (
            "challenges_fastest_legendary",
            lambda s: s.notna().astype("Int64").sum()
        ),

        # Item Tags
        avg_items_completed = ("number_of_items_completed", "mean"),

        # Jungle farm
        avg_ally_jungle_minions_killed     = ("total_ally_jungle_minions_killed", "mean"),
        avg_enemy_jungle_minions_killed    = ("total_enemy_jungle_minions_killed", "mean"),
        avg_enemy_jungle_cs_differential_early = ("more_enemy_jungle_cs_than_opponent_as_jungler", "mean"),
        avg_jungle_monsters_cs             = ("neutral_minions_killed", "mean"),
        avg_buffs_stolen                   = ("challenges_buffs_stolen", "mean"),
        avg_initial_buff_count             = ("challenges_initial_buff_count", "mean"),
        avg_epic_monster_kills_within_30s_of_spawn = ("challenges_epic_monster_kills_within30_seconds_of_spawn", "mean"),
        avg_initial_crab_count             = ("challenges_initial_crab_count", "mean"),
        avg_crabs_per_game                 = ("challenges_scuttle_crab_kills", "mean"),
        avg_jg_cs_before_10m               = ("challenges_jungle_cs_before10_minutes", "mean"),

        # Jungle combat / related
        avg_jungler_kills_early_jungle     = ("jungler_kills_early_jungle", "mean"),
        avg_jungler_early_kills_on_laners  = ("kills_on_laners_early_jungle_as_jungler", "mean"),
        avg_times_had_early_takedowns_in_all_lanes_as_laner = ("takedowns_in_all_lanes_early_as_laner", "mean"),
        avg_jungler_takedowns_near_damaged_epic_monsters    = ("challenges_jungler_takedowns_near_damaged_epic_monster", "mean"),
        avg_kills_with_help_from_epic_monster = ("challenges_kills_with_help_from_epic_monster", "mean"),
            
        # --- Vision stats ---
        avg_vision_score                                   = ("vision_score", "mean"),
        avg_vision_score_per_min                           = ("challenges_vision_score_per_minute", "mean"),
        avg_vision_score_advantage_over_lane_opponent      = ("challenges_vision_score_advantage_lane_opponent", "mean"),
        avg_stealth_wards_placed                           = ("challenges_stealth_wards_placed", "mean"),
        avg_wards_placed                                   = ("wards_placed", "mean"),
        avg_wards_guarded                                  = ("challenges_wards_guarded", "mean"),
        avg_control_wards_placed                           = ("detector_wards_placed", "mean"),
        avg_control_ward_time_coverage_in_river_or_enemy_half = ("challenges_control_ward_time_coverage_in_river_or_enemy_half", "mean"),
        avg_unseen_recalls                                 = ("challenges_unseen_recalls", "mean"),

        # Wards killed
        sum_of_games_with_highest_wards_killed             = ("challenges_highest_ward_kills",
                                                            lambda s: s.astype("Int64").sum()),   # keep SUM (not % yet)
        avg_wards_killed                                   = ("wards_killed", "mean"),
        avg_ward_takedowns                                 = ("challenges_ward_takedowns", "mean"),
        avg_ward_takedowns_before_20m                      = ("challenges_ward_takedowns_before20_m", "mean"),
        avg_times_2_wards_killed_with_1_sweeper            = ("challenges_two_wards_one_sweeper_count", "mean"),
        avg_control_wards_bought                           = ("vision_wards_bought_in_game", "mean"),

        # --- Teamwide: first objective rates (avg(int)*100) ---
        pct_of_games_team_took_first_baron     = ("objectives_baron_first",
                                                lambda s: (s.astype("Int64") * 100).mean()),
        avg_earliest_baron_by_team_time        = ("challenges_earliest_baron", "mean"),
        pct_of_games_team_took_first_dragon    = ("objectives_dragon_first",
                                                lambda s: (s.astype("Int64") * 100).mean()),
        pct_of_games_team_took_first_inhib     = ("objectives_inhibitor_first",
                                                lambda s: (s.astype("Int64") * 100).mean()),
        pct_of_games_team_took_first_herald    = ("objectives_rift_herald_first",
                                                lambda s: (s.astype("Int64") * 100).mean()),
        pct_of_games_team_took_first_turret    = ("objectives_tower_first",
                                                lambda s: (s.astype("Int64") * 100).mean()),

        # Team objectives
        avg_baron_kills_by_team                = ("objectives_baron_kills", "mean"),
        avg_herald_kills_by_team               = ("objectives_rift_herald_kills", "mean"),
        avg_dragon_kills_by_team               = ("objectives_dragon_kills", "mean"),
        pct_of_games_with_perfect_drag_soul_taken = ("challenges_perfect_dragon_souls_taken",
                                                    lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()),
        avg_elder_dragon_kills_by_team         = ("challenges_team_elder_dragon_kills", "mean"),
        avg_elder_dragon_kills_w_opposing_soul = ("challenges_elder_dragon_kills_with_opposing_soul", "mean"),

        # Team structures
        avg_inhib_kills_by_team                = ("objectives_inhibitor_kills", "mean"),
        avg_tower_kills_by_team                = ("objectives_tower_kills", "mean"),
        avg_inhibs_lost_by_team                = ("inhibitors_lost", "mean"),
        pct_of_games_with_nexus_lost_by_team   = ("nexus_lost",
                                                lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()),
        avg_turrets_lost_by_team               = ("turrets_lost", "mean"),
        pct_of_games_first_turret_taken_by_team= ("challenges_first_turret_killed",
                                                lambda s: (s.astype("Int64") * 100).mean()),
        avg_first_turret_kill_time_by_team     = ("challenges_first_turret_killed_time", "mean"),

        # Team kills
        avg_total_team_champ_kills             = ("objectives_champion_kills", "mean"),
        avg_team_aces_before_15_by_team        = ("challenges_aces_before15_minutes", "mean"),
        avg_flawless_aces_by_team              = ("challenges_flawless_aces", "mean"),
        avg_shortest_time_to_ace_from_1st_takedown = ("challenges_shortest_time_to_ace_from_first_takedown", "mean"),
        avg_max_kill_deficit                   = ("challenges_max_kill_deficit", "mean"),
        pct_of_games_that_are_perfect_games    = ("challenges_perfect_game",
                                                lambda s: (s.astype("Int64") * 100).mean()),

        # --- Individual participant damage to structures ---
        avg_indiv_dmg_dealt_to_buildings       = ("damage_dealt_to_buildings", "mean"),
        avg_indiv_dmg_dealth_to_turrets        = ("damage_dealt_to_turrets", "mean"),  # alias kept as given
        avg_indiv_turret_plates_taken          = ("challenges_turret_plates_taken", "mean"),

        # First tower (booleans cast to int * 100 then averaged)
        pct_of_games_indiv_killed_1st_tower        = ("first_tower_kill",
                                                    lambda s: (s.astype("Int64") * 100).mean()),
        pct_of_games_individual_takedown_1st_tower = ("challenges_takedown_on_first_turret",
                                                    lambda s: (s.astype("Int64") * 100).mean()),
        pct_of_games_individual_took_1st_tower_quick = ("challenges_quick_first_turret",
                                                        lambda s: (s.astype("Int64") * 100).mean()),
        pct_of_games_individual_had_1st_turret_assist = ("first_tower_assist",
                                                        lambda s: (s.astype("Int64") * 100).mean()),

        # Turrets kills/takedowns
        avg_turrets_killed_before_plates_fell  = ("challenges_k_turrets_destroyed_before_plates_fall", "mean"),
        avg_individual_tower_kills             = ("turret_kills", "mean"),
        avg_individual_tower_takedowns         = ("turret_takedowns", "mean"),
        avg_individual_tower_takedowns2        = ("challenges_turret_takedowns", "mean"),
        avg_individual_solo_towers_kills_late_game = ("challenges_solo_turrets_lategame", "mean"),
        avg_indiv_towers_taken_w_rift_herald   = ("challenges_turrets_taken_with_rift_herald", "mean"),
        avg_indiv_multi_towers_taken_w_rift_herald = ("challenges_multi_turret_rift_herald_count", "mean"),

        # Inhibitor & nexus
        avg_individual_inhibitor_kills         = ("inhibitor_kills", "mean"),
        avg_individual_inhibitor_takedowns     = ("inhibitor_takedowns", "mean"),
        pct_of_games_individual_killed_nexus   = ("nexus_kills",
                                                lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()),
        avg_individual_nexus_takedowns         = ("nexus_takedowns",
                                                lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()),
        pct_of_games_with_open_nexus           = ("challenges_had_open_nexus",
                                                lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()),

        # --- Individual participant objectives: kills/takedowns ---
        avg_individual_dmg_dealt_to_objectives = ("damage_dealt_to_objectives", "mean"),
        avg_individual_baron_kills             = ("baron_kills", "mean"),
        avg_individual_solo_baron_kills        = ("challenges_solo_baron_kills", "mean"),
        avg_individual_baron_takedowns         = ("challenges_baron_takedowns", "mean"),
        avg_individual_dragon_kills            = ("dragon_kills", "mean"),
        avg_individual_dragon_takedowns        = ("challenges_dragon_takedowns", "mean"),
        avg_individual_rift_herald_takedowns   = ("challenges_rift_herald_takedowns", "mean"),
        avg_individual_void_monster_kills      = ("challenges_void_monster_kill", "mean"),

        # --- Objective steals ---
        avg_objectives_stolen                  = ("objectives_stolen", "mean"),
        avg_objectives_stolen_assists          = ("objectives_stolen_assists", "mean"),
        avg_epic_monster_steals                = ("challenges_epic_monster_steals", "mean"),
        avg_epic_monster_steals_without_smite  = ("challenges_epic_monster_stolen_without_smite", "mean"),
        avg_epic_monsters_killed_near_enemy_jgler = ("challenges_epic_monster_kills_near_enemy_jungler", "mean"),

        # --- Earliest dragon / derived flags (Spark uses avg(col * 100) for flags) ---
        avg_earliest_drag_takedown             = ("challenges_earliest_dragon_takedown", "mean"),
        pct_of_games_had_drag_takedown         = ("had_dragon_takedown",
                                                lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()),
        pct_of_games_had_drag_takedown_min_5_to_7  = ("first_drag_takedown_min_5_to_7",
                                                    lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()),
        pct_of_games_had_drag_takedown_min_7_to_11 = ("first_drag_takedown_min_7_to_11",
                                                    lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()),
        pct_of_games_had_drag_takedown_min_11_to_15= ("first_drag_takedown_min_11_to_15",
                                                    lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()),
        pct_of_games_had_drag_takedown_min_15_plus = ("first_drag_takedown_min_15_plus",
                                                    lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()),

        # --- Game length related ---
        avg_time_played_per_game_minutes       = ("time_played", lambda s: s.mean() / 60.0),
        avg_game_length                        = ("challenges_game_length", "mean"),
        pct_of_games_ended_in_early_ff         = ("game_ended_in_early_surrender",
                                                lambda s: (s.astype("Int64") * 100).mean()),
        pct_of_games_ended_in_ff               = ("game_ended_in_surrender",
                                                lambda s: (s.astype("Int64") * 100).mean()),
        pct_of_games_team_ffd                  = ("team_early_surrendered",
                                                lambda s: (s.astype("Int64") * 100).mean()),

        # --- Multikills ---
        avg_doublekills                        = ("double_kills", "mean"),
        avg_triplekills                        = ("triple_kills", "mean"),
        avg_quadrakills                        = ("quadra_kills", "mean"),
        avg_pentakills                         = ("penta_kills", "mean"),
        avg_largest_multikill                  = ("largest_multi_kill", "mean"),
        avg_number_of_multikills               = ("challenges_multikills", "mean"),
        avg_multikills_with_one_spell          = ("challenges_multi_kill_one_spell", "mean"),

        # --- Killing sprees ---
        avg_killing_sprees                     = ("killing_sprees", "mean"),
        avg_killing_sprees2                    = ("challenges_killing_sprees", "mean"),
        avg_legendary_count                    = ("challenges_legendary_count", "mean"),
        avg_largest_killing_spee               = ("largest_killing_spree", "mean"),  # alias kept as provided

        # --- Misc ---
        avg_unreal_kills                       = ("unreal_kills", "mean"),
        avg_12_assist_streaks                  = ("challenges_12_assist_streak_count", "mean"),
        avg_elder_drag_multikills              = ("challenges_elder_dragon_multikills", "mean"),
        avg_full_team_takedowns                = ("challenges_full_team_takedown", "mean"),

        avg_times_blast_cone_enemy             = ("challenges_blast_cone_opposite_opponent_count", "mean"),
        pct_of_games_danced_with_rift_herald   = ("challenges_danced_with_rift_herald",
                                                lambda s: (pd.to_numeric(s, errors="coerce") * 100).mean()),
        avg_double_aces                        = ("challenges_double_aces", "mean"),
        avg_fist_bump_participations           = ("challenges_fist_bump_participation", "mean"),

        # Spark used avg("fully_stacked_mejais") — mean of 0/1/NA (no *100 here to match Spark)
        percent_of_games_with_fully_stacked_mejais = ("fully_stacked_mejais", "mean"),

        # Spark logic here effectively avg((challenges_mejais_full_stack_in_time != 0).cast(int))
        # i.e., fraction of games where the value is non-zero.
        avg_mejai_full_stack_time = ("challenges_mejais_full_stack_in_time",
                                    lambda s: s.ne(0).astype("Int64").mean()),

        avg_outer_turret_executes_before_10m   = ("challenges_outer_turret_executes_before10_minutes", "mean"),
        avg_takedowns_in_enemy_fountain        = ("challenges_takedowns_in_enemy_fountain", "mean"),

        pct_of_games_played_champ_select_position = ("challenges_played_champ_select_position",
                                                    lambda s: (pd.to_numeric(s, errors="coerce")).sum())
        )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
        intermediate_df = grouped_df.agg(**{**agg_map, **tag_aggs, **summ_spells_aggs})
    
    intermediate_df = intermediate_df.copy()
    
    new_columns = {
        "role_play_rate": (
            (intermediate_df["total_games_played_in_role"].astype("Float64")).mul(100)
            .div(intermediate_df["total_games_per_champion"].astype("Float64"))
            .where(intermediate_df["total_games_per_champion"].ne(0))
        ),

        "pct_of_games_with_highest_damage_dealt": (
            (intermediate_df["sum_of_games_with_highest_damage_dealt"].astype("Float64")).mul(100)
            .div(intermediate_df["total_games_played_in_role"].astype("Float64"))
            .where(intermediate_df["total_games_played_in_role"].ne(0))
        ),

        "pct_of_games_with_highest_crowd_control_score": (
            (intermediate_df["sum_of_games_with_highest_crowd_control_score"].astype("Float64")).mul(100)
            .div(intermediate_df["total_games_played_in_role"].astype("Float64"))
            .where(intermediate_df["total_games_played_in_role"].ne(0))
        ),
        # Change name to pct when we change spark
        "total_games_completed_supp_quest_first": (
            (intermediate_df["sum_of_games_completed_supp_quest_first"].astype("Float64")).mul(100)
            .div(intermediate_df["total_games_played_in_role"].astype("Float64"))
            .where(intermediate_df["total_games_played_in_role"].ne(0))
        ),

        "pct_of_games_with_highest_wards_killed": (
            (intermediate_df["sum_of_games_with_highest_wards_killed"].astype("Float64")).mul(100)
            .div(intermediate_df["total_games_played_in_role"].astype("Float64"))
            .where(intermediate_df["total_games_played_in_role"].ne(0))
        ),

        "kda": (
            ((intermediate_df["avg_kills"].astype("Float64"))
            .add(intermediate_df["avg_assists"].astype("Float64"))
            .div(intermediate_df["avg_deaths"].astype("Float64")))
            .where(intermediate_df["avg_deaths"].ne(0))
        ),

        "win_rate": (
            (intermediate_df["total_wins"].astype("Float64")).mul(100)
            .div(intermediate_df["total_games_played_in_role"].astype("Float64"))
            .where(intermediate_df["total_games_played_in_role"].ne(0))
        ),

        "avg_cs": (
            (intermediate_df["avg_minions_killed"]).astype("Float64")
            .add(intermediate_df["avg_jungle_monsters_cs"]).astype("Float64")
        ),

        "pct_games_first_to_complete_item": (
            (intermediate_df["total_games_fastest_item_completion"].astype("Float64")).mul(100)
            .div(intermediate_df["total_games_played_in_role"].astype("Float64"))
            .where(intermediate_df["total_games_played_in_role"].ne(0))
        )
    }

    new_columns_df = pd.DataFrame(new_columns)

    intermediate_df = pd.concat([intermediate_df, new_columns_df], axis=1)

    # Create column that is derived from one of the new columns separately 
    intermediate_df["avg_cs_per_minute"] = (
        intermediate_df["avg_cs"].astype("Float64")
        .div(intermediate_df["avg_time_played_per_game_minutes"].astype("Float64"))
        .where(intermediate_df["avg_time_played_per_game_minutes"].ne(0))
    )

    denominator = intermediate_df["avg_items_completed"].astype("Float64")
    pct_cols = {
        f"pct_of_matches_with_{tag}":
            intermediate_df[f"avg_{tag}_count"].astype("Float64").mul(100).div(denominator).mask(denominator.eq(0))
        for tag in all_item_tags
    }

    columns_to_drop = [f"avg_{tag}_count" for tag in all_item_tags] + [
        "sum_of_games_with_highest_damage_dealt", "sum_of_games_with_highest_crowd_control_score", 
        "sum_of_games_completed_supp_quest_first", "sum_of_games_with_highest_wards_killed"]

    final_df = (
        intermediate_df
        .drop(columns=columns_to_drop)
        .assign(**pct_cols)
    )

    return final_df
        

def derive_counter_stats_pd(
    raw_counter_stats_df: pd.DataFrame,
    desired_team_positions: List[str],
    min_games: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Pandas equivalent of derive_counter_stats (Spark).
    For each role:
      - filter rows
      - keep distinct (match_id, team_id, champion_name)
      - self-join on match to pair opponents (exclude same team)
      - aggregate games & wins
      - compute win_rate only when number_of_games >= min_games; else NA
    Returns: { role -> DataFrame(champion_name, opp_champion_name, number_of_games, wins, win_rate) }
    """
    counter_stats_dfs_by_role: Dict[str, pd.DataFrame] = {}

    for role in desired_team_positions:
        # Filter + select + distinct
        role_df = (
            raw_counter_stats_df.loc[
                raw_counter_stats_df["team_position"] == role,
                ["match_id", "team_id", "champion_name", "win"],
            ]
            .drop_duplicates(subset=["match_id", "team_id", "champion_name"])
            .copy()
        )

        if role_df.empty:
            raise ValueError(f"No rows found for team_position={role}")

        # Cast win to int
        win_int = pd.to_numeric(role_df["win"], errors="coerce").fillna(0).astype("Int64")
        role_df["win"] = win_int

        # Self-join by match to pair opponents (exclude same team)
        b = role_df[["match_id", "team_id", "champion_name"]].rename(
            columns={"team_id": "opp_team_id", "champion_name": "opp_champion_name"}
        )
        pairs = (
            role_df.merge(b, on="match_id", how="inner")
                   .loc[lambda d: d["team_id"] != d["opp_team_id"],
                        ["champion_name", "opp_champion_name", "win"]]
        )

        # Aggregate
        grp = pairs.groupby(["champion_name", "opp_champion_name"], as_index=False)
        agg = grp.agg(
            number_of_games=("win", "size"),
            wins=("win", "sum"),     
        )

        # win_rate: NULL (NA) if number_of_games < min_games
        den = agg["number_of_games"].astype("Float64")
        num = agg["wins"].astype("Float64").mul(100)
        win_rate = num.div(den)
        win_rate = win_rate.mask(den.lt(min_games))
        agg["win_rate"] = win_rate

        counter_stats_dfs_by_role[role] = agg[["champion_name", "opp_champion_name", "number_of_games", "wins", "win_rate"]]

    return counter_stats_dfs_by_role


"""
Main aggregating functions, uses helpers as needed, idea is to pull all wanted keys from match_data struct into columns (some keys will be modified, derived)
Finally, match_data will be dropped
"""
def main_aggregator(
    raw_master_df: pd.DataFrame,
    queue_type: str,
    items_dict: dict,
    user_puuid: str
) -> pd.DataFrame:
    
    participants_df, teams_df = create_matches_df(raw_master_df, queue_type)

    # Create an indicator column for games where champion had a dragon takedown, and subsequent columns with the timing of first dragon takedown
    participants_df = derive_participant_dragon_stats(participants_df)

    participants_df, all_item_tags, all_summoner_spells = map_tags_and_summoner_spells_to_df(
        participants_df, 
        items_dict, 
        SUMMONER_SPELLS_DICT, 
    )

    participants_df = extract_fields_with_exclusions(participants_df)

    merged_df = participants_df.merge(
        teams_df.drop(columns=["win"], errors="ignore"),  # drop to avoid duplicate col
        on=["match_id", "team_id"],
        how="left",
    )

    single_user_df = aggregate_champion_data(merged_df, all_item_tags, all_summoner_spells, user_puuid)

    return single_user_df

