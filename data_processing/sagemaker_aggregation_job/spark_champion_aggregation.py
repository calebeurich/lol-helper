import re, json
import boto3
from typing import Dict, List, Set, Tuple, Any
from urllib.parse import urlparse
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from enum import Enum

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


# — UDF definitions unchanged —
def camel_to_snake(name: str) -> str:
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def rename_keys(obj):
    if isinstance(obj, dict):
        return { camel_to_snake(k): rename_keys(v) for k, v in obj.items() }
    elif isinstance(obj, list):
        return [rename_keys(x) for x in obj]
    else:
        return obj


def safe_snake_json(s: str) -> str:
    if not s or not s.strip():
        return None
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        return None
    return json.dumps(rename_keys(parsed))


def format_df_to_snake(raw_df: DataFrame) -> DataFrame:

    snake_json = F.udf(safe_snake_json, StringType())
    # apply UDF + rename columns
    snake_formatted_df = (
        raw_df
            .withColumn("match_data", snake_json(F.col("match_data")))
    )

    return snake_formatted_df


def explode_and_flatten_struct(
        all_matches_df: DataFrame,
        desired_struct_name: str,
        base_path: str = "match_data_struct.info"
    ) -> DataFrame:
    """
    Explode and flatten a nested struct array from match data into separate rows.
    
    This function takes a DataFrame containing match data with nested struct arrays
    and explodes a specific struct field into individual rows, then flattens the 
    struct columns into top-level columns.
    
    Parameters
    ----------
    all_matches_df : pyspark.sql.DataFrame
        DataFrame containing match data with a nested structure at 
        "match_data_struct.info.{desired_struct_name}". Must contain at least:
        - match_id: Identifier for each match
        - match_data_struct.info.{desired_struct_name}: Array of structs to explode
    desired_struct_name : str
        Name of the struct array to explode within match_data_struct.info.
        Should be plural (e.g., "participants", "teams", "bans").
        The function will remove the trailing "s" for the alias.
    base_path: str
        Optional configurable path of match_data struct
        Defaults to current path "match_data_struct.info" 
        
    Returns
    -------
    pyspark.sql.DataFrame
        Flattened DataFrame with one row per array element containing:
        - match_id: Original match identifier (duplicated for each exploded row)
        - All top-level fields from the exploded struct as columns
        
    Examples
    --------
    >>> # Assuming df has structure: match_data_struct.info.participants
    >>> participants_df = explode_and_flatten_struct(all_matches_df, "participants")
    >>> participants_df.printSchema()
    root
     |-- match_id: string (nullable = true)
     |-- championId: integer (nullable = true)
     |-- summonerId: string (nullable = true)
     |-- ...
     
    >>> # For a match with 10 participants, this creates 10 rows
    >>> participants_df.groupBy("match_id").count().show()
    +----------+-----+
    |match_id  |count|
    +----------+-----+
    |MATCH_001 |   10|
    |MATCH_002 |   10|
    +----------+-----+
    
    Notes
    -----
    - The function assumes a specific nested structure: match_data_struct.info.{struct_name}
    - The struct name should be plural as the function strips the trailing "s" for aliasing
    - Each match_id will be duplicated for each element in the exploded array
    - All nested fields within the struct are promoted to top-level columns
    
    Raises
    ------
    AnalysisException
        If the specified struct path doesn't exist in the DataFrame
    """

    struct_path = f"{base_path}.{desired_struct_name}"
    element_alias = desired_struct_name.rstrip("s")

    return (
        all_matches_df
        .select(
            F.col(MATCH_ID),
            F.explode(F.col(struct_path)).alias(element_alias)
        )
        .select(
            MATCH_ID,
            F.col(element_alias + ".*")
        )
    )

def create_matches_df(
    spark: SparkSession,
    csv_file_path_or_user_df: str,
    npartitions: int = DEFAULT_PARTITIONS,
    queue_id: int = 420,
    single_user_flag = False
) -> Tuple[DataFrame, DataFrame]:
    """
    Load and process match data from CSV containing JSON strings into structured DataFrames.
    
    This function reads a CSV file containing match data as JSON strings, filters out
    error records, parses the JSON into structured format, and extracts participant
    and team information for ranked solo/duo queue matches (queueId 420).
    
    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session for DataFrame operations.
    csv_file_path : str
        Path to the CSV file containing match data. Expected columns:
        - match_id: Unique identifier for each match
        - match_data: JSON string containing full match information
    npartitions : int, default=200
        Number of partitions to use after filtering. Adjust based on data size
        and cluster resources.
    queue_id: int, default=420
        Identifier representing ranked games for filtering purposes.
        
    Returns
    -------
    participants_df : pyspark.sql.DataFrame
        DataFrame with one row per participant containing:
        - match_id: Match identifier
        - All participant fields from the JSON structure
        - Filtered to only include participants with non-null team_position
    teams_df : pyspark.sql.DataFrame
        DataFrame with one row per team (2 per match) containing:
        - match_id: Match identifier
        - All team fields from the JSON structure
        
    Examples
    --------
    >>> spark = SparkSession.builder.appName("MatchAnalysis").getOrCreate()
    >>> participants, teams = create_matches_df(
    ...     spark, 
    ...     "hdfs://data/matches.csv",
    ...     npartitions=100
    ... )
    >>> participants.groupBy("team_position").count().show()
    +-------------+-------+
    |team_position|  count|
    +-------------+-------+
    |          TOP| 123456|
    |       JUNGLE| 123456|
    |          MID| 123456|
    |          BOT| 123456|
    |      SUPPORT| 123456|
    +-------------+-------+
    
    Notes
    -----
    - The function caches the cleaned DataFrame after filtering and repartitioning
    - Only processes ranked solo/duo queue matches (queueId == 420)
    - Removes duplicate matches based on match_id
    - Filters out "Patch Error" records which indicate incomplete data
    - Schema inference reads all JSON data (samplingRatio=1.0) for accuracy
    - Participants without a team_position (e.g., spectators) are excluded
    
    Raises
    ------
    AnalysisException
        If the CSV file cannot be read or required columns are missing
    """

    # ========== Extract df From All CSV Files ==========
    if not single_user_flag:
        raw_df = (
            spark.read
                .option("header", True)
                .option("multiLine", True)
                .option("escape", "\"")
                .option("quote", "\"")
                .csv(f"file://{csv_file_path_or_user_df}/*.csv")
        )
        sampling_ratio = 0.2
    else:
        raw_df = csv_file_path_or_user_df
        sampling_ratio = 1

    # ========== Filtering by Current Patch ==========
    # Drop "Patch Error" rows, repartition & cache
    filtered_by_patch_df = (
        raw_df
            .filter(F.col(MATCH_DATA).isNotNull())
            .filter(~F.col(MATCH_DATA).rlike(PATCH_ERROR_PATTERN))
            .repartition(npartitions)
            .cache()
    )

    # ========== Format DataFrame from camel to snake case ==========
    filtered_by_patch_df = format_df_to_snake(filtered_by_patch_df)

    # ========== Full JSON Schema Inference ==========
    # Build an RDD[String] of all the match_data JSON texts
    json_rdd = (
        filtered_by_patch_df
        .select(MATCH_DATA)
        .filter(F.col(MATCH_DATA).isNotNull())
        .filter(F.length(F.col(MATCH_DATA)) > 10)
        .rdd
        .map(lambda r: r[MATCH_DATA])
        )
    
    # Have Spark read *all* of it as JSON, sampling 100%, so it can build a complete schema
    inferred_schema = (
        spark.read
             .option("multiLine", True)
             .option("samplingRatio", sampling_ratio)
             .json(json_rdd)
             .schema
    )

    # ========== Parsing JSON Dict ==========
    # Parse into nested struct & drop raw text
    parsed_struct_df = (
        filtered_by_patch_df
          .withColumn(f"{MATCH_DATA}_struct", F.from_json(F.col(MATCH_DATA), inferred_schema))
          .drop(MATCH_DATA)
    )

    # ========== Filtering Distinct Ranked Games ==========
    # Keep only queueId == 420 (i.e. ranked games), then drop duplicates on matchId
    if queue_id == 420:
        ranked_matches_df = (
            parsed_struct_df.filter(F.col(f"{MATCH_DATA}_struct.info.queue_id") == queue_id)
        )

    ranked_matches_df = ranked_matches_df.dropDuplicates(["match_id"])

    participants_df = explode_and_flatten_struct(ranked_matches_df, "participants")
    participants_df = participants_df.filter(
            (F.col("team_position").isNotNull()) &
            (F.col("team_position") != "")
    )

    # ========== Explode and Flatten Struct ==========
    teams_df = explode_and_flatten_struct(ranked_matches_df, "teams")

    filtered_by_patch_df.unpersist()
    json_rdd.unpersist()

    return participants_df, teams_df

def get_items_dict_from_s3(
    s3_uri: str,
    s3_client: boto3.client = None
) -> Dict[str, Any]:
    """
    Load item-to-tags mapping JSON dict from S3.

    Parameters
    ----------
    s3_uri: path, e.g. "s3://my-bucket/path/to/item_id_tags.json"
        Path to S3 repository containing desired JSON file
    s3_client: optional boto3 S3 client; if None, one will be created.
        
    Returns:
    dict[str, Any]: python dict object
        A dict mapping item_id (str) to list of tags.
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")
    return json.loads(body)


def map_tags_and_summoner_spells_to_df(
    participants_df: DataFrame,
    items_dict: Dict[str, List[str]],
    summoner_spells_dict: Dict[str, str],
    spark: SparkSession
) -> Tuple[DataFrame, Set[str], Set[str]]:
    """
    Map item tags and summoner spell names to participants dataframe with feature columns.
    
    This function enriches the participants dataframe by mapping item IDs to their associated
    tags and summoner spell IDs to their names. It creates feature columns for each unique
    tag count and binary indicators for each summoner spell, all aimed for aggregation.
    
    Parameters
    ----------
    participants_df : pyspark.sql.DataFrame
        DataFrame containing participant data with columns:
        - item0, item1, ..., item5: Item IDs for each item slot
        - summoner1_id, summoner2_id: Summoner spell IDs
    items_dict : dict[str, list[str]]
        Dictionary mapping item IDs (as strings) to lists of associated tags.
        Example: {"1001": ["Boots", "Movement"], "1004": ["Mana", "Regen"]}
    summoner_spells_dict : dict[str, str]
        Dictionary mapping summoner spell IDs (as strings) to spell names.
        Example: {"4": "flash", "12": "teleport"}
    spark : pyspark.sql.SparkSession
        Active Spark session for creating broadcast DataFrames.
        
    Returns
    -------
    participants_df : pyspark.sql.DataFrame
        Enhanced DataFrame with additional columns:
        - number_of_items_completed: Count of non-empty item slots (0-6)
        - item_tags: Flattened array of all tags from equipped items
        - tag_[{tag_name}]_count: Count of each unique tag across all items
        - summoner_spells_per_game: Array of summoner spell names
        - has_{spell_name}: Binary indicator (0/1) for each summoner spell
    unique_item_tags : set[str]
        Set of all unique item tags found in items_dict.
    unique_summoner_spells : set[str]
        Set of all unique summoner spell names found in summoner_spells_dict.
        
    Examples
    --------
    >>> # Assuming spark session and data are prepared
    >>> items = {"1001": ["Boots"], "3078": ["Health", "Damage"]}
    >>> spells = {"4": "Flash", "12": "Teleport"}
    >>> result_df, tags, spells = map_tags_and_summoner_spells_to_df(
    ...     participants_df, items, spells, spark
    ... )
    >>> result_df.select("tag_[Boots]_count", "has_flash").show()
    
    Notes
    -----
    - The function assumes exactly 6 item slots (item0 through item5).
    - Empty item slots are handled gracefully with empty arrays.
    - Broadcast joins are used for performance optimization.
    - Column names with special characters are wrapped in brackets: tag_[{name}]_count
    """
    NUM_ITEM_SLOTS = 6
    EMPTY_STRING = ""

    # ========== Dict Data Validation ==========
    if not items_dict:
        raise ValueError("items_dict cannot be empty")
    if not summoner_spells_dict:
        raise ValueError("summ_spells_dict cannot be empty")
    
    # ========== Prepare lookup DataFrames ==========
    # Create broadcast df for items_dict and summoner_spells_dict
    items_data = [(int(item_id), tags) for item_id, tags in items_dict.items()]
    items_lookup_df = spark.createDataFrame(items_data, ["item_id", "tags"])
    unique_item_tags = {tag for _, tags in items_dict.items() for tag in tags}
    items_lookup_df = F.broadcast(items_lookup_df)

    summoner_spell_data = [
        (int(summ_spell_id), summ_spell_name) 
        for summ_spell_id, summ_spell_name in summoner_spells_dict.items()
        ]
    summoner_spell_lookup_df = spark.createDataFrame(
        summoner_spell_data, ["summoner_spell_id", "summoner_spell_name"]
        )
    unique_summoner_spells = {
        spell_name for _, spell_name in summoner_spells_dict.items()
        }  
    summoner_spell_lookup_df = F.broadcast(summoner_spell_lookup_df)

    # ========== Process Item Tags ==========
    # Create a new tags column per item and add item count features
    for i in range(NUM_ITEM_SLOTS):
        participants_df = (
            participants_df
            .join( 
                items_lookup_df.withColumnRenamed("tags", f"tags_{i}"), 
                F.col(f"item{i}") == F.col("item_id"), 
                "left" 
            )
            .drop("item_id")
        )
 
    participants_df = (
        participants_df
        # Create column to count completed items
        .withColumn(
            "number_of_items_completed",
            (
            F.when(F.size(F.col("tags_0")) > 0, 1).otherwise(0) +
            F.when(F.size(F.col("tags_1")) > 0, 1).otherwise(0) +
            F.when(F.size(F.col("tags_2")) > 0, 1).otherwise(0) +
            F.when(F.size(F.col("tags_3")) > 0, 1).otherwise(0) +
            F.when(F.size(F.col("tags_4")) > 0, 1).otherwise(0) +
            F.when(F.size(F.col("tags_5")) > 0, 1).otherwise(0)
            )
        )
        # Flatten all item columns into one
        .withColumn(
            "item_tags",
            F.flatten(
                F.array(*[F.coalesce(F.col(f"tags_{i}"), F.array()) 
                for i in range(NUM_ITEM_SLOTS)])
            )
        )
        .drop(*[f"tags_{i}" for i in range(NUM_ITEM_SLOTS)])
    )
    
    # Adding a binary column per tag to facilitate later aggregation
    for tag in unique_item_tags:
        participants_df = participants_df.withColumn(
            f"tag_[{tag}]_count",
            F.size(F.expr(f"filter(item_tags, x -> x = '{tag}')"))
        )

    # ========== Process Summoner Spells ==========
    # Process all summoner spell columns
    participants_df = (
        participants_df
        .join(
            summoner_spell_lookup_df.alias("s1"), # Alias is necessary for the second join
            F.col("summoner1_id") == F.col("s1.summoner_spell_id"),
            "left"
        )
        .join(
            summoner_spell_lookup_df.alias("s2"),
            F.col("summoner2_id") == F.col("s2.summoner_spell_id"),
            "left"
        )
        .withColumn(
            "summoner_spells_per_game",
            F.array(
                F.coalesce(F.col("s1.summoner_spell_name"),
                           F.lit(EMPTY_STRING)),
                F.coalesce(F.col("s2.summoner_spell_name"),
                           F.lit(EMPTY_STRING))
            )
        )
        .drop("s1.summoner_spell_id", "s2.summoner_spell_id",
              "s1.summoner_spell_name", "s2.summoner_spell_name")
    )

    # Adding a binary column per summoner spell for later aggregation
    for summoner_spell in unique_summoner_spells:
        participants_df = participants_df.withColumn(
            f"has_{summoner_spell}", 
            F.array_contains("summoner_spells_per_game",
                             summoner_spell).cast("int")
    )
    
    return participants_df, unique_item_tags, unique_summoner_spells


def derive_participant_dragon_stats(participants_df: DataFrame):
    """Helper function to create dragon statistics columns"""
    # Note: for participants with no dragon takedown in match, the "earliest_dragon_takedown" has a NULL value
    ## For now we will keep it as is because PySpark ignore nulls and since these columns are used for summation statistics it doesn't matter
    drag_takedown = F.col("challenges.earliest_dragon_takedown")
    
    participants_df_with_dragon_stats = (
        participants_df
        .withColumn("had_dragon_takedown", drag_takedown.isNotNull().cast("int"))
        # Flag drag_takedown between 5 and 7 minutes (i.e. 301–420 seconds)
        .withColumn(
            "first_drag_takedown_min_5_to_7",
            drag_takedown.between(301, 420).cast("int")
        )
        .withColumn(
            "first_drag_takedown_min_7_to_11",
            drag_takedown.between(421, 660).cast("int")
        )
        .withColumn(
            "first_drag_takedown_min_11_to_15",       
            drag_takedown.between(661, 900).cast("int")
        )
        .withColumn(
            "first_drag_takedown_min_15+",
            F.when((drag_takedown > 900), F.lit(1))
        )
    )

    return participants_df_with_dragon_stats


def extract_fields_with_exclusions(
    df: DataFrame,
    position_specific_fields: dict = POSITION_SPECIFIC_METRICS,
    laner_specific_fields: dict = LANER_SPECIFIC_METRICS,
    all_positions: list = None,
    position_column: str = "team_position",
    match_data_column: str = "match_data"
) -> DataFrame:
    """
    Extracts fields with position-specific logic and exclusions.
    """
    if all_positions is None:
        all_positions = ["JUNGLE", "MID", "BOTTOM", "UTILITY", "TOP"]
    
    # Collect all operations
    operations = []
    
    # Add position-specific fields
    for position, field_mappings in position_specific_fields.items():
        for new_col, field_path in field_mappings.items():
            operations.append({
                "new_col": new_col,
                "condition": F.col(position_column) == position,
                "field_path": field_path
            })
    
    # Add laner specific fields
    non_jungle_positions = [pos for pos in all_positions if pos != "JUNGLE"]
    
    for new_col, field_path in laner_specific_fields.items():
        for position in non_jungle_positions:
            position_col_name = new_col
            operations.append({
                "new_col": position_col_name,
                "condition": F.col(position_column) == position,
                "field_path": field_path
            })
    
    # Create all columns in one select - WITH NULL SAFETY
    select_exprs = [df["*"]] + [
        F.when(
            op["condition"],
            # Use coalesce to handle missing fields - returns 0 if field doesn't exist
            F.coalesce(
                F.col(f"{match_data_column}.{op['field_path']}"),
                F.lit(0)
            )
        ).otherwise(0).alias(op["new_col"])
        for op in operations
    ]
    
    # Use try-except to handle schema mismatches
    try:
        return df.select(*select_exprs)
    except Exception as e:
        print(f"Schema issue detected: {str(e)}")
        # Fallback: Add columns one by one with error handling
        result_df = df
        for op in operations:
            try:
                result_df = result_df.withColumn(
                    op["new_col"],
                    F.when(
                        op["condition"],
                        F.coalesce(
                            F.col(f"{match_data_column}.{op['field_path']}"),
                            F.lit(0)
                        )
                    ).otherwise(0)
                )
            except:
                # If field doesn't exist at all, just create column with 0
                print(f"Field {op['field_path']} not found, creating column with default 0")
                result_df = result_df.withColumn(
                    op["new_col"],
                    F.when(op["condition"], F.lit(0)).otherwise(0)
                )
        return result_df


def aggregate_champion_data(merged_df, all_item_tags, all_summoner_spells, granularity="champion_x_role", single_user_puuid = None): # Add granularity input
    if granularity == "champion_x_role":
        grouping = ["champion_id", "champion_name","team_position"]
        window = Window.partitionBy("champion_id")
    elif granularity == "champion_x_role_x_user":
        grouping = ["champion_id", "champion_name","team_position", "puuid"]
        window = Window.partitionBy("puuid", "champion_id")
    elif granularity == "single_user":
        merged_df = merged_df.filter(merged_df.puuid == single_user_puuid)
        grouping = ["champion_id", "champion_name","team_position"]
        window = Window.partitionBy("champion_id")
    else:
        raise ValueError("Incorrect granularity input, must be 'champion_x_role', 'champion_x_role_x_user' or 'single_user'")        

    intermediate_df = (
        merged_df
        .withColumn("total_games_per_champion", F.count("*").over(window))
        .withColumn("fully_stacked_mejais", F.when(F.col("challenges.mejais_full_stack_in_time") > 0, 1).otherwise(0))
        .groupBy(*grouping)
        # moreEnemyJungleThanOpponent -> As a jungler, before 10 minutes, take more of the opponent's jungle than them
        .agg(
            F.count("*").alias("total_games_played_in_role"),
            F.first("total_games_per_champion").alias("total_games_per_champion"), # Used for filtering minimum games in role
            
            # Core stats (KDA and number of games)
            F.avg("kills").alias("avg_kills"),
            F.avg("deaths").alias("avg_deaths"),
            F.avg("challenges.deaths_by_enemy_champs").alias("avg_deaths_by_enemy_champs"),
            F.avg("assists").alias("avg_assists"),
            F.avg("challenges.kill_participation").alias("avg_kill_participation"),
            F.avg("challenges.takedowns").alias("avg_takedowns"),
            F.sum(F.col("win").cast("int") * 100).alias("total_wins"), # Original column is boolean, need to cast to int before summing

            # Damage dealt stats
            F.sum("challenges.highest_champion_damage").alias("pct_of_games_with_highest_damage_dealt"), # Need to stay as sum and derive later due to NULLS
            F.avg("challenges.team_damage_percentage").alias("avg_pct_damage_dealt_in_team"),
            F.avg("challenges.damage_per_minute").alias("average_damage_per_minute"),
            F.avg("total_damage_dealt_to_champions").alias("avg_damage_dealt_to_champions"),
            F.avg("total_damage_dealt").alias("avg_total_damage_dealt"),
            F.avg("magic_damage_dealt_to_champions").alias("avg_magic_damage_dealt_to_champions"),
            F.avg("magic_damage_dealt").alias("avg_total_magic_damage_dealt"),
            F.avg("physical_damage_dealt_to_champions").alias("avg_physical_damage_dealt_to_champions"),
            F.avg("physical_damage_dealt").alias("avg_total_physical_damage_dealt"),
            F.avg("true_damage_dealt_to_champions").alias("avg_true_damage_dealt_to_champions"),
            F.avg("true_damage_dealt").alias("avg_total_true_damage_dealt"),
            F.avg("largest_critical_strike").alias("avg_largest_critical_strike"),

            # Damage Taken
            F.avg("challenges.damage_taken_on_team_percentage").alias("avg_pct_damage_taken_in_team"),
            F.avg("total_damage_taken").alias("avg_damage_taken"),
            F.avg("magic_damage_taken").alias("avg_magic_damage_taken"),
            F.avg("physical_damage_taken").alias("avg_physical_damage_taken"),
            F.avg("true_damage_taken").alias("avg_true_damage_taken"),
            F.avg("damage_self_mitigated").alias("avg_damage_self_mitigated"),
            # Situational Damage Taken (e.g. in teamfights)
            F.avg("challenges.killed_champ_took_full_team_damage_survived").alias("avg_times_killed_champ_took_full_team_damage_and_survived"), # DATA VALIDATION: check if always zero
            F.avg("challenges.survived_single_digit_hp_count").alias("avg_times_survived_single_digit_hp"),
            F.avg("challenges.survived_three_immobilizes_in_fight").alias("avg_times_survived_three_immobilizes_in_fight"),
            F.avg("challenges.took_large_damage_survived").alias("avg_times_took_large_damage_survived"),

            # Crowd Control         
            F.sum("challenges.highest_crowd_control_score").alias("pct_of_games_with_highest_crowd_control_score"), # Need to stay as sum and derive later due to NULLS
            F.avg("time_c_cing_others").alias("avg_time_ccing_others"), # Measure of time
            F.avg("total_time_cc_dealt").alias("avg_times_applied_cc_on_others"), # Number of times cc'd others
            F.avg("challenges.enemy_champion_immobilizations").alias("avg_enemy_champion_immobilizations"),

            # Healing and Shielding + Support
            F.avg("total_heal").alias("avg_total_healing"),
            F.avg("total_heals_on_teammates").alias("avg_heals_on_teammate"),
            F.avg("total_units_healed").alias("avg_total_units_healed"),
            F.avg('total_damage_shielded_on_teammates').alias("avg_dmg_shielded_on_team"),
            F.avg("challenges.effective_heal_and_shielding").alias("avg_effective_heal_and_shield"),
            F.sum(F.col("challenges.faster_support_quest_completion").cast("int")).alias("total_games_completed_supp_quest_first"), # Need to stay as sum and derive later due to NULLS
            F.avg("complete_support_quest_in_time").alias("avg_supp_quest_completion_time"),

            # Misc
            F.avg("longest_time_spent_living").alias("avg_longest_time_spent_alive"),
            F.avg("total_time_spent_dead").alias("avg_time_spent_dead"),

            # Spell Casts
            F.avg("spell1_casts").alias("avg_spell1_casts"),
            F.avg("spell2_casts").alias("avg_spell2_casts"),
            F.avg("spell3_casts").alias("avg_spell3_casts"),
            F.avg("spell4_casts").alias("avg_spell4_casts"),
            F.avg("challenges.ability_uses").alias("avg_ability_uses"),

            # Skillshot related (dodging and hitting)
            F.avg("challenges.dodge_skill_shots_small_window").alias("avg_times_dodged_skillshot_in_small_window"),
            F.avg("challenges.skillshots_dodged").alias("avg_skillshots_dodged"),
            F.avg("challenges.land_skill_shots_early_game").alias("avg_skillshots_landed_early_game"),
            F.avg("challenges.skillshots_hit").alias("avg_skillshots_hit"),

            # Picks
            F.avg("challenges.immobilize_and_kill_with_ally").alias("avg_times_immobilize_and_kill_with_ally"),
            F.avg("challenges.kill_after_hidden_with_ally").alias("avg_times_got_kill_after_hidden_with_ally"),
            F.avg("challenges.pick_kill_with_ally").alias("avg_times_pick_kill_with_ally"),
            F.avg("challenges.knock_enemy_into_team_and_kill").alias("avg_times_knock_enemy_into_team_and_kill"),

            # Kills under or near turret
            F.avg("challenges.kills_near_enemy_turret").alias("avg_kills_near_enemy_turret"),
            F.avg("challenges.kills_under_own_turret").alias("avg_kills_under_own_turret"),

            # Misc mechanics
            F.avg("challenges.multikills_after_aggressive_flash").alias("avg_multikills_after_aggressive_flash"),
            F.avg("challenges.outnumbered_kills").alias("avg_outnumbered_kills"),
            F.avg("challenges.outnumbered_nexus_kill").alias("avg_times_outnumbered_nexus_kill"),
            F.avg("challenges.quick_cleanse").alias("avg_times_quick_cleanse"),

            # Misc laning
                # Kills, takedowns and plays 
            F.avg("challenges.quick_solo_kills").alias("avg_quick_solo_kills"),
            F.avg("challenges.solo_kills").alias("avg_solo_kills"),
            F.avg("challenges.takedowns_after_gaining_level_advantage").alias("avg_takedowns_after_gaining_lvl_advantage"),
            F.avg("kills_on_other_lanes_early_as_laner").alias("avg_kills_on_other_lanes_early_as_laner"), # As a laner, in a single game, get kills before 10 minutes outside your lane (anyone but your lane opponent)
            F.avg("challenges.save_ally_from_death").alias("avg_times_save_ally_from_death"),
            F.avg("challenges.takedowns_in_alcove").alias("avg_takedowns_in_alcove"),
                # First blood and early kills
            F.avg(F.col("first_blood_kill").cast("int") * 100).alias("pct_of_games_first_blood_kill"),
            F.avg(F.col("first_blood_assist").cast("int") * 100).alias("pct_of_games_first_blood_assist"),
            F.avg("challenges.takedowns_before_jungle_minion_spawn").alias("avg_takedowns_before_jungle_camps_spawn"), # Get takedowns on enemy champions before jungle camps spawn (1:30)
            F.avg("challenges.takedowns_first_x_minutes").alias("avg_first_takedown_time"),

            # Summoner Spells
            F.avg("summoner1_casts").alias("avg_summoner_spell1_casts_per_game"),
            F.avg("summoner2_casts").alias("avg_summoner_spell2_casts_per_game"),
            *[(F.avg(f"has_{summoner_spell}") * 100).alias(f"pct_of_matches_with_{summoner_spell}") for summoner_spell in all_summoner_spells],

            # Experience and Gold
            F.avg("champ_experience").alias("avg_champ_exp_at_game_end"),
            F.avg("champ_level").alias("avg_champ_level_at_game_end"),
            F.avg("gold_earned").alias("avg_gold_earned_per_game"),
            F.avg("challenges.gold_per_minute").alias("avg_gold_per_minute"),
            F.avg("gold_spent").alias("avg_gold_spent"),
            F.avg("bounty_level").alias("avg_bounty_lvl"),
            F.avg("challenges.bounty_gold").alias("avg_bounty_gold"),
                # Laning Specific   
            F.avg(F.col("challenges.early_laning_phase_gold_exp_advantage") * 100).alias("pct_of_games_with_early_lane_phase_gold_exp_adv"), # Seems to only give "1" to a single player per match: End the early laning phase (7 minutes) with 20% more gold and experience than your role opponent on Summoner's
            F.avg(F.col("challenges.laning_phase_gold_exp_advantage") * 100).alias("pct_of_games_with_lanephase_gold_exp_adv"), # End the laning phase (14 minutes) with 20% more gold and experience than your role opponent 
            F.avg("challenges.max_level_lead_lane_opponent").alias("avg_max_level_lead_over_lane_opp"),
                # Minions Specific
            F.avg("total_minions_killed").alias("avg_minions_killed"),
            F.avg("challenges.lane_minions_first10_minutes").alias("avg_minions_killed_by_10_mins"),
            F.avg("challenges.max_cs_advantage_on_lane_opponent").alias("avg_max_cs_lead_over_lane_opponent"),
                # Item Purchases
            F.avg("consumables_purchased").alias("avg_consumables_purchased"),
            F.avg("items_purchased").alias("avg_number_of_items_purchased"),
            F.sum(F.col("challenges.fastest_legendary").isNotNull().cast("int")).alias("total_games_fastest_item_completion"), # DATA VALIDATION: Maybe do NULLS for non supports? - likely earlier in code / also, seems to be 1 per team, check

            # Item Tags
            F.avg("number_of_items_completed").alias("avg_items_completed"),
            *[
                F.avg(f"tag_[{tag}]_count").alias(f"avg_{tag}_count")
                for tag in all_item_tags
            ],

            # Jungle related stats
                # Jungle farm
            F.avg("total_ally_jungle_minions_killed").alias("avg_ally_jungle_minions_killed"), # Don'ty necessarily add up to total CS (might be counting a buff as 1 cs for example), use as standalone jungle farm distribution stat (ally jg vs invade, etc)
            F.avg("total_enemy_jungle_minions_killed").alias("avg_enemy_jungle_minions_killed"),
            F.avg("more_enemy_jungle_cs_than_opponent_as_jungler").alias("avg_enemy_jungle_cs_differential_early"), # As a jungler, before 10 minutes, take more of the opponent's jungle than them
            F.avg("neutral_minions_killed").alias("avg_jungle_monsters_cs"), # Jungle monsters/farm - note that this shows for all players
            F.avg("challenges.buffs_stolen").alias("avg_buffs_stolen"),
            F.avg("challenges.initial_buff_count").alias("avg_initial_buff_count"), # Decided not to add NULLs to non junglers for now
            F.avg("challenges.epic_monster_kills_within30_seconds_of_spawn").alias("avg_epic_monster_kills_within_30s_of_spawn"),
            F.avg("challenges.initial_crab_count").alias("avg_initial_crab_count"), # Decided not to add NULLs to non junglers for now
            F.avg("challenges.scuttle_crab_kills").alias("avg_crabs_per_game"),
            F.avg("challenges.jungle_cs_before10_minutes").alias("avg_jg_cs_before_10m"), # Decided not to add NULLs to non junglers for now
                # Jungle Combat
            F.avg("jungler_kills_early_jungle").alias("avg_jungler_kills_early_jungle"), # As a jungler, get kills on the enemy jungler in their own jungle before 10 minutes
            F.avg("kills_on_laners_early_jungle_as_jungler").alias("avg_jungler_early_kills_on_laners"), # As a jungler, get kills on top lane, mid lane, bot lane, or support players before 10 minutes
            F.avg("takedowns_in_all_lanes_early_as_laner").alias("avg_times_had_early_takedowns_in_all_lanes_as_laner"), # As a laner, get a takedown in all three lanes within 10 minutes
            F.avg("challenges.jungler_takedowns_near_damaged_epic_monster").alias("avg_jungler_takedowns_near_damaged_epic_monsters"), # Take down junglers near a damaged Epic Monster before it is killed. Epic Monsters include Dragons, the Rift Herald, and Baron Nashor.
            F.avg("challenges.kills_with_help_from_epic_monster").alias("avg_kills_with_help_from_epic_monster"),
            
            # Vision Stats
                # Vision score and wards placed + unseen recalls
            F.avg("vision_score").alias("avg_vision_score"),
            F.avg("challenges.vision_score_per_minute").alias("avg_vision_score_per_min"),
            F.avg("challenges.vision_score_advantage_lane_opponent").alias("avg_vision_score_advantage_over_lane_opponent"),
            F.avg("challenges.stealth_wards_placed").alias("avg_stealth_wards_placed"),
            F.avg("wards_placed").alias("avg_wards_placed"), # DATA VALIDATION: Need to check if duplicate of above
            F.avg("challenges.wards_guarded").alias("avg_wards_guarded"),
            F.avg("detector_wards_placed").alias("avg_control_wards_placed"), # Same as control wards
            F.avg("challenges.control_ward_time_coverage_in_river_or_enemy_half").alias("avg_control_ward_time_coverage_in_river_or_enemy_half"),
            F.avg("challenges.unseen_recalls").alias("avg_unseen_recalls"),
                # Wards killed
            F.sum(F.col("challenges.highest_ward_kills").cast("int")).alias("pct_of_games_with_highest_wards_killed"), # Need to stay as sum and derive later due to NULLS
            F.avg("wards_killed").alias("avg_wards_killed"),
            F.avg("challenges.ward_takedowns").alias("avg_ward_takedowns"),
            F.avg("challenges.ward_takedowns_before20_m").alias("avg_ward_takedowns_before_20m"),
            F.avg("challenges.two_wards_one_sweeper_count").alias("avg_times_2_wards_killed_with_1_sweeper"),
            F.avg("vision_wards_bought_in_game").alias("avg_control_wards_bought"),

            # Teamwide stats (mostly from teams_df with some from participants_df)
                # First objective rates
            F.avg(F.col("objectives.baron.first").cast("int") * 100).alias("pct_of_games_team_took_first_baron"),
            F.avg("challenges.earliest_baron").alias("avg_earliest_baron_by_team_time"),
            F.avg(F.col("objectives.dragon.first").cast("int") * 100).alias("pct_of_games_team_took_first_dragon"),
            F.avg(F.col("objectives.inhibitor.first").cast("int") * 100).alias("pct_of_games_team_took_first_inhib"),
            F.avg(F.col("objectives.rift_herald.first").cast("int") * 100).alias("pct_of_games_team_took_first_herald"),
            F.avg(F.col("objectives.tower.first").cast("int") * 100).alias("pct_of_games_team_took_first_turret"),
                # Team Objectives
            F.avg("objectives.baron.kills").alias("avg_baron_kills_by_team"),
            F.avg("objectives.rift_herald.kills").alias("avg_herald_kills_by_team"),
            F.avg("objectives.dragon.kills").alias("avg_dragon_kills_by_team"),
            F.avg(F.col("challenges.perfect_dragon_souls_taken") * 100).alias("pct_of_games_with_perfect_drag_soul_taken"),
            F.avg("challenges.team_elder_dragon_kills").alias("avg_elder_dragon_kills_by_team"),
            F.avg("challenges.elder_dragon_kills_with_opposing_soul").alias("avg_elder_dragon_kills_w_opposing_soul"),
                # Team Structures
            F.avg("objectives.inhibitor.kills").alias("avg_inhib_kills_by_team"),
            F.avg("objectives.tower.kills").alias("avg_tower_kills_by_team"),
            F.avg("inhibitors_lost").alias("avg_inhibs_lost_by_team"),
            F.avg(F.col("nexus_lost") * 100).alias("pct_of_games_with_nexus_lost_by_team"),
            F.avg("turrets_lost").alias("avg_turrets_lost_by_team"),
            F.avg(F.col("challenges.first_turret_killed").cast("int") * 100).alias("pct_of_games_first_turret_taken_by_team"),
            F.avg("challenges.first_turret_killed_time").alias("avg_first_turret_kill_time_by_team"),
                # Team Kills
            F.avg("objectives.champion.kills").alias("avg_total_team_champ_kills"),
            F.avg("challenges.aces_before15_minutes").alias("avg_team_aces_before_15_by_team"),
            F.avg("challenges.flawless_aces").alias("avg_flawless_aces_by_team"),
            F.avg("challenges.shortest_time_to_ace_from_first_takedown").alias("avg_shortest_time_to_ace_from_1st_takedown"),
            F.avg("challenges.max_kill_deficit").alias("avg_max_kill_deficit"), # DATA VALIDATION: CHECK ? see if always zero values
            F.avg(F.col("challenges.perfect_game").cast("int") * 100).alias("pct_of_games_that_are_perfect_games"),

            # Individual participant damage to structures
                # Damage dealt to structures
            F.avg("damage_dealt_to_buildings").alias("avg_indiv_dmg_dealt_to_buildings"),
            F.avg("damage_dealt_to_turrets").alias("avg_indiv_dmg_dealth_to_turrets"),
            F.avg("challenges.turret_plates_taken").alias("avg_indiv_turret_plates_taken"),
                # First tower
            F.avg(F.col("first_tower_kill").cast("int") * 100).alias("pct_of_games_indiv_killed_1st_tower"), # Boolean for champ who took first tower
            F.avg(F.col("challenges.takedown_on_first_turret").cast("int") * 100).alias("pct_of_games_individual_takedown_1st_tower"), # Boolean value, states whether participant had takedown on first turret
            F.avg(F.col("challenges.quick_first_turret").cast("int") * 100).alias("pct_of_games_individual_took_1st_tower_quick"), # DATA VALIDATION: Boolean value, we don't know what quick means here in terms of time
            F.avg(F.col("first_tower_assist").cast("int") * 100).alias("pct_of_games_individual_had_1st_turret_assist"),
                # Turrets kills/takedowns
            F.avg("challenges.k_turrets_destroyed_before_plates_fall").alias("avg_turrets_killed_before_plates_fell"),
            F.avg("turret_kills").alias("avg_individual_tower_kills"),
            F.avg("turret_takedowns").alias("avg_individual_tower_takedowns"),
            F.avg("challenges.turret_takedowns").alias("avg_individual_tower_takedowns2"), # DATA VALIDATION: Compare with above
            F.avg("challenges.solo_turrets_lategame").alias("avg_individual_solo_towers_kills_late_game"), # Destroy side lane turrets solo (majority damage dealt by you) after 14 minutes without dying
            F.avg("challenges.turrets_taken_with_rift_herald").alias("avg_indiv_towers_taken_w_rift_herald"),
            F.avg("challenges.multi_turret_rift_herald_count").alias("avg_indiv_multi_towers_taken_w_rift_herald"),
                # Inhibitor and nexus kills/takedowns + misc
            F.avg("inhibitor_kills").alias("avg_individual_inhibitor_kills"),
            F.avg("inhibitor_takedowns").alias("avg_individual_inhibitor_takedowns"),
            F.avg(F.col("nexus_kills") * 100).alias("pct_of_games_individual_killed_nexus"),
            F.avg(F.col("nexus_takedowns") * 100).alias("avg_individual_nexus_takedowns"),
            F.avg(F.col("challenges.had_open_nexus") * 100).alias("pct_of_games_with_open_nexus"),

            # Individual participant objectives
                # Objective kills/takedowns
            F.avg("damage_dealt_to_objectives").alias("avg_individual_dmg_dealt_to_objectives"),
            F.avg("baron_kills").alias("avg_individual_baron_kills"),
            F.avg("challenges.solo_baron_kills").alias("avg_individual_solo_baron_kills"),
            F.avg("challenges.baron_takedowns").alias("avg_individual_baron_takedowns"),
            F.avg("dragon_kills").alias("avg_individual_dragon_kills"),
            F.avg("challenges.dragon_takedowns").alias("avg_individual_dragon_takedowns"),
            F.avg("challenges.rift_herald_takedowns").alias("avg_individual_rift_herald_takedowns"),
            F.avg("challenges.void_monster_kill").alias("avg_individual_void_monster_kills"), # DATA VALIDATION: Void grubs? I have seen this have a valye of 7 when the champion did NOT have a rift herald takedown
                # Objective steals
            F.avg("objectives_stolen").alias("avg_objectives_stolen"),
            F.avg("objectives_stolen_assists").alias("avg_objectives_stolen_assists"),
            F.avg("challenges.epic_monster_steals").alias("avg_epic_monster_steals"), # DATA VALIDATION: Check if duplicate of the one above
            F.avg("challenges.epic_monster_stolen_without_smite").alias("avg_epic_monster_steals_without_smite"), # Steal an Epic jungle monster without using Summoner Smite. Epic Monsters include Dragons, the Rift Herald, and Baron Nashor. 
            F.avg("challenges.epic_monster_kills_near_enemy_jungler").alias("avg_epic_monsters_killed_near_enemy_jgler"), # Secure Epic Monsters with the enemy jungler nearby. Epic Monsters include Dragons, the Rift Herald, and Baron Nashor.
                # Earliest dragon takedown stats (used for derived stats)
            F.avg("challenges.earliest_dragon_takedown").alias("avg_earliest_drag_takedown"),
            F.avg(F.col("had_dragon_takedown") * 100).alias("pct_of_games_had_drag_takedown"),
            F.avg(F.col("first_drag_takedown_min_5_to_7") * 100).alias("pct_of_games_had_drag_takedown_min_5_to_7"), 
            F.avg(F.col("first_drag_takedown_min_7_to_11") * 100).alias("pct_of_games_had_drag_takedown_min_7_to_11"),
            F.avg(F.col("first_drag_takedown_min_11_to_15") * 100).alias("pct_of_games_had_drag_takedown_min_11_to_15"),
            F.avg(F.col("first_drag_takedown_min_15+") * 100).alias("pct_of_games_had_drag_takedown_min_15_plus"),

            # Game length related
            (F.avg("time_played") / 60).alias("avg_time_played_per_game_minutes"),
            F.avg("challenges.game_length").alias("avg_game_length"), # DATA VALIDATION: Compare with above and delete one
            F.avg(F.col("game_ended_in_early_surrender").cast("int") * 100).alias("pct_of_games_ended_in_early_ff"),
            F.avg(F.col("game_ended_in_surrender").cast("int") * 100).alias("pct_of_games_ended_in_ff"),
            F.avg(F.col("team_early_surrendered").cast("int") * 100).alias("pct_of_games_team_ffd"),

            # Multikill and killing spree stats
                # Multikills
            F.avg("double_kills").alias("avg_doublekills"),
            F.avg("triple_kills").alias("avg_triplekills"),
            F.avg("quadra_kills").alias("avg_quadrakills"),
            F.avg("penta_kills").alias("avg_pentakills"),
            F.avg("largest_multi_kill").alias("avg_largest_multikill"),
            F.avg("challenges.multikills").alias("avg_number_of_multikills"),
            F.avg("challenges.multi_kill_one_spell").alias("avg_multikills_with_one_spell"),
                # Killing sprees stats
            F.avg("killing_sprees").alias("avg_killing_sprees"),
            F.avg("challenges.killing_sprees").alias("avg_killing_sprees2"), # DATA VALIDATION: Check if equal and delete one
            F.avg("challenges.legendary_count").alias("avg_legendary_count"), # DATA VALIDATION: Not too sure what this is, my current guess is how many times champ was legendary (do they need to die and get leg again or it counts kills above leg?)
            F.avg("largest_killing_spree").alias("avg_largest_killing_spee"),
                # Misc
            F.avg("unreal_kills").alias("avg_unreal_kills"), # DATA VALIDATION: Might be a zero stat, need to check
            F.avg("challenges.12_assist_streak_count").alias("avg_12_assist_streaks"),
            F.avg("challenges.elder_dragon_multikills").alias("avg_elder_drag_multikills"),
            F.avg("challenges.full_team_takedown").alias("avg_full_team_takedowns"),

            # Items - decide whether to use data from match or from timeline and make sure to indluce item tags in df 

            # Misc
            F.avg("challenges.blast_cone_opposite_opponent_count").alias("avg_times_blast_cone_enemy"),
            F.avg(F.col("challenges.danced_with_rift_herald") * 100).alias("pct_of_games_danced_with_rift_herald"),
            F.avg("challenges.double_aces").alias("avg_double_aces"),
            F.avg("challenges.fist_bump_participation").alias("avg_fist_bump_participations"),
            F.avg("fully_stacked_mejais").alias("percent_of_games_with_fully_stacked_mejais"),
            
            F.avg(
                F.when(F.col("challenges.mejais_full_stack_in_time") != 0,
                       F.col("challenges.mejais_full_stack_in_time") != 0).cast("int")
            ).alias("avg_mejai_full_stack_time"),

            F.avg("challenges.outer_turret_executes_before10_minutes").alias("avg_outer_turret_executes_before_10m"),
            F.avg("challenges.takedowns_in_enemy_fountain").alias("avg_takedowns_in_enemy_fountain"),

            # Position/Role - find aggregation key word for most common string -> DATA VALIDATION
            F.sort_array(
                F.collect_list("individual_position")
            )[0].alias("mode_individual_position"),

            F.sort_array(
                F.collect_list("lane")
            )[0].alias("mode_lane"),

            F.sort_array(
                F.collect_list("role")
            )[0].alias("mode_role"),

            #F.sort_array(
                #F.collect_list("team_position")
            #)[0].alias("mode_team_position"),

            # CONSIDER EXPLORATORY FEATURE ANALYSIS TO EVALUATE

            ##### Had to use alternate methor as SageMaker does not support Spark 3.4, which is where F.mode was added
            #F.mode("individual_position").alias("mode_individual_position"), # Best guess for which position the player actually played in isolation of anything else
            #F.mode("lane").alias("mode_lane"), # Gives slightly different string than above, might have something to do with where champ spent most of the time
            #F.mode("role").alias("mode_role"),
            #F.mode("team_position").alias("mode_team_position"), # The teamPosition is the best guess for which position the player actually played if we add the constraint that each team must have one top player, one jungle, one middle, etc
            
            F.sum(F.col("challenges.played_champ_select_position").cast("int")).alias("pct_of_games_played_champ_select_position") # only present if no roleswap - seems useless
        )
        .withColumn(
            "role_play_rate",
            (F.col("total_games_played_in_role") * 100 / F.col("total_games_per_champion"))
        )
        # Converting "sum" columns into percentages (couldn't do it in one step due to NULLS)
        .withColumn(
            "pct_of_games_with_highest_damage_dealt",
            (F.col("pct_of_games_with_highest_damage_dealt") * 100 / F.col("total_games_played_in_role"))
        )
        .withColumn(
            "pct_of_games_with_highest_crowd_control_score",
            (F.col("pct_of_games_with_highest_crowd_control_score") * 100 / F.col("total_games_played_in_role"))
        )
        .withColumn(
            "total_games_completed_supp_quest_first",
            (F.col("total_games_completed_supp_quest_first")* 100 / F.col("total_games_played_in_role"))
        )
        .withColumn(
            "pct_of_games_with_highest_wards_killed",
            (F.col("pct_of_games_with_highest_wards_killed")* 100 / F.col("total_games_played_in_role"))
        )
        # Core derived stats
        .withColumn(
            "kda",
            ((F.col("avg_kills") + F.col("avg_assists")) / F.col("avg_deaths"))
        )
        .withColumn(
            "win_rate",
            (F.col("total_wins") * 100 / F.col("total_games_played_in_role"))
        )
        .withColumn(
            "avg_cs",
            (F.col("avg_minions_killed") + F.col("avg_jungle_monsters_cs"))
        )
        .withColumn(
            "avg_cs_per_minute",
            (F.col("avg_cs") / F.col("avg_time_played_per_game_minutes"))
        )
        # Misc?
        .withColumn(
            "pct_games_first_to_complete_item",
            (F.col("total_games_fastest_item_completion") * 100 / F.col("total_games_played_in_role"))
        )
    )

    final_df = intermediate_df
    for tag in all_item_tags:
        final_df = final_df.withColumn(
            f"pct_of_matches_with_{tag}",
            F.col(f"avg_{tag}_count") * 100 / F.col("avg_items_completed")
        ).drop(f"avg_{tag}_count")
    
    return final_df
        

def derive_counter_stats(
        raw_counter_stats_df: DataFrame, desired_team_positions: list = ALL_TEAM_POSITIONS, min_games: int = 10
) -> dict[str, DataFrame]:
        
    counter_stats_dfs_by_role = {}

    for role in desired_team_positions:

        role_filtered_df = (raw_counter_stats_df
                            .filter(F.col("team_position") == F.lit(role))
                            .select("match_id", "team_id", "champion_name", "win")
                            .dropDuplicates(["match_id","team_id","champion_name"]))

        if not role_filtered_df.take(1):
            raise ValueError(f"No rows found for team_position={role}")

        role_filtered_df = role_filtered_df.withColumn("win", F.col("win").cast("int"))

        # Self-join by match to pair opponents
        a = role_filtered_df.alias("a")
        b = (role_filtered_df
                .select(
                    F.col("match_id"),
                    F.col("team_id").alias("opp_team_id"),
                    F.col("champion_name").alias("opp_champion_name")
                )
                .alias("b"))
        
        pairs = (
            a.join(b, on="match_id", how="inner")
            .where(F.col("a.team_id") != F.col("b.opp_team_id")) 
        )

        # Aggregate to wins/games for champion and opponent
        agg = (pairs.groupBy(
                            F.col("a.champion_name").alias("champion_name"),
                            F.col("b.opp_champion_name").alias("opp_champion_name"))
                    .agg(F.count(F.lit(1)).alias("number_of_games"),
                         F.sum(F.col("a.win")).alias("wins"))
        )
    
        agg = agg.withColumn(
            "win_rate",
            F.when(F.col("number_of_games") >= F.lit(min_games),
                   (F.col("wins").cast("double") / F.col("number_of_games")) * F.lit(100.00)
            ).otherwise(F.lit(None).cast("double"))
        )

        counter_stats_dfs_by_role[role] = agg.select("champion_name", "opp_champion_name",
                                                  "number_of_games", "wins", "win_rate")

    return counter_stats_dfs_by_role


"""
Main aggregating functions, uses helpers as needed, idea is to pull all wanted keys from match_data struct into columns (some keys will be modified, derived)
Finally, match_data will be dropped
"""
def main_aggregator(
    spark: SparkSession,
    csv_file_path_or_user_df: str,
    items_json_path: str,
    single_user_flag: bool = False,
    single_user_puuid: str = None,
    desired_queue_id: int = 420
) -> DataFrame:
    
    participants_df, teams_df = create_matches_df(
        spark=spark, csv_file_path_or_user_df=csv_file_path_or_user_df,
        npartitions=DEFAULT_PARTITIONS, queue_id=desired_queue_id, 
        single_user_flag=single_user_flag
    )

    # Create an indicator column for games where champion had a dragon takedown, and subsequent columns with the timing of first dragon takedown
    participants_df = derive_participant_dragon_stats(participants_df)

    # Load JSON from local file instead of S3
    with open(items_json_path, "r") as f:
        items_dict = json.loads(f.read())

    participants_df, all_item_tags, all_summoner_spells = map_tags_and_summoner_spells_to_df(
        participants_df, 
        items_dict, 
        SUMMONER_SPELLS_DICT, 
        spark
    )

    participants_df = extract_fields_with_exclusions(participants_df)

    merged_df = participants_df.join(
        teams_df.drop("win"), # Remove column that appears in both DataFrames
        on = ["match_id", "team_id"],
        how = "left"
    )

    if single_user_flag:
        single_user_df = aggregate_champion_data(
            merged_df, 
            all_item_tags, 
            all_summoner_spells, 
            "single_user",
            single_user_puuid
        )

        return single_user_df

    else:
        counter_stats_cols = ["match_id", "team_position", "champion_name", "win", "team_id"]
        raw_counter_stats_df = merged_df[counter_stats_cols].drop_duplicates(subset=counter_stats_cols)
        counter_stats_dfs_by_role = derive_counter_stats(raw_counter_stats_df)

        champion_x_role_df = aggregate_champion_data(
            merged_df, 
            all_item_tags, 
            all_summoner_spells, 
            granularity="champion_x_role"
        )
        champion_x_role_x_user_df = aggregate_champion_data(
            merged_df, 
            all_item_tags, 
            all_summoner_spells, 
            granularity="champion_x_role_x_user"
        )

        return champion_x_role_df, champion_x_role_x_user_df, counter_stats_dfs_by_role