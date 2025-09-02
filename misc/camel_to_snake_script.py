import os, shutil, re, json, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

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

snake_json = udf(safe_snake_json, StringType())

def format_to_snake():
    # stop any old session
    try: spark.stop()
    except: pass

    spark = (
    SparkSession.builder
        .appName("CleanRawMatchData")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )

    INPUT_PATH_LARGE  = "C:/Users/17862/Desktop/SnexCode/lol-helper/csv_files/batches_patch_15_6"
    OUTPUT_PATH_LARGE = "C:/Users/17862/Desktop/SnexCode/lol-helper/csv_files/formatted_15_6"
    os.makedirs(OUTPUT_PATH_LARGE, exist_ok=True)

    for fname in os.listdir(INPUT_PATH_LARGE):
        if not fname.lower().endswith(".csv"):
            continue

        in_path = os.path.join(INPUT_PATH_LARGE, fname)
        base = os.path.splitext(fname)[0]
        tmp_dir = os.path.join(OUTPUT_PATH_LARGE, f".tmp_{base}")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        print(f"→ Processing {in_path}")
        df_raw = (
            spark.read
                 .option("header", True)
                 .option("multiLine", True)
                 .option("quote", '"')
                 .option("escape", '"')
                 .csv(in_path)
        )

        # apply UDF + rename columns
        df_clean = (
            df_raw
              .withColumn("matchData", snake_json(col("matchData")))
              .withColumnRenamed("summonerId", "summoner_id")
              .withColumnRenamed("puuid",      "puuid")
              .withColumnRenamed("matchId",    "match_id")
              .withColumnRenamed("matchData",  "match_data")
        )

        # write single-part CSV into tmp_dir
        (
            df_clean
              .coalesce(1)
              .write
              .mode("overwrite")
              .option("header", True)
              .option("quote", "\"")
              .option("escape", "\"")
              .option("quoteAll", True)
              .csv(tmp_dir)
        )

        # find the one part file and move it
        part_file = next(
            f for f in os.listdir(tmp_dir)
            if f.startswith("part-") and f.endswith(".csv")
        )
        src = os.path.join(tmp_dir, part_file)
        dst = os.path.join(OUTPUT_PATH_LARGE, fname)
        print(f"   moving → {dst}")
        shutil.move(src, dst)

        # clean up tmp folder
        shutil.rmtree(tmp_dir)
        print(f"✓ Wrote {dst}\n")

    spark.stop()