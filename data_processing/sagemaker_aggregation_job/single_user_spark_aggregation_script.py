from dotenv import load_dotenv
from pyspark.sql import SparkSession
from spark_champion_aggregation import main_aggregator

import argparse, os, glob

# Load environment variables and set up
load_dotenv()
TEST_USER_PUUID = os.getenv("TEST_USER_PUUID")
PATCH = "15_6"

if __name__ == "__main__":
    # Use --input-path and --output-path from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",  type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--items-json-path", type=str, required=True)
    args = parser.parse_args()

    print(f"Starting Spark job: reading CSVs from {args.input_path}")
    print(f"Reading items JSON from {args.items_json_path}")

    spark = (
        SparkSession.builder
            .appName("single_user_aggregation")
            .getOrCreate()
    )

    # Hand SparkSession + folder‑prefix into aggregator
    single_user_df = main_aggregator(
        spark, 
        args.input_path,
        args.items_json_path,
        single_user_flag = True,
        single_user_puuid = TEST_USER_PUUID
    )

    print(f"Aggregation completed; writing output as csv to {args.output_path}")

    # Create a subdirectory that Spark can manage
    single_user_user_output_path = f"{args.output_path}/single_user_data/patch_{PATCH}"
    print(f"Writing to: file://{single_user_user_output_path}")
    
    (single_user_df
        .coalesce(1)
        .write
        .option("header", True)
        .mode("overwrite")
        .csv(f"file://{single_user_user_output_path}")) # "file://" prefix needed for Spark to output to S3

    print("Spark aggregation complete.")

    # Find the part file
    part_files = glob.glob(f"{single_user_user_output_path}/part-*.csv")
    if part_files:
        part_file = part_files[0]
        new_name = f"{single_user_user_output_path}/single_user_aggregated_data.csv"

        print(f"Renaming {part_file} to {new_name}")
        os.rename(part_file, new_name)

        # Remove .crc files
        for crc_file in glob.glob(f"{single_user_user_output_path}/.*.crc"):
            os.remove(crc_file)

        print("File renamed successfully")
    else:
        print("Warning: No part file found to rename")

    spark.stop()