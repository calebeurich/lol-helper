from pyspark.sql import SparkSession
from spark_champion_aggregation import main_aggregator

import argparse, os, glob

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
            .appName("champion_aggregation")
            .getOrCreate()
    )

    # Hand SparkSession + folder‑prefix into aggregator
    champion_x_role_df, champion_x_role_x_user_df, counter_stats_dfs_by_role = main_aggregator(
        spark, 
        args.input_path,
        args.items_json_path
    )

    print(f"Aggregation completed; writing output as csv to {args.output_path}")

    # Create a subdirectory that Spark can manage
    champion_x_role_output_path = f"{args.output_path}/champion_x_role/patch_{PATCH}"
    champion_x_role_x_user_output_path = f"{args.output_path}/champion_x_role_x_user/patch_{PATCH}"
    counter_stats_output_path = f"{args.output_path}/counter_stats_dfs_by_role/patch_{PATCH}"
    
    print(f"Writing to: file://{champion_x_role_output_path}")
    print(f"Writing to: file://{champion_x_role_x_user_output_path}")

    (champion_x_role_df
        .coalesce(1)
        .write
        .option("header", True)
        .mode("overwrite")
        .csv(f"file://{champion_x_role_output_path}")) # "file://" prefix needed for Spark to output to S3
    
    (champion_x_role_x_user_df
        .coalesce(1)
        .write
        .option("header", True)
        .mode("overwrite")
        .csv(f"file://{champion_x_role_x_user_output_path}"))
    
    for role, df in counter_stats_dfs_by_role.items():
        (df
            .coalesce(1)
            .write
            .option("header", True)
            .mode("overwrite")
            .csv(f"file://{counter_stats_output_path}/{role}"))

    print("Spark aggregation complete.")

    path_mapping = {
        champion_x_role_output_path: "champion_x_role_aggregated_data.csv",
        champion_x_role_x_user_output_path: "champion_x_role_x_user_aggregated_data.csv"
    }

    for path in path_mapping:
        # Find the part file
        part_files = glob.glob(f"{path}/part-*.csv")
        if part_files:
            part_file = part_files[0]
            new_name = f"{path}/{path_mapping[path]}"

            print(f"Renaming {part_file} to {new_name}")
            os.rename(part_file, new_name)

            # Remove .crc files
            for crc_file in glob.glob(f"{path}/.*.crc"):
                os.remove(crc_file)

            print("File renamed successfully")
        else:
            print("Warning: No part file found to rename")

    spark.stop()
