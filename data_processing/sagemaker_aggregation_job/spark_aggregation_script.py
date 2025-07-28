from pyspark.sql import SparkSession
from spark_champion_aggregation import main_aggregator

import argparse

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
    champion_x_role_df, champion_x_role_x_user_df = main_aggregator(
        spark, 
        args.input_path,
        args.items_json_path
    )

    print(f"Aggregation completed; writing output as csv to {args.output_path}")

    # Create a subdirectory that Spark can manage
    champion_x_role_output_path = f"file://{args.output_path}/champion_x_role/patch_{PATCH}"
    champion_x_role_x_user_output_path = f"file://{args.output_path}/champion_x_role_x_user/patch_{PATCH}"
    print(f"Writing to: {champion_x_role_output_path}")
    print(f"Writing to: {champion_x_role_x_user_output_path}")

    (champion_x_role_df
        .coalesce(1)
        .write
        .option("header", True)
        .mode("overwrite")
        .csv(champion_x_role_output_path))
    
    (champion_x_role_x_user_df
        .coalesce(1)
        .write
        .option("header", True)
        .mode("overwrite")
        .csv(champion_x_role_x_user_output_path))

    print("Spark aggregation complete.")
    spark.stop()
