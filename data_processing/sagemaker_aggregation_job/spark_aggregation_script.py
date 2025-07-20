from pyspark.sql import SparkSession
from spark_champion_aggregation import main_aggregator

import argparse

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
    aggregated_by_champion_df = main_aggregator(
        spark, 
        args.input_path,
        args.items_json_path
    )

    print(f"Aggregation completed; writing output as csv to {args.output_path}")

    # Create a subdirectory that Spark can manage
    output_path = f"file://{args.output_path}/spark_output"
    print(f"Writing to: {output_path}")

    (aggregated_by_champion_df
        .coalesce(1)
        .write
        .option("header", True)
        .mode("overwrite")
        .csv(output_path))

    print("Spark aggregation complete.")
    spark.stop()
