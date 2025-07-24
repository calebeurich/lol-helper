from pyspark.sql import SparkSession
from spark_champion_aggregation import main_aggregator

import argparse

TEST_USER_PUUID = "Ou6LPc4Q_QF6qOQ69SBz5oZAY3dnaniTyKH9hE8fsGBWFveaXiYtrL_sQizh5_tPb6BUP3QHieQVAA"

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
    single_user_user_output_path = f"file://{args.output_path}/single_user_output"
    print(f"Writing to: {single_user_user_output_path}")
    
    (single_user_df
        .coalesce(1)
        .write
        .option("header", True)
        .mode("overwrite")
        .csv(single_user_user_output_path))

    print("Spark aggregation complete.")
    spark.stop()