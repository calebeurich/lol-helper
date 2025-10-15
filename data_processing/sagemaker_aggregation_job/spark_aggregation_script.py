from pyspark.sql import SparkSession
from spark_champion_aggregation import main_aggregator

import argparse, os, glob, shutil, uuid

PATCH = "15_6"

def write_and_rename_parquet(df, role: str, base_output_path: str, filename: str):
    """
    Write a Spark DataFrame as parquet and rename to a specific filename.

    Args:
        df: Spark DataFrame
        role: Role name (e.g., 'top', 'jungle')
        base_output_path: Base output directory path
        filename: Desired final filename (e.g., 'top_champion_x_role_aggregated_data.parquet')
    """
    role_dir = f"{base_output_path}/{role}"
    tmp_dir = f"{role_dir}/_tmp_write_{uuid.uuid4().hex[:8]}"

    # Ensure parent directory exists
    os.makedirs(role_dir, exist_ok=True)

    try:
        # Write to temporary directory
        (df
            .coalesce(1)
            .write
            .mode("overwrite")
            .parquet(f"file://{tmp_dir}"))

        # Find the parquet file (choose largest part just in case)
        parquet_files = glob.glob(f"{tmp_dir}/part-*.parquet")
        if not parquet_files:
            raise FileNotFoundError(f"No parquet file found in {tmp_dir}")
        part_file = max(parquet_files, key=os.path.getsize)

        final_file = f"{role_dir}/{filename}"

        # Rename the file
        os.replace(part_file, final_file)
        print(f"Success, written: {final_file}")

    except Exception as e:
        print(f"Error writing {role} to {role_dir}: {e}")
        raise

    finally:
        # Clean up temporary directory
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
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

    # Run aggregation (returns dicts keyed by role)
    champion_x_role_dfs, champion_x_role_x_user_dfs, counter_stats_dfs_by_role = main_aggregator(
        spark,
        args.input_path,
        args.items_json_path
    )

    print(f"Aggregation completed; writing output as parquet to {args.output_path}")

    # Define output paths
    champion_x_role_output_path = f"{args.output_path}/champion_x_role/patch_{PATCH}"
    champion_x_role_x_user_output_path = f"{args.output_path}/champion_x_role_x_user/patch_{PATCH}"
    counter_stats_output_path = f"{args.output_path}/counter_stats_dfs_by_role/patch_{PATCH}"

    print(f"Writing to: file://{champion_x_role_output_path}")
    print(f"Writing to: file://{champion_x_role_x_user_output_path}")
    print(f"Writing to: file://{counter_stats_output_path}")

    # Write champion_x_role per role
    print("\n Writing champion_x_role DataFrames")
    for role, df in champion_x_role_dfs.items():
        filename = f"{role}_champion_x_role_aggregated_data.parquet"
        write_and_rename_parquet(df, role, champion_x_role_output_path, filename)

    # Write champion_x_role_x_user per role
    print("\n Writing champion_x_role_x_user DataFrames")
    for role, df in champion_x_role_x_user_dfs.items():
        filename = f"{role}_champion_x_role_x_user_aggregated_data.parquet"
        write_and_rename_parquet(df, role, champion_x_role_x_user_output_path, filename)

    # Write counter_stats per role
    print("\n Writing counter_stats DataFrames")
    for role, df in counter_stats_dfs_by_role.items():
        filename = f"{role}_counter_stats.parquet"
        write_and_rename_parquet(df, role, counter_stats_output_path, filename)

    print("\n Spark aggregation complete - all files written successfully!")
    spark.stop()
