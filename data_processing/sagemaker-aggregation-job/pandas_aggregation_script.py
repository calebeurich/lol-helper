import sys
sys.path.append("/opt/ml/processing/input/modules")  # Ensure modules are in path for SageMaker

import pandas as pd
import os
import argparse
from items_and_summs_module import tag_finder, item_filter, get_summ_spell_name
from champion_aggregation import main_aggregator

def main(input_path, output_path):
    # Read all CSV files in the input directory
    all_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith(".csv")]

    if not all_files:
        print("⚠️ No CSV files found in input path:", input_path)
        return

    df_list = [pd.read_csv(file) for file in all_files]

    # Concat into master_df
    master_df = pd.concat(df_list, ignore_index=True)

    # Perform full aggregation on master_df
    agg_champion_stats_df = main_aggregator(master_df)

    # Save to output path
    output_file = os.path.join(output_path, "aggregated_output.csv")
    agg_champion_stats_df.to_csv(output_file, index=False)
    print(f"✅ Aggregation complete. Output written to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    main(args.input_path, args.output_path)


