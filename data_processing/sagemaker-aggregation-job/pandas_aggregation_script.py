import sys
import os
import zipfile

# Unzip modules.zip if running in SageMaker
zip_path = '/opt/ml/processing/code/modules.zip'
extract_path = '/opt/ml/processing/code/modules'
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    sys.path.insert(0, extract_path)

# Optional: for local dev only
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

import pandas as pd
import argparse
from items_and_summs_module import tag_finder, item_filter, get_summ_spell_name
from champion_aggregation import main_aggregator

def main(input_path, output_path):
    all_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.csv')]
    if not all_files:
        print('⚠️ No CSV files found in input path:', input_path)
        return
    df_list = [pd.read_csv(file) for file in all_files]
    master_df = pd.concat(df_list, ignore_index=True)
    agg_champion_stats_df = main_aggregator(master_df)
    output_file = os.path.join(output_path, 'aggregated_output.csv')
    agg_champion_stats_df.to_csv(output_file, index=True)
    print(f'✅ Aggregation complete. Output written to: {output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()
    main(args.input_path, args.output_path)


