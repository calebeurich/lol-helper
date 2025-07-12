import sys
import os
import zipfile
import os, sys
from typing import Dict, List, Set, Tuple, Any
from pyspark.sql import SparkSession
import boto3
from urllib.parse import urlparse
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from enum import Enum

import pandas as pd
import argparse
from spark_champion_aggregation import main_aggregator

def main(input_path, output_path):
    all_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.csv')]
    if not all_files:
        print('⚠️ No CSV files found in input path:', input_path)
        return
    list_df = [pd.read_csv(file) for file in all_files]
    master_df = pd.concat(list_df, ignore_index=True)
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