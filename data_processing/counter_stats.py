from IPython.display import display
import pandas as pd
import json
from dotenv import load_dotenv
from data_processing.champion_aggregation import merged_df

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

with open('item_id_tags.json', 'r') as data:
    items_dict = json.load(data)

display(merged_df.head())