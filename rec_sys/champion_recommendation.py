#%%
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import os, boto3, io

pd.set_option('display.max_columns', None)


load_dotenv()
BUCKET = os.getenv("BUCKET")
PROCESSED_DATA_FOLDER = os.getenv("PROCESSED_DATA_FOLDER")
PATCH = "patch_15_6"
GRANULARITY_LIST = ["single_user", "champion_x_role", "champion_x_role_x_user"]
AWS_KEY = os.getenv("AWS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET")
#%%
# Test your credentials
print(f"AWS_KEY loaded: {'Yes' if AWS_KEY else 'No'}")
print(f"AWS_SECRET loaded: {'Yes' if AWS_SECRET else 'No'}")

# Test S3 connection
try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name="us-east-2"
    )
    response = s3.list_buckets()
    print("✅ AWS credentials working!")
    print(f"Available buckets: {[bucket['Name'] for bucket in response['Buckets']]}")
except Exception as e:
    print(f"❌ AWS credentials error: {e}")
# %%
# Code in case we want to pull all dfs in a single function

def get_processed_data_file(granularity: str, role=None) -> pd.DataFrame:

    if granularity == "single_user":
        key = f"{PROCESSED_DATA_FOLDER}/single_user_data/{PATCH}/single_user_aggregated_data.csv"
    elif granularity == "champion_x_role":
        key = f"{PROCESSED_DATA_FOLDER}/champion_x_role/{PATCH}/champion_x_role_aggregated_data.csv"
    elif granularity == "champion_x_role_x_user":
        key = f"{PROCESSED_DATA_FOLDER}/champion_x_role_x_user/{PATCH}/champion_x_role_x_user_aggregated_data.csv"
    elif granularity == "cluster":
        key = f"{PROCESSED_DATA_FOLDER}/clusters/{PATCH}/{role}_clusters_df.csv"
    elif granularity == "residuals":
        key = f"{PROCESSED_DATA_FOLDER}/clusters/{PATCH}/{role}_champion_residuals_df.csv"
    elif granularity == "semantic_champion":
        key = f"{PROCESSED_DATA_FOLDER}/bedrock_output/{PATCH}/{role}_champion_semantic_tags_and_descriptions.csv"
    elif granularity == "semantic_cluster":
        key = f"{PROCESSED_DATA_FOLDER}/bedrock_output/{PATCH}/{role}_cluster_semantic_tags_and_descriptions.csv"

    #add all granularities
    else:
        raise ValueError("Incorrect granularity input, must be 'champion_x_role', 'champion_x_role_x_user', 'single_user', or 'cluster'")
    
    # Pull the object
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name="us-east-2"
    )

    print(key)
    obj = s3.get_object(Bucket=BUCKET, Key=key)

    # Read it straight into pandas
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    print(df.shape)

    return df

#%%
champ_x_role_df = get_processed_data_file("champion_x_role")
display(champ_x_role_df.head())

#%%
sngle_user_df = get_processed_data_file("single_user")
display(sngle_user_df.head())
#%%
top_cluster_df = get_processed_data_file("cluster", "top")
display(top_cluster_df.head())
#%%
top_residuals_df = get_processed_data_file("residuals", "top")
display(top_residuals_df.head())
#%%
top_semantic_cluster_df = get_processed_data_file("semantic_cluster", "top")
display(top_semantic_cluster_df.head())
#%%
top_semantic_champion_df = get_processed_data_file("semantic_champion", "top")
display(top_semantic_champion_df.head())
#%%
#bigger dataframe for user comparisons
champion_x_role_x_user_df = get_processed_data_file("champion_x_role_x_user")
display(champion_x_role_x_user_df.head())
#%%
#%%
