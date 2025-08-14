from dotenv import load_dotenv
import numpy as np
import pandas as pd
import os, boto3, io
import ast

load_dotenv()
BUCKET = os.getenv("BUCKET")
PROCESSED_DATA_FOLDER = os.getenv("PROCESSED_DATA_FOLDER")
AWS_KEY = os.getenv("AWS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET")
PATCH = "patch_15_6"

def get_processed_data_file(granularity: str, role=None) -> pd.DataFrame:
    """Load data from S3 based on granularity and role"""
    
    if granularity == "single_user":
        key = f"{PROCESSED_DATA_FOLDER}/single_user_data/{PATCH}/single_user_aggregated_data.csv"
    elif granularity == "champion_x_role":
        key = f"{PROCESSED_DATA_FOLDER}/champion_x_role/{PATCH}/champion_x_role_aggregated_data.csv"
    elif granularity == "champion_x_role_x_user":
        key = f"{PROCESSED_DATA_FOLDER}/champion_x_role_x_user/{PATCH}/champion_x_role_x_user_aggregated_data.csv"
    elif granularity == "cluster":
        key = f"{PROCESSED_DATA_FOLDER}/clusters/{PATCH}/{role}_vectors_df.csv"
    else:
        raise ValueError("Incorrect granularity input, must be 'champion_x_role', 'champion_x_role_x_user', 'single_user', or 'cluster'")
    
    # Pull the object
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name="us-east-2"
    )
    
    print(f"Loading from: {key}")
    obj = s3.get_object(Bucket=BUCKET, Key=key)

    # Read it straight into pandas
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    print(f"Loaded data shape: {df.shape}")
    return df

def get_champion_recommendations(num_recs=5):
    """Get champion recommendations based on the user's most played champion"""
    
    # Load single user data and find top row by total_games_played_in_role
    print("Loading single user data...")
    single_user_df = get_processed_data_file("single_user")

    # Find the top row by total_games_played_in_role
    top_row = single_user_df.loc[single_user_df['total_games_played_in_role'].idxmax()]
    champion_name = top_row['champion_name']
    team_position = top_row['team_position']

    print(f"Top champion by games played: {champion_name} in {team_position} position")
    print(f"Total games played in role: {top_row['total_games_played_in_role']}")

    # Load the cluster data for the specific position
    print(f"Loading cluster data for {team_position.lower()}...")
    top_cluster_df = get_processed_data_file("cluster", team_position.lower())

    # Find the row for our champion in the cluster data
    champion_cluster_row = top_cluster_df.loc[
        (top_cluster_df['champion_name'] == champion_name) & 
        (top_cluster_df['team_position'] == team_position)
    ]

    if len(champion_cluster_row) == 0:
        print(f"Champion {champion_name} not found in {team_position} cluster data")
        return None
    else:
        # Get the residual vector for this champion
        residual_vec = champion_cluster_row['residual_vec_scaled'].iloc[0]
        
        # Convert string representation back to list if needed
        if isinstance(residual_vec, str):
            residual_vec = ast.literal_eval(residual_vec)
        
        print(f"Found residual vector for {champion_name}")
        
        # Get the cluster of the original champion
        original_cluster = champion_cluster_row['cluster'].iloc[0]
        print(f"Original champion {champion_name} is in cluster {original_cluster}")
        
        # Calculate distances to all other champions in the same position
        distances = []
        for idx, row in top_cluster_df.iterrows():
            if row['champion_name'] != champion_name:  # Exclude the original champion
                other_residual = row['residual_vec_scaled']
                if isinstance(other_residual, str):
                    other_residual = ast.literal_eval(other_residual)
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(np.array(residual_vec) - np.array(other_residual))
                distances.append({
                    'champion_name': row['champion_name'],
                    'cluster': row['cluster'],
                    'distance': distance,
                    'euclidean_distance_to_centroid': row['euclidean_distance_to_centroid']
                })
        
        # Sort by distance and get top recommendations
        distances.sort(key=lambda x: x['distance'])
        
        # Handle case where there aren't enough champions to recommend
        available_champions = len(distances)
        if available_champions < num_recs:
            print(f"Warning: Only {available_champions} champions available for recommendation (requested {num_recs})")
            actual_num_recs = available_champions
        else:
            actual_num_recs = num_recs
        
        top_recommendations = distances[:actual_num_recs]
        
        print(f"\nTop {actual_num_recs} champion recommendations for {champion_name} ({team_position}):")
        for i, rec in enumerate(top_recommendations, 1):
            print(f"{i}. {rec['champion_name']} (cluster {rec['cluster']}, distance: {rec['distance']:.4f})")
        
        return {
            'original_champion': champion_name,
            'position': team_position,
            'original_cluster': original_cluster,
            'recommendations': top_recommendations,
            'requested_recs': num_recs,
            'actual_recs': actual_num_recs,
            'available_champions': available_champions
        }

if __name__ == "__main__":
    recommendations = get_champion_recommendations(5)
    if recommendations:
        print(f"\nSummary:")
        print(f"Based on your most played champion ({recommendations['original_champion']} in {recommendations['position']}, cluster {recommendations['original_cluster']}),")
        print(f"here are {recommendations['actual_recs']} similar champions you might enjoy:")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"  {i}. {rec['champion_name']} (cluster {rec['cluster']})")
        
        # Show cluster distribution
        clusters = [rec['cluster'] for rec in recommendations['recommendations']]
        cluster_counts = {}
        for cluster in clusters:
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        
        print(f"\nCluster distribution of recommendations:")
        for cluster, count in cluster_counts.items():
            print(f"  Cluster {cluster}: {count} champions")
        
        # Show availability info
        if recommendations['actual_recs'] < recommendations['requested_recs']:
            print(f"\nNote: Only {recommendations['available_champions']} champions available in {recommendations['position']} position")
    else:
        print("No recommendations found.")
