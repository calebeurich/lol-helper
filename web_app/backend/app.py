from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
logging.basicConfig(level=logging.DEBUG)

# Load the data
def load_data():
    try:
        # Load champion stats
        df = pd.read_csv('../../data_processing/champion_stats.csv')
        app.logger.debug(f"Successfully loaded champion_stats.csv with {len(df)} rows")
        
        # Load cluster data if it exists
        cluster_file = '../clustering/champion_cluster_profiles.pkl'
        cluster_profiles = None
        if os.path.exists(cluster_file):
            cluster_profiles = pd.read_pickle(cluster_file)
            app.logger.debug("Successfully loaded cluster profiles")
        else:
            app.logger.debug("No cluster profiles found")
        
        return df, cluster_profiles
    except Exception as e:
        app.logger.error(f"Error loading data: {str(e)}")
        return None, None

# Load data at startup
df, cluster_profiles = load_data()

@app.route('/api/champion_stats')
def champion_stats():
    try:
        if df is None:
            return jsonify({"error": "Data not loaded"}), 500
        # Convert DataFrame to JSON
        stats = df.to_dict(orient='records')
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error in champion_stats: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cluster_visualization')
def cluster_visualization():
    try:
        if df is None:
            return jsonify({"error": "Data not loaded"}), 500

        # Prepare data for clustering visualization
        features = [
            'kills', 'deaths', 'assists', 'kda',
            'totalDamageDealtToChampions', 'totalDamageTaken',
            'visionScore', 'goldEarned', 'totalMinionsKilled'
        ]
        
        # Select features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Standardize the data
        X_scaled = StandardScaler().fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            z=X_pca[:, 2],
            color=clusters,
            hover_data={'championName': df['championName']},
            title='Champion Clusters (3D PCA)',
            labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'}
        )
        
        return jsonify(fig.to_dict())
    except Exception as e:
        app.logger.error(f"Error in cluster_visualization: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/champion_comparison')
def champion_comparison():
    try:
        if df is None:
            return jsonify({"error": "Data not loaded"}), 500

        # Create a radar chart comparing champions
        features = [
            'kills', 'deaths', 'assists', 'kda',
            'totalDamageDealtToChampions', 'totalDamageTaken',
            'visionScore', 'goldEarned', 'totalMinionsKilled'
        ]
        
        # Select features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        
        # Normalize the data for the radar chart
        X = df[available_features].copy()
        X = X.fillna(X.mean())
        X_scaled = StandardScaler().fit_transform(X)
        
        # Create a sample radar chart for the first two champions
        fig = go.Figure()
        
        for i in range(2):
            fig.add_trace(go.Scatterpolar(
                r=X_scaled[i],
                theta=available_features,
                fill='toself',
                name=df['championName'].iloc[i]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-2, 2]
                )),
            showlegend=True,
            title='Champion Comparison (Radar Chart)'
        )
        
        return jsonify(fig.to_dict())
    except Exception as e:
        app.logger.error(f"Error in champion_comparison: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 