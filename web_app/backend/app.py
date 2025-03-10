from flask import Flask, jsonify, request
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
# Enable CORS with specific settings
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
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

        # Get champion names from query parameters
        champion1 = request.args.get('champion1', type=str)
        champion2 = request.args.get('champion2', type=str)

        if not champion1 or not champion2:
            # If no champions specified, return list of available champions
            available_champions = sorted(df['championName'].unique().tolist())
            return jsonify({
                "message": "Please specify two champions to compare",
                "available_champions": available_champions
            })

        # Verify champions exist
        if champion1 not in df['championName'].values or champion2 not in df['championName'].values:
            return jsonify({"error": "One or both champions not found"}), 400

        # Create a radar chart comparing champions
        features = [
            'avg_kills', 'avg_deaths', 'avg_assists', 'kda',
            'avg_dmg_dealt_to_champions', 'avg_dmg_taken',
            'avg_vision_score', 'avg_gold_earned_per_game', 'avg_cs'
        ]
        
        # Create more readable labels for the features
        feature_labels = {
            'avg_kills': 'Kills',
            'avg_deaths': 'Deaths',
            'avg_assists': 'Assists',
            'kda': 'KDA',
            'avg_dmg_dealt_to_champions': 'Damage to Champions',
            'avg_dmg_taken': 'Damage Taken',
            'avg_vision_score': 'Vision Score',
            'avg_gold_earned_per_game': 'Gold Earned',
            'avg_cs': 'CS'
        }
        
        # Select features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        
        # Get data for selected champions
        champ1_data = df[df['championName'] == champion1].iloc[0]
        champ2_data = df[df['championName'] == champion2].iloc[0]
        
        # Create array for normalization
        X = pd.DataFrame([champ1_data[available_features], champ2_data[available_features]])
        X = X.fillna(X.mean())
        X_scaled = StandardScaler().fit_transform(X)
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=X_scaled[0],
            theta=[feature_labels[f] for f in available_features],
            fill='toself',
            name=champion1
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=X_scaled[1],
            theta=[feature_labels[f] for f in available_features],
            fill='toself',
            name=champion2
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-2, 2]
                )),
            showlegend=True,
            title=f'Champion Comparison: {champion1} vs {champion2}'
        )
        
        return jsonify(fig.to_dict())
    except Exception as e:
        app.logger.error(f"Error in champion_comparison: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000) 