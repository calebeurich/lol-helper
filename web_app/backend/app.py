from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import json
import os

app = Flask(__name__)

# Load the data
def load_data():
    # Load champion stats
    df = pd.read_csv('../../data_processing/champion_stats.csv')
    
    # Load cluster data if it exists
    cluster_file = '../clustering/champion_cluster_profiles.pkl'
    if os.path.exists(cluster_file):
        cluster_profiles = pd.read_pickle(cluster_file)
    else:
        cluster_profiles = None
    
    return df, cluster_profiles

# Load data at startup
df, cluster_profiles = load_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/champion_stats')
def champion_stats():
    # Convert DataFrame to JSON
    stats = df.to_dict(orient='records')
    return jsonify(stats)

@app.route('/api/cluster_visualization')
def cluster_visualization():
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

@app.route('/api/champion_comparison')
def champion_comparison():
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

if __name__ == '__main__':
    app.run(debug=True) 