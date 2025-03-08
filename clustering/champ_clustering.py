# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import pickle

# %%
#%pip install seaborn

# %%
# Read the data and filter out low playrate champions
df = pd.read_csv('data_processing/champion_stats.csv')
df = df[df['total_games_played_in_role'] >= 5]  # Filter out champions with fewer than 5 games

# Select features for clustering
features = [
    # Core Stats
    'avg_kills', 'avg_deaths', 'avg_assists', 'kda',
    'avg_kill_participation', 'avg_takedowns',
    'avg_deaths_by_enemy_champs',
    'winrate',
    
    # Damage Stats
    'avg_dmg_dealt_to_champions', 'avg_dmg_taken',
    'avg_magic_dmg_to_champs', 'avg_physical_dmg_to_champs', 'avg_true_dmg_to_champs',
    'avg_dmg_self_mitigated', 'damage_per_minute',
    'avg_largest_crit', 'avg_pct_damage_in_team', 'avg_dmg_taken_team_pct',
    'pct_highest_dmg_in_match',
    
    # Support/Utility Stats
    'avg_time_ccing_champs', 'avg_heals_on_teammate', 'avg_dmg_shielded_on_team',
    'avg_effective_heal_and_shield', 'avg_champ_immobilizations',
    'pct_highest_cc_in_match', 'avg_times_save_ally_from_death',
    
    # Economy Stats
    'avg_gold_earned_per_game', 'avg_gold_spent', 'avg_cs',
    'avg_neutral_monsters_cs', 'cs_per_minute', 'gold_per_minute',
    'avg_cs_10_mins', 'avg_jg_cs_before_10m',
    'avg_max_cs_over_lane_opp',
    
    # Vision Stats
    'avg_vision_score', 'avg_wards_placed', 'avg_wards_killed', 'avg_ctrl_wards_bought',
    'avg_ctrol_wards_placed', 'avg_vision_score_per_min', 'pct_highest_ward_kills_in_match',
    'avg_ctrl_ward_time_coverage_in_river_or_enemy_half',
    
    # Objective Stats
    'pct_of_games_team_took_first_baron', 'pct_of_games_team_took_first_drag', 
    'pct_of_games_team_took_first_turret', 'pct_games_team_took_first_herald',
    'avg_indiv_dmg_dealt_to_buildings', 'avg_dmg_dealt_to_objs', 'avg_indiv_turret_plates_taken',
    'avg_epic_monster_steals', 'avg_epic_monster_kills_within_30s_of_spawn',
    
    # Jungle Stats
    'avg_buffs_stolen', 'avg_initial_buff_count', 'avg_initial_crab_count',
    'avg_crabs_per_game', 'avg_jgler_kills_early_jungle',
    'avg_jgler_early_kills_on_laners',
    
    # Survival Stats
    'avg_longest_time_alive', 'avg_bounty_lvl', 'avg_time_spent_dead',
    'avg_times_survived_single_digit_hp', 'avg_times_survived_3_immobilizes_in_fight',
    'avg_times_took_large_dmg_survived',
    
    # Early Game Stats
    'pct_of_games_with_early_lanephase_gold_exp_adv', 'pct_of_games_with_lanephase_gold_exp_adv',
    'avg_max_lvl_lead_lane_opp', 'pct_games_first_blood_kill', 'pct_of_games_indiv_killed_1st_tower',
    
    # Multikill Stats
    'avg_killing_sprees', 'avg_largest_killing_spee', 'avg_number_of_multikills',
    'avg_multikills_with_one_spell', 'avg_legendary_count'
]

# Define role-specific cluster counts for more detailed archetypes
ROLE_CLUSTERS = {
    'TOP': 6,      # Split pushers, Tanks, Bruisers, AP carries, Lane bullies, Utility tanks
    'JUNGLE': 6,   # Assassins, Power farmers, Gankers, Tank/CC, Objective control, Utility
    'MIDDLE': 6,   # Assassins, Control mages, Burst mages, Roamers, Artillery mages, Utility mages
    'BOTTOM': 5,   # Hypercarries, Lane bullies, Utility ADCs, Poke ADCs, Early game ADCs
    'UTILITY': 6   # Enchanters, Engage tanks, Poke supports, Wardens, Utility mages, Roaming supports
}

def get_distinctive_features(cluster_features, all_clusters_features, n_features=5):
    """Identify the most distinctive features of a cluster compared to others."""
    # Calculate z-scores for this cluster's features compared to all clusters
    cluster_means = all_clusters_features.mean()
    cluster_stds = all_clusters_features.std()
    
    z_scores = {}
    for feature in cluster_features.index:
        if cluster_stds[feature] != 0:  # Avoid division by zero
            z_scores[feature] = (cluster_features[feature] - cluster_means[feature]) / cluster_stds[feature]
    
    # Sort features by absolute z-score
    sorted_features = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_features[:n_features]

def get_cluster_description(role, cluster_features):
    """Generate a meaningful description of the cluster based on its distinctive features."""
    description = []
    
    # Damage profile
    if 'avg_magic_dmg_to_champs' in cluster_features and 'avg_physical_dmg_to_champs' in cluster_features:
        if cluster_features['avg_magic_dmg_to_champs'] > cluster_features['avg_physical_dmg_to_champs']:
            damage_type = "Magic damage"
        else:
            damage_type = "Physical damage"
    else:
        damage_type = "Mixed damage"
    
    # Playstyle indicators
    if ('avg_dmg_taken' in cluster_features and 'avg_dmg_self_mitigated' in cluster_features and
            cluster_features['avg_dmg_taken'] > 2000 and cluster_features['avg_dmg_self_mitigated'] > 15000):
        description.append("Tank")
    
    if 'avg_kills' in cluster_features and cluster_features['avg_kills'] > 7:
        description.append("High kill potential")
    
    if 'avg_assists' in cluster_features and cluster_features['avg_assists'] > 12:
        description.append("Team-oriented")
    
    if 'avg_time_ccing_champs' in cluster_features and cluster_features['avg_time_ccing_champs'] > 25:
        description.append("CC-heavy")
    
    if 'avg_effective_heal_and_shield' in cluster_features and cluster_features['avg_effective_heal_and_shield'] > 5000:
        description.append("Supportive")
    
    if 'cs_per_minute' in cluster_features and cluster_features['cs_per_minute'] > 7:
        description.append("Farm-focused")
    
    if 'avg_vision_score' in cluster_features and cluster_features['avg_vision_score'] > 30:
        description.append("Vision-focused")
    
    if 'pct_games_first_blood_kill' in cluster_features and cluster_features['pct_games_first_blood_kill'] > 20:
        description.append("Early game aggressor")
    
    if 'avg_indiv_dmg_dealt_to_buildings' in cluster_features and cluster_features['avg_indiv_dmg_dealt_to_buildings'] > 2000:
        description.append("Split push potential")
    
    # Role-specific descriptions
    if role == 'JUNGLE' and 'avg_jgler_early_kills_on_laners' in cluster_features and cluster_features['avg_jgler_early_kills_on_laners'] > 1:
        description.append("Strong ganker")
    
    if role == 'MIDDLE' and 'avg_roam_score' in cluster_features and cluster_features['avg_roam_score'] > 5:
        description.append("Roaming playstyle")
    
    return ", ".join(description) + f" ({damage_type} focused)"

def format_feature_name(feature_name):
    """Convert feature names to more readable format."""
    name = feature_name.replace('avg_', '').replace('pct_', '% ').replace('_', ' ')
    return name.title()

def cluster_role(role_df, role_name, features):
    print(f"\n=== Clustering for {role_name} ===")
    print(f"Number of champions: {len(role_df)}")
    
    # Check which features exist in the dataframe
    available_features = [f for f in features if f in role_df.columns]
    missing_features = [f for f in features if f not in role_df.columns]

    if missing_features:
        print(f"Warning: The following features are not in the dataset and will be skipped: {missing_features}")

    # Prepare the data
    X = role_df[available_features].copy()

    # Handle missing values by imputing with mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = StandardScaler().fit_transform(X_imputed)

    # Use predetermined number of clusters for the role
    n_clusters = ROLE_CLUSTERS.get(role_name, 5)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    role_df['Cluster'] = kmeans.fit_predict(X)

    # Calculate cluster characteristics
    cluster_means = role_df.groupby('Cluster')[available_features].mean()
    
    print(f"\n=== {role_name} Archetypes Analysis ===")
    for cluster in range(n_clusters):
        print(f"\nArchetype {cluster + 1}:")
        cluster_champs = role_df[role_df['Cluster'] == cluster]['championName'].tolist()
        
        # Get cluster characteristics
        cluster_features = cluster_means.loc[cluster]
        description = get_cluster_description(role_name, cluster_features)
        
        # Get distinctive features
        distinctive_features = get_distinctive_features(cluster_features, cluster_means)
        
        print(f"Description: {description}")
        print("Champions:", ', '.join(sorted(cluster_champs)))
        
        print("\nDistinctive Characteristics:")
        for feature, z_score in distinctive_features:
            feature_name = format_feature_name(feature)
            value = cluster_features[feature]
            if 'pct' in feature or 'winrate' in feature:
                print(f"- {feature_name}: {value*100:.1f}% (z-score: {z_score:.2f})")
            else:
                print(f"- {feature_name}: {value:.1f} (z-score: {z_score:.2f})")
        
        print("\nKey Performance Stats:")
        if 'winrate' in cluster_features:
            print(f"Winrate: {cluster_features['winrate']*100:.1f}%")
        if all(stat in cluster_features for stat in ['avg_kills', 'avg_deaths', 'avg_assists']):
            print(f"KDA: {cluster_features['avg_kills']:.1f}/{cluster_features['avg_deaths']:.1f}/{cluster_features['avg_assists']:.1f}")
        if 'damage_per_minute' in cluster_features:
            print(f"Damage/min: {cluster_features['damage_per_minute']:.0f}")
        if 'cs_per_minute' in cluster_features:
            print(f"CS/min: {cluster_features['cs_per_minute']:.1f}")
        print("-" * 50)
    
    return role_df, kmeans, None  # Return None for PCA to save memory

# Get unique roles
roles = df['mode_team_position'].unique()

# Dictionary to store results for each role
role_results = {}

# Perform clustering for each role
for role in roles:
    role_df = df[df['mode_team_position'] == role].copy()
    role_results[role] = cluster_role(role_df, role, features)

# Save results
results = {
    'role_clusters': {role: results[0]['Cluster'].to_dict() for role, results in role_results.items()},
    'role_models': {role: results[1] for role, results in role_results.items()}
}

with open('clustering/champion_role_clusters.pkl', 'wb') as f:
    pickle.dump(results, f)

def find_similar_champions_in_role(champ_name, role, n=5):
    """Find similar champions within the same role based on playstyle metrics."""
    role_df = df[df['mode_team_position'] == role]
    if champ_name not in role_df['championName'].values:
        print(f"Champion '{champ_name}' not found for role '{role}'")
        return
    
    # Get the features for the target champion
    target_features = role_df[role_df['championName'] == champ_name][features].iloc[0]
    
    # Calculate distances to all other champions in the same role
    distances = []
    for idx, row in role_df.iterrows():
        if row['championName'] != champ_name:
            dist = np.linalg.norm(row[features] - target_features)
            distances.append((row['championName'], dist))
    
    # Sort by distance and get top N
    similar_champs = sorted(distances, key=lambda x: x[1])[:n]
    
    # Get the cluster of the target champion
    target_cluster = role_df[role_df['championName'] == champ_name]['Cluster'].iloc[0]
    cluster_features = role_df.groupby('Cluster')[features].mean().loc[target_cluster]
    
    print(f"\nChampion Analysis: {champ_name} ({role})")
    print(f"Archetype: {get_cluster_description(role, cluster_features)}")
    print("\nSimilar champions by playstyle:")
    for champ, dist in similar_champs:
        champ_cluster = role_df[role_df['championName'] == champ]['Cluster'].iloc[0]
        champ_features = role_df.groupby('Cluster')[features].mean().loc[champ_cluster]
        print(f"{champ}: {get_cluster_description(role, champ_features)}")

print("\nClustering analysis complete. Use find_similar_champions_in_role(champ_name, role) to find similar champions.")



