import pandas as pd

def assign_partner(raw_df):
    # Group by matchId and teamPosition
    grouped = raw_df.groupby(['match_id', 'team_position'], group_keys=False)
    
    # Create a helper function to find opponent champion
    def find_opponent_champion(group):
        if len(group) != 2:
            return pd.Series(['Opp Champ Error'] * len(group), index=group.index)
        
        # Get champion names in the group
        champs = group['champion_name'].tolist()
        
        # Create a Series with opponent champions
        opponent_champs = pd.Series(
            [champs[1] if champ == champs[0] else champs[0] 
             for champ in group['champion_name']],
            index=group.index
        )
        
        return opponent_champs
    
    # Apply the function to each group with include_groups=False
    return grouped.apply(find_opponent_champion, include_groups=False)

def create_counter_stats_df(raw_df):

    raw_df['opponent_champion'] = assign_partner(raw_df)
    raw_df['champion_and_role'] = raw_df['champion_name'] + '_' + raw_df['team_position']

    agg_counters_df = raw_df.groupby(['champion_and_role', 'opponent_champion']).agg(
        win_rate = ('win_x', 'mean'),
        game_count = ('matchId', 'count')
    ).reset_index()

    win_rate_pivot_df = agg_counters_df.pivot(index='champion_and_role', columns='opponent_champion', values='win_rate')
    game_count_pivot_df = agg_counters_df.pivot(index='champion_and_role', columns='opponent_champion', values='game_count')

    #testing_df = filtered_df.filter(['match_id', 'champion_name', 'opponent_champion'])

    return win_rate_pivot_df, game_count_pivot_df