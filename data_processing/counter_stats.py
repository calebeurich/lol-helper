from IPython.display import display
import pandas as pd
import json
from dotenv import load_dotenv
#from champion_aggregation import filtered_df

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#with open('filtered_df.csv', 'r') as data:
    #filtered_df = json.load(data)

filtered_df = pd.read_csv('filtered_df.csv')

def assign_partner_optimized(df):
    # Group by matchId and teamPosition
    grouped = df.groupby(['matchId', 'teamPosition'], group_keys=False)
    
    # Create a helper function to find opponent champion
    def find_opponent_champion(group):
        if len(group) != 2:
            return pd.Series(['Opp Champ Error'] * len(group), index=group.index)
        
        # Get champion names in the group
        champs = group['championName'].tolist()
        
        # Create a Series with opponent champions
        opponent_champs = pd.Series(
            [champs[1] if champ == champs[0] else champs[0] 
             for champ in group['championName']],
            index=group.index
        )
        
        return opponent_champs
    
    # Apply the function to each group with include_groups=False
    return grouped.apply(find_opponent_champion, include_groups=False)

filtered_df['opponent_champion'] = assign_partner_optimized(filtered_df)
filtered_df['champion_and_role'] = filtered_df['championName'] + '_' + filtered_df['teamPosition']

agg_df = filtered_df.groupby(['champion_and_role', 'opponent_champion']).agg(
    win_rate = ('win_x', 'mean'),
    game_count = ('matchId', 'count')
).reset_index()

win_rate_pivot = agg_df.pivot(index='champion_and_role', columns='opponent_champion', values='win_rate')
game_count_pivot = agg_df.pivot(index='champion_and_role', columns='opponent_champion', values='game_count')

testing_df = filtered_df.filter(['matchId', 'championName', 'opponent_champion'])