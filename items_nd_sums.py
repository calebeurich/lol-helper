import json
import pandas as pd

#  Open JSON file with item data
with open('items_info.json', 'r') as data:
    item_data = json.load(data)['data']

items_dict = {}
tags_set = set()

summ_spells_dict = {
    21 : 'barrier',
    1 : 'cleanse',
    3 : 'exhaust',
    4 : 'flash',
    6 : 'ghost', 
    7 : 'heal',
    14 : 'ignite',
    11 : 'smite', 
    12 : 'teleport'  
}

# Populate dict with item ids (keys) and item tags (values)- tags can be a list of more than 1
for item_id in item_data.keys():
    try:
        item_data[item_id]['into']
    except KeyError:
        items_dict[item_id] = item_data[item_id]['tags']

# Create entry to account for missing values (i.e. item slot was empty at game end)
items_dict["0"] = []

# Save file as JSON
with open('item_id_tags.json', 'w') as json_file: 
    json.dump(items_dict, json_file)

for tags in items_dict.values():
    tags_set.update(tags)

#for tag in tags_set:
    #print(f"champion_stats['pct_items_{tag.lower()}_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('{tag}')) / champion_stats['completed_items'] * 100)")