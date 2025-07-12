import json, re
import pandas as pd
import os

module_dir = os.path.dirname(__file__)
json_path = os.path.join(module_dir, "items_info.json")

def camel_to_snake(tags_list: list)-> list:
    for index, tag in enumerate(tags_list):
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', tag)
        s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
        tags_list[index] = s2.lower()
    return tags_list

#  Open JSON file with item data
with open(json_path, "r") as data:
    item_data = json.load(data)["data"]

items_dict = {}
tags_set = set()

# Populate dict with item ids (keys) and item tags (values)- tags can be a list of more than 1
for item_id in item_data.keys():
    try:
        item_data[item_id]["into"]
    except KeyError:
        items_dict[item_id] = camel_to_snake(item_data[item_id]["tags"])

# Create entry to account for missing values (i.e. item slot was empty at game end)
items_dict["0"] = []

# Save file as JSON
with open("item_id_tags.json", "w") as json_file: 
    json.dump(items_dict, json_file)

for tags in items_dict.values():
    tags_set.update(tags)

#for tag in tags_set:
    #print(f"champion_stats['pct_items_{tag.lower()}_tag'] = (champion_stats['item_tags'].apply(lambda tags: tags.count('{tag}')) / champion_stats['completed_items'] * 100)")