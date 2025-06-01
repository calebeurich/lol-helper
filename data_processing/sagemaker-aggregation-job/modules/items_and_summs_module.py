import json
import pandas as pd
import items_nd_sums
import os

#  Open JSON file with item data
json_path = os.path.join(os.path.dirname(__file__), "items_info.json")
with open(json_path, "r") as data:
    item_data = json.load(data)["data"]
items_dict = items_nd_sums.items_dict
summ_spells_dict = items_nd_sums.summ_spells_dict

def tag_finder(item_id):
    try:
        tag = items_dict[item_id]
    except KeyError:
        tag = []
    return tag

def item_filter(item_id):
    if item_id in items_dict:
        return True
    else:
        return False
    
def get_summ_spell_name(summ_spell_id):
    return summ_spells_dict[summ_spell_id]



