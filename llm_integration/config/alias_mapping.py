ROLES = {
    "top":"TOP",

    "jungle":"JUNGLE", "jg":"JUNGLE", "jung":"JUNGLE","jgl":"JUNGLE",

    "middle":"MIDDLE","mid":"MIDDLE", 

    "bottom":"BOTTOM", "bot":"BOTTOM", "adc":"BOTTOM", 
    "adcarry":"BOTTOM", "ad carry":"BOTTOM", "ad_carry":"BOTTOM",

    "support":"UTILITY", "supp":"UTILITY", "sup":"UTILITY", "utility": "UTILITY"
}

BINARY_REPLIES = {
    "yes": True, "y": True,
    "no": False, "n":False
}

QUEUES = {
    "Ranked Solo Queue":"ranked_solo_queue", "solo queue":"ranked_solo_queue", "solo":"ranked_solo_queue",
    "Ranked Including Flex":"ranked_including_flex", "Ranked Flex": "ranked_including_flex", "all ranked": "ranked_including_flex",
    "draft":"draft", "norm":"draft", "norms":"draft", "normal":"draft",
    "All Queues":"all_queues", "All":"all_queues"
}

CHAMPION_CRITERIA = {
    "Win Rate": "win_rate",
    "Play Rate": "role_play_rate",
    "Manual Champion Selection": "manual_selection"
}

METHODOLOGIES = {
    "Compare with similar players": "collaborative_filtering",
    "Champion pool optimization": "mathematical_optimization",
    "Qualitative exploration": "natural_language_exploration"
}
