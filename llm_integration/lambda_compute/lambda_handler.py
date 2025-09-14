# handler.py
import json
import importlib.resources as res

# Loading items JSON into zip allows for faster retrieval instead of using S3
# Load once at init to avoid re-reading on every invocation
with res.files("myfunc.data").joinpath("items_info.json").open("r", encoding="utf-8") as f:
    ITEMS_INFO = json.load(f)

def lambda_handler(event, context):
    # Use ITEMS_INFO freely; it's cached across warm invocations
    item = ITEMS_INFO.get("some_key")
    return {"ok": True, "item": item}