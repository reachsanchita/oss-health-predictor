"""MongoDB connection (collections: repos, monthly_snapshots, live_counters)."""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=3000)
db = _client[os.getenv("MONGODB_DB", "oss_health")]
repos = db["repos"]
snapshots = db["monthly_snapshots"]
counters = db["live_counters"]


def init_indexes():
    repos.create_index("full_name", unique=True)
    snapshots.create_index([("repository", 1), ("year_month", 1)], unique=True)
    counters.create_index("key", unique=True)
