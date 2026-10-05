import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not found. Check your .env file.")

DB_NAME = "oss_health_predictor"

_client = None


def get_db():
    """
    Returns the MongoDB database object, reusing one client connection.
    """
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
    return _client[DB_NAME]