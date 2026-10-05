import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB", "vivasensei")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

sessions_collection = db["viva_sessions"]


def test_connection():
    client.admin.command("ping")
    return True


def create_session(session_id, viva_settings, messages=None):
    document = {
        "session_id": session_id,
        "viva": viva_settings,
        "messages": messages or [],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    sessions_collection.insert_one(document)


def save_session(session_id, viva_settings, messages):
    sessions_collection.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "viva": viva_settings,
                "messages": messages,
                "updated_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )


def load_session(session_id):
    return sessions_collection.find_one(
        {"session_id": session_id},
        {"_id": 0},
    )