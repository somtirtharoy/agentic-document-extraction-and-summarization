from datetime import datetime, timezone
from typing import Any

from google.cloud import firestore

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SESSIONS_COLLECTION = "sessions"
_CACHE_COLLECTION = "doc_cache"


class FirestoreClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._db = firestore.Client(project=settings.gcp_project_id)

    # ── Session memory ────────────────────────────────────────────────────────

    def get_history(self, session_id: str) -> list[dict]:
        """Return all turns for a session, ordered by timestamp."""
        turns = (
            self._db.collection(_SESSIONS_COLLECTION)
            .document(session_id)
            .collection("turns")
            .order_by("ts")
            .stream()
        )
        return [t.to_dict() for t in turns]

    def append_turn(self, session_id: str, role: str, content: Any) -> None:
        """Append one turn to the session history."""
        self._db.collection(_SESSIONS_COLLECTION).document(session_id).collection(
            "turns"
        ).add(
            {
                "role": role,
                "content": content if isinstance(content, str) else str(content),
                "ts": datetime.now(timezone.utc),
            }
        )

    def append_tool_turn(
        self, session_id: str, tool_name: str, args: dict, observation: dict
    ) -> None:
        """Append a tool-call + observation pair to the session history."""
        self._db.collection(_SESSIONS_COLLECTION).document(session_id).collection(
            "turns"
        ).add(
            {
                "role": "tool",
                "tool_name": tool_name,
                "args": args,
                "observation": observation,
                "ts": datetime.now(timezone.utc),
            }
        )

    # ── Doc-level cache ───────────────────────────────────────────────────────

    def cache_get(self, doc_id: str, key: str) -> dict | None:
        """Return cached value for (doc_id, key) or None on miss."""
        ref = self._db.collection(_CACHE_COLLECTION).document(doc_id)
        snap = ref.get()
        if snap.exists:
            data = snap.to_dict() or {}
            if key in data:
                logger.info("Cache hit", extra={"doc_id": doc_id, "key": key})
                return data[key]
        return None

    def cache_set(self, doc_id: str, key: str, value: Any) -> None:
        """Store value in cache for (doc_id, key). Merges with existing keys."""
        self._db.collection(_CACHE_COLLECTION).document(doc_id).set(
            {key: value}, merge=True
        )
        logger.info("Cache set", extra={"doc_id": doc_id, "key": key})
