from typing import Any

from src.gcp.firestore_client import FirestoreClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AgentMemory:
    """Thin interface over FirestoreClient for agent-specific memory operations."""

    def __init__(self) -> None:
        self._store = FirestoreClient()

    def get_history(self, session_id: str) -> list[dict]:
        return self._store.get_history(session_id)

    def append_user(self, session_id: str, message: str) -> None:
        self._store.append_turn(session_id, "user", message)

    def append_model(self, session_id: str, message: str) -> None:
        self._store.append_turn(session_id, "model", message)

    def append_tool(
        self, session_id: str, tool_name: str, args: dict, observation: dict
    ) -> None:
        self._store.append_tool_turn(session_id, tool_name, args, observation)

    def cache_get(self, doc_id: str, key: str) -> Any | None:
        return self._store.cache_get(doc_id, key)

    def cache_set(self, doc_id: str, key: str, value: Any) -> None:
        self._store.cache_set(doc_id, key, value)
