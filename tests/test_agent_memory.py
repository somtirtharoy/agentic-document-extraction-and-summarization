"""Tests for AgentMemory — verifies delegation to FirestoreClient."""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def memory():
    with patch("src.gcp.firestore_client.firestore.Client"):
        from src.agent.memory import AgentMemory

        mem = AgentMemory()
        mock_store = MagicMock()
        mem._store = mock_store
        yield mem, mock_store


class TestGetHistory:
    def test_delegates_to_store(self, memory):
        mem, mock_store = memory
        expected = [{"role": "user", "content": "hi"}]
        mock_store.get_history.return_value = expected
        result = mem.get_history("session-1")
        mock_store.get_history.assert_called_once_with("session-1")
        assert result == expected

    def test_empty_history_returned(self, memory):
        mem, mock_store = memory
        mock_store.get_history.return_value = []
        assert mem.get_history("empty-session") == []


class TestAppendTurns:
    def test_append_user_delegates(self, memory):
        mem, mock_store = memory
        mem.append_user("s1", "hello")
        mock_store.append_turn.assert_called_once_with("s1", "user", "hello")

    def test_append_model_delegates(self, memory):
        mem, mock_store = memory
        mem.append_model("s1", "response text")
        mock_store.append_turn.assert_called_once_with("s1", "model", "response text")

    def test_append_tool_delegates(self, memory):
        mem, mock_store = memory
        args = {"query": "test"}
        obs = {"results": []}
        mem.append_tool("s1", "search_documents", args, obs)
        mock_store.append_tool_turn.assert_called_once_with("s1", "search_documents", args, obs)


class TestCache:
    def test_cache_get_hit(self, memory):
        mem, mock_store = memory
        mock_store.cache_get.return_value = {"entities": [{"name": "Google"}]}
        result = mem.cache_get("doc-1", "entities")
        mock_store.cache_get.assert_called_once_with("doc-1", "entities")
        assert result["entities"][0]["name"] == "Google"

    def test_cache_get_miss_returns_none(self, memory):
        mem, mock_store = memory
        mock_store.cache_get.return_value = None
        assert mem.cache_get("doc-1", "missing") is None

    def test_cache_set_delegates(self, memory):
        mem, mock_store = memory
        value = {"entities": []}
        mem.cache_set("doc-1", "entities", value)
        mock_store.cache_set.assert_called_once_with("doc-1", "entities", value)
