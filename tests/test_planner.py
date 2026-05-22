"""Tests for the planner's history-to-contents conversion and ReAct loop."""
from unittest.mock import MagicMock, patch


class TestHistoryToContents:
    """_history_to_contents converts Firestore dicts to Vertex AI Content objects."""

    def _run(self, history):
        with patch("src.agent.planner.Content") as MockContent, \
             patch("src.agent.planner.Part") as MockPart:
            from src.agent.planner import _history_to_contents
            result = _history_to_contents(history)
            return result, MockContent, MockPart

    def test_user_turn_creates_one_content(self):
        history = [{"role": "user", "content": "hello"}]
        result, MockContent, MockPart = self._run(history)
        assert len(result) == 1
        MockPart.from_text.assert_called_with("hello")
        MockContent.assert_called_with(role="user", parts=[MockPart.from_text.return_value])

    def test_model_turn_creates_one_content(self):
        history = [{"role": "model", "content": "I will help"}]
        result, MockContent, MockPart = self._run(history)
        assert len(result) == 1
        MockContent.assert_called_with(role="model", parts=[MockPart.from_text.return_value])

    def test_tool_turn_creates_two_contents(self):
        history = [{"role": "tool", "tool_name": "search_documents",
                    "args": {"query": "test"}, "observation": {"results": []}}]
        result, MockContent, MockPart = self._run(history)
        # One Content for the function_call (model role) + one for the function_response (user role)
        assert len(result) == 2

    def test_tool_turn_calls_from_function_call(self):
        history = [{"role": "tool", "tool_name": "search_documents",
                    "args": {"query": "test"}, "observation": {"results": []}}]
        _, _, MockPart = self._run(history)
        MockPart.from_function_call.assert_called_once_with(
            name="search_documents", args={"query": "test"}
        )

    def test_tool_turn_calls_from_function_response(self):
        obs = {"results": [{"doc_id": "abc"}]}
        history = [{"role": "tool", "tool_name": "search_documents",
                    "args": {}, "observation": obs}]
        _, _, MockPart = self._run(history)
        MockPart.from_function_response.assert_called_once_with(
            name="search_documents", response={"result": obs}
        )

    def test_unknown_role_is_skipped(self):
        history = [{"role": "system", "content": "ignored"}]
        result, _, _ = self._run(history)
        assert result == []

    def test_empty_history_returns_empty_list(self):
        result, _, _ = self._run([])
        assert result == []

    def test_mixed_history_correct_content_count(self):
        history = [
            {"role": "user", "content": "find articles"},           # 1 Content
            {"role": "tool", "tool_name": "search_documents",        # 2 Contents
             "args": {}, "observation": {}},
            {"role": "model", "content": "Here are the results"},   # 1 Content
        ]
        result, _, _ = self._run(history)
        assert len(result) == 4

    def test_tool_turn_missing_args_defaults_to_empty_dict(self):
        history = [{"role": "tool", "tool_name": "search_documents", "observation": {}}]
        # Should not raise — args defaults to {}
        result, _, MockPart = self._run(history)
        MockPart.from_function_call.assert_called_once_with(
            name="search_documents", args={}
        )


class TestPlannerInit:
    def test_planner_instantiates_model_with_system_instruction(self):
        with patch("src.agent.planner._ensure_init"), \
             patch("src.agent.planner.GenerativeModel") as MockModel:
            mock_tools = MagicMock()
            mock_memory = MagicMock()
            from src.agent.planner import Planner

            Planner(tools=mock_tools, memory=mock_memory)
            # GenerativeModel should be called with system_instruction kwarg
            call_kwargs = MockModel.call_args[1]
            assert "system_instruction" in call_kwargs
