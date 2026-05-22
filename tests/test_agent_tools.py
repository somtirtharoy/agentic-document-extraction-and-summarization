"""Tests for AgentTools with all GCP dependencies mocked."""
import json
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.models import Entity
from src.summarization.gemini_summarizer import SummaryOutput


@pytest.fixture
def tools():
    """AgentTools with BQ, Cloud NL, Gemini extractor, and summarizer all mocked."""
    patches = [
        patch("src.agent.tools.BQClient"),
        patch("src.agent.tools.GCPNLExtractor"),
        patch("src.agent.tools.GeminiExtractor"),
        patch("src.agent.tools.GeminiSummarizer"),
        patch("src.agent.tools.get_model"),
    ]
    [p.start() for p in patches]
    try:
        mock_memory = MagicMock()
        mock_memory.cache_get.return_value = None

        from src.agent.tools import AgentTools

        agent_tools = AgentTools(memory=mock_memory)
        agent_tools._memory = mock_memory
        agent_tools._bq = MagicMock()
        agent_tools._nl_extractor = MagicMock()
        agent_tools._gemini_extractor = MagicMock()
        agent_tools._summarizer = MagicMock()
        agent_tools._project = "test-project"
        yield agent_tools, mock_memory
    finally:
        for p in patches:
            p.stop()


# ── dispatch ──────────────────────────────────────────────────────────────────


class TestDispatch:
    def test_unknown_tool_returns_error(self, tools):
        at, _ = tools
        result = at.dispatch("nonexistent_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_known_tool_is_dispatched(self, tools):
        at, _ = tools
        at._bq.query.return_value = []
        result = at.dispatch("search_documents", {"query": "test"})
        assert "Unknown tool" not in result.get("error", "")

    def test_all_five_tools_registered(self, tools):
        at, _ = tools
        at._bq.query.return_value = []
        tool_args = {
            "search_documents": {"query": "test"},
            "extract_entities": {"doc_id": "d"},
            "summarize_document": {"doc_id": "d"},
            "aggregate_entities": {"doc_ids": ["d"]},
            "compare_summaries": {"doc_ids": ["d"], "focus": "themes"},
        }
        for name, args in tool_args.items():
            result = at.dispatch(name, args)
            assert "Unknown tool" not in result.get("error", "")


# ── search_documents ──────────────────────────────────────────────────────────


class TestSearchDocuments:
    def test_returns_results_list(self, tools):
        at, _ = tools
        at._bq.query.return_value = [
            {"doc_id": "abc-123", "snippet": "test snippet", "score": 3}
        ]
        result = at.search_documents("test query")
        assert "results" in result
        assert result["results"][0]["doc_id"] == "abc-123"

    def test_empty_results_includes_message(self, tools):
        at, _ = tools
        at._bq.query.return_value = []
        result = at.search_documents("nonexistent topic")
        assert result["results"] == []
        assert "message" in result

    def test_bq_error_returns_error_dict(self, tools):
        at, _ = tools
        at._bq.query.side_effect = Exception("BQ unavailable")
        result = at.search_documents("test")
        assert "error" in result
        assert "hint" in result

    def test_top_k_passed_to_query(self, tools):
        at, _ = tools
        at._bq.query.return_value = []
        at.search_documents("query", top_k=3)
        at._bq.query.assert_called_once()


# ── extract_entities ──────────────────────────────────────────────────────────


class TestExtractEntities:
    def test_cache_hit_skips_api_call(self, tools):
        at, mock_memory = tools
        cached = [{"name": "Google", "type": "ORGANIZATION"}]
        mock_memory.cache_get.return_value = cached
        result = at.extract_entities("doc-1")
        assert result["source"] == "cache"
        assert result["entities"] == cached
        at._nl_extractor.extract.assert_not_called()

    def test_doc_not_found_returns_error(self, tools):
        at, mock_memory = tools
        mock_memory.cache_get.return_value = None
        at._bq.query.return_value = []
        result = at.extract_entities("missing-doc")
        assert "error" in result
        assert "not found" in result["error"]

    def test_cloud_nl_extraction_success(self, tools):
        at, mock_memory = tools
        mock_memory.cache_get.return_value = None
        at._bq.query.return_value = [{"text": "article text about Google"}]
        mock_entity = Entity(name="Google", type="ORGANIZATION", salience=0.9)
        at._nl_extractor.extract.return_value = ([mock_entity], ["Technology"])
        result = at.extract_entities("doc-1")
        assert result["source"] == "cloud_nl"
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Google"

    def test_result_cached_after_api_call(self, tools):
        at, mock_memory = tools
        mock_memory.cache_get.return_value = None
        at._bq.query.return_value = [{"text": "text"}]
        at._nl_extractor.extract.return_value = ([], [])
        at.extract_entities("doc-1")
        mock_memory.cache_set.assert_called_once_with("doc-1", "entities", {"entities": [], "categories": []})

    def test_api_error_returns_error_dict(self, tools):
        at, mock_memory = tools
        mock_memory.cache_get.return_value = None
        at._bq.query.return_value = [{"text": "text"}]
        at._nl_extractor.extract.side_effect = Exception("API quota exceeded")
        result = at.extract_entities("doc-1")
        assert "error" in result


# ── summarize_document ────────────────────────────────────────────────────────


class TestSummarizeDocument:
    def test_returns_from_bq_cache(self, tools):
        at, _ = tools
        at._bq.query.return_value = [{
            "tldr": "Short summary.",
            "bullets": json.dumps(["Point 1", "Point 2"]),
            "abstract": "Abstract text.",
        }]
        result = at.summarize_document("doc-1", style="tldr")
        assert result["source"] == "cache"
        assert result["summary"] == "Short summary."

    def test_bullets_style_from_cache(self, tools):
        at, _ = tools
        at._bq.query.return_value = [{
            "tldr": "TLDR.",
            "bullets": json.dumps(["P1", "P2"]),
            "abstract": "Abstract.",
        }]
        result = at.summarize_document("doc-1", style="bullets")
        assert result["summary"] == ["P1", "P2"]

    def test_live_summarization_on_cache_miss(self, tools):
        at, _ = tools
        at._bq.query.side_effect = [[], [{"text": "article text"}]]
        mock_output = SummaryOutput(tldr="Live summary.", bullets=[], abstract="")
        at._summarizer.summarize.return_value = mock_output
        result = at.summarize_document("doc-1", style="tldr")
        assert result["source"] == "live"
        assert result["summary"] == "Live summary."

    def test_invalid_style_returns_error(self, tools):
        at, _ = tools
        result = at.summarize_document("doc-1", style="invalid_style")
        assert "error" in result

    def test_doc_not_found_on_live_path_returns_error(self, tools):
        at, _ = tools
        at._bq.query.side_effect = [[], []]
        result = at.summarize_document("doc-1")
        assert "error" in result


# ── aggregate_entities ────────────────────────────────────────────────────────


class TestAggregateEntities:
    def test_empty_doc_ids_returns_error(self, tools):
        at, _ = tools
        result = at.aggregate_entities([])
        assert "error" in result

    def test_no_entity_rows_in_bq(self, tools):
        at, _ = tools
        at._bq.query.return_value = []
        result = at.aggregate_entities(["doc-1"])
        assert result["top_entities"] == []

    def test_aggregation_ranks_by_frequency(self, tools):
        at, _ = tools
        at._bq.query.return_value = [
            {"entity_name": "Google", "entity_type": "ORGANIZATION", "sentiment_score": 0.5},
            {"entity_name": "Google", "entity_type": "ORGANIZATION", "sentiment_score": 0.3},
            {"entity_name": "CEO", "entity_type": "PERSON", "sentiment_score": 0.0},
        ]
        result = at.aggregate_entities(["doc-1"])
        top = result["top_entities"]
        assert top[0]["name"] == "Google"
        assert top[0]["frequency"] == 2

    def test_avg_sentiment_computed(self, tools):
        at, _ = tools
        at._bq.query.return_value = [
            {"entity_name": "A", "entity_type": "ORG", "sentiment_score": 0.4},
            {"entity_name": "A", "entity_type": "ORG", "sentiment_score": 0.6},
        ]
        result = at.aggregate_entities(["doc-1"])
        assert result["top_entities"][0]["avg_sentiment"] == pytest.approx(0.5, rel=1e-3)

    def test_sentiment_distribution_counts(self, tools):
        at, _ = tools
        at._bq.query.return_value = [
            {"entity_name": "Pos", "entity_type": "OTHER", "sentiment_score": 0.5},
            {"entity_name": "Neg", "entity_type": "OTHER", "sentiment_score": -0.5},
            {"entity_name": "Neu", "entity_type": "OTHER", "sentiment_score": 0.0},
        ]
        result = at.aggregate_entities(["doc-1"])
        dist = result["sentiment_distribution"]
        assert dist["positive"] == 1
        assert dist["negative"] == 1
        assert dist["neutral"] == 1

    def test_bq_error_returns_error_dict(self, tools):
        at, _ = tools
        at._bq.query.side_effect = Exception("BQ error")
        result = at.aggregate_entities(["doc-1"])
        assert "error" in result

    def test_top_n_limits_results(self, tools):
        at, _ = tools
        at._bq.query.return_value = [
            {"entity_name": f"Entity{i}", "entity_type": "OTHER", "sentiment_score": 0.0}
            for i in range(20)
        ]
        result = at.aggregate_entities(["doc-1"], top_n=5)
        assert len(result["top_entities"]) == 5


# ── compare_summaries ─────────────────────────────────────────────────────────


class TestCompareSummaries:
    def test_no_summaries_returns_error(self, tools):
        at, _ = tools
        at._bq.query.return_value = []
        result = at.compare_summaries(["doc-1"], focus="climate impact")
        assert "error" in result

    def test_synthesis_returned_on_success(self, tools):
        at, _ = tools
        at._bq.query.return_value = [{
            "tldr": "T", "bullets": "[]", "abstract": "This is the abstract."
        }]
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = "Synthesis result."
        with patch("src.agent.tools.get_model", return_value=mock_model):
            result = at.compare_summaries(["doc-1"], focus="climate")
        assert result["synthesis"] == "Synthesis result."
        assert "doc-1" in result["doc_ids_used"]
