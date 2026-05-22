"""Tests for Pydantic input-validation schemas in src/agent/schemas.py."""
import pytest

from src.agent.schemas import (
    AggregateEntitiesInput,
    CompareSummariesInput,
    ExtractEntitiesInput,
    SearchDocumentsInput,
    SummarizeDocumentInput,
)


class TestSearchDocumentsInput:
    def test_valid_defaults(self):
        s = SearchDocumentsInput(query="climate change")
        assert s.query == "climate change"
        assert s.top_k == 5

    def test_explicit_top_k(self):
        s = SearchDocumentsInput(query="test", top_k=3)
        assert s.top_k == 3

    def test_top_k_at_max_boundary(self):
        s = SearchDocumentsInput(query="test", top_k=10)
        assert s.top_k == 10

    def test_top_k_exceeds_max_raises(self):
        with pytest.raises(Exception):
            SearchDocumentsInput(query="test", top_k=11)

    def test_top_k_zero_raises(self):
        with pytest.raises(Exception):
            SearchDocumentsInput(query="test", top_k=0)


class TestExtractEntitiesInput:
    def test_valid(self):
        e = ExtractEntitiesInput(doc_id="abc-123")
        assert e.doc_id == "abc-123"


class TestSummarizeDocumentInput:
    def test_default_style_is_tldr(self):
        s = SummarizeDocumentInput(doc_id="abc")
        assert s.style == "tldr"

    def test_bullets_style(self):
        s = SummarizeDocumentInput(doc_id="abc", style="bullets")
        assert s.style == "bullets"

    def test_abstract_style(self):
        s = SummarizeDocumentInput(doc_id="abc", style="abstract")
        assert s.style == "abstract"

    def test_invalid_style_raises(self):
        with pytest.raises(Exception):
            SummarizeDocumentInput(doc_id="abc", style="essay")

    def test_invalid_style_raises_for_short_name(self):
        with pytest.raises(Exception):
            SummarizeDocumentInput(doc_id="abc", style="tl;dr")


class TestAggregateEntitiesInput:
    def test_valid(self):
        a = AggregateEntitiesInput(doc_ids=["id1", "id2"])
        assert len(a.doc_ids) == 2
        assert a.top_n == 10

    def test_custom_top_n(self):
        a = AggregateEntitiesInput(doc_ids=["id1"], top_n=25)
        assert a.top_n == 25

    def test_top_n_at_max_boundary(self):
        a = AggregateEntitiesInput(doc_ids=["id1"], top_n=50)
        assert a.top_n == 50

    def test_top_n_exceeds_max_raises(self):
        with pytest.raises(Exception):
            AggregateEntitiesInput(doc_ids=["id1"], top_n=51)

    def test_top_n_zero_raises(self):
        with pytest.raises(Exception):
            AggregateEntitiesInput(doc_ids=["id1"], top_n=0)


class TestCompareSummariesInput:
    def test_valid(self):
        c = CompareSummariesInput(doc_ids=["a", "b"], focus="common themes")
        assert c.focus == "common themes"
        assert len(c.doc_ids) == 2

    def test_single_doc_id(self):
        c = CompareSummariesInput(doc_ids=["only-one"], focus="sentiment")
        assert len(c.doc_ids) == 1
