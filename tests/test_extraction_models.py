"""Tests for Pydantic models in src/extraction/models.py."""
import pytest

from src.extraction.models import (
    Actor,
    DateEvent,
    Entity,
    ExtractionResult,
    GeminiExtraction,
)


class TestEntity:
    def test_required_fields(self):
        e = Entity(name="Google", type="ORGANIZATION")
        assert e.name == "Google"
        assert e.type == "ORGANIZATION"

    def test_default_numeric_fields(self):
        e = Entity(name="X", type="OTHER")
        assert e.salience == 0.0
        assert e.sentiment_score == 0.0
        assert e.sentiment_magnitude == 0.0

    def test_all_fields_set(self):
        e = Entity(
            name="Joe Biden",
            type="PERSON",
            salience=0.75,
            sentiment_score=0.2,
            sentiment_magnitude=0.9,
        )
        assert e.salience == 0.75
        assert e.sentiment_magnitude == 0.9


class TestActor:
    def test_basic_construction(self):
        a = Actor(name="Alice", role="CEO")
        assert a.name == "Alice"
        assert a.role == "CEO"


class TestDateEvent:
    def test_basic_construction(self):
        d = DateEvent(date="2024-01-15", event="Election day")
        assert d.date == "2024-01-15"
        assert d.event == "Election day"


class TestGeminiExtraction:
    def test_all_defaults(self):
        g = GeminiExtraction()
        assert g.core_issue == ""
        assert g.actors == []
        assert g.key_metrics == []
        assert g.dates == []
        assert g.sentiment_label == "neutral"
        assert g.sentiment_reason == ""

    def test_positive_sentiment(self):
        g = GeminiExtraction(sentiment_label="positive")
        assert g.sentiment_label == "positive"

    def test_negative_sentiment(self):
        g = GeminiExtraction(sentiment_label="negative")
        assert g.sentiment_label == "negative"

    def test_invalid_sentiment_raises(self):
        with pytest.raises(Exception):
            GeminiExtraction(sentiment_label="unknown")

    def test_nested_actors_constructed(self):
        g = GeminiExtraction(actors=[{"name": "Alice", "role": "CEO"}])
        assert isinstance(g.actors[0], Actor)
        assert g.actors[0].name == "Alice"

    def test_nested_dates_constructed(self):
        g = GeminiExtraction(dates=[{"date": "2024-01", "event": "Budget vote"}])
        assert isinstance(g.dates[0], DateEvent)
        assert g.dates[0].event == "Budget vote"

    def test_key_metrics_list(self):
        g = GeminiExtraction(key_metrics=["GDP +3.2%", "Unemployment 4.1%"])
        assert len(g.key_metrics) == 2


class TestExtractionResult:
    def test_minimal_construction(self):
        r = ExtractionResult(doc_id="abc-123")
        assert r.doc_id == "abc-123"
        assert r.entities == []
        assert r.categories == []
        assert r.gemini is None
        assert r.error is None

    def test_with_entities(self):
        entities = [Entity(name="Google", type="ORGANIZATION")]
        r = ExtractionResult(doc_id="abc-123", entities=entities)
        assert len(r.entities) == 1

    def test_with_gemini_result(self):
        gemini = GeminiExtraction(core_issue="Budget crisis")
        r = ExtractionResult(doc_id="abc-123", gemini=gemini)
        assert r.gemini.core_issue == "Budget crisis"

    def test_error_field(self):
        r = ExtractionResult(doc_id="abc-123", error="API timeout")
        assert r.error == "API timeout"
