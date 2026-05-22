from typing import Literal

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A single entity extracted by the Cloud Natural Language API or spaCy."""
    name: str
    type: str
    salience: float = 0.0
    sentiment_score: float = 0.0
    sentiment_magnitude: float = 0.0


class Actor(BaseModel):
    name: str
    role: str


class DateEvent(BaseModel):
    date: str
    event: str


class GeminiExtraction(BaseModel):
    """Structured extraction result from Gemini — one record per document."""
    core_issue: str = Field(default="")
    actors: list[Actor] = Field(default_factory=list)
    key_metrics: list[str] = Field(default_factory=list)
    dates: list[DateEvent] = Field(default_factory=list)
    sentiment_label: Literal["positive", "negative", "neutral"] = "neutral"
    sentiment_reason: str = Field(default="")


class ExtractionResult(BaseModel):
    """Full extraction output for one document — both extractors combined."""
    doc_id: str
    entities: list[Entity] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    gemini: GeminiExtraction | None = None
    error: str | None = None
