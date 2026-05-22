from typing import Literal

from pydantic import BaseModel, Field
from vertexai.generative_models import FunctionDeclaration, Tool

# ── Tool input schemas (validated before execution) ───────────────────────────

class SearchDocumentsInput(BaseModel):
    query: str = Field(description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of results to return")


class ExtractEntitiesInput(BaseModel):
    doc_id: str = Field(description="Document ID to extract entities from")


class SummarizeDocumentInput(BaseModel):
    doc_id: str = Field(description="Document ID to summarize")
    style: Literal["tldr", "bullets", "abstract"] = Field(
        default="tldr", description="Summary style: tldr, bullets, or abstract"
    )


class AggregateEntitiesInput(BaseModel):
    doc_ids: list[str] = Field(description="List of document IDs to aggregate entities across")
    top_n: int = Field(default=10, ge=1, le=50, description="Number of top entities to return")


class CompareSummariesInput(BaseModel):
    doc_ids: list[str] = Field(description="List of document IDs to compare")
    focus: str = Field(description="Specific aspect or question to focus the comparison on")


# ── Vertex AI FunctionDeclarations ────────────────────────────────────────────

TOOL_DECLARATIONS = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="search_documents",
            description="Search the document corpus for articles relevant to a query. Returns doc_id, snippet, and relevance score.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "top_k": {"type": "integer", "description": "Number of results to return (default 5, max 10)"},
                },
                "required": ["query"],
            },
        ),
        FunctionDeclaration(
            name="extract_entities",
            description="Extract named entities, sentiment, and key facts from a specific document. Results are cached.",
            parameters={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document ID to extract entities from"},
                },
                "required": ["doc_id"],
            },
        ),
        FunctionDeclaration(
            name="summarize_document",
            description="Get a summary of a specific document. Cached after first call.",
            parameters={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document ID to summarize"},
                    "style": {
                        "type": "string",
                        "enum": ["tldr", "bullets", "abstract"],
                        "description": "tldr=one sentence, bullets=3 key points, abstract=75-word paragraph",
                    },
                },
                "required": ["doc_id"],
            },
        ),
        FunctionDeclaration(
            name="aggregate_entities",
            description="Count and rank named entities across multiple documents, with sentiment distribution.",
            parameters={
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to aggregate",
                    },
                    "top_n": {"type": "integer", "description": "Number of top entities to return (default 10)"},
                },
                "required": ["doc_ids"],
            },
        ),
        FunctionDeclaration(
            name="compare_summaries",
            description="Synthesise a cross-document answer by comparing summaries of multiple articles on a specific focus.",
            parameters={
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to compare",
                    },
                    "focus": {"type": "string", "description": "Specific aspect or question to focus the comparison on"},
                },
                "required": ["doc_ids", "focus"],
            },
        ),
    ]
)
