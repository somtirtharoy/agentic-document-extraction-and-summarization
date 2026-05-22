"""Tests for _pydantic_to_vertex_schema — the $ref inliner used by both extractors."""
import copy

from pydantic import BaseModel

from src.extraction.gemini_extractor import _pydantic_to_vertex_schema


class Leaf(BaseModel):
    value: str
    count: int = 0


class Middle(BaseModel):
    label: str
    leaf: Leaf


class Root(BaseModel):
    name: str
    middle: Middle
    items: list[Leaf]


class TestPydanticToVertexSchema:
    def test_simple_model_has_no_refs(self):
        schema = _pydantic_to_vertex_schema(Leaf)
        assert "$ref" not in str(schema)
        assert "$defs" not in schema

    def test_nested_model_refs_inlined(self):
        schema = _pydantic_to_vertex_schema(Middle)
        assert "$ref" not in str(schema)
        assert "$defs" not in schema

    def test_deeply_nested_refs_inlined(self):
        schema = _pydantic_to_vertex_schema(Root)
        assert "$ref" not in str(schema)
        assert "$defs" not in schema

    def test_nested_properties_accessible(self):
        schema = _pydantic_to_vertex_schema(Middle)
        leaf_props = schema["properties"]["leaf"]["properties"]
        assert "value" in leaf_props
        assert "count" in leaf_props

    def test_list_of_nested_model_inlined(self):
        schema = _pydantic_to_vertex_schema(Root)
        items_schema = schema["properties"]["items"]
        assert "items" in items_schema
        assert "$ref" not in str(items_schema)

    def test_title_fields_removed(self):
        schema = _pydantic_to_vertex_schema(Middle)
        assert "title" not in str(schema)

    def test_returns_dict(self):
        assert isinstance(_pydantic_to_vertex_schema(Leaf), dict)

    def test_does_not_mutate_original_schema(self):
        original = copy.deepcopy(Root.model_json_schema())
        _pydantic_to_vertex_schema(Root)
        assert Root.model_json_schema() == original

    def test_gemini_extraction_schema_has_no_refs(self):
        from src.extraction.models import GeminiExtraction

        schema = _pydantic_to_vertex_schema(GeminiExtraction)
        assert "$ref" not in str(schema)
        assert "$defs" not in schema
