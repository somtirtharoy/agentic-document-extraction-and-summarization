import copy
from pathlib import Path

import yaml
from pydantic import BaseModel
from vertexai.generative_models import GenerationConfig

from config.settings import get_settings
from src.extraction.models import GeminiExtraction
from src.gcp.vertex_client import get_model
from src.utils.logging import get_logger
from src.utils.retry import gcp_retry

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "config" / "prompts" / "entity_extraction.yaml"
_MAX_CHARS = 12_000  # stay well within Gemini context for this task


def _load_prompt() -> dict:
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)


def _pydantic_to_vertex_schema(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to a Vertex-compatible JSON schema dict.

    Vertex GenerationConfig expects a plain dict, not a Pydantic class, and
    does not support JSON Schema $ref / $defs — nested models are inlined.
    """
    schema = copy.deepcopy(model.model_json_schema())
    defs = schema.pop("$defs", {})

    def resolve(node: object) -> object:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return resolve(copy.deepcopy(defs[name]))
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


class GeminiExtractor:
    def __init__(self) -> None:
        settings = get_settings()
        self._model = get_model(settings.gemini_model_flash)
        self._prompt = _load_prompt()
        self._generation_config = GenerationConfig(
            temperature=0.0,        # deterministic extraction
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_schema=_pydantic_to_vertex_schema(GeminiExtraction),
        )

    @gcp_retry
    def extract(self, doc_id: str, text: str) -> GeminiExtraction:
        """Extract structured information from a document using Gemini."""
        truncated = text[:_MAX_CHARS]
        prompt = self._prompt["user_template"].format(text=truncated)

        response = self._model.generate_content(
            [self._prompt["system"], prompt],
            generation_config=self._generation_config,
        )

        result = GeminiExtraction.model_validate_json(response.text)
        logger.info(
            "Gemini extraction complete",
            extra={
                "doc_id": doc_id,
                "actors": len(result.actors),
                "metrics": len(result.key_metrics),
            },
        )
        return result
