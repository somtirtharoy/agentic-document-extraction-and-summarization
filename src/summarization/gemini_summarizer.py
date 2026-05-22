import copy
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from vertexai.generative_models import GenerationConfig

from config.settings import get_settings
from src.gcp.vertex_client import get_model
from src.utils.logging import get_logger
from src.utils.retry import gcp_retry

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "config" / "prompts" / "summarization.yaml"

# ~4k tokens in chars — docs above this threshold go through map-reduce
_MAP_REDUCE_THRESHOLD = 16_000
_CHUNK_SIZE = 12_000


class SummaryOutput(BaseModel):
    tldr: str = Field(default="")
    bullets: list[str] = Field(default_factory=list)
    abstract: str = Field(default="")


def _load_prompt() -> dict:
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)


def _pydantic_to_vertex_schema(model: type[BaseModel]) -> dict:
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


class GeminiSummarizer:
    def __init__(self) -> None:
        settings = get_settings()
        self._model = get_model(settings.gemini_model_flash)
        self._prompt = _load_prompt()
        self._generation_config = GenerationConfig(
            temperature=0.2,
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_schema=_pydantic_to_vertex_schema(SummaryOutput),
        )

    def summarize(self, doc_id: str, text: str) -> SummaryOutput:
        """Summarize a document. Uses map-reduce for long texts."""
        if len(text) <= _MAP_REDUCE_THRESHOLD:
            return self._summarize_direct(doc_id, text)
        return self._summarize_map_reduce(doc_id, text)

    @gcp_retry
    def _summarize_direct(self, doc_id: str, text: str) -> SummaryOutput:
        prompt = self._prompt["user_template"].format(text=text)
        response = self._model.generate_content(
            [self._prompt["system"], prompt],
            generation_config=self._generation_config,
        )
        result = SummaryOutput.model_validate_json(response.text)
        logger.info("Summarized (direct)", extra={"doc_id": doc_id})
        return result

    def _summarize_map_reduce(self, doc_id: str, text: str) -> SummaryOutput:
        """Split → summarize each chunk → reduce into final summary."""
        chunks = [text[i: i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]
        logger.info("Map-reduce summarization", extra={"doc_id": doc_id, "chunks": len(chunks)})

        chunk_summaries = [self._summarize_direct(doc_id, chunk).abstract for chunk in chunks]

        combined = "\n\n".join(f"[Part {i+1}]: {s}" for i, s in enumerate(chunk_summaries))
        reduce_prompt = self._prompt["reduce_template"].format(chunk_summaries=combined)

        response = self._model.generate_content(
            [self._prompt["system"], reduce_prompt],
            generation_config=self._generation_config,
        )
        result = SummaryOutput.model_validate_json(response.text)
        logger.info("Map-reduce complete", extra={"doc_id": doc_id})
        return result
