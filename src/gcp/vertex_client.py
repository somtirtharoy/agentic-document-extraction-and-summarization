import vertexai
from vertexai.generative_models import GenerativeModel

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_initialized = False
_model_cache: dict[str, GenerativeModel] = {}


def _ensure_init() -> None:
    global _initialized
    if _initialized:
        return
    settings = get_settings()
    vertexai.init(project=settings.gcp_project_id, location=settings.gcp_region)
    _initialized = True
    logger.info("Vertex AI initialized", extra={"project": settings.gcp_project_id})


def get_model(model_name: str | None = None) -> GenerativeModel:
    """Return a cached GenerativeModel instance for the given model name."""
    _ensure_init()
    settings = get_settings()
    name = model_name or settings.gemini_model

    if name not in _model_cache:
        _model_cache[name] = GenerativeModel(name)
        logger.info("Loaded Gemini model", extra={"model": name})

    return _model_cache[name]
