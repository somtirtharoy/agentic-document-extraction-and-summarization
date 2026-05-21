from google.cloud import language_v1

from config.settings import get_settings
from src.extraction.models import Entity
from src.utils.logging import get_logger
from src.utils.retry import gcp_retry

logger = get_logger(__name__)

# Cloud NL API hard limit
_MAX_BYTES = 99_000


class GCPNLExtractor:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = language_v1.LanguageServiceClient()
        self._project = settings.gcp_project_id

    @gcp_retry
    def extract(self, doc_id: str, text: str) -> tuple[list[Entity], list[str]]:
        """Run entity-sentiment analysis and text classification.

        Returns (entities, categories).
        """
        # Truncate to API byte limit
        encoded = text.encode("utf-8")[:_MAX_BYTES].decode("utf-8", errors="ignore")
        document = language_v1.Document(
            content=encoded,
            type_=language_v1.Document.Type.PLAIN_TEXT,
            language="en",
        )

        entities = self._get_entities(document)
        categories = self._get_categories(document)

        logger.info(
            "Cloud NL extraction complete",
            extra={"doc_id": doc_id, "entities": len(entities), "categories": len(categories)},
        )
        return entities, categories

    def _get_entities(self, document: language_v1.Document) -> list[Entity]:
        response = self._client.analyze_entity_sentiment(
            request={"document": document, "encoding_type": language_v1.EncodingType.UTF8}
        )
        return [
            Entity(
                name=e.name,
                type=language_v1.Entity.Type(e.type_).name,
                salience=round(e.salience, 4),
                sentiment_score=round(e.sentiment.score, 4),
                sentiment_magnitude=round(e.sentiment.magnitude, 4),
            )
            for e in response.entities
        ]

    def _get_categories(self, document: language_v1.Document) -> list[str]:
        try:
            response = self._client.classify_text(request={"document": document})
            return [c.name for c in response.categories]
        except Exception:
            # classify_text requires ~20+ sentences; silently skip short docs
            return []
