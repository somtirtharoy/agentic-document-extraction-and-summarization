from src.extraction.models import Entity
from src.utils.logging import get_logger

logger = get_logger(__name__)

# spaCy label → Cloud NL API-style type name for consistency in comparison
_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "OTHER",
    "TIME": "OTHER",
    "MONEY": "OTHER",
    "PERCENT": "OTHER",
    "PRODUCT": "CONSUMER_GOOD",
    "EVENT": "EVENT",
    "WORK_OF_ART": "WORK_OF_ART",
    "LAW": "OTHER",
    "NORP": "ORGANIZATION",
}


class SpacyExtractor:
    def __init__(self) -> None:
        import spacy
        self._nlp = spacy.load("en_core_web_sm")

    def extract(self, doc_id: str, text: str) -> list[Entity]:
        """Extract named entities using spaCy en_core_web_sm."""
        # Truncate to keep inference fast
        doc = self._nlp(text[:10_000])

        seen: set[str] = set()
        entities: list[Entity] = []

        for ent in doc.ents:
            key = (ent.text.strip().lower(), ent.label_)
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                Entity(
                    name=ent.text.strip(),
                    type=_LABEL_MAP.get(ent.label_, "OTHER"),
                    salience=0.0,        # spaCy doesn't compute salience
                    sentiment_score=0.0,
                    sentiment_magnitude=0.0,
                )
            )

        logger.info(
            "spaCy extraction complete",
            extra={"doc_id": doc_id, "entities": len(entities)},
        )
        return entities
