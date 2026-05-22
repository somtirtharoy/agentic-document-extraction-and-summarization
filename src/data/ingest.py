import uuid
from datetime import UTC, datetime

from datasets import load_dataset
from tqdm import tqdm

from config.settings import get_settings
from src.data.schema import DOCUMENTS_SCHEMA
from src.gcp.bq_client import BQClient
from src.gcp.gcs_client import GCSClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

GCS_RAW_PATH = "raw/documents.jsonl"
BQ_TABLE = "documents"


def _normalize_record(row: dict, source: str) -> dict:
    """Map a HuggingFace CNN/DailyMail row to the documents schema."""
    return {
        "doc_id": str(uuid.uuid4()),
        "text": row["article"].strip(),
        "reference_summary": row["highlights"].strip(),
        "source": source,
        "ingested_at": datetime.now(UTC).isoformat(),
    }


def run(limit: int = 1500) -> int:
    """Download CNN/DailyMail, upload to GCS, load into BQ. Returns row count."""
    settings = get_settings()
    gcs = GCSClient()
    bq = BQClient()

    logger.info("Loading CNN/DailyMail dataset from HuggingFace", extra={"limit": limit})
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", trust_remote_code=True)
    dataset = dataset.select(range(min(limit, len(dataset))))

    records = [
        _normalize_record(row, source="cnn_dailymail")
        for row in tqdm(dataset, desc="Normalizing")
    ]

    # Drop records where article text is empty
    records = [r for r in records if r["text"]]
    logger.info("Normalized records", extra={"count": len(records)})

    # Upload JSONL to GCS
    gcs_uri = gcs.upload_jsonl(records, GCS_RAW_PATH)

    # Ensure BQ table exists then load from GCS
    bq.create_table(BQ_TABLE, DOCUMENTS_SCHEMA)
    bq.load_from_gcs(gcs_uri, BQ_TABLE, DOCUMENTS_SCHEMA, write_disposition="WRITE_TRUNCATE")

    logger.info("Ingestion complete", extra={"project": settings.gcp_project_id, "rows": len(records)})
    return len(records)
