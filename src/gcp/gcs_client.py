import json
from pathlib import Path

from google.cloud import storage

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GCSClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = storage.Client(project=settings.gcp_project_id)
        self._bucket_name = settings.gcs_bucket

    @property
    def _bucket(self) -> storage.Bucket:
        return self._client.bucket(self._bucket_name)

    def upload_jsonl(self, records: list[dict], gcs_path: str) -> str:
        """Serialize records as newline-delimited JSON and upload to GCS.

        Returns the gs:// URI of the uploaded object.
        """
        blob = self._bucket.blob(gcs_path)
        payload = "\n".join(json.dumps(r) for r in records)
        blob.upload_from_string(payload, content_type="application/jsonl")
        uri = f"gs://{self._bucket_name}/{gcs_path}"
        logger.info("Uploaded JSONL to GCS", extra={"uri": uri, "rows": len(records)})
        return uri

    def upload_file(self, local_path: str | Path, gcs_path: str) -> str:
        """Upload a local file to GCS. Returns the gs:// URI."""
        blob = self._bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        uri = f"gs://{self._bucket_name}/{gcs_path}"
        logger.info("Uploaded file to GCS", extra={"uri": uri})
        return uri

    def download_as_text(self, gcs_path: str) -> str:
        return self._bucket.blob(gcs_path).download_as_text()

    def blob_exists(self, gcs_path: str) -> bool:
        return self._bucket.blob(gcs_path).exists()
