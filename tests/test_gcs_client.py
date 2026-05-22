"""Tests for GCSClient with mocked google.cloud.storage."""
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def gcs():
    with patch("src.gcp.gcs_client.storage") as mock_storage:
        mock_client = MagicMock()
        mock_storage.Client.return_value = mock_client
        from src.gcp.gcs_client import GCSClient

        client = GCSClient()
        client._bucket_name = "test-bucket"
        client._client = mock_client
        yield client, mock_client


class TestUploadJsonl:
    def test_returns_gs_uri(self, gcs):
        client, mock_client = gcs
        mock_blob = MagicMock()
        mock_client.bucket.return_value.blob.return_value = mock_blob
        uri = client.upload_jsonl([{"a": 1}], "raw/docs.jsonl")
        assert uri == "gs://test-bucket/raw/docs.jsonl"

    def test_uploads_newline_delimited_json(self, gcs):
        client, mock_client = gcs
        mock_blob = MagicMock()
        mock_client.bucket.return_value.blob.return_value = mock_blob
        records = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
        client.upload_jsonl(records, "raw/docs.jsonl")
        uploaded = mock_blob.upload_from_string.call_args[0][0]
        lines = uploaded.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == records[0]
        assert json.loads(lines[1]) == records[1]


class TestBlobExists:
    def test_returns_true_when_blob_exists(self, gcs):
        client, mock_client = gcs
        mock_client.bucket.return_value.blob.return_value.exists.return_value = True
        assert client.blob_exists("raw/docs.jsonl") is True

    def test_returns_false_when_blob_missing(self, gcs):
        client, mock_client = gcs
        mock_client.bucket.return_value.blob.return_value.exists.return_value = False
        assert client.blob_exists("missing/file.jsonl") is False
