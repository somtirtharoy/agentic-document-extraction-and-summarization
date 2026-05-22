"""Tests for BQClient with mocked google.cloud.bigquery."""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def bq(monkeypatch):
    """BQClient backed by a mocked bigquery.Client."""
    with patch("src.gcp.bq_client.bigquery") as mock_bq_module:
        mock_client = MagicMock()
        mock_bq_module.Client.return_value = mock_client
        mock_bq_module.Table = MagicMock()
        mock_bq_module.SourceFormat.NEWLINE_DELIMITED_JSON = "NEWLINE_DELIMITED_JSON"
        from src.gcp.bq_client import BQClient

        client = BQClient()
        client._project = "test-project"
        client._dataset = "test_dataset"
        client._client = mock_client
        yield client, mock_client


class TestTableRef:
    def test_format_is_project_dataset_table(self, bq):
        client, _ = bq
        assert client.table_ref("documents") == "test-project.test_dataset.documents"

    def test_different_table_names(self, bq):
        client, _ = bq
        assert client.table_ref("summaries") == "test-project.test_dataset.summaries"


class TestInsertRows:
    def test_raises_on_bq_errors(self, bq):
        client, mock_client = bq
        mock_client.insert_rows_json.return_value = [{"errors": [{"reason": "invalid"}]}]
        with pytest.raises(RuntimeError, match="BQ insert errors"):
            client.insert_rows("some_table", [{"key": "val"}])

    def test_succeeds_with_no_errors(self, bq):
        client, mock_client = bq
        mock_client.insert_rows_json.return_value = []
        client.insert_rows("some_table", [{"key": "val"}])
        mock_client.insert_rows_json.assert_called_once()

    def test_passes_rows_to_underlying_client(self, bq):
        client, mock_client = bq
        mock_client.insert_rows_json.return_value = []
        rows = [{"a": 1}, {"b": 2}]
        client.insert_rows("my_table", rows)
        call_args = mock_client.insert_rows_json.call_args
        assert call_args[0][1] == rows


class TestQuery:
    def test_returns_list_of_dicts(self, bq):
        client, mock_client = bq
        mock_row_1 = {"col": "value1"}
        mock_row_2 = {"col": "value2"}
        mock_client.query.return_value.result.return_value = [mock_row_1, mock_row_2]
        result = client.query("SELECT 1")
        assert result == [{"col": "value1"}, {"col": "value2"}]

    def test_empty_result_set(self, bq):
        client, mock_client = bq
        mock_client.query.return_value.result.return_value = []
        result = client.query("SELECT 1 WHERE FALSE")
        assert result == []

    def test_passes_params_to_job_config(self, bq):
        client, mock_client = bq
        mock_client.query.return_value.result.return_value = []
        from google.cloud.bigquery import ScalarQueryParameter

        params = [ScalarQueryParameter("p", "STRING", "val")]
        client.query("SELECT @p", params)
        call_args = mock_client.query.call_args
        assert call_args[1]["job_config"].query_parameters == params
