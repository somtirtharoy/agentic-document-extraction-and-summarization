from google.cloud import bigquery
from google.cloud.bigquery import LoadJobConfig, QueryJobConfig, SchemaField

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BQClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = bigquery.Client(project=settings.gcp_project_id)
        self._dataset = settings.bq_dataset
        self._project = settings.gcp_project_id

    def table_ref(self, table_name: str) -> str:
        return f"{self._project}.{self._dataset}.{table_name}"

    def create_table(self, table_name: str, schema: list[SchemaField], exists_ok: bool = True) -> None:
        table = bigquery.Table(self.table_ref(table_name), schema=schema)
        self._client.create_table(table, exists_ok=exists_ok)
        logger.info("Ensured BQ table exists", extra={"table": self.table_ref(table_name)})

    def load_from_gcs(
        self,
        gcs_uri: str,
        table_name: str,
        schema: list[SchemaField],
        write_disposition: str = "WRITE_TRUNCATE",
    ) -> None:
        """Load a JSONL file from GCS into a BQ table."""
        job_config = LoadJobConfig(
            schema=schema,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=write_disposition,
        )
        load_job = self._client.load_table_from_uri(
            gcs_uri, self.table_ref(table_name), job_config=job_config
        )
        load_job.result()
        table = self._client.get_table(self.table_ref(table_name))
        logger.info(
            "Loaded data into BQ",
            extra={"table": self.table_ref(table_name), "rows": table.num_rows},
        )

    def insert_rows(self, table_name: str, rows: list[dict]) -> None:
        """Stream-insert rows into a BQ table (for small batches)."""
        errors = self._client.insert_rows_json(self.table_ref(table_name), rows)
        if errors:
            raise RuntimeError(f"BQ insert errors: {errors}")
        logger.info("Inserted rows via streaming", extra={"table": table_name, "rows": len(rows)})

    def query(self, sql: str, params: list | None = None) -> list[dict]:
        """Run a SQL query and return results as a list of dicts."""
        job_config = QueryJobConfig(query_parameters=params or [])
        rows = self._client.query(sql, job_config=job_config).result()
        return [dict(row) for row in rows]

    def load_table_from_dataframe(
        self,
        df,
        table_name: str,
        schema: list[SchemaField],
        write_disposition: str = "WRITE_TRUNCATE",
    ) -> None:
        job_config = LoadJobConfig(schema=schema, write_disposition=write_disposition)
        load_job = self._client.load_table_from_dataframe(
            df, self.table_ref(table_name), job_config=job_config
        )
        load_job.result()
        logger.info(
            "Loaded dataframe into BQ",
            extra={"table": self.table_ref(table_name), "rows": len(df)},
        )
