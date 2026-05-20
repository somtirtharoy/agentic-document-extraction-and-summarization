from google.cloud.bigquery import SchemaField

DOCUMENTS_SCHEMA: list[SchemaField] = [
    SchemaField("doc_id", "STRING", mode="REQUIRED"),
    SchemaField("text", "STRING", mode="REQUIRED"),
    SchemaField("reference_summary", "STRING", mode="NULLABLE"),
    SchemaField("source", "STRING", mode="NULLABLE"),
    SchemaField("ingested_at", "TIMESTAMP", mode="REQUIRED"),
]

DOCUMENTS_CLEAN_SCHEMA: list[SchemaField] = [
    SchemaField("doc_id", "STRING", mode="REQUIRED"),
    SchemaField("text", "STRING", mode="REQUIRED"),
    SchemaField("reference_summary", "STRING", mode="NULLABLE"),
    SchemaField("source", "STRING", mode="NULLABLE"),
    SchemaField("ingested_at", "TIMESTAMP", mode="REQUIRED"),
    SchemaField("token_count", "INTEGER", mode="REQUIRED"),
    SchemaField("lang", "STRING", mode="REQUIRED"),
    SchemaField("text_hash", "STRING", mode="REQUIRED"),
]
