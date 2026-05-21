from google.cloud.bigquery import SchemaField

# ── Phase 1: ingestion ────────────────────────────────────────────────────────

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

# ── Phase 2: extraction ───────────────────────────────────────────────────────

ENTITIES_SCHEMA: list[SchemaField] = [
    SchemaField("doc_id", "STRING", mode="REQUIRED"),
    SchemaField("entity_name", "STRING", mode="REQUIRED"),
    SchemaField("entity_type", "STRING", mode="NULLABLE"),
    SchemaField("salience", "FLOAT", mode="NULLABLE"),
    SchemaField("sentiment_score", "FLOAT", mode="NULLABLE"),
    SchemaField("sentiment_magnitude", "FLOAT", mode="NULLABLE"),
    SchemaField("extractor", "STRING", mode="REQUIRED"),  # "cloud_nl" | "spacy"
    SchemaField("extracted_at", "TIMESTAMP", mode="REQUIRED"),
]

EXTRACTIONS_GEMINI_SCHEMA: list[SchemaField] = [
    SchemaField("doc_id", "STRING", mode="REQUIRED"),
    SchemaField("core_issue", "STRING", mode="NULLABLE"),
    SchemaField("actors", "STRING", mode="NULLABLE"),       # JSON-encoded list[{name, role}]
    SchemaField("key_metrics", "STRING", mode="NULLABLE"),  # JSON-encoded list[str]
    SchemaField("dates", "STRING", mode="NULLABLE"),        # JSON-encoded list[{date, event}]
    SchemaField("sentiment_label", "STRING", mode="NULLABLE"),
    SchemaField("sentiment_reason", "STRING", mode="NULLABLE"),
    SchemaField("extracted_at", "TIMESTAMP", mode="REQUIRED"),
]
