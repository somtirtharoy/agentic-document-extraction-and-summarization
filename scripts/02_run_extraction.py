"""
Run entity extraction on documents_clean using Cloud NL API + Gemini in parallel.

Usage:
    python -m scripts.02_run_extraction
    python -m scripts.02_run_extraction --limit 200 --workers 4 --skip-spacy
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from tqdm import tqdm

from src.data.schema import ENTITIES_SCHEMA, EXTRACTIONS_GEMINI_SCHEMA
from src.extraction.gemini_extractor import GeminiExtractor
from src.extraction.gcp_nl_extractor import GCPNLExtractor
from src.gcp.bq_client import BQClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

BQ_SOURCE = "documents_clean"
BQ_ENTITIES = "entities"
BQ_GEMINI = "extractions_gemini"


def _extract_one(
    row: dict,
    nl_extractor: GCPNLExtractor,
    gemini_extractor: GeminiExtractor,
    spacy_extractor,
    run_spacy: bool,
) -> tuple[list[dict], dict | None]:
    """Run all extractors on a single document. Returns (entity_rows, gemini_row)."""
    doc_id = row["doc_id"]
    text = row["text"]
    now = datetime.now(timezone.utc).isoformat()

    entity_rows: list[dict] = []
    gemini_row: dict | None = None

    # Cloud NL API
    try:
        entities, _categories = nl_extractor.extract(doc_id, text)
        for e in entities:
            entity_rows.append({
                "doc_id": doc_id,
                "entity_name": e.name,
                "entity_type": e.type,
                "salience": e.salience,
                "sentiment_score": e.sentiment_score,
                "sentiment_magnitude": e.sentiment_magnitude,
                "extractor": "cloud_nl",
                "extracted_at": now,
            })
    except Exception as exc:
        logger.error("Cloud NL failed", extra={"doc_id": doc_id, "error": str(exc)})

    # Gemini structured extraction
    try:
        result = gemini_extractor.extract(doc_id, text)
        gemini_row = {
            "doc_id": doc_id,
            "core_issue": result.core_issue,
            "actors": json.dumps([a.model_dump() for a in result.actors]),
            "key_metrics": json.dumps(result.key_metrics),
            "dates": json.dumps([d.model_dump() for d in result.dates]),
            "sentiment_label": result.sentiment_label,
            "sentiment_reason": result.sentiment_reason,
            "extracted_at": now,
        }
    except Exception as exc:
        logger.error("Gemini extraction failed", extra={"doc_id": doc_id, "error": str(exc)})

    # Optional spaCy baseline
    if run_spacy and spacy_extractor:
        try:
            spacy_entities = spacy_extractor.extract(doc_id, text)
            for e in spacy_entities:
                entity_rows.append({
                    "doc_id": doc_id,
                    "entity_name": e.name,
                    "entity_type": e.type,
                    "salience": e.salience,
                    "sentiment_score": e.sentiment_score,
                    "sentiment_magnitude": e.sentiment_magnitude,
                    "extractor": "spacy",
                    "extracted_at": now,
                })
        except Exception as exc:
            logger.error("spaCy extraction failed", extra={"doc_id": doc_id, "error": str(exc)})

    return entity_rows, gemini_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run information extraction on cleaned documents.")
    parser.add_argument("--limit", type=int, default=500, help="Max documents to process (default: 500)")
    parser.add_argument("--workers", type=int, default=4, help="ThreadPoolExecutor workers (default: 4)")
    parser.add_argument("--skip-spacy", action="store_true", help="Skip spaCy baseline extraction")
    args = parser.parse_args()

    bq = BQClient()

    # Load documents
    sql = f"""
        SELECT doc_id, text
        FROM `{bq.table_ref(BQ_SOURCE)}`
        LIMIT {args.limit}
    """
    print(f"\n{'='*55}")
    print(f" Loading up to {args.limit:,} documents from BQ...")
    print(f"{'='*55}")
    rows = bq.query(sql)
    print(f" Loaded {len(rows):,} documents\n")

    # Init extractors
    nl_extractor = GCPNLExtractor()
    gemini_extractor = GeminiExtractor()
    spacy_extractor = None

    if not args.skip_spacy:
        try:
            from src.extraction.spacy_baseline import SpacyExtractor
            spacy_extractor = SpacyExtractor()
            print(" spaCy extractor loaded")
        except Exception as e:
            print(f" spaCy unavailable ({e}), skipping")

    # Ensure BQ tables exist
    bq.create_table(BQ_ENTITIES, ENTITIES_SCHEMA)
    bq.create_table(BQ_GEMINI, EXTRACTIONS_GEMINI_SCHEMA)

    # Batch extraction with ThreadPoolExecutor
    all_entity_rows: list[dict] = []
    all_gemini_rows: list[dict] = []

    print(f"{'='*55}")
    print(f" Extracting with {args.workers} workers...")
    print(f"{'='*55}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _extract_one, row, nl_extractor, gemini_extractor, spacy_extractor, not args.skip_spacy
            ): row["doc_id"]
            for row in rows
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            doc_id = futures[future]
            try:
                entity_rows, gemini_row = future.result()
                all_entity_rows.extend(entity_rows)
                if gemini_row:
                    all_gemini_rows.append(gemini_row)
            except Exception as exc:
                logger.error("Unexpected error", extra={"doc_id": doc_id, "error": str(exc)})

    # Write to BQ
    print(f"\n{'='*55}")
    print(" Writing results to BigQuery...")
    print(f"{'='*55}")

    if all_entity_rows:
        import pandas as pd
        df_entities = pd.DataFrame(all_entity_rows)
        df_entities["extracted_at"] = pd.to_datetime(df_entities["extracted_at"], utc=True)
        bq.load_table_from_dataframe(df_entities, BQ_ENTITIES, ENTITIES_SCHEMA, write_disposition="WRITE_TRUNCATE")

    if all_gemini_rows:
        import pandas as pd
        df_gemini = pd.DataFrame(all_gemini_rows)
        df_gemini["extracted_at"] = pd.to_datetime(df_gemini["extracted_at"], utc=True)
        bq.load_table_from_dataframe(df_gemini, BQ_GEMINI, EXTRACTIONS_GEMINI_SCHEMA, write_disposition="WRITE_TRUNCATE")

    print(f"\n{'='*55}")
    print(" Summary")
    print(f"{'='*55}")
    print(f"  Documents processed : {len(rows):,}")
    print(f"  Entity rows written : {len(all_entity_rows):,}")
    print(f"  Gemini rows written : {len(all_gemini_rows):,}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
