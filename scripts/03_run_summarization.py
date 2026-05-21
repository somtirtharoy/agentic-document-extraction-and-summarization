"""
Batch summarize documents_clean using Gemini and write to nlp_demo.summaries.

Usage:
    python -m scripts.03_run_summarization
    python -m scripts.03_run_summarization --limit 300 --workers 4
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from src.data.schema import SUMMARIES_SCHEMA
from src.gcp.bq_client import BQClient
from src.summarization.gemini_summarizer import GeminiSummarizer
from src.utils.logging import get_logger

logger = get_logger(__name__)

BQ_SOURCE = "documents_clean"
BQ_TARGET = "summaries"


def _summarize_one(row: dict, summarizer: GeminiSummarizer) -> dict | None:
    doc_id = row["doc_id"]
    try:
        result = summarizer.summarize(doc_id, row["text"])
        return {
            "doc_id": doc_id,
            "tldr": result.tldr,
            "bullets": json.dumps(result.bullets),
            "abstract": result.abstract,
            "summarized_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.error("Summarization failed", extra={"doc_id": doc_id, "error": str(exc)})
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch summarize documents with Gemini.")
    parser.add_argument("--limit", type=int, default=500, help="Max documents to process (default: 500)")
    parser.add_argument("--workers", type=int, default=4, help="ThreadPoolExecutor workers (default: 4)")
    args = parser.parse_args()

    bq = BQClient()

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

    summarizer = GeminiSummarizer()
    bq.create_table(BQ_TARGET, SUMMARIES_SCHEMA)

    results: list[dict] = []

    print(f"{'='*55}")
    print(f" Summarizing with {args.workers} workers...")
    print(f"{'='*55}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_summarize_one, row, summarizer): row["doc_id"]
            for row in rows
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing"):
            result = future.result()
            if result:
                results.append(result)

    if results:
        df = pd.DataFrame(results)
        df["summarized_at"] = pd.to_datetime(df["summarized_at"], utc=True)
        bq.load_table_from_dataframe(df, BQ_TARGET, SUMMARIES_SCHEMA, write_disposition="WRITE_TRUNCATE")

    print(f"\n{'='*55}")
    print(" Summary")
    print(f"{'='*55}")
    print(f"  Documents processed  : {len(rows):,}")
    print(f"  Summaries written    : {len(results):,}")
    print(f"  Failed               : {len(rows) - len(results):,}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
