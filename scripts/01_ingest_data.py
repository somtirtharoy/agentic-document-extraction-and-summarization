"""
Ingest CNN/DailyMail articles into GCS + BigQuery, then clean and deduplicate.

Usage:
    python -m scripts.01_ingest_data
    python -m scripts.01_ingest_data --limit 500
"""
import argparse

from src.data import ingest, preprocess
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and preprocess CNN/DailyMail documents.")
    parser.add_argument(
        "--limit",
        type=int,
        default=1500,
        help="Number of articles to pull from HuggingFace (default: 1500)",
    )
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(" Step 1/2 — Ingesting from HuggingFace → GCS → BigQuery")
    print(f"{'='*55}")
    rows_ingested = ingest.run(limit=args.limit)

    print(f"\n{'='*55}")
    print(" Step 2/2 — Preprocessing → documents_clean")
    print(f"{'='*55}")
    rows_in, rows_out = preprocess.run()

    print(f"\n{'='*55}")
    print(" Summary")
    print(f"{'='*55}")
    print(f"  Ingested  : {rows_ingested:,} documents")
    print(f"  After clean: {rows_out:,} documents")
    print(f"  Dropped   : {rows_in - rows_out:,} documents (non-English, out-of-range, dupes)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
