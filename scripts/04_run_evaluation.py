"""
Evaluate summarization quality via ROUGE-1/2/L on a sample of documents.
Compares Gemini (abstractive) vs TextRank (extractive) against CNN/DM reference summaries.

Usage:
    python -m scripts.04_run_evaluation
    python -m scripts.04_run_evaluation --sample 100
"""
import argparse
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from src.data.schema import EVALUATION_RESULTS_SCHEMA
from src.gcp.bq_client import BQClient
from src.summarization.evaluator import Evaluator
from src.summarization.gemini_summarizer import GeminiSummarizer
from src.summarization.textrank_baseline import TextRankSummarizerBaseline
from src.utils.logging import get_logger

logger = get_logger(__name__)

BQ_SOURCE = "documents_clean"
BQ_TARGET = "evaluation_results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROUGE evaluation on summarization models.")
    parser.add_argument("--sample", type=int, default=100, help="Number of docs to evaluate (default: 100)")
    args = parser.parse_args()

    bq = BQClient()

    sql = f"""
        SELECT doc_id, text, reference_summary
        FROM `{bq.table_ref(BQ_SOURCE)}`
        WHERE reference_summary IS NOT NULL
        LIMIT {args.sample}
    """
    print(f"\n{'='*55}")
    print(f" Loading {args.sample} documents for evaluation...")
    print(f"{'='*55}")
    rows = bq.query(sql)
    print(f" Loaded {len(rows):,} documents\n")

    gemini = GeminiSummarizer()
    textrank = TextRankSummarizerBaseline()
    evaluator = Evaluator()

    gemini_hypotheses: list[str] = []
    textrank_hypotheses: list[str] = []
    references: list[str] = []
    doc_ids: list[str] = []
    eval_rows: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()

    print(f"{'='*55}")
    print(" Running summarizers...")
    print(f"{'='*55}")

    for row in tqdm(rows, desc="Evaluating"):
        doc_id = row["doc_id"]
        text = row["text"]
        reference = row["reference_summary"]

        try:
            gemini_summary = gemini.summarize(doc_id, text).abstract
        except Exception as exc:
            logger.error("Gemini failed during eval", extra={"doc_id": doc_id, "error": str(exc)})
            gemini_summary = ""

        try:
            tr_summary = textrank.summarize(doc_id, text)
        except Exception as exc:
            logger.error("TextRank failed", extra={"doc_id": doc_id, "error": str(exc)})
            tr_summary = ""

        gemini_scores = evaluator.score(reference, gemini_summary)
        textrank_scores = evaluator.score(reference, tr_summary)

        gemini_hypotheses.append(gemini_summary)
        textrank_hypotheses.append(tr_summary)
        references.append(reference)
        doc_ids.append(doc_id)

        eval_rows.append({
            "doc_id": doc_id, "model": "gemini",
            "rouge1": gemini_scores.rouge1, "rouge2": gemini_scores.rouge2,
            "rougeL": gemini_scores.rougeL, "evaluated_at": now,
        })
        eval_rows.append({
            "doc_id": doc_id, "model": "textrank",
            "rouge1": textrank_scores.rouge1, "rouge2": textrank_scores.rouge2,
            "rougeL": textrank_scores.rougeL, "evaluated_at": now,
        })

    # Macro-averaged corpus scores
    gemini_corpus = evaluator.score_corpus(references, gemini_hypotheses)
    textrank_corpus = evaluator.score_corpus(references, textrank_hypotheses)

    # Write per-doc scores to BQ
    bq.create_table(BQ_TARGET, EVALUATION_RESULTS_SCHEMA)
    df = pd.DataFrame(eval_rows)
    df["evaluated_at"] = pd.to_datetime(df["evaluated_at"], utc=True)
    bq.load_table_from_dataframe(df, BQ_TARGET, EVALUATION_RESULTS_SCHEMA, write_disposition="WRITE_TRUNCATE")

    # Print comparison table
    print(f"\n{'='*55}")
    print(f" ROUGE Results  (n={len(rows)} documents)")
    print(f"{'='*55}")
    print(f"  {'Model':<12}  {'ROUGE-1':>8}  {'ROUGE-2':>8}  {'ROUGE-L':>8}")
    print(f"  {'-'*42}")
    print(f"  {'Gemini':<12}  {gemini_corpus.rouge1:>8.3f}  {gemini_corpus.rouge2:>8.3f}  {gemini_corpus.rougeL:>8.3f}")
    print(f"  {'TextRank':<12}  {textrank_corpus.rouge1:>8.3f}  {textrank_corpus.rouge2:>8.3f}  {textrank_corpus.rougeL:>8.3f}")
    print(f"{'='*55}\n")
    print(" Note: TextRank (extractive) typically scores higher on ROUGE because")
    print(" it reuses original words. Gemini (abstractive) produces more fluent,")
    print(" human-readable summaries at the cost of lower lexical overlap.\n")


if __name__ == "__main__":
    main()
