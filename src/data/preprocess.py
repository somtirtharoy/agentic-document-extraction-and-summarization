import hashlib
import re

from langdetect import detect, LangDetectException
from tqdm import tqdm

from config.settings import get_settings
from src.data.schema import DOCUMENTS_CLEAN_SCHEMA
from src.gcp.bq_client import BQClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

BQ_SOURCE_TABLE = "documents"
BQ_CLEAN_TABLE = "documents_clean"

TOKEN_MIN = 100
TOKEN_MAX = 5_000


def _count_tokens(text: str) -> int:
    """Whitespace-based token count — fast proxy; no tokenizer dependency needed here."""
    return len(text.split())


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_lang(text: str) -> str | None:
    try:
        return detect(text[:500])  # sample first 500 chars — langdetect is slow on full text
    except LangDetectException:
        return None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _clean_record(row: dict) -> dict | None:
    """Return a cleaned record or None if it should be dropped."""
    text = _normalize_whitespace(row["text"])

    lang = _detect_lang(text)
    if lang != "en":
        return None

    token_count = _count_tokens(text)
    if not (TOKEN_MIN <= token_count <= TOKEN_MAX):
        return None

    return {
        **row,
        "text": text,
        "token_count": token_count,
        "lang": lang,
        "text_hash": _sha256(text),
    }


def run() -> tuple[int, int]:
    """Read documents from BQ, clean, dedupe, write to documents_clean.

    Returns (rows_in, rows_out).
    """
    settings = get_settings()
    bq = BQClient()

    sql = f"SELECT * FROM `{bq.table_ref(BQ_SOURCE_TABLE)}`"
    logger.info("Reading documents from BQ", extra={"table": bq.table_ref(BQ_SOURCE_TABLE)})
    rows = bq.query(sql)
    rows_in = len(rows)
    logger.info("Fetched rows", extra={"count": rows_in})

    cleaned: list[dict] = []
    seen_hashes: set[str] = set()

    for row in tqdm(rows, desc="Cleaning"):
        result = _clean_record(row)
        if result is None:
            continue
        if result["text_hash"] in seen_hashes:
            continue
        seen_hashes.add(result["text_hash"])
        cleaned.append(result)

    rows_out = len(cleaned)
    dropped = rows_in - rows_out
    logger.info(
        "Cleaning complete",
        extra={"rows_in": rows_in, "rows_out": rows_out, "dropped": dropped},
    )

    bq.create_table(BQ_CLEAN_TABLE, DOCUMENTS_CLEAN_SCHEMA)
    bq.load_table_from_dataframe(
        _to_dataframe(cleaned),
        BQ_CLEAN_TABLE,
        DOCUMENTS_CLEAN_SCHEMA,
        write_disposition="WRITE_TRUNCATE",
    )

    return rows_in, rows_out


def _to_dataframe(records: list[dict]):
    import pandas as pd
    df = pd.DataFrame(records)
    # BQ expects TIMESTAMP as a proper datetime, not a raw ISO string
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
    return df
