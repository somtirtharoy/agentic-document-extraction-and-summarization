"""Tests for pure functions in src/data/preprocess.py — no GCP required."""
from unittest.mock import patch

from langdetect import LangDetectException

from src.data.preprocess import (
    TOKEN_MAX,
    TOKEN_MIN,
    _clean_record,
    _count_tokens,
    _detect_lang,
    _normalize_whitespace,
    _sha256,
)

# ── _normalize_whitespace ─────────────────────────────────────────────────────


class TestNormalizeWhitespace:
    def test_crlf_converted_to_lf(self):
        assert _normalize_whitespace("line1\r\nline2") == "line1\nline2"

    def test_cr_only_converted_to_lf(self):
        assert _normalize_whitespace("a\rb") == "a\nb"

    def test_multiple_spaces_collapsed(self):
        assert _normalize_whitespace("a   b") == "a b"

    def test_tabs_collapsed_to_space(self):
        assert _normalize_whitespace("a\t\tb") == "a b"

    def test_three_or_more_newlines_collapsed(self):
        assert _normalize_whitespace("a\n\n\n\nb") == "a\n\nb"

    def test_exactly_two_newlines_preserved(self):
        assert _normalize_whitespace("a\n\nb") == "a\n\nb"

    def test_leading_and_trailing_whitespace_stripped(self):
        assert _normalize_whitespace("  hello  ") == "hello"

    def test_already_clean_text_unchanged(self):
        clean = "Hello world.\nSecond line."
        assert _normalize_whitespace(clean) == clean


# ── _count_tokens ─────────────────────────────────────────────────────────────


class TestCountTokens:
    def test_basic_word_count(self):
        assert _count_tokens("hello world foo") == 3

    def test_empty_string_returns_zero(self):
        assert _count_tokens("") == 0

    def test_single_word(self):
        assert _count_tokens("word") == 1

    def test_extra_spaces_not_counted(self):
        assert _count_tokens("  a  b  ") == 2

    def test_newlines_as_separators(self):
        assert _count_tokens("a\nb\nc") == 3


# ── _sha256 ───────────────────────────────────────────────────────────────────


class TestSha256:
    def test_deterministic(self):
        assert _sha256("hello") == _sha256("hello")

    def test_different_inputs_produce_different_hashes(self):
        assert _sha256("hello") != _sha256("world")

    def test_returns_64_char_hex_string(self):
        result = _sha256("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string_has_known_hash(self):
        import hashlib

        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256("") == expected


# ── _detect_lang ──────────────────────────────────────────────────────────────


class TestDetectLang:
    def test_english_returned_as_en(self):
        with patch("src.data.preprocess.detect", return_value="en"):
            assert _detect_lang("some english text") == "en"

    def test_non_english_code_passed_through(self):
        with patch("src.data.preprocess.detect", return_value="fr"):
            assert _detect_lang("du texte en français") == "fr"

    def test_lang_detect_exception_returns_none(self):
        with patch("src.data.preprocess.detect", side_effect=LangDetectException(0, "err")):
            assert _detect_lang("???") is None

    def test_only_first_500_chars_sampled(self):
        long_text = "x" * 1000
        with patch("src.data.preprocess.detect", return_value="en") as mock_detect:
            _detect_lang(long_text)
            mock_detect.assert_called_once_with("x" * 500)


# ── _clean_record ─────────────────────────────────────────────────────────────


def _make_row(text: str) -> dict:
    return {
        "doc_id": "abc-123",
        "text": text,
        "ingested_at": "2024-01-01T00:00:00Z",
        "reference_summary": "summary text",
    }


class TestCleanRecord:
    def test_valid_english_doc_passes(self):
        text = " ".join(["word"] * 150)
        with patch("src.data.preprocess.detect", return_value="en"):
            result = _clean_record(_make_row(text))
        assert result is not None
        assert result["token_count"] == 150
        assert result["lang"] == "en"
        assert len(result["text_hash"]) == 64

    def test_non_english_dropped(self):
        text = " ".join(["word"] * 150)
        with patch("src.data.preprocess.detect", return_value="de"):
            assert _clean_record(_make_row(text)) is None

    def test_below_token_min_dropped(self):
        text = " ".join(["word"] * (TOKEN_MIN - 1))
        with patch("src.data.preprocess.detect", return_value="en"):
            assert _clean_record(_make_row(text)) is None

    def test_at_token_min_passes(self):
        text = " ".join(["word"] * TOKEN_MIN)
        with patch("src.data.preprocess.detect", return_value="en"):
            assert _clean_record(_make_row(text)) is not None

    def test_above_token_max_dropped(self):
        text = " ".join(["word"] * (TOKEN_MAX + 1))
        with patch("src.data.preprocess.detect", return_value="en"):
            assert _clean_record(_make_row(text)) is None

    def test_at_token_max_passes(self):
        text = " ".join(["word"] * TOKEN_MAX)
        with patch("src.data.preprocess.detect", return_value="en"):
            assert _clean_record(_make_row(text)) is not None

    def test_whitespace_normalized_in_output(self):
        raw = "  " + "  ".join(["word"] * 150) + "  "
        with patch("src.data.preprocess.detect", return_value="en"):
            result = _clean_record(_make_row(raw))
        assert result is not None
        assert not result["text"].startswith(" ")
        assert "  " not in result["text"]

    def test_original_fields_preserved(self):
        text = " ".join(["word"] * 150)
        with patch("src.data.preprocess.detect", return_value="en"):
            result = _clean_record(_make_row(text))
        assert result["doc_id"] == "abc-123"
        assert result["reference_summary"] == "summary text"
