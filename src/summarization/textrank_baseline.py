from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

from src.utils.logging import get_logger

logger = get_logger(__name__)

_SENTENCE_COUNT = 3


class TextRankSummarizerBaseline:
    def __init__(self) -> None:
        self._summarizer = TextRankSummarizer()
        self._tokenizer = Tokenizer("english")

    def summarize(self, doc_id: str, text: str) -> str:
        """Return a multi-sentence extractive summary using TextRank."""
        parser = PlaintextParser.from_string(text, self._tokenizer)
        sentences = self._summarizer(parser.document, _SENTENCE_COUNT)
        summary = " ".join(str(s) for s in sentences)
        logger.info("TextRank summarized", extra={"doc_id": doc_id})
        return summary
