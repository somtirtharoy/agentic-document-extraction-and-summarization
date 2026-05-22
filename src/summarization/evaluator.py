from dataclasses import dataclass

from rouge_score import rouge_scorer

from src.utils.logging import get_logger

logger = get_logger(__name__)

_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


@dataclass
class RougeScores:
    rouge1: float
    rouge2: float
    rougeL: float

    def __str__(self) -> str:
        return f"R1={self.rouge1:.3f}  R2={self.rouge2:.3f}  RL={self.rougeL:.3f}"


class Evaluator:
    def __init__(self) -> None:
        self._scorer = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=True)

    def score(self, reference: str, hypothesis: str) -> RougeScores:
        """Compute ROUGE-1/2/L F1 scores for a single hypothesis."""
        result = self._scorer.score(reference, hypothesis)
        return RougeScores(
            rouge1=round(result["rouge1"].fmeasure, 4),
            rouge2=round(result["rouge2"].fmeasure, 4),
            rougeL=round(result["rougeL"].fmeasure, 4),
        )

    def score_corpus(
        self, references: list[str], hypotheses: list[str]
    ) -> RougeScores:
        """Compute macro-averaged ROUGE scores across a corpus."""
        assert len(references) == len(hypotheses)
        scores = [self.score(r, h) for r, h in zip(references, hypotheses)]
        n = len(scores)
        return RougeScores(
            rouge1=round(sum(s.rouge1 for s in scores) / n, 4),
            rouge2=round(sum(s.rouge2 for s in scores) / n, 4),
            rougeL=round(sum(s.rougeL for s in scores) / n, 4),
        )
