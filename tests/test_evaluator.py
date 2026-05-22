"""Tests for ROUGE evaluator — pure computation, no GCP."""
import pytest

from src.summarization.evaluator import Evaluator, RougeScores


class TestRougeScores:
    def test_str_contains_all_metrics(self):
        s = RougeScores(rouge1=0.5, rouge2=0.3, rougeL=0.4)
        text = str(s)
        assert "R1=0.500" in text
        assert "R2=0.300" in text
        assert "RL=0.400" in text

    def test_dataclass_field_access(self):
        s = RougeScores(rouge1=0.1, rouge2=0.2, rougeL=0.3)
        assert s.rouge1 == 0.1
        assert s.rouge2 == 0.2
        assert s.rougeL == 0.3


class TestEvaluator:
    def setup_method(self):
        self.ev = Evaluator()

    def test_identical_text_scores_perfect(self):
        text = "the quick brown fox jumps over the lazy dog"
        scores = self.ev.score(text, text)
        assert scores.rouge1 == 1.0
        assert scores.rouge2 == 1.0
        assert scores.rougeL == 1.0

    def test_empty_hypothesis_scores_zero(self):
        scores = self.ev.score("some reference text here", "")
        assert scores.rouge1 == 0.0
        assert scores.rouge2 == 0.0
        assert scores.rougeL == 0.0

    def test_completely_different_text_scores_low(self):
        scores = self.ev.score("cats sleep on mats", "dogs run in parks")
        assert scores.rouge1 < 0.5

    def test_partial_overlap_is_between_zero_and_one(self):
        reference = "the cat sat on the mat"
        hypothesis = "the cat is on a mat"
        scores = self.ev.score(reference, hypothesis)
        assert 0.0 < scores.rouge1 < 1.0
        assert 0.0 < scores.rougeL < 1.0

    def test_scores_are_rounded_to_4_decimal_places(self):
        text = "the cat sat on the mat"
        scores = self.ev.score(text, text)
        assert scores.rouge1 == round(scores.rouge1, 4)

    def test_score_corpus_averages_correctly(self):
        refs = ["hello world", "foo bar baz"]
        hyps = ["hello world", "something else entirely"]
        corpus_scores = self.ev.score_corpus(refs, hyps)
        s1 = self.ev.score(refs[0], hyps[0])
        s2 = self.ev.score(refs[1], hyps[1])
        expected_r1 = round((s1.rouge1 + s2.rouge1) / 2, 4)
        assert corpus_scores.rouge1 == expected_r1

    def test_score_corpus_single_doc(self):
        text = "this is a test sentence with enough words"
        scores = self.ev.score_corpus([text], [text])
        assert scores.rouge1 == 1.0

    def test_score_corpus_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            self.ev.score_corpus(["ref1", "ref2"], ["hyp1"])
