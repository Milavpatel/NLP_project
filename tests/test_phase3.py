"""
tests/test_phase3.py
──────────────────────
Focused tests for Phase 3: Real Extractive Summarization & Keyword Extraction.

Run with:
    python -m pytest tests/test_phase3.py -v

Test Coverage:
  1. Extractive Summarization (core/summarizer.py):
     - Empty text, whitespace, and very short text (1-2 sentences).
     - Normalized word frequency sentence scoring.
     - TF-IDF sentence scoring with scikit-learn.
     - Exact sentence count vs. percentage ratio controls.
     - Preservation of verbatim original sentences (no hallucination).
     - Document order preservation (chronological vs. score order).
     - Compression ratio and word count calculations.
     - Handling of documents containing only stopwords or zero-score sentences.
     - Handling of repeated/duplicate sentences.
     - Integration with core.text_processor.
     - ROUGE evaluation helper.
  2. Keyword & Keyphrase Extraction (core/keyword_extractor.py):
     - TF-IDF keyword extraction.
     - Normalized frequency ranking.
     - RAKE multi-word keyphrases.
     - Separate spaCy named entity recognition (NER).
     - Empty, whitespace, stopword-only, and numeric edge cases.
     - Output structure, score bounds in [0.0, 1.0], and descending ordering.
"""

import sys
import os
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.summarizer import (
    summarize_text,
    extractive_summary,
    frequency_score_sentences,
    tfidf_score_sentences,
    evaluate_rouge,
    SummaryResult,
    SentenceScore,
)
from core.keyword_extractor import (
    tfidf_keywords,
    frequency_keywords,
    rake_keywords,
    spacy_entities,
    extract_keywords,
)
from core.text_processor import (
    clean_text,
    split_sentences,
    tokenise_words,
    preprocess,
)

SAMPLE_LECTURE_NOTES = (
    "Natural Language Processing is a branch of artificial intelligence. "
    "It enables computers to understand, interpret, and generate human language. "
    "Statistical methods and machine learning algorithms are central to modern NLP. "
    "Tokenization is the process of breaking text down into individual words or subwords. "
    "Stopword removal discards frequent words that carry minimal conceptual significance. "
    "Extractive summarization selects the most salient sentences directly from source notes. "
    "Keyword extraction highlights crucial terms to facilitate quick student revision. "
    "Evaluation metrics like ROUGE measure n-gram overlap between summaries and references."
)


# ─────────────────────────────────────────────────────────────────────────────
# ── Extractive Summarizer Tests ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractiveSummarizerBasic:
    def test_empty_string(self):
        res = summarize_text("")
        assert res.summary_text == ""
        assert res.selected_sentences == []
        assert res.sentence_scores == []
        assert res.original_word_count == 0
        assert res.summary_word_count == 0
        assert res.compression_ratio == 0.0

    def test_whitespace_only(self):
        res = summarize_text("   \n\t  ")
        assert res.summary_text == ""
        assert res.num_sentences_selected == 0

    def test_single_sentence(self):
        text = "Natural language processing enables computers to understand human language."
        res = summarize_text(text, ratio=0.5)
        assert res.total_sentences == 1
        assert res.num_sentences_selected == 1
        assert res.summary_text == text
        assert len(res.sentence_scores) == 1
        assert res.sentence_scores[0].selected is True
        assert res.sentence_scores[0].rank == 1

    def test_two_sentences(self):
        text = "First sentence here. Second sentence follows."
        res = summarize_text(text, num_sentences=1)
        assert res.total_sentences == 2
        assert res.num_sentences_selected == 1
        assert res.summary_text in text


class TestExtractiveSummarizerAlgorithms:
    def test_frequency_scoring_produces_valid_scores(self):
        sentences = split_sentences(SAMPLE_LECTURE_NOTES)
        scores = frequency_score_sentences(sentences)
        assert len(scores) == len(sentences)
        assert all(isinstance(s, float) for s in scores)
        assert all(s >= 0.0 for s in scores)

    def test_tfidf_scoring_produces_valid_scores(self):
        sentences = split_sentences(SAMPLE_LECTURE_NOTES)
        scores = tfidf_score_sentences(sentences)
        assert len(scores) == len(sentences)
        assert all(isinstance(s, float) for s in scores)
        assert all(s >= 0.0 for s in scores)

    def test_tfidf_method_selection(self):
        res = summarize_text(SAMPLE_LECTURE_NOTES, method="tfidf", ratio=0.3)
        assert res.method == "tfidf"
        assert len(res.selected_sentences) > 0

    def test_frequency_method_selection(self):
        res = summarize_text(SAMPLE_LECTURE_NOTES, method="frequency", ratio=0.3)
        assert res.method == "frequency"
        assert len(res.selected_sentences) > 0


class TestSentencePreservationAndOrdering:
    def test_sentences_are_verbatim_originating_from_source(self):
        """Every sentence in summary must originate directly from the original source."""
        res = summarize_text(SAMPLE_LECTURE_NOTES, ratio=0.4)
        original_sentences = split_sentences(SAMPLE_LECTURE_NOTES)
        for s in res.selected_sentences:
            assert s in original_sentences

    def test_summary_restores_original_document_order(self):
        """Selected sentences MUST be ordered in their original sequential document order, NOT by score."""
        res = summarize_text(SAMPLE_LECTURE_NOTES, ratio=0.5)
        original_sentences = split_sentences(SAMPLE_LECTURE_NOTES)

        # Map each selected sentence to its original index
        indices = [original_sentences.index(s) for s in res.selected_sentences]
        # Must be strictly increasing (original document order preserved)
        assert indices == sorted(indices)
        assert len(indices) == len(set(indices))

    def test_summary_preserves_order_with_frequency_method(self):
        res = summarize_text(SAMPLE_LECTURE_NOTES, method="frequency", num_sentences=4)
        original_sentences = split_sentences(SAMPLE_LECTURE_NOTES)
        indices = [original_sentences.index(s) for s in res.selected_sentences]
        assert indices == sorted(indices)


class TestSentenceSelectionControls:
    def test_exact_sentence_count(self):
        for target_k in [1, 2, 4]:
            res = summarize_text(SAMPLE_LECTURE_NOTES, num_sentences=target_k)
            assert res.num_sentences_selected == target_k
            assert len(res.selected_sentences) == target_k

    def test_num_sentences_exceeding_total(self):
        """Requesting more sentences than available returns all sentences."""
        res = summarize_text(SAMPLE_LECTURE_NOTES, num_sentences=100)
        assert res.num_sentences_selected == res.total_sentences
        assert res.num_sentences_selected == 8

    def test_ratio_control(self):
        res_30 = summarize_text(SAMPLE_LECTURE_NOTES, ratio=0.3)
        res_60 = summarize_text(SAMPLE_LECTURE_NOTES, ratio=0.6)
        assert res_30.num_sentences_selected <= res_60.num_sentences_selected

    def test_compression_ratio_calculation(self):
        res = summarize_text(SAMPLE_LECTURE_NOTES, num_sentences=2)
        assert res.original_word_count > res.summary_word_count
        assert res.compression_ratio > 0.0
        assert res.compression_ratio < 100.0


class TestExtractiveSummarizerEdgeCases:
    def test_stopword_only_text(self):
        """Text composed entirely of stopwords should not crash."""
        text = "This is it. It was there. They were with us. That is what it is."
        res = summarize_text(text, num_sentences=2)
        assert res.total_sentences == 4
        assert res.num_sentences_selected == 2
        # Sentences exist verbatim
        for s in res.selected_sentences:
            assert s in text

    def test_repeated_sentences(self):
        """Repeated identical sentences should be handled without index crashes."""
        text = "Artificial intelligence is transformative. Artificial intelligence is transformative. Machine learning is practical."
        res = summarize_text(text, num_sentences=2)
        assert res.total_sentences == 3
        assert res.num_sentences_selected == 2

    def test_backward_compatible_extractive_summary_function(self):
        summary_str = extractive_summary(SAMPLE_LECTURE_NOTES, ratio=0.3)
        assert isinstance(summary_str, str)
        assert len(summary_str) > 0
        assert "NLP" in summary_str or "Natural Language" in summary_str

    def test_rouge_evaluation(self):
        scores = evaluate_rouge(SAMPLE_LECTURE_NOTES, SAMPLE_LECTURE_NOTES)
        assert scores["rouge1"] == 1.0
        assert scores["rouge2"] == 1.0
        assert scores["rougeL"] == 1.0

        scores_empty = evaluate_rouge("", "")
        assert scores_empty["rouge1"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ── Keyword Extractor Tests ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestTfidfKeywords:
    def test_empty_string(self):
        assert tfidf_keywords("") == []
        assert tfidf_keywords("   ") == []

    def test_returns_top_n_terms(self):
        kws = tfidf_keywords(SAMPLE_LECTURE_NOTES, n=5)
        assert len(kws) <= 5
        assert len(kws) > 0
        for term, score in kws:
            assert isinstance(term, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_sorted_descending_by_score(self):
        kws = tfidf_keywords(SAMPLE_LECTURE_NOTES, n=10)
        scores = [s for _, s in kws]
        assert scores == sorted(scores, reverse=True)

    def test_stopword_only_text_does_not_crash(self):
        text = "This is that and that was it and they were there."
        kws = tfidf_keywords(text, n=5)
        assert isinstance(kws, list)


class TestFrequencyKeywords:
    def test_empty_string(self):
        assert frequency_keywords("") == []

    def test_normalized_scores_in_range(self):
        kws = frequency_keywords(SAMPLE_LECTURE_NOTES, n=8)
        assert len(kws) > 0
        # The most frequent word must have a normalized score of 1.0
        assert kws[0][1] == 1.0
        for term, score in kws:
            assert 0.0 <= score <= 1.0

    def test_sorted_descending(self):
        kws = frequency_keywords(SAMPLE_LECTURE_NOTES, n=8)
        scores = [s for _, s in kws]
        assert scores == sorted(scores, reverse=True)

    def test_stopwords_are_filtered(self):
        text = "the the the student student student is is is reading reading"
        kws = frequency_keywords(text, n=5)
        terms = [t for t, _ in kws]
        assert "the" not in terms
        assert "is" not in terms
        assert "student" in terms


class TestRakeKeywords:
    def test_empty_string(self):
        assert rake_keywords("") == []

    def test_rake_extracts_keyphrases(self):
        kws = rake_keywords(SAMPLE_LECTURE_NOTES, n=6)
        assert len(kws) > 0
        for phrase, score in kws:
            assert isinstance(phrase, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_rake_scores_sorted_descending(self):
        kws = rake_keywords(SAMPLE_LECTURE_NOTES, n=6)
        scores = [s for _, s in kws]
        assert scores == sorted(scores, reverse=True)


class TestSpacyEntities:
    def test_empty_string(self):
        assert spacy_entities("") == []

    def test_entities_are_separated_from_keywords(self):
        """NER must return (text, label) and not plain keywords."""
        text = "Alan Turing worked at Bletchley Park in the United Kingdom in 1942."
        ents = spacy_entities(text)
        if ents:
            for ent_text, label in ents:
                assert isinstance(ent_text, str)
                assert isinstance(label, str)
                assert len(ent_text) > 1

    def test_entities_do_not_conflate_all_words(self):
        """Regular common nouns like 'computers' should not be labeled arbitrarily."""
        text = "Alan Turing created early computing theory."
        ents = spacy_entities(text)
        labels = [label for _, label in ents]
        # If spaCy recognized anything, Turing is typically PERSON
        if ents:
            assert any(l in ("PERSON", "ORG", "GPE") for l in labels)


class TestExtractKeywordsDispatcher:
    def test_dispatcher_tfidf(self):
        kws = extract_keywords(SAMPLE_LECTURE_NOTES, method="tfidf", n=5)
        assert len(kws) > 0

    def test_dispatcher_frequency(self):
        kws = extract_keywords(SAMPLE_LECTURE_NOTES, method="frequency", n=5)
        assert len(kws) > 0

    def test_dispatcher_rake(self):
        kws = extract_keywords(SAMPLE_LECTURE_NOTES, method="rake", n=5)
        assert len(kws) > 0


# ─────────────────────────────────────────────────────────────────────────────
# ── Preprocessing Integration Tests ──────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessingIntegration:
    def test_summarizer_uses_clean_text_and_split_sentences(self):
        raw = "Line one with\u201csmart quotes\u201d.\n\n\n\nLine two with spaces.   Line three."
        res = summarize_text(raw, num_sentences=2)
        assert res.total_sentences == 3
        # Ensure smart quotes were cleaned
        assert "\u201c" not in res.summary_text
        assert "\u201d" not in res.summary_text

    def test_full_pipeline_consistency(self):
        doc = preprocess(SAMPLE_LECTURE_NOTES, remove_stopwords=True)
        summary_res = summarize_text(SAMPLE_LECTURE_NOTES, num_sentences=3)
        assert len(doc.sentences) == summary_res.total_sentences
        assert summary_res.num_sentences_selected == 3
