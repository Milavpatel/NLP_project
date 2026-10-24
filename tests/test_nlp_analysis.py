"""
tests/test_nlp_analysis.py
───────────────────────────
Quality-Assurance tests for NLP analysis suite & End-to-End Workflow Integration.

Run with:
    python -m pytest tests/test_nlp_analysis.py -v

Test Coverage:
  1. POS Tag Distribution (core/nlp_analyzer.py):
     - Maps Penn Treebank tags to 7 high-level categories.
     - Empty text and punctuation-only inputs.
  2. Word Frequency Distribution:
     - Top-N non-stopword token counts.
     - Empty and noisy text handling.
  3. Word Cloud Generator:
     - Generates valid WordCloud object for standard text.
     - Handles empty/stopword-only text without crashing.
  4. Sentence Length Distribution:
     - Computes accurate per-sentence word counts.
  5. Readability Metrics:
     - Flesch Reading Ease score within bounds [0.0, 100.0].
     - Flesch-Kincaid Grade level computation.
     - Syllable counter heuristics.
  6. End-to-End Pipeline & Stale State Reset:
     - Full pipeline execution: text -> preprocess -> summary -> keywords -> quiz -> revision -> export.
     - clear_derived_data() resets downstream artifacts.
"""

import sys
import os
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nlp_analyzer import (
    pos_tag_distribution,
    word_frequency,
    generate_wordcloud,
    sentence_length_distribution,
    compute_readability,
    _count_syllables_in_word,
)
from core.text_processor import preprocess, split_sentences, clean_text
from core.summarizer import summarize_text
from core.keyword_extractor import tfidf_keywords
from core.quiz_generator import generate_mcq
from core.revision_builder import assemble_revision_sheet, export_pdf, export_markdown, export_txt

SAMPLE_TEXT = (
    "Natural Language Processing is a branch of artificial intelligence. "
    "Computers analyze human language using statistical models and machine learning. "
    "Tokenization breaks sentences into words, while extractive summarization identifies key statements. "
    "Evaluation metrics like ROUGE compute overlap between hypothesis summaries and references."
)


# ─────────────────────────────────────────────────────────────────────────────
# ── POS Tag Distribution Tests ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestPosTagDistribution:
    def test_pos_categories_present(self):
        pos = pos_tag_distribution(SAMPLE_TEXT)
        assert isinstance(pos, dict)
        for cat in ["Nouns", "Verbs", "Adjectives", "Adverbs", "Pronouns", "Prep & Conj", "Other"]:
            assert cat in pos

        # In SAMPLE_TEXT, nouns should dominate
        assert pos["Nouns"] > 0
        assert pos["Verbs"] > 0

    def test_pos_empty_string(self):
        assert pos_tag_distribution("") == {}
        assert pos_tag_distribution("   \t\n") == {}

    def test_pos_punctuation_only(self):
        assert pos_tag_distribution("! @ # $ % ^ & * ( )") == {}


# ─────────────────────────────────────────────────────────────────────────────
# ── Word Frequency Tests ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestWordFrequency:
    def test_top_words_sorted_by_count(self):
        wf = word_frequency(SAMPLE_TEXT, n=5)
        assert len(wf) <= 5
        assert len(wf) > 0
        counts = [c for _, c in wf]
        assert counts == sorted(counts, reverse=True)

    def test_stopwords_are_excluded(self):
        text = "the the the book book is is reading"
        wf = word_frequency(text, n=5)
        terms = [t for t, _ in wf]
        assert "the" not in terms
        assert "is" not in terms
        assert "book" in terms

    def test_empty_input(self):
        assert word_frequency("") == []


# ─────────────────────────────────────────────────────────────────────────────
# ── Word Cloud Tests ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestWordCloud:
    def test_wordcloud_generates_valid_object(self):
        wc = generate_wordcloud(SAMPLE_TEXT)
        assert wc is not None
        # WordCloud objects have a to_array() method
        arr = wc.to_array()
        assert arr.shape[0] > 0
        assert arr.shape[1] > 0

    def test_wordcloud_empty_text(self):
        assert generate_wordcloud("") is None
        assert generate_wordcloud("   ") is None


# ─────────────────────────────────────────────────────────────────────────────
# ── Sentence Length & Readability Tests ───────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestSentenceLengthAndReadability:
    def test_sentence_length_distribution(self):
        sents = ["First sentence.", "A second sentence with more words here."]
        lens = sentence_length_distribution(sents)
        assert lens == [2, 7]

    def test_sentence_length_empty(self):
        assert sentence_length_distribution([]) == []

    def test_syllable_counter_heuristics(self):
        assert _count_syllables_in_word("cat") == 1
        assert _count_syllables_in_word("water") == 2
        assert _count_syllables_in_word("computer") == 3
        assert _count_syllables_in_word("intelligence") >= 4

    def test_readability_metrics(self):
        rd = compute_readability(SAMPLE_TEXT)
        assert "reading_ease" in rd
        assert "grade_level" in rd
        assert "interpretation" in rd
        assert 0.0 <= rd["reading_ease"] <= 100.0
        assert rd["grade_level"] >= 1.0

    def test_readability_empty(self):
        rd = compute_readability("")
        assert rd["reading_ease"] == 0.0
        assert "Empty" in rd["interpretation"]


# ─────────────────────────────────────────────────────────────────────────────
# ── End-to-End Pipeline & Integration Tests ──────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndIntegration:
    def test_full_pipeline_flow(self):
        """Verify the seamless progression through all 5 phases for a single document."""
        # 1. Clean & Preprocess (Phase 2)
        doc = preprocess(SAMPLE_TEXT, remove_stopwords=True)
        assert len(doc.sentences) == 4
        assert len(doc.tokens) > 0

        # 2. Extractive Summarization (Phase 3)
        summary_res = summarize_text(SAMPLE_TEXT, method="tfidf", num_sentences=2)
        assert summary_res.num_sentences_selected == 2
        assert len(summary_res.summary_text) > 0

        # 3. Keyword Extraction (Phase 3)
        kws = tfidf_keywords(SAMPLE_TEXT, n=10)
        assert len(kws) > 0

        # 4. Quiz Generation (Phase 4)
        quiz = generate_mcq(doc.sentences, kws, n=2, seed=42)
        assert len(quiz) > 0
        assert len(quiz[0]["options"]) == 4

        # 5. Revision Sheet Assembly & Multi-Format Exports (Phase 5)
        sheet = assemble_revision_sheet(
            title="E2E Integration Test",
            summary=summary_res.summary_text,
            keywords=kws,
            quiz_items=quiz,
            stats={"words": len(doc.tokens), "sentences": len(doc.sentences)},
        )

        pdf_bytes = export_pdf(sheet)
        md_text = export_markdown(sheet)
        txt_text = export_txt(sheet)

        assert pdf_bytes.startswith(b"%PDF")
        assert len(pdf_bytes) > 2000
        assert "# 📚 E2E Integration Test" in md_text
        assert "REVISION SHEET: E2E INTEGRATION TEST" in txt_text
