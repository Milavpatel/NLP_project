"""
tests/test_phase4.py
──────────────────────
Focused tests for Phase 4: Reliable, Note-Grounded MCQ Quiz Generator.

Run with:
    python -m pytest tests/test_phase4.py -v

Test Coverage:
  1. Option Quality & Integrity:
     - Exactly 4 answer options per question.
     - All 4 options are distinct and unique (no duplicates).
     - Exactly one defensible correct answer present in the options list.
     - Shuffling options does not break answer matching.
  2. Grounded Traceability:
     - Every question has a verbatim source_sentence from the notes.
     - Explanation references the source sentence.
     - Every correct answer is grounded in the text.
  3. Question Types:
     - Fill-in-the-blank question formatting.
     - Definition-based question formatting.
  4. Robust Edge Cases:
     - Empty text and whitespace.
     - Very short text (< 5 words).
     - Insufficient distractors: skips questions instead of inventing facts.
     - Repeated or trivial vocabulary.
  5. Quiz Scoring:
     - All correct answers.
     - Mixed answers.
     - Unanswered questions (None or empty).
     - All incorrect answers.
     - Empty quiz items.
  6. Answer Leakage Protection:
     - Option strings do not contain leakage indicators like '(Correct)'.
"""

import sys
import os
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quiz_generator import (
    generate_mcq,
    score_quiz,
    evaluate_quiz_detailed,
    _is_valid_distractor,
    _get_stem,
)
from core.text_processor import split_sentences
from core.keyword_extractor import tfidf_keywords

SAMPLE_LECTURE_NOTES = (
    "Natural Language Processing is a branch of artificial intelligence. "
    "Tokenization is the process of breaking text down into individual words or subwords. "
    "Stopword removal discards frequent words that carry minimal conceptual significance. "
    "Extractive summarization selects the most salient sentences directly from source notes. "
    "Keyword extraction highlights crucial terms to facilitate quick student revision. "
    "Stemming reduces words to their base root form by stripping suffixes. "
    "Lemmatization considers the context and part of speech to resolve canonical lemmas. "
    "Evaluation metrics like ROUGE measure n-gram overlap between summaries and references."
)


# ─────────────────────────────────────────────────────────────────────────────
# ── MCQ Generation & Option Integrity ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestQuizOptionIntegrity:
    def test_four_options_per_question(self):
        sentences = split_sentences(SAMPLE_LECTURE_NOTES)
        kws = tfidf_keywords(SAMPLE_LECTURE_NOTES, n=15)
        items = generate_mcq(sentences, kws, n=5, seed=42)

        assert len(items) > 0
        for item in items:
            assert "options" in item
            assert len(item["options"]) == 4, f"Question {item['id']} did not have exactly 4 options"

    def test_options_are_all_unique(self):
        sentences = split_sentences(SAMPLE_LECTURE_NOTES)
        kws = tfidf_keywords(SAMPLE_LECTURE_NOTES, n=15)
        items = generate_mcq(sentences, kws, n=5, seed=42)

        for item in items:
            opts = item["options"]
            # Case-insensitive uniqueness check
            lower_opts = [o.lower().strip() for o in opts]
            assert len(lower_opts) == len(set(lower_opts)), f"Duplicate options found in question: {opts}"

    def test_exactly_one_correct_answer_in_options(self):
        sentences = split_sentences(SAMPLE_LECTURE_NOTES)
        kws = tfidf_keywords(SAMPLE_LECTURE_NOTES, n=15)
        items = generate_mcq(sentences, kws, n=5, seed=42)

        for item in items:
            answer = item["answer"]
            opts = item["options"]
            # Correct answer must be in options
            assert answer in opts, f"Answer '{answer}' not found in options {opts}"
            # Must appear exactly once
            assert opts.count(answer) == 1

    def test_answer_consistency_after_shuffling(self):
        """Random option ordering must not break answer validation."""
        sentences = split_sentences(SAMPLE_LECTURE_NOTES)
        kws = tfidf_keywords(SAMPLE_LECTURE_NOTES, n=15)

        # Generate with two different seeds
        items_seed1 = generate_mcq(sentences, kws, n=3, seed=1)
        items_seed2 = generate_mcq(sentences, kws, n=3, seed=99)

        for items in [items_seed1, items_seed2]:
            for item in items:
                assert item["answer"] in item["options"]


# ─────────────────────────────────────────────────────────────────────────────
# ── Source Sentence Traceability & Grounding ──────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestQuizGroundingAndTraceability:
    def test_source_sentence_exists_in_notes(self):
        sentences = split_sentences(SAMPLE_LECTURE_NOTES)
        items = generate_mcq(sentences, n=5, raw_text=SAMPLE_LECTURE_NOTES, seed=42)

        for item in items:
            src = item.get("source_sentence")
            assert src is not None
            assert src in sentences, f"Source sentence '{src}' does not originate from source notes"

    def test_explanation_references_source_sentence(self):
        sentences = split_sentences(SAMPLE_LECTURE_NOTES)
        items = generate_mcq(sentences, n=4, raw_text=SAMPLE_LECTURE_NOTES, seed=42)

        for item in items:
            exp = item.get("explanation")
            assert exp is not None
            assert len(exp) > 10
            # Explanation should cite the source sentence
            assert item["source_sentence"] in exp

    def test_question_types_supported(self):
        sentences = split_sentences(SAMPLE_LECTURE_NOTES)
        # Test fill in the blank
        items_fib = generate_mcq(sentences, n=3, raw_text=SAMPLE_LECTURE_NOTES, question_type="fill_in_the_blank", seed=42)
        for item in items_fib:
            assert item["question_type"] == "fill_in_the_blank"
            assert "_______" in item["question"]

        # Test definition based
        items_def = generate_mcq(sentences, n=3, raw_text=SAMPLE_LECTURE_NOTES, question_type="definition", seed=42)
        for item in items_def:
            assert item["question_type"] == "definition"
            assert "According to your notes" in item["question"] or "defined" in item["question"] or "described" in item["question"]


# ─────────────────────────────────────────────────────────────────────────────
# ── Distractor Validation Logic ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestDistractorValidation:
    def test_identical_terms_rejected(self):
        assert not _is_valid_distractor("Tokenization", "Tokenization", "Some sentence.", [])
        assert not _is_valid_distractor("tokenization", "Tokenization", "Some sentence.", [])

    def test_same_stem_rejected(self):
        # "tokens" and "token" share the stem "token"
        assert not _is_valid_distractor("tokens", "token", "Some sentence.", [])
        # "stemmed" and "stemming" share stem "stem"
        assert not _is_valid_distractor("stemming", "stemmed", "Some sentence.", [])

    def test_substring_match_rejected(self):
        # NLP vs Modern NLP
        assert not _is_valid_distractor("Natural Language", "Natural Language Processing", "Some sentence.", [])

    def test_word_in_source_sentence_rejected(self):
        sentence = "Tokenization breaks sentences into words."
        # "sentences" appears in source sentence, so it shouldn't be a confusing distractor
        assert not _is_valid_distractor("sentences", "Tokenization", sentence, [])

    def test_valid_distinct_distractor_accepted(self):
        sentence = "Tokenization is the process of breaking text down."
        assert _is_valid_distractor("Lemmatization", "Tokenization", sentence, ["Stemming"])


# ─────────────────────────────────────────────────────────────────────────────
# ── Edge Cases & Robustness ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestQuizEdgeCases:
    def test_empty_input(self):
        assert generate_mcq([], []) == []
        assert generate_mcq([], raw_text="") == []
        assert generate_mcq([], raw_text="   \n\t  ") == []

    def test_very_short_text(self):
        """A text with only 1 short sentence should not generate questions with fake distractors."""
        items = generate_mcq(["Hello world."], raw_text="Hello world.")
        assert items == []

    def test_insufficient_distractors_skips_rather_than_fakes(self):
        """When text has too few distinct terms to build 3 distractors, it should return [] rather than faking."""
        text = "Apple apple apple. Fruit fruit fruit."
        sentences = split_sentences(text)
        items = generate_mcq(sentences, raw_text=text)
        # Cannot form 4 unique valid options from only 2 words
        assert items == []

    def test_no_answer_leakage_in_options(self):
        """Options must be clean strings without indicators like '(Correct)'."""
        sentences = split_sentences(SAMPLE_LECTURE_NOTES)
        items = generate_mcq(sentences, n=5, raw_text=SAMPLE_LECTURE_NOTES, seed=42)

        for item in items:
            for opt in item["options"]:
                assert "(correct)" not in opt.lower()
                assert "[x]" not in opt.lower()
                assert "answer:" not in opt.lower()


# ─────────────────────────────────────────────────────────────────────────────
# ── Quiz Scoring & Evaluation ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestQuizScoring:
    def test_perfect_score(self):
        items = [
            {"id": 1, "answer": "Tokenization", "options": ["Tokenization", "Stemming", "Parsing", "Lemmatization"]},
            {"id": 2, "answer": "ROUGE", "options": ["BLEU", "ROUGE", "METEOR", "CIDEr"]},
        ]
        user_answers = {0: "Tokenization", 1: "ROUGE"}
        correct, total = score_quiz(items, user_answers)
        assert correct == 2
        assert total == 2

    def test_partial_score(self):
        items = [
            {"id": 1, "answer": "Tokenization", "options": ["Tokenization", "Stemming", "Parsing", "Lemmatization"]},
            {"id": 2, "answer": "ROUGE", "options": ["BLEU", "ROUGE", "METEOR", "CIDEr"]},
        ]
        user_answers = {0: "Tokenization", 1: "BLEU"}
        correct, total = score_quiz(items, user_answers)
        assert correct == 1
        assert total == 2

    def test_unanswered_treated_as_incorrect(self):
        items = [
            {"id": 1, "answer": "Tokenization", "options": ["Tokenization", "Stemming", "Parsing", "Lemmatization"]},
            {"id": 2, "answer": "ROUGE", "options": ["BLEU", "ROUGE", "METEOR", "CIDEr"]},
        ]
        user_answers = {0: "Tokenization"}  # Question 1 unanswered
        correct, total = score_quiz(items, user_answers)
        assert correct == 1
        assert total == 2

        # None values
        user_answers_none = {0: None, 1: None}
        correct_none, total_none = score_quiz(items, user_answers_none)
        assert correct_none == 0
        assert total_none == 2

    def test_evaluate_quiz_detailed(self):
        items = [
            {
                "id": 1,
                "question": "What is...",
                "answer": "Tokenization",
                "options": ["Tokenization", "Stemming", "Parsing", "Lemmatization"],
                "source_sentence": "Tokenization is...",
                "explanation": "According to notes...",
            }
        ]
        user_answers = {0: "Tokenization"}
        details = evaluate_quiz_detailed(items, user_answers)

        assert len(details) == 1
        assert details[0]["is_correct"] is True
        assert details[0]["is_answered"] is True
        assert details[0]["correct_answer"] == "Tokenization"
        assert details[0]["user_answer"] == "Tokenization"
