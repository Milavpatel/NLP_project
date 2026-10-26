"""
tests/test_quiz_extensions.py
───────────────────────────────
Dedicated unit tests for True/False question generation, interactive Flashcards,
session state lifecycle, and multi-format revision exports.

Run with:
    python -m pytest tests/test_quiz_extensions.py -v
"""

import sys
import os
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quiz_generator import (
    generate_true_false,
    score_true_false,
    evaluate_true_false_detailed,
    generate_flashcards,
    TrueFalseQuestion,
    Flashcard,
)
from core.revision_builder import (
    assemble_revision_sheet,
    export_markdown,
    export_txt,
    export_pdf,
    RevisionSheet,
)

SAMPLE_LECTURE_NOTES = (
    "Natural Language Processing is a branch of artificial intelligence. "
    "Tokenization is defined as the process of breaking text into words or subwords. "
    "Extractive summarization selects the most salient sentences directly from source notes. "
    "Stopword removal discards frequent words that carry minimal conceptual significance. "
    "Keyword extraction highlights crucial terms to facilitate quick student revision. "
    "Named Entity Recognition refers to the detection of real-world objects like organizations and places."
)


# ─────────────────────────────────────────────────────────────────────────────
# ── True / False Tests ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class TestTrueFalseGeneration:
    def test_generate_tf_basic_structure(self):
        items = generate_true_false([], raw_text=SAMPLE_LECTURE_NOTES, n=4, seed=42)
        assert len(items) >= 2
        for item in items:
            assert "id" in item
            assert "statement" in item
            assert isinstance(item["correct_answer"], bool)
            assert "source_sentence" in item
            assert "explanation" in item
            assert len(item["statement"].strip()) > 5

    def test_tf_true_statements_are_verbatim(self):
        items = generate_true_false([], raw_text=SAMPLE_LECTURE_NOTES, n=6, seed=42)
        true_items = [it for it in items if it["correct_answer"] is True]
        assert len(true_items) > 0
        for it in true_items:
            assert it["statement"] == it["source_sentence"]

    def test_tf_false_statements_differ_from_source(self):
        items = generate_true_false([], raw_text=SAMPLE_LECTURE_NOTES, n=6, seed=42)
        false_items = [it for it in items if it["correct_answer"] is False]
        assert len(false_items) > 0
        for it in false_items:
            assert it["statement"] != it["source_sentence"]
            assert "False" in it["explanation"]

    def test_tf_distribution_and_verified_falsity(self):
        items = generate_true_false([], raw_text=SAMPLE_LECTURE_NOTES, n=4, seed=42)
        true_count = sum(1 for it in items if it["correct_answer"] is True)
        false_count = sum(1 for it in items if it["correct_answer"] is False)
        assert true_count == 2
        assert false_count == 2
        for it in items:
            if not it["correct_answer"]:
                assert it["statement"].lower() not in SAMPLE_LECTURE_NOTES.lower()

    def test_tf_scoring_all_correct(self):
        items = [
            {"id": 1, "statement": "Statement 1", "correct_answer": True},
            {"id": 2, "statement": "Statement 2", "correct_answer": False},
            {"id": 3, "statement": "Statement 3", "correct_answer": True},
        ]
        user_answers = {0: True, 1: False, 2: True}
        correct, total = score_true_false(items, user_answers)
        assert correct == 3
        assert total == 3

    def test_tf_scoring_mixed_and_unanswered(self):
        items = [
            {"id": 1, "statement": "Statement 1", "correct_answer": True},
            {"id": 2, "statement": "Statement 2", "correct_answer": False},
            {"id": 3, "statement": "Statement 3", "correct_answer": True},
        ]
        user_answers = {0: True, 1: True}  # Q1 correct, Q2 wrong, Q3 unanswered
        correct, total = score_true_false(items, user_answers)
        assert correct == 1
        assert total == 3

    def test_tf_detailed_feedback(self):
        items = [
            {
                "id": 1,
                "statement": "Statement 1",
                "correct_answer": True,
                "source_sentence": "Original sentence 1.",
                "explanation": "True because it matches notes.",
            }
        ]
        user_answers = {0: True}
        detailed = evaluate_true_false_detailed(items, user_answers)
        assert len(detailed) == 1
        assert detailed[0]["is_correct"] is True
        assert detailed[0]["is_answered"] is True


# ─────────────────────────────────────────────────────────────────────────────
# ── Flashcard Generation Tests ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class TestFlashcardGeneration:
    def test_generate_flashcards_basic(self):
        cards = generate_flashcards([], raw_text=SAMPLE_LECTURE_NOTES, n=5)
        assert len(cards) >= 2
        for card in cards:
            assert "id" in card
            assert "term" in card
            assert "definition" in card
            assert "source_sentence" in card
            assert "category" in card
            assert len(card["term"]) >= 2
            assert len(card["definition"]) >= 5

    def test_flashcard_definition_detection(self):
        cards = generate_flashcards([], raw_text=SAMPLE_LECTURE_NOTES, n=6)
        categories = [c["category"] for c in cards]
        # Should detect explicit definition from "Tokenization is defined as..." or "refers to..."
        assert "Definition" in categories

    def test_flashcard_empty_input(self):
        cards = generate_flashcards([], raw_text="")
        assert cards == []


# ─────────────────────────────────────────────────────────────────────────────
# ── Revision Export Integration Tests ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class TestRevisionWithExtensions:
    def test_markdown_export_includes_tf_and_flashcards(self):
        sheet = assemble_revision_sheet(
            title="NLP Revision",
            summary="This is a summary.",
            tf_items=[{
                "id": 1,
                "statement": "NLP is a branch of AI.",
                "correct_answer": True,
                "explanation": "Source notes confirm.",
            }],
            flashcards=[{
                "id": 1,
                "term": "Tokenization",
                "category": "Definition",
                "definition": "Breaking text into words.",
            }],
        )
        md = export_markdown(sheet)
        assert "True / False Practice Questions" in md
        assert "Key Concept Flashcards" in md
        assert "Tokenization" in md
        assert "NLP is a branch of AI." in md

    def test_txt_export_includes_tf_and_flashcards(self):
        sheet = assemble_revision_sheet(
            title="NLP Revision",
            summary="This is a summary.",
            tf_items=[{
                "id": 1,
                "statement": "NLP is a branch of AI.",
                "correct_answer": True,
                "explanation": "Source notes confirm.",
            }],
            flashcards=[{
                "id": 1,
                "term": "Tokenization",
                "category": "Definition",
                "definition": "Breaking text into words.",
            }],
        )
        txt = export_txt(sheet)
        assert "TRUE / FALSE PRACTICE QUESTIONS" in txt
        assert "CONCEPT STUDY FLASHCARDS" in txt

    def test_pdf_export_includes_tf_and_flashcards(self):
        sheet = assemble_revision_sheet(
            title="NLP Revision",
            summary="This is a summary.",
            tf_items=[{
                "id": 1,
                "statement": "NLP is a branch of AI.",
                "correct_answer": True,
                "explanation": "Source notes confirm.",
            }],
            flashcards=[{
                "id": 1,
                "term": "Tokenization",
                "category": "Definition",
                "definition": "Breaking text into words.",
            }],
        )
        pdf_bytes = export_pdf(sheet)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF")
