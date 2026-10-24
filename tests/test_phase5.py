"""
tests/test_phase5.py
──────────────────────
Focused tests for Phase 5: Complete Revision Sheet Builder & Multi-Format Exporter.

Run with:
    python -m pytest tests/test_phase5.py -v

Test Coverage:
  1. Revision Sheet Assembly:
     - Full assembly with all sections (summary, keywords, entities, quiz, stats).
     - Assembly with missing/empty optional sections (no summary, no quiz, no keywords).
     - Title sanitation and fallback.
  2. Multi-Format Exports:
     - Markdown export content, structure, tables, and headers.
     - Plain text export content and formatting.
     - ReportLab PDF export: non-empty binary output starting with '%PDF'.
  3. Content Fidelity & Answer Key:
     - Correct inclusion of quiz answers, options, and source sentences.
     - Explanations matched 1:1 with quiz items.
  4. Robustness & Special Characters:
     - Long multi-paragraph notes.
     - Special characters, unicode smart quotes, ampersands, angle brackets.
     - Numeric formatting of stats with missing keys.
"""

import sys
import os
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.revision_builder import (
    assemble_revision_sheet,
    export_markdown,
    export_txt,
    export_pdf,
    RevisionSheet,
    _format_stat_num,
)

SAMPLE_TITLE = "Natural Language Processing 101"
SAMPLE_SUMMARY = (
    "Natural Language Processing is a branch of artificial intelligence. "
    "Tokenization breaks text into words, while extractive summarization selects key statements."
)
SAMPLE_KEYWORDS = [
    ("natural language processing", 1.0000),
    ("tokenization", 0.8542),
    ("summarization", 0.7231),
    ("artificial intelligence", 0.6510),
]
SAMPLE_ENTITIES = [
    ("Alan Turing", "PERSON"),
    ("Bletchley Park", "GPE"),
    ("1950", "DATE"),
]
SAMPLE_STATS = {
    "words": 1540,
    "sentences": 82,
    "paragraphs": 12,
    "vocab_size": 420,
    "reading_time": 8,
}
SAMPLE_QUIZ = [
    {
        "id": 1,
        "question": "Complete the statement:\n\"_______ is a branch of artificial intelligence.\"",
        "options": ["Robotics", "Computer Vision", "Natural Language Processing", "Reinforcement Learning"],
        "answer": "Natural Language Processing",
        "source_sentence": "Natural Language Processing is a branch of artificial intelligence.",
        "explanation": "According to your notes: \"Natural Language Processing is a branch of artificial intelligence.\"",
        "question_type": "fill_in_the_blank",
    },
    {
        "id": 2,
        "question": "Which term is defined as the process of breaking text down into individual words?",
        "options": ["Tokenization", "Stemming", "Parsing", "Lemmatization"],
        "answer": "Tokenization",
        "source_sentence": "Tokenization is the process of breaking text down into individual words.",
        "explanation": "According to your notes: \"Tokenization is the process of breaking text down into individual words.\"",
        "question_type": "definition",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# ── Revision Sheet Assembly ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestRevisionSheetAssembly:
    def test_full_assembly(self):
        sheet = assemble_revision_sheet(
            title=SAMPLE_TITLE,
            summary=SAMPLE_SUMMARY,
            summary_method="tfidf",
            keywords=SAMPLE_KEYWORDS,
            entities=SAMPLE_ENTITIES,
            stats=SAMPLE_STATS,
            quiz_items=SAMPLE_QUIZ,
        )

        assert sheet.title == SAMPLE_TITLE
        assert sheet.summary == SAMPLE_SUMMARY
        assert len(sheet.keywords) == len(SAMPLE_KEYWORDS)
        assert len(sheet.entities) == len(SAMPLE_ENTITIES)
        assert len(sheet.quiz_items) == len(SAMPLE_QUIZ)
        assert sheet.stats["words"] == 1540
        assert sheet.date_str != ""

    def test_assembly_missing_optional_sections(self):
        sheet = assemble_revision_sheet(
            title="Minimal Guide",
            summary=None,
            keywords=None,
            entities=None,
            quiz_items=None,
        )

        assert sheet.summary is None
        assert sheet.keywords == []
        assert sheet.entities == []
        assert sheet.quiz_items == []
        assert sheet.stats == {}

    def test_title_sanitization(self):
        sheet = assemble_revision_sheet(title="lecture_notes_nlp.docx")
        assert sheet.title == "Lecture Notes Nlp"

        sheet_empty = assemble_revision_sheet(title="")
        assert "Revision Guide" in sheet_empty.title


# ─────────────────────────────────────────────────────────────────────────────
# ── Markdown Export ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestMarkdownExport:
    def test_markdown_full_export(self):
        sheet = assemble_revision_sheet(
            title=SAMPLE_TITLE,
            summary=SAMPLE_SUMMARY,
            keywords=SAMPLE_KEYWORDS,
            entities=SAMPLE_ENTITIES,
            stats=SAMPLE_STATS,
            quiz_items=SAMPLE_QUIZ,
        )
        md = export_markdown(sheet)

        assert f"# 📚 {SAMPLE_TITLE}" in md
        assert "Document Overview" in md
        assert "1,540" in md
        assert SAMPLE_SUMMARY in md
        assert "natural language processing" in md
        assert "Alan Turing" in md
        assert "Q1. Complete the statement" in md
        assert "**Answer:** Natural Language Processing" in md

    def test_markdown_empty_sections(self):
        sheet = assemble_revision_sheet(title="Empty Notes")
        md = export_markdown(sheet)

        assert "Summary not generated" in md
        assert "No keywords extracted" in md
        assert "No practice quiz questions generated" in md


# ─────────────────────────────────────────────────────────────────────────────
# ── Plain Text Export ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestTxtExport:
    def test_txt_full_export(self):
        sheet = assemble_revision_sheet(
            title=SAMPLE_TITLE,
            summary=SAMPLE_SUMMARY,
            keywords=SAMPLE_KEYWORDS,
            entities=SAMPLE_ENTITIES,
            stats=SAMPLE_STATS,
            quiz_items=SAMPLE_QUIZ,
        )
        txt = export_txt(sheet)

        assert "REVISION SHEET:" in txt
        assert SAMPLE_TITLE.upper() in txt
        assert "EXECUTIVE SUMMARY" in txt
        assert SAMPLE_SUMMARY in txt
        assert "KEY CONCEPTS & VOCABULARY" in txt
        assert "Tokenization" in txt or "tokenization" in txt
        assert "CORRECT ANSWER: Natural Language Processing" in txt
        assert "SOURCE NOTE:" in txt

    def test_txt_empty_sections(self):
        sheet = assemble_revision_sheet(title="Sparse Notes")
        txt = export_txt(sheet)

        assert "[Summary was not generated for this document]" in txt
        assert "[No keywords extracted]" in txt
        assert "[No practice quiz generated]" in txt


# ─────────────────────────────────────────────────────────────────────────────
# ── ReportLab PDF Export ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestPdfExport:
    def test_pdf_generation_returns_valid_pdf_bytes(self):
        sheet = assemble_revision_sheet(
            title=SAMPLE_TITLE,
            summary=SAMPLE_SUMMARY,
            keywords=SAMPLE_KEYWORDS,
            entities=SAMPLE_ENTITIES,
            stats=SAMPLE_STATS,
            quiz_items=SAMPLE_QUIZ,
        )
        pdf_bytes = export_pdf(sheet)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        # Valid PDF files start with %PDF magic bytes
        assert pdf_bytes.startswith(b"%PDF")

    def test_pdf_handles_empty_sections(self):
        sheet = assemble_revision_sheet(title="Blank Revision Sheet")
        pdf_bytes = export_pdf(sheet)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 500
        assert pdf_bytes.startswith(b"%PDF")

    def test_pdf_handles_special_characters_and_xml(self):
        """ReportLab XML parser must not crash on &, <, >, smart quotes, and dashes."""
        sheet = assemble_revision_sheet(
            title="NLP & AI: <Basics> & 'Advances'—2026",
            summary="Testing special chars: & < > \"quotes\" \u2018single\u2019 \u201cdouble\u201d \u2014 emdash.",
            keywords=[("c++ & python", 0.95), ("<token>", 0.85)],
            stats={"words": "100", "sentences": 10},
            quiz_items=[{
                "question": "Is A < B & C > D?",
                "options": ["A & B", "C < D", "Both & neither", "None"],
                "answer": "A & B",
                "source_sentence": "Formula states A < B & C > D.",
                "explanation": "Special chars & < > handled safely.",
            }],
        )
        pdf_bytes = export_pdf(sheet)
        assert pdf_bytes.startswith(b"%PDF")

    def test_pdf_multi_page_handling(self):
        """Ensure multi-page documents generate with running headers/footers cleanly."""
        long_summary = " ".join([SAMPLE_SUMMARY] * 12)
        many_keywords = [(f"Concept {i}", 1.0 - (i * 0.03)) for i in range(24)]
        many_quiz = [SAMPLE_QUIZ[0], SAMPLE_QUIZ[1]] * 4  # 8 questions

        sheet = assemble_revision_sheet(
            title="Comprehensive NLP Final Exam Revision Guide",
            summary=long_summary,
            keywords=many_keywords,
            entities=SAMPLE_ENTITIES * 4,
            stats=SAMPLE_STATS,
            quiz_items=many_quiz,
        )
        pdf_bytes = export_pdf(sheet)
        assert len(pdf_bytes) > 5000
        assert pdf_bytes.startswith(b"%PDF")


# ─────────────────────────────────────────────────────────────────────────────
# ── Consistency & Utility Tests ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestRevisionSheetConsistency:
    def test_quiz_answers_included_in_exports(self):
        sheet = assemble_revision_sheet(
            title="Quiz Check",
            quiz_items=SAMPLE_QUIZ,
        )
        md = export_markdown(sheet)
        txt = export_txt(sheet)

        assert "Natural Language Processing" in md
        assert "Tokenization" in md
        assert "Natural Language Processing" in txt
        assert "Tokenization" in txt

    def test_format_stat_num_helper(self):
        assert _format_stat_num(1000) == "1,000"
        assert _format_stat_num("2500") == "2,500"
        assert _format_stat_num(None) == "N/A"
        assert _format_stat_num("N/A") == "N/A"
        assert _format_stat_num("unknown") == "unknown"
