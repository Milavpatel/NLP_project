"""
tests/test_quiz_slider.py
─────────────────────────
Verify that MCQ generation generates the exact number of requested questions
specified by the UI slider on real study documents.
"""

import pytest
from core.text_processor import clean_text, split_sentences
from core.quiz_generator import generate_mcq
from utils.file_parser import parse_pdf


@pytest.fixture(scope="module")
def sample_pdf_text():
    with open("sample_nlp_notes.pdf", "rb") as f:
        return parse_pdf(f.read())


@pytest.mark.parametrize("requested_n", [2, 3, 4, 5, 8, 10])
def test_mcq_generation_fulfills_slider_count_mixed(sample_pdf_text, requested_n):
    """Verify that 'mixed' question format generates the exact requested n questions."""
    sentences = split_sentences(clean_text(sample_pdf_text))
    items = generate_mcq(sentences, n=requested_n, raw_text=sample_pdf_text, question_type="mixed", seed=42)
    
    assert len(items) == requested_n, (
        f"Expected {requested_n} MCQs for mixed format, but got {len(items)}"
    )
    for q in items:
        assert len(q["options"]) == 4
        assert q["answer"] in q["options"]
        assert len(set(q["options"])) == 4


@pytest.mark.parametrize("requested_n", [2, 3, 5])
def test_mcq_generation_fulfills_slider_count_fib(sample_pdf_text, requested_n):
    """Verify that 'fill_in_the_blank' generates the exact requested n questions."""
    sentences = split_sentences(clean_text(sample_pdf_text))
    items = generate_mcq(sentences, n=requested_n, raw_text=sample_pdf_text, question_type="fill_in_the_blank", seed=42)
    
    assert len(items) == requested_n
    for q in items:
        assert q["question_type"] == "fill_in_the_blank"
        assert "_______" in q["question"]


@pytest.mark.parametrize("requested_n", [2, 3, 5])
def test_mcq_generation_fulfills_slider_count_def(sample_pdf_text, requested_n):
    """Verify that 'definition' generates the exact requested n questions."""
    sentences = split_sentences(clean_text(sample_pdf_text))
    items = generate_mcq(sentences, n=requested_n, raw_text=sample_pdf_text, question_type="definition", seed=42)
    
    assert len(items) == requested_n
    for q in items:
        assert q["question_type"] == "definition"
