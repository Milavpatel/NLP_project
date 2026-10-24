"""
tests/test_phase2.py
──────────────────────
Focused tests for Phase 2: file parsing and text preprocessing.

Run with:
    python -m pytest tests/test_phase2.py -v

Tests:
  TXT / PDF / DOCX / PPTX extraction
  Empty inputs and error paths
  NLTK tokenisation
  Preprocessing pipeline options
"""

import io
import sys
import os
import textwrap
import pytest

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_parser import (
    parse_txt, parse_pdf, parse_docx, parse_pptx, parse_file,
    ParseError, SUPPORTED_EXTENSIONS,
)
from core.text_processor import (
    clean_text, split_sentences, tokenise_words,
    stem_tokens, preprocess, get_basic_stats,
    _get_stopwords,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build minimal valid in-memory files
# ─────────────────────────────────────────────────────────────────────────────

def _make_docx(paragraphs: list[str]) -> bytes:
    """Build a minimal DOCX in memory using python-docx."""
    import docx, io
    doc = docx.Document()
    for para in paragraphs:
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_pptx(slides: list[list[str]]) -> bytes:
    """Build a minimal PPTX in memory using python-pptx."""
    from pptx import Presentation
    from pptx.util import Inches
    import io
    prs = Presentation()
    blank_layout = prs.slide_layouts[6]   # completely blank
    for texts in slides:
        slide = prs.slides.add_slide(blank_layout)
        from pptx.util import Pt
        for i, text in enumerate(texts):
            txBox = slide.shapes.add_textbox(
                Inches(1), Inches(1 + i * 1.2), Inches(8), Inches(1)
            )
            tf = txBox.text_frame
            tf.text = text
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _make_pdf_text(text: str) -> bytes:
    """Build a minimal single-page PDF with a text layer using PyMuPDF."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# ── TXT Parsing ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestParseTxt:
    def test_basic_utf8(self):
        content = "Hello World. This is a test.".encode("utf-8")
        result = parse_txt(content)
        assert "Hello World" in result

    def test_latin1_fallback(self):
        content = "Caf\xe9 notes".encode("latin-1")   # é in latin-1
        result = parse_txt(content)
        assert "Caf" in result    # at minimum the prefix survives

    def test_empty_raises(self):
        with pytest.raises(ParseError, match="empty"):
            parse_txt(b"   \n  \t  ")

    def test_whitespace_only_raises(self):
        with pytest.raises(ParseError):
            parse_txt(b"")

    def test_multiline_preserved(self):
        content = "Line one.\nLine two.\nLine three.".encode()
        result = parse_txt(content)
        assert "Line one" in result
        assert "Line two" in result


# ─────────────────────────────────────────────────────────────────────────────
# ── PDF Parsing ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestParsePdf:
    def test_basic_pdf(self):
        pdf_bytes = _make_pdf_text("Natural Language Processing lecture notes.")
        result = parse_pdf(pdf_bytes)
        assert "Natural Language Processing" in result

    def test_empty_pdf_raises(self):
        """A PDF with no text layer should raise ParseError."""
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        doc = fitz.open()
        doc.new_page()          # blank page, no text
        buf = io.BytesIO()
        doc.save(buf)
        with pytest.raises(ParseError, match="No text could be extracted"):
            parse_pdf(buf.getvalue())

    def test_invalid_bytes_raises(self):
        with pytest.raises(ParseError):
            parse_pdf(b"this is not a pdf")

    def test_zero_bytes_raises(self):
        with pytest.raises(ParseError):
            parse_file("test.pdf", b"")

    def test_multiple_pages(self):
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 100), f"Page {i+1} content here.", fontsize=12)
        buf = io.BytesIO()
        doc.save(buf)
        result = parse_pdf(buf.getvalue())
        assert "Page 1" in result
        assert "Page 2" in result
        assert "Page 3" in result


# ─────────────────────────────────────────────────────────────────────────────
# ── DOCX Parsing ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestParseDocx:
    def test_basic_docx(self):
        content = _make_docx(["Introduction to NLP", "Tokenisation splits text into tokens."])
        result = parse_docx(content)
        assert "Introduction to NLP" in result
        assert "Tokenisation" in result

    def test_empty_docx_raises(self):
        """A DOCX with no paragraph content should raise ParseError."""
        import docx, io
        doc = docx.Document()
        buf = io.BytesIO()
        doc.save(buf)
        with pytest.raises(ParseError, match="No text"):
            parse_docx(buf.getvalue())

    def test_invalid_bytes_raises(self):
        with pytest.raises(ParseError):
            parse_docx(b"not a docx file")

    def test_multiline_paragraphs(self):
        paras = [f"Paragraph {i}" for i in range(5)]
        content = _make_docx(paras)
        result = parse_docx(content)
        for p in paras:
            assert p in result

    def test_dispatch_to_docx(self):
        content = _make_docx(["Hello from dispatch."])
        text, ftype = parse_file("notes.docx", content)
        assert ftype == "docx"
        assert "Hello from dispatch" in text


# ─────────────────────────────────────────────────────────────────────────────
# ── PPTX Parsing ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestParsePptx:
    def test_basic_pptx(self):
        content = _make_pptx([["Slide 1 Title", "Slide 1 Body"]])
        result = parse_pptx(content)
        assert "Slide 1 Title" in result
        assert "Slide 1 Body" in result

    def test_multiple_slides(self):
        content = _make_pptx([
            ["Introduction", "NLP basics"],
            ["Tokenisation", "Breaking text into tokens"],
        ])
        result = parse_pptx(content)
        assert "[Slide 1]" in result
        assert "[Slide 2]" in result
        assert "Introduction" in result
        assert "Tokenisation" in result

    def test_dispatch_to_pptx(self):
        content = _make_pptx([["Hello PPTX"]])
        text, ftype = parse_file("lecture.pptx", content)
        assert ftype == "pptx"
        assert "Hello PPTX" in text

    def test_invalid_bytes_raises(self):
        with pytest.raises(ParseError):
            parse_pptx(b"not a pptx")


# ─────────────────────────────────────────────────────────────────────────────
# ── parse_file dispatcher ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestParseFileDispatcher:
    def test_txt_dispatch(self):
        text, ftype = parse_file("notes.txt", b"Hello from txt.")
        assert ftype == "txt"
        assert "Hello" in text

    def test_unsupported_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_file("notes.xyz", b"some content")

    def test_zero_bytes_raises(self):
        with pytest.raises(ParseError, match="empty"):
            parse_file("notes.txt", b"")

    def test_supported_extensions_set(self):
        expected = {"txt", "pdf", "docx", "doc", "pptx"}
        assert expected == SUPPORTED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# ── clean_text ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_unicode_quotes_normalised(self):
        result = clean_text("\u201cHello\u201d \u2018world\u2019")
        assert '"Hello"' in result
        assert "'world'" in result

    def test_excessive_blank_lines_collapsed(self):
        result = clean_text("Para one.\n\n\n\n\nPara two.")
        assert "\n\n\n" not in result

    def test_trailing_whitespace_stripped(self):
        result = clean_text("  hello   ")
        assert result == "hello"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_non_breaking_space(self):
        result = clean_text("hello\u00a0world")
        assert "\u00a0" not in result


# ─────────────────────────────────────────────────────────────────────────────
# ── split_sentences ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitSentences:
    def test_basic_sentences(self):
        text = "The cat sat on the mat. The dog lay on the rug. Hello there."
        sents = split_sentences(text)
        assert len(sents) == 3

    def test_empty_input(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   ") == []

    def test_single_sentence(self):
        sents = split_sentences("Only one sentence here.")
        assert len(sents) == 1

    def test_returns_list_of_strings(self):
        sents = split_sentences("Sentence one. Sentence two!")
        assert all(isinstance(s, str) for s in sents)


# ─────────────────────────────────────────────────────────────────────────────
# ── tokenise_words ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestTokeniseWords:
    def test_basic_tokens(self):
        tokens = tokenise_words("Hello world test")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lowercase(self):
        tokens = tokenise_words("Hello WORLD")
        assert all(t == t.lower() for t in tokens)

    def test_stopwords_removed(self):
        tokens = tokenise_words("The cat sat on the mat", remove_stopwords=True)
        stopwords = _get_stopwords()
        for t in tokens:
            assert t not in stopwords

    def test_stopwords_kept(self):
        tokens = tokenise_words("the cat and the dog", remove_stopwords=False)
        # "the" should appear at least once
        assert "the" in tokens

    def test_punctuation_removed_by_default(self):
        tokens = tokenise_words("Hello, world! How are you?")
        for t in tokens:
            assert t not in (",", "!", "?", ".")

    def test_empty_input(self):
        assert tokenise_words("") == []

    def test_numeric_tokens_kept(self):
        tokens = tokenise_words("Chapter 3 covers NLP")
        assert "3" in tokens


# ─────────────────────────────────────────────────────────────────────────────
# ── stem_tokens ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestStemTokens:
    def test_basic_stemming(self):
        stemmed = stem_tokens(["running", "runs", "runner"])
        # Porter stemmer should produce the same or shorter root
        assert all(len(s) <= len(orig) + 2 for s, orig in
                   zip(stemmed, ["running", "runs", "runner"]))

    def test_empty_list(self):
        assert stem_tokens([]) == []

    def test_same_length_output(self):
        tokens = ["tokenize", "processing", "learning"]
        assert len(stem_tokens(tokens)) == len(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# ── get_basic_stats ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestGetBasicStats:
    def test_word_count(self):
        stats = get_basic_stats("Hello world foo bar")
        assert stats["words"] == 4

    def test_empty_text(self):
        stats = get_basic_stats("")
        assert stats["words"] == 0
        assert stats["sentences"] == 0

    def test_vocab_size_present(self):
        stats = get_basic_stats("cat cat dog dog bird")
        assert "vocab_size" in stats
        assert stats["vocab_size"] == 3   # cat, dog, bird

    def test_reading_time_at_least_one(self):
        stats = get_basic_stats("one")
        assert stats["reading_time"] >= 1

    def test_keys_present(self):
        stats = get_basic_stats("Some text here.")
        for key in ("words", "sentences", "characters", "paragraphs",
                    "avg_word_len", "reading_time", "vocab_size"):
            assert key in stats, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# ── preprocess (full pipeline) ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocess:
    TEXT = textwrap.dedent("""
        Natural Language Processing is a field of artificial intelligence.
        It enables computers to understand human language.
        Applications include machine translation and sentiment analysis.
    """).strip()

    def test_returns_processed_doc(self):
        from core.text_processor import ProcessedDoc
        doc = preprocess(self.TEXT)
        assert isinstance(doc, ProcessedDoc)

    def test_sentences_populated(self):
        doc = preprocess(self.TEXT)
        assert len(doc.sentences) >= 2

    def test_tokens_populated(self):
        doc = preprocess(self.TEXT)
        assert len(doc.tokens) > 0

    def test_stopwords_removed_when_requested(self):
        doc = preprocess(self.TEXT, remove_stopwords=True)
        stop = _get_stopwords()
        for t in doc.filtered:
            assert t not in stop

    def test_filtered_shorter_than_tokens(self):
        doc = preprocess(self.TEXT, remove_stopwords=True)
        assert len(doc.filtered) <= len(doc.tokens)

    def test_stemming_pipeline(self):
        doc = preprocess(self.TEXT, use_stemming=True)
        assert doc.stemming_used is True
        assert len(doc.stemmed) > 0

    def test_no_stemming_by_default(self):
        doc = preprocess(self.TEXT)
        assert doc.stemmed == []

    def test_vocab_is_set(self):
        doc = preprocess(self.TEXT)
        assert isinstance(doc.vocab, set)
        assert len(doc.vocab) > 0

    def test_empty_text(self):
        doc = preprocess("")
        assert doc.sentences == []
        assert doc.tokens == []
        assert doc.filtered == []

    def test_cleaned_text_preserved(self):
        doc = preprocess(self.TEXT)
        assert len(doc.cleaned_text) > 0


# ─────────────────────────────────────────────────────────────────────────────
# ── Run directly ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
