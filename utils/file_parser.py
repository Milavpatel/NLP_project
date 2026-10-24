"""
utils/file_parser.py
──────────────────────
Phase 2: Real text extraction for TXT, PDF, DOCX, PPTX.

Design decisions:
  - PDF: uses PyMuPDF (fitz) — better text layer coverage than PyPDF2.
    Raises ParseError for encrypted / image-only / empty PDFs so the UI
    can show a clear, honest error instead of returning fake text.
  - DOCX: python-docx, includes table cell text.
  - PPTX: python-pptx, extracts all text frames across all slides.
  - TXT: tries UTF-8 then latin-1 fallback.
  - All parsers raise ParseError on unrecoverable problems.
"""

from __future__ import annotations

import io
import re
from pathlib import Path


# ── Custom exception ──────────────────────────────────────────────────────────
class ParseError(Exception):
    """Raised when a file cannot be parsed cleanly."""


# ── TXT ───────────────────────────────────────────────────────────────────────
def parse_txt(content: bytes, encoding: str = "utf-8") -> str:
    """Decode plain-text bytes, falling back to latin-1 on encoding error."""
    if not content.strip():
        raise ParseError("The uploaded text file is empty.")
    try:
        text = content.decode(encoding)
    except UnicodeDecodeError:
        text = content.decode("latin-1", errors="replace")
    text = text.strip()
    if not text:
        raise ParseError("The uploaded text file contains no readable text.")
    return text


# ── PDF ───────────────────────────────────────────────────────────────────────
def parse_pdf(content: bytes) -> str:
    """
    Extract text from PDF bytes using PyMuPDF (fitz).

    Raises ParseError for:
      - Encrypted / password-protected PDFs
      - PDFs with no text layer (image-only / scanned)
      - PDFs that produce no content after extraction
    """
    try:
        import pymupdf as fitz  # PyMuPDF 2.0+ preferred import
    except ImportError:
        try:
            import fitz  # fallback for older PyMuPDF
        except ImportError:
            raise ParseError(
                "PyMuPDF is not installed. Run: pip install PyMuPDF"
            )

    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ParseError(f"Could not open PDF: {exc}") from exc

    if doc.is_encrypted:
        raise ParseError(
            "This PDF is password-protected. Please decrypt it before uploading."
        )

    pages_text: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        try:
            page_text = page.get_text("text")          # plain text
        except Exception:
            page_text = ""
        if page_text.strip():
            pages_text.append(page_text.strip())

    doc.close()

    if not pages_text:
        raise ParseError(
            "No text could be extracted from this PDF. It may be a scanned "
            "image-only PDF with no text layer. Try running OCR first, or "
            "copy-paste the text into the 'Paste notes' box."
        )

    full_text = "\n\n".join(pages_text)
    # Collapse excessive blank lines left by PDF layout
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()
    return full_text


# ── DOCX ──────────────────────────────────────────────────────────────────────
def parse_docx(content: bytes) -> str:
    """
    Extract text from DOCX bytes using python-docx.
    Includes paragraph text AND all table cell text.
    """
    try:
        import docx
    except ImportError:
        raise ParseError(
            "python-docx is not installed. Run: pip install python-docx"
        )

    try:
        doc = docx.Document(io.BytesIO(content))
    except Exception as exc:
        raise ParseError(f"Could not open DOCX file: {exc}") from exc

    chunks: list[str] = []

    # Paragraphs
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            chunks.append(text)

    # Tables (flatten all cells)
    for table in doc.tables:
        for row in table.rows:
            row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_texts:
                chunks.append(" | ".join(row_texts))

    if not chunks:
        raise ParseError(
            "No text could be extracted from this DOCX file. "
            "The document may be empty or contain only images."
        )

    return "\n\n".join(chunks)


# ── PPTX ──────────────────────────────────────────────────────────────────────
def parse_pptx(content: bytes) -> str:
    """
    Extract text from PPTX bytes using python-pptx.
    Iterates every slide and every text frame within each shape.
    """
    try:
        from pptx import Presentation
    except ImportError:
        raise ParseError(
            "python-pptx is not installed. Run: pip install python-pptx"
        )

    try:
        prs = Presentation(io.BytesIO(content))
    except Exception as exc:
        raise ParseError(f"Could not open PPTX file: {exc}") from exc

    slide_chunks: list[str] = []
    for slide_num, slide in enumerate(prs.slides, start=1):
        texts: list[str] = []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                line = "".join(run.text for run in para.runs).strip()
                if line:
                    texts.append(line)
        if texts:
            slide_chunks.append(f"[Slide {slide_num}]\n" + "\n".join(texts))

    if not slide_chunks:
        raise ParseError(
            "No text could be extracted from this PPTX file. "
            "The presentation may be empty or contain only images."
        )

    return "\n\n".join(slide_chunks)


# ── Dispatcher ────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {"txt", "pdf", "docx", "doc", "pptx"}


def parse_file(filename: str, content: bytes) -> tuple[str, str]:
    """
    Dispatch file parsing based on file extension.

    Returns:
        (extracted_text, file_type_str)

    Raises:
        ParseError — on unrecoverable parse failures (caller should display
                     the error message directly to the user).
        ValueError  — for unsupported file types.
    """
    if not content:
        raise ParseError("The uploaded file is empty (0 bytes).")

    ext = Path(filename).suffix.lstrip(".").lower()

    if ext == "txt":
        return parse_txt(content), "txt"
    elif ext == "pdf":
        return parse_pdf(content), "pdf"
    elif ext in ("docx", "doc"):
        return parse_docx(content), "docx"
    elif ext == "pptx":
        return parse_pptx(content), "pptx"
    else:
        raise ValueError(
            f"Unsupported file type '.{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
