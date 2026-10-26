"""
core/revision_builder.py
─────────────────────────
Phase 5: Complete Revision Sheet Builder.

Assembles all extracted NLP knowledge into a structured, unified study guide:
  - Document Title & Timestamp
  - Document Statistics
  - Concise Extractive Summary
  - Key Concepts & Importance Scores
  - Named Entities (when available)
  - Practice MCQs with Answer Key & Explanations

Provides multi-format exports:
  - export_markdown() — Clean, structured GitHub-flavored Markdown
  - export_txt()      — Clean formatted plain text
  - export_pdf()      — Multi-page PDF using ReportLab with headers/footers,
                        page numbers ("Page X of Y"), elegant styling, and
                        graceful handling of long text and missing sections.
"""

from __future__ import annotations

import datetime
import html
import io
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RevisionSheet:
    """Consolidated revision sheet containing all grounded study materials."""
    title: str
    date_str: str = ""
    summary: Optional[str] = None
    summary_method: Optional[str] = None
    keywords: list[tuple[str, float]] = field(default_factory=list)
    keyword_method: Optional[str] = None
    entities: list[tuple[str, str]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    quiz_items: list[dict] = field(default_factory=list)
    tf_items: list[dict] = field(default_factory=list)
    flashcards: list[dict] = field(default_factory=list)
    raw_text: Optional[str] = None

    def __post_init__(self):
        if not self.date_str:
            self.date_str = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")


# ── Assembly Helper ───────────────────────────────────────────────────────────
def assemble_revision_sheet(
    title: str = "Study Notes",
    summary: Optional[str] = None,
    summary_method: Optional[str] = None,
    keywords: Optional[list[tuple[str, float]]] = None,
    keyword_method: Optional[str] = None,
    entities: Optional[list[tuple[str, str]]] = None,
    stats: Optional[dict] = None,
    quiz_items: Optional[list[dict]] = None,
    tf_items: Optional[list[dict]] = None,
    flashcards: Optional[list[dict]] = None,
    raw_text: Optional[str] = None,
    date_str: Optional[str] = None,
) -> RevisionSheet:
    """
    Construct a RevisionSheet instance from available session or document data.
    Never fabricates missing sections — empty sections remain None or empty list.
    """
    clean_title = title.strip() if title and title.strip() else "Study Notes Revision Guide"
    if clean_title.endswith((".txt", ".pdf", ".docx", ".pptx", ".doc")):
        clean_title = clean_title.rsplit(".", 1)[0].replace("_", " ").title()

    return RevisionSheet(
        title=clean_title,
        date_str=date_str if date_str else datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        summary=summary.strip() if (summary and summary.strip()) else None,
        summary_method=summary_method if summary_method else None,
        keywords=list(keywords) if keywords else [],
        keyword_method=keyword_method if keyword_method else None,
        entities=list(entities) if entities else [],
        stats=dict(stats) if stats else {},
        quiz_items=list(quiz_items) if quiz_items else [],
        tf_items=list(tf_items) if tf_items else [],
        flashcards=list(flashcards) if flashcards else [],
        raw_text=raw_text if raw_text else None,
    )


def _format_stat_num(val, default="N/A") -> str:
    """Safely format an integer or numeric string with comma separators."""
    if val is None or val == "N/A":
        return default
    try:
        return f"{int(val):,}"
    except (ValueError, TypeError):
        return str(val)


# ── 1. Markdown Export ────────────────────────────────────────────────────────
def export_markdown(sheet: RevisionSheet) -> str:
    """
    Convert a RevisionSheet into a structured GitHub-flavored Markdown document.
    """
    lines: list[str] = []
    lines.append(f"# 📚 {sheet.title} — Revision Guide")
    lines.append(f"**Generated on:** {sheet.date_str}\n")
    lines.append("---\n")

    # Document Statistics
    if sheet.stats:
        lines.append("## 📊 Document Overview\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        if "words" in sheet.stats:
            lines.append(f"| Total Words | {_format_stat_num(sheet.stats['words'])} |")
        if "sentences" in sheet.stats:
            lines.append(f"| Sentences | {_format_stat_num(sheet.stats['sentences'])} |")
        if "reading_time" in sheet.stats:
            lines.append(f"| Est. Reading Time | {sheet.stats['reading_time']} min |")
        if "vocab_size" in sheet.stats:
            lines.append(f"| Vocabulary Size | {_format_stat_num(sheet.stats['vocab_size'])} unique words |")
        lines.append("")

    # Summary
    lines.append("## 📋 Key Summary\n")
    if sheet.summary:
        method_str = f" *({sheet.summary_method.upper()})*" if sheet.summary_method else ""
        lines.append(f"> {sheet.summary}{method_str}\n")
    else:
        lines.append("*Summary not generated for this revision sheet.*\n")

    # Key Terms
    lines.append("## 🔑 Crucial Concepts & Keywords\n")
    if sheet.keywords:
        lines.append("| # | Concept / Term | Salience Score |")
        lines.append("|---|---|---|")
        for idx, (kw, score) in enumerate(sheet.keywords, start=1):
            lines.append(f"| {idx} | **{kw}** | {score:.4f} |")
        lines.append("")
    else:
        lines.append("*No keywords extracted.*\n")

    # Named Entities
    if sheet.entities:
        lines.append("## 🏢 Named Entities Identified\n")
        lines.append("| Entity | Category |")
        lines.append("|---|---|")
        for ent, label in sheet.entities:
            lines.append(f"| {ent} | `{label}` |")
        lines.append("\n")

    # Practice Quiz
    lines.append("## 🧠 Practice Questions & Answer Key\n")
    if sheet.quiz_items:
        letters = ["A", "B", "C", "D"]
        for idx, item in enumerate(sheet.quiz_items, start=1):
            q_text = item.get("question", "").strip()
            options = item.get("options", [])
            answer = item.get("answer", "")
            src = item.get("source_sentence", "")
            exp = item.get("explanation", "")

            lines.append(f"### Q{idx}. {q_text}\n")
            for l_idx, opt in enumerate(options):
                prefix = letters[l_idx] if l_idx < len(letters) else "-"
                is_correct = " *(Correct Answer)*" if opt.strip().lower() == answer.strip().lower() else ""
                lines.append(f"- **{prefix})** {opt}{is_correct}")

            lines.append("")
            lines.append(f"**Answer:** {answer}  ")
            if src:
                lines.append(f"**Source from notes:** *\"{src}\"*  ")
            if exp and exp != src:
                lines.append(f"**Explanation:** {exp}  ")
            lines.append("")
    else:
        lines.append("*No practice quiz questions generated.*\n")

    # True / False
    if sheet.tf_items:
        lines.append("## ⚖️ True / False Practice Questions\n")
        lines.append("| # | Statement | Expected Answer | Explanation / Citation |")
        lines.append("|---|---|---|---|")
        for idx, item in enumerate(sheet.tf_items, start=1):
            stmt = item.get("statement", "").strip()
            ans = "**True**" if item.get("correct_answer") else "**False**"
            exp = item.get("explanation", "").strip()
            lines.append(f"| {idx} | {stmt} | {ans} | *{exp}* |")
        lines.append("")

    # Flashcards
    if sheet.flashcards:
        lines.append("## 📇 Key Concept Flashcards\n")
        lines.append("| # | Concept / Term | Category | Definition / Grounded Statement |")
        lines.append("|---|---|---|---|")
        for idx, fc in enumerate(sheet.flashcards, start=1):
            term = fc.get("term", "").strip()
            cat = fc.get("category", "Key Concept")
            definition = fc.get("definition", "").strip()
            lines.append(f"| {idx} | **{term}** | `{cat}` | {definition} |")
        lines.append("")

    lines.append("---\n*Generated by Student Notes Summarizer & Quiz Generator*")
    return "\n".join(lines)


# ── 2. Plain Text Export ──────────────────────────────────────────────────────
def export_txt(sheet: RevisionSheet) -> str:
    """
    Convert a RevisionSheet into a clean, human-readable plain text document.
    """
    lines: list[str] = []
    bar = "=" * 68
    sub_bar = "-" * 68

    lines.append(bar)
    lines.append(f"REVISION SHEET: {sheet.title.upper()}")
    lines.append(f"Generated on: {sheet.date_str}")
    lines.append(bar)
    lines.append("")

    # Stats
    if sheet.stats:
        lines.append("DOCUMENT OVERVIEW")
        lines.append(sub_bar)
        w = sheet.stats.get("words", "N/A")
        s = sheet.stats.get("sentences", "N/A")
        rt = sheet.stats.get("reading_time", "N/A")
        vs = sheet.stats.get("vocab_size", "N/A")
        lines.append(f"Words: {w} | Sentences: {s} | Est. Reading Time: {rt} min | Vocabulary: {vs}")
        lines.append("")

    # Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append(sub_bar)
    if sheet.summary:
        lines.append(sheet.summary)
    else:
        lines.append("[Summary was not generated for this document]")
    lines.append("")

    # Keywords
    lines.append("KEY CONCEPTS & VOCABULARY")
    lines.append(sub_bar)
    if sheet.keywords:
        for idx, (kw, score) in enumerate(sheet.keywords, start=1):
            lines.append(f"{idx:2d}. {kw:<30} (Score: {score:.4f})")
    else:
        lines.append("[No keywords extracted]")
    lines.append("")

    # Entities
    if sheet.entities:
        lines.append("NAMED ENTITIES")
        lines.append(sub_bar)
        for ent, label in sheet.entities:
            lines.append(f"• {ent} [{label}]")
        lines.append("")

    # Quiz
    lines.append("PRACTICE QUESTIONS & DETAILED ANSWER KEY")
    lines.append(sub_bar)
    if sheet.quiz_items:
        letters = ["A", "B", "C", "D"]
        for idx, item in enumerate(sheet.quiz_items, start=1):
            lines.append(f"Q{idx}. {item.get('question', '').strip()}")
            for l_idx, opt in enumerate(item.get("options", [])):
                prefix = letters[l_idx] if l_idx < len(letters) else "-"
                lines.append(f"    {prefix}) {opt}")
            lines.append(f"    CORRECT ANSWER: {item.get('answer', '')}")
            if item.get("source_sentence"):
                lines.append(f"    SOURCE NOTE:    \"{item.get('source_sentence', '')}\"")
            if item.get("explanation"):
                lines.append(f"    EXPLANATION:    {item.get('explanation', '')}")
            lines.append("")
    else:
        lines.append("[No practice quiz generated]")
        lines.append("")

    # True / False
    if sheet.tf_items:
        lines.append("TRUE / FALSE PRACTICE QUESTIONS")
        lines.append(sub_bar)
        for idx, item in enumerate(sheet.tf_items, start=1):
            ans_str = "TRUE" if item.get("correct_answer") else "FALSE"
            lines.append(f"TF{idx}. [ {ans_str:5s} ] {item.get('statement', '').strip()}")
            if item.get("explanation"):
                lines.append(f"        Citation: {item.get('explanation')}")
        lines.append("")

    # Flashcards
    if sheet.flashcards:
        lines.append("CONCEPT STUDY FLASHCARDS")
        lines.append(sub_bar)
        for idx, fc in enumerate(sheet.flashcards, start=1):
            lines.append(f"CARD {idx:2d}. [{fc.get('category', 'Concept')}] {fc.get('term', '')}")
            lines.append(f"         Definition: {fc.get('definition', '')}")
        lines.append("")

    lines.append(bar)
    lines.append("NLP Summarizer & Quiz Generator · Revision Guide")
    lines.append(bar)
    return "\n".join(lines)


# ── 3. ReportLab Multi-Page PDF Export ─────────────────────────────────────────
def export_pdf(sheet: RevisionSheet) -> bytes:
    """
    Generate a professional, publication-quality multi-page PDF revision guide.

    Features:
      - Clean palette (deep indigo primary, slate neutrals, emerald accents).
      - Running headers on subsequent pages.
      - Dynamic page count footer ("Page X of Y").
      - XML/HTML character escaping to prevent markup syntax crashes.
      - Table wrapping and page-break guards (keepWithNext=True on headings).
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        KeepTogether,
        HRFlowable,
    )
    from reportlab.pdfgen import canvas

    # Dynamic 2-pass Canvas for exact page count
    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_page_decorations(num_pages)
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)

        def draw_page_decorations(self, page_count):
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))

            # Running header (pages 2+)
            if self._pageNumber > 1:
                self.setStrokeColor(colors.HexColor("#cbd5e1"))
                self.setLineWidth(0.5)
                self.line(40, 755, 572, 755)
                self.drawString(40, 760, f"{sheet.title} — Revision Sheet")

            # Running footer (all pages)
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(40, 42, 572, 42)
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(572, 30, page_text)
            self.drawString(40, 30, "Student Notes Summarizer & Quiz Generator · Academic Revision Guide")
            self.restoreState()

    def _safe_xml(text: str) -> str:
        """Escape text safely for ReportLab Paragraph XML parser."""
        if not text:
            return ""
        # Clean control characters
        t = str(text)
        t = html.escape(t, quote=False)
        t = t.replace("\u2018", "'").replace("\u2019", "'")
        t = t.replace("\u201c", '"').replace("\u201d", '"')
        t = t.replace("\u2013", "-").replace("\u2014", "-")
        return t

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=48,
        bottomMargin=54,
    )

    base_styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#4c1d95"),  # Deep Violet
        spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e1b4b"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )
    summary_box_style = ParagraphStyle(
        "SummaryText",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
    )
    q_title_style = ParagraphStyle(
        "QuestionTitle",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1e1b4b"),
        spaceAfter=3,
        keepWithNext=True,
    )
    option_style = ParagraphStyle(
        "OptionText",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    answer_style = ParagraphStyle(
        "AnswerText",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#059669"),  # Emerald Green
    )
    source_style = ParagraphStyle(
        "SourceText",
        parent=base_styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
    )

    story = []

    # 1. Document Title & Header
    story.append(Paragraph(_safe_xml(sheet.title), title_style))
    story.append(Paragraph(_safe_xml(f"Academic Revision Guide · Generated on {sheet.date_str}"), meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#7c3aed"), spaceAfter=10))

    # 2. Document Overview Stats Grid
    if sheet.stats:
        story.append(Paragraph("Document Overview", h2_style))
        stats_data = [
            [
                Paragraph("<b>Total Words:</b>", table_cell_bold),
                Paragraph(_safe_xml(_format_stat_num(sheet.stats.get("words"))), table_cell_style),
                Paragraph("<b>Sentences:</b>", table_cell_bold),
                Paragraph(_safe_xml(_format_stat_num(sheet.stats.get("sentences"))), table_cell_style),
            ],
            [
                Paragraph("<b>Est. Reading Time:</b>", table_cell_bold),
                Paragraph(_safe_xml(f"{sheet.stats.get('reading_time', 'N/A')} min"), table_cell_style),
                Paragraph("<b>Unique Vocabulary:</b>", table_cell_bold),
                Paragraph(_safe_xml(_format_stat_num(sheet.stats.get("vocab_size"))), table_cell_style),
            ],
        ]
        stats_table = Table(stats_data, colWidths=[110, 150, 110, 160])
        stats_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 8))

    # 3. Executive Summary Block
    story.append(Paragraph("Executive Summary", h2_style))
    if sheet.summary:
        summary_content = [[Paragraph(_safe_xml(sheet.summary), summary_box_style)]]
        summary_table = Table(summary_content, colWidths=[532])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f3ff")),  # Soft violet tint
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#c4b5fd")),
            ("LINELEFT", (0, 0), (0, -1), 3, colors.HexColor("#7c3aed")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(summary_table)
    else:
        story.append(Paragraph("<i>No summary was generated for this document.</i>", body_style))
    story.append(Spacer(1, 8))

    # 4. Key Concepts & Vocabulary (2-column table)
    story.append(Paragraph("Key Concepts & Salience Scores", h2_style))
    if sheet.keywords:
        kw_rows = [
            [
                Paragraph("<b>#</b>", table_cell_bold),
                Paragraph("<b>Key Term / Concept</b>", table_cell_bold),
                Paragraph("<b>Score</b>", table_cell_bold),
                Paragraph("<b>#</b>", table_cell_bold),
                Paragraph("<b>Key Term / Concept</b>", table_cell_bold),
                Paragraph("<b>Score</b>", table_cell_bold),
            ]
        ]
        # Split into two columns for compact page usage
        half = (len(sheet.keywords) + 1) // 2
        col_left = sheet.keywords[:half]
        col_right = sheet.keywords[half:]

        for i in range(half):
            idx_l = i + 1
            term_l, score_l = col_left[i]
            if i < len(col_right):
                idx_r = half + i + 1
                term_r, score_r = col_right[i]
                row = [
                    Paragraph(str(idx_l), table_cell_style),
                    Paragraph(_safe_xml(term_l), table_cell_bold),
                    Paragraph(f"{score_l:.4f}", table_cell_style),
                    Paragraph(str(idx_r), table_cell_style),
                    Paragraph(_safe_xml(term_r), table_cell_bold),
                    Paragraph(f"{score_r:.4f}", table_cell_style),
                ]
            else:
                row = [
                    Paragraph(str(idx_l), table_cell_style),
                    Paragraph(_safe_xml(term_l), table_cell_bold),
                    Paragraph(f"{score_l:.4f}", table_cell_style),
                    Paragraph("", table_cell_style),
                    Paragraph("", table_cell_style),
                    Paragraph("", table_cell_style),
                ]
            kw_rows.append(row)

        kw_table = Table(kw_rows, colWidths=[24, 180, 56, 24, 180, 56])
        kw_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ede9fe")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(kw_table)
    else:
        story.append(Paragraph("<i>No keywords were extracted for this document.</i>", body_style))
    story.append(Spacer(1, 8))

    # 5. Named Entities (if any)
    if sheet.entities:
        story.append(Paragraph("Named Entities", h2_style))
        ent_rows = [[
            Paragraph("<b>Entity</b>", table_cell_bold),
            Paragraph("<b>Category</b>", table_cell_bold),
            Paragraph("<b>Entity</b>", table_cell_bold),
            Paragraph("<b>Category</b>", table_cell_bold),
        ]]
        half_e = (len(sheet.entities) + 1) // 2
        e_left = sheet.entities[:half_e]
        e_right = sheet.entities[half_e:]
        for i in range(half_e):
            ent_l, cat_l = e_left[i]
            if i < len(e_right):
                ent_r, cat_r = e_right[i]
                row = [
                    Paragraph(_safe_xml(ent_l), table_cell_bold),
                    Paragraph(_safe_xml(cat_l), table_cell_style),
                    Paragraph(_safe_xml(ent_r), table_cell_bold),
                    Paragraph(_safe_xml(cat_r), table_cell_style),
                ]
            else:
                row = [
                    Paragraph(_safe_xml(ent_l), table_cell_bold),
                    Paragraph(_safe_xml(cat_l), table_cell_style),
                    Paragraph("", table_cell_style),
                    Paragraph("", table_cell_style),
                ]
            ent_rows.append(row)

        ent_table = Table(ent_rows, colWidths=[170, 96, 170, 96])
        ent_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(ent_table)
        story.append(Spacer(1, 8))

    # 6. Practice Questions & Detailed Answer Key
    story.append(Paragraph("Practice Questions & Answer Key", h2_style))
    if sheet.quiz_items:
        letters = ["A", "B", "C", "D"]
        for idx, item in enumerate(sheet.quiz_items, start=1):
            q_flowables = []
            q_text = item.get("question", "").strip()
            q_flowables.append(Paragraph(_safe_xml(f"Q{idx}. {q_text}"), q_title_style))

            # Options
            opt_texts = []
            for l_idx, opt in enumerate(item.get("options", [])):
                letter_p = letters[l_idx] if l_idx < len(letters) else "-"
                is_correct = opt.strip().lower() == item.get("answer", "").strip().lower()
                mark = " [CORRECT]" if is_correct else ""
                opt_texts.append(f"<b>{letter_p})</b> {opt}{mark}")

            for opt_str in opt_texts:
                q_flowables.append(Paragraph(_safe_xml(opt_str), option_style))

            # Answer and Source Note
            q_flowables.append(Spacer(1, 2))
            q_flowables.append(Paragraph(_safe_xml(f"Answer Key: {item.get('answer', '')}"), answer_style))
            if item.get("source_sentence"):
                q_flowables.append(Paragraph(_safe_xml(f"Source Note: \"{item.get('source_sentence', '')}\""), source_style))

            q_flowables.append(Spacer(1, 8))
            # Keep question block together to prevent page breaks across a question
            story.append(KeepTogether(q_flowables))
    else:
        story.append(Paragraph("<i>No practice quiz questions were generated.</i>", body_style))

    # 7. True / False Questions (if any)
    if sheet.tf_items:
        story.append(Spacer(1, 6))
        story.append(Paragraph("True / False Practice", h2_style))
        for idx, item in enumerate(sheet.tf_items, start=1):
            tf_flowables = []
            stmt = item.get("statement", "").strip()
            ans = "TRUE" if item.get("correct_answer") else "FALSE"
            tf_flowables.append(Paragraph(_safe_xml(f"TF{idx}. {stmt}"), q_title_style))
            tf_flowables.append(Paragraph(_safe_xml(f"Expected: {ans}"), answer_style))
            if item.get("explanation"):
                tf_flowables.append(Paragraph(_safe_xml(f"Citation: {item.get('explanation')}"), source_style))
            tf_flowables.append(Spacer(1, 6))
            story.append(KeepTogether(tf_flowables))
        story.append(Spacer(1, 8))

    # 8. Flashcards (if any)
    if sheet.flashcards:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Concept Study Flashcards", h2_style))
        fc_rows = [[
            Paragraph("<b>#</b>", table_cell_bold),
            Paragraph("<b>Concept / Term</b>", table_cell_bold),
            Paragraph("<b>Category</b>", table_cell_bold),
            Paragraph("<b>Definition / Grounded Statement</b>", table_cell_bold),
        ]]
        for idx, fc in enumerate(sheet.flashcards[:15], start=1):
            fc_rows.append([
                Paragraph(str(idx), table_cell_style),
                Paragraph(_safe_xml(fc.get("term", "")), table_cell_bold),
                Paragraph(_safe_xml(fc.get("category", "Concept")), table_cell_style),
                Paragraph(_safe_xml(fc.get("definition", "")), table_cell_style),
            ])
        fc_table = Table(fc_rows, colWidths=[25, 120, 85, 302])
        fc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(fc_table)
        story.append(Spacer(1, 8))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
