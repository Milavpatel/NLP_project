"""
pages/05_revision.py — Consolidated Revision Sheet
Phase 5: Complete Revision Sheet Builder & Multi-Format Exporter (PDF, Markdown, TXT).
Preserves UI theme, styling, session-state keys, and responsive layout.
"""

import streamlit as st
import pandas as pd

from utils.session import (
    init_session,
    KEY_RAW_TEXT,
    KEY_FILE_NAME,
    KEY_SUMMARY,
    KEY_SUMMARY_METHOD,
    KEY_KEYWORDS,
    KEY_KEYWORD_METHOD,
    KEY_ENTITIES,
    KEY_NLP_STATS,
    KEY_QUIZ_ITEMS,
    KEY_SENTENCES,
)
from utils.ui_helpers import (
    load_css,
    render_sidebar,
    page_header,
    section_divider,
    empty_state,
    chip_row,
)
from core.revision_builder import (
    assemble_revision_sheet,
    export_markdown,
    export_txt,
    export_pdf,
    RevisionSheet,
)
from core.text_processor import get_basic_stats, split_sentences
from core.summarizer import summarize_text
from core.keyword_extractor import tfidf_keywords, spacy_entities
from core.quiz_generator import generate_mcq

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Revision Sheet · NLP Summarizer",
    page_icon="🎯",
    layout="wide",
)
init_session()
load_css()
render_sidebar(active_page="revision")

page_header(
    "🎯",
    "Consolidated Revision Sheet",
    "An all-in-one study guide combining your executive summary, key concepts, named entities, document statistics, and practice MCQs.",
)

raw = st.session_state.get(KEY_RAW_TEXT)
file_name = st.session_state.get(KEY_FILE_NAME, "Study Notes")

if not raw or not raw.strip():
    empty_state(
        icon="🎯",
        title="No Document Loaded",
        description="Please upload or paste your study notes on the Upload page first to generate a revision sheet.",
        cta_label="📄 Go to Upload →",
        cta_page="01_upload",
    )
else:
    # Gather currently generated session data
    summary = st.session_state.get(KEY_SUMMARY)
    summary_method = st.session_state.get(KEY_SUMMARY_METHOD, "tfidf")
    keywords = st.session_state.get(KEY_KEYWORDS, [])
    keyword_method = st.session_state.get(KEY_KEYWORD_METHOD, "tfidf")
    entities = st.session_state.get(KEY_ENTITIES, [])
    quiz_items = st.session_state.get(KEY_QUIZ_ITEMS, [])
    stats = st.session_state.get(KEY_NLP_STATS)

    if not stats:
        stats = get_basic_stats(raw)
        st.session_state[KEY_NLP_STATS] = stats

    sentences = st.session_state.get(KEY_SENTENCES)
    if not sentences:
        sentences = split_sentences(raw)
        st.session_state[KEY_SENTENCES] = sentences

    # ── Quick Generation Controls for Missing Sections ────────────────────────
    has_all_sections = bool(summary and keywords and quiz_items)

    if not has_all_sections:
        with st.expander("⚡ Auto-Generate Missing Sections", expanded=True):
            st.caption(
                "You can generate any missing components right here, or build the entire "
                "revision guide in one click from your loaded notes."
            )
            qcol1, qcol2, qcol3, qcol4 = st.columns(4)

            with qcol1:
                if not summary:
                    if st.button("➕ Generate Summary", key="quick_gen_summ", use_container_width=True):
                        with st.spinner("Summarizing notes …"):
                            res = summarize_text(raw, method="tfidf", ratio=0.3)
                            st.session_state[KEY_SUMMARY] = res.summary_text
                            st.session_state[KEY_SUMMARY_METHOD] = res.method
                            st.rerun()
                else:
                    st.caption("✅ Summary Ready")

            with qcol2:
                if not keywords:
                    if st.button("➕ Extract Keywords", key="quick_gen_kws", use_container_width=True):
                        with st.spinner("Extracting keywords …"):
                            kws = tfidf_keywords(raw, n=15)
                            st.session_state[KEY_KEYWORDS] = kws
                            st.session_state[KEY_KEYWORD_METHOD] = "tfidf"
                            st.rerun()
                else:
                    st.caption(f"✅ {len(keywords)} Keywords Ready")

            with qcol3:
                if not entities:
                    if st.button("➕ Extract Entities", key="quick_gen_ents", use_container_width=True):
                        with st.spinner("Detecting named entities …"):
                            ents = spacy_entities(raw)
                            st.session_state[KEY_ENTITIES] = ents
                            st.rerun()
                else:
                    st.caption(f"✅ {len(entities)} Entities Ready")

            with qcol4:
                if not quiz_items:
                    if st.button("➕ Generate Practice MCQs", key="quick_gen_quiz", use_container_width=True):
                        with st.spinner("Creating practice quiz …"):
                            mcqs = generate_mcq(sentences, keywords if keywords else None, n=5, raw_text=raw)
                            st.session_state[KEY_QUIZ_ITEMS] = mcqs
                            st.rerun()
                else:
                    st.caption(f"✅ {len(quiz_items)} MCQs Ready")

            # Master button
            if st.button("🚀 Build Complete Revision Guide (Generate All Missing)", key="quick_gen_all", use_container_width=True):
                with st.spinner("Assembling complete revision sheet …"):
                    if not summary:
                        res = summarize_text(raw, method="tfidf", ratio=0.3)
                        st.session_state[KEY_SUMMARY] = res.summary_text
                        st.session_state[KEY_SUMMARY_METHOD] = res.method
                    if not keywords:
                        kws = tfidf_keywords(raw, n=15)
                        st.session_state[KEY_KEYWORDS] = kws
                        st.session_state[KEY_KEYWORD_METHOD] = "tfidf"
                    if not entities:
                        ents = spacy_entities(raw)
                        st.session_state[KEY_ENTITIES] = ents
                    if not quiz_items:
                        curr_kws = st.session_state.get(KEY_KEYWORDS, [])
                        mcqs = generate_mcq(sentences, curr_kws if curr_kws else None, n=5, raw_text=raw)
                        st.session_state[KEY_QUIZ_ITEMS] = mcqs
                st.success("Complete revision sheet assembled!")
                st.rerun()

    # Refresh data references
    summary = st.session_state.get(KEY_SUMMARY)
    keywords = st.session_state.get(KEY_KEYWORDS, [])
    entities = st.session_state.get(KEY_ENTITIES, [])
    quiz_items = st.session_state.get(KEY_QUIZ_ITEMS, [])

    # Assemble revision sheet instance
    sheet: RevisionSheet = assemble_revision_sheet(
        title=file_name,
        summary=summary,
        summary_method=summary_method,
        keywords=keywords,
        keyword_method=keyword_method,
        entities=entities,
        stats=stats,
        quiz_items=quiz_items,
        raw_text=raw,
    )

    # ── Header Banner ─────────────────────────────────────────────────────────
    section_divider("Revision Sheet Preview")

    st.markdown(
        f"""
        <div class="nlp-card nlp-animate" style="background:linear-gradient(135deg, #1e1b4b 0%, #161b22 100%);
             border: 1px solid rgba(124, 58, 237, 0.4); padding: 1.5rem; margin-bottom: 1.5rem;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.25rem;">
                📚 {sheet.title}
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 1rem;">
                Academic Study Guide · {sheet.date_str}
            </div>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                {'<span class="nlp-badge nlp-badge-ready">✅ Summary Included</span>' if sheet.summary else '<span class="nlp-badge nlp-badge-warning">⏳ No Summary</span>'}
                {'<span class="nlp-badge nlp-badge-ready">✅ ' + str(len(sheet.keywords)) + ' Key Concepts</span>' if sheet.keywords else '<span class="nlp-badge nlp-badge-warning">⏳ No Keywords</span>'}
                {'<span class="nlp-badge nlp-badge-info">🏢 ' + str(len(sheet.entities)) + ' Named Entities</span>' if sheet.entities else ''}
                {'<span class="nlp-badge nlp-badge-ready">🧠 ' + str(len(sheet.quiz_items)) + ' Practice MCQs</span>' if sheet.quiz_items else '<span class="nlp-badge nlp-badge-warning">⏳ No Quiz</span>'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 1: Document Overview ──────────────────────────────────────────
    st.markdown("#### 📊 1. Document Overview")
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Words", f"{stats.get('words', 0):,}")
    ov2.metric("Sentences", f"{stats.get('sentences', 0):,}")
    ov3.metric("Est. Reading Time", f"{stats.get('reading_time', 1)} min")
    ov4.metric("Unique Vocabulary", f"{stats.get('vocab_size', 0):,}")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Executive Summary ──────────────────────────────────────────
    st.markdown("#### 📋 2. Executive Summary")
    if sheet.summary:
        st.markdown(
            f"""
            <div class="nlp-card" style="font-size:0.92rem; line-height:1.8; color:#e2e8f0; border-left:3px solid #7c3aed;">
                {sheet.summary}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Summary not yet generated for this document. Use the quick generator above or visit the Summary page.", icon="ℹ️")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Key Concepts & Salience Scores ──────────────────────────────
    st.markdown("#### 🔑 3. Key Concepts & Vocabulary")
    if sheet.keywords:
        # Display as chips first
        chip_row([kw for kw, _ in sheet.keywords])
        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("🔍 View Concept Salience Scores Table", expanded=False):
            df_kws = pd.DataFrame(sheet.keywords, columns=["Concept / Term", "Importance Score"])
            df_kws.index = df_kws.index + 1
            st.dataframe(
                df_kws,
                use_container_width=True,
                column_config={
                    "Importance Score": st.column_config.ProgressColumn(
                        "Importance Score",
                        format="%.4f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                },
            )
    else:
        st.info("Keywords not yet extracted. Use the quick generator above or visit the Keywords page.", icon="ℹ️")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 4: Named Entities ─────────────────────────────────────────────
    if sheet.entities:
        st.markdown("#### 🏢 4. Named Entities Identified")
        df_ents = pd.DataFrame(sheet.entities, columns=["Entity", "Category"])
        df_ents.index = df_ents.index + 1
        st.dataframe(df_ents, use_container_width=True, height=200)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 5: Practice MCQs & Answer Key ─────────────────────────────────
    st.markdown("#### 🧠 5. Practice Questions & Complete Answer Key")
    if sheet.quiz_items:
        letters = ["A", "B", "C", "D"]
        for idx, item in enumerate(sheet.quiz_items, start=1):
            q_text = item.get("question", "").strip()
            options = item.get("options", [])
            answer = item.get("answer", "")
            src = item.get("source_sentence", "")
            exp = item.get("explanation", "")

            opts_html = "".join([
                f'<div style="margin:0.2rem 0; font-size:0.85rem; color:{"#10b981; font-weight:600;" if opt.strip().lower() == answer.strip().lower() else "#94a3b8;"}">'
                f'<strong>{letters[l_idx]})</strong> {opt} '
                f'{"<span class=\"nlp-badge nlp-badge-ready\" style=\"font-size:0.65rem;\">Answer Key</span>" if opt.strip().lower() == answer.strip().lower() else ""}'
                f'</div>'
                for l_idx, opt in enumerate(options)
            ])

            st.markdown(
                f"""
                <div class="nlp-card" style="margin-bottom:0.9rem; border-left:3px solid #059669;">
                    <div style="font-weight:600; color:#f8fafc; font-size:0.92rem; margin-bottom:0.5rem;">
                        Q{idx}. {q_text}
                    </div>
                    <div style="margin-bottom:0.6rem;">
                        {opts_html}
                    </div>
                    <div style="font-size:0.8rem; color:#8b949e; border-top:1px solid #30363d; padding-top:0.5rem;">
                        📖 <strong>Source from notes:</strong> <em>"{src}"</em><br>
                        💡 <strong>Explanation:</strong> {exp}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Practice questions not yet generated. Use the quick generator above or visit the Quiz page.", icon="ℹ️")

    # ── Section 6: Multi-Format Exports ───────────────────────────────────────
    section_divider("Export Revision Guide")

    st.markdown("Download your complete revision sheet in your preferred format:")

    # Generate export payloads
    with st.spinner("Preparing export bundles …"):
        md_content = export_markdown(sheet)
        txt_content = export_txt(sheet)
        pdf_bytes = export_pdf(sheet)

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        st.download_button(
            label="📕 Download PDF Guide (.pdf)",
            data=pdf_bytes,
            file_name=f"{sheet.title.lower().replace(' ', '_')}_revision_sheet.pdf",
            mime="application/pdf",
            key="dl_pdf_guide",
            use_container_width=True,
            help="High-quality multi-page ReportLab PDF with page numbers and formatted tables.",
        )

    with exp_col2:
        st.download_button(
            label="📝 Download Markdown (.md)",
            data=md_content,
            file_name=f"{sheet.title.lower().replace(' ', '_')}_revision_sheet.md",
            mime="text/markdown",
            key="dl_md_guide",
            use_container_width=True,
            help="GitHub-flavored Markdown format for Notion, Obsidian, or GitHub.",
        )

    with exp_col3:
        st.download_button(
            label="📄 Download Plain Text (.txt)",
            data=txt_content,
            file_name=f"{sheet.title.lower().replace(' ', '_')}_revision_sheet.txt",
            mime="text/plain",
            key="dl_txt_guide",
            use_container_width=True,
            help="Clean text guide compatible with any text editor or mobile notes app.",
        )
