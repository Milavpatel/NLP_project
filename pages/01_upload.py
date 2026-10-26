"""
pages/01_upload.py — Upload Notes
Phase 2: Connected to real file parser and preprocessing pipeline.
Preserves all NLP preprocessing logic and session keys.
"""

import streamlit as st

from utils.session    import (init_session, KEY_RAW_TEXT, KEY_FILE_NAME,
                               KEY_UPLOADED_FILE, KEY_PROCESSED_TEXT,
                               KEY_SENTENCES, KEY_NLP_STATS,
                               reset_all, clear_derived_data)
from utils.ui_helpers import (load_css, render_sidebar, page_header,
                               section_divider)
from utils.file_parser import parse_file, ParseError, SUPPORTED_EXTENSIONS
from core.text_processor import (get_basic_stats, preprocess,
                                  split_sentences, spacy_status)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Upload Notes · NLP Studio",
    page_icon="📄",
    layout="wide",
)
init_session()
load_css()
render_sidebar(active_page="upload")

page_header(
    "📄",
    "Upload & Ingest Study Notes",
    "Import lecture notes in PDF, Word, PowerPoint, or Plain Text format to run the NLP pipeline.",
)

# ── spaCy status notice ───────────────────────────────────────────────────────
_spacy_ok, _spacy_msg = spacy_status()
if not _spacy_ok:
    st.warning(
        f"⚠️ **spaCy lemmatisation unavailable.** {_spacy_msg}  \n"
        "Stemming and tokenisation will still work normally.",
        icon="⚠️",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Ingestion Source Tabs (File Upload vs Direct Paste)
# ─────────────────────────────────────────────────────────────────────────────
tab_upload, tab_paste = st.tabs(["📁 File Uploader (.pdf, .docx, .pptx, .txt)", "✏️ Direct Paste Notes"])

with tab_upload:
    uploaded = st.file_uploader(
        label="Drag & drop your study notes here, or browse files",
        type=["txt", "pdf", "docx", "doc", "pptx"],
        key="file_uploader_widget",
        help="Supported formats: Searchable PDF, Microsoft Word (.docx), PowerPoint (.pptx), Plain Text (.txt)",
    )

with tab_paste:
    pasted_text = st.text_area(
        label="Paste text directly",
        placeholder="Type or paste lecture notes, book excerpts, or study material here …",
        height=220,
        key="paste_text_widget",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Pipeline Configuration
# ─────────────────────────────────────────────────────────────────────────────
section_divider("Preprocessing Pipeline Configuration")

prep_col1, prep_col2, prep_col3 = st.columns(3)
with prep_col1:
    st.markdown("""
    <div class="nlp-card" style="padding:1rem; margin-bottom:0.5rem;">
        <div style="font-weight:700; color:#f1f5f9; font-size:0.92rem; margin-bottom:0.2rem;">Stopword Removal</div>
        <div style="font-size:0.78rem; color:#94a3b8; line-height:1.45;">Filters 179 high-frequency function words via NLTK corpus.</div>
    </div>""", unsafe_allow_html=True)
    opt_stopwords = st.toggle("Enable Stopword Filtering", value=True, key="opt_stopwords")

with prep_col2:
    st.markdown("""
    <div class="nlp-card" style="padding:1rem; margin-bottom:0.5rem;">
        <div style="font-weight:700; color:#f1f5f9; font-size:0.92rem; margin-bottom:0.2rem;">Porter Stemming</div>
        <div style="font-size:0.78rem; color:#94a3b8; line-height:1.45;">Heuristic suffix stripping algorithm (running → run, study → studi).</div>
    </div>""", unsafe_allow_html=True)
    opt_stemming = st.toggle("Enable Porter Stemmer", value=False, key="opt_stemming")

with prep_col3:
    st.markdown("""
    <div class="nlp-card" style="padding:1rem; margin-bottom:0.5rem;">
        <div style="font-weight:700; color:#f1f5f9; font-size:0.92rem; margin-bottom:0.2rem;">spaCy Lemmatization</div>
        <div style="font-size:0.78rem; color:#94a3b8; line-height:1.45;">Morphological reduction to canonical vocabulary headwords.</div>
    </div>""", unsafe_allow_html=True)
    opt_lemma = st.toggle("Enable spaCy Lemmatizer", value=False, key="opt_lemma", disabled=not _spacy_ok)

# ─────────────────────────────────────────────────────────────────────────────
# Action Buttons
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
process_col1, process_col2, _ = st.columns([1.5, 1, 3])
with process_col1:
    process_btn = st.button("⚡ Ingest & Process Notes", key="process_btn", use_container_width=True)
with process_col2:
    clear_btn = st.button("🗑️ Clear Input", key="clear_btn", use_container_width=True, type="secondary")

if clear_btn:
    reset_all()
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Document Ingestion Handler
# ─────────────────────────────────────────────────────────────────────────────
if process_btn:
    raw_text  = None
    file_name = None

    if uploaded is not None:
        with st.spinner(f"Extracting text from **{uploaded.name}** …"):
            try:
                content   = uploaded.read()
                raw_text, ftype = parse_file(uploaded.name, content)
                file_name = uploaded.name
                st.session_state[KEY_UPLOADED_FILE] = content
                st.toast(f"Parsed {file_name} successfully ({ftype.upper()})", icon="✅")
            except ParseError as e:
                st.error(f"❌ **Parse error:** {e}")
                raw_text = None
            except ValueError as e:
                st.error(f"❌ **Unsupported format:** {e}")
                raw_text = None
            except Exception as e:
                st.error(f"❌ **Unexpected error:** {e}")
                raw_text = None

    elif pasted_text.strip():
        raw_text  = pasted_text.strip()
        file_name = "pasted_notes.txt"
        st.toast("Loaded notes from direct paste.", icon="✅")

    else:
        st.warning("Please upload a file or paste some notes first.", icon="ℹ️")

    if raw_text:
        clear_derived_data()

        st.session_state[KEY_RAW_TEXT]  = raw_text
        st.session_state[KEY_FILE_NAME] = file_name

        with st.spinner("Executing linguistic preprocessing …"):
            try:
                doc = preprocess(
                    raw_text,
                    remove_stopwords = opt_stopwords,
                    use_stemming     = opt_stemming,
                    use_lemma        = opt_lemma,
                )
                st.session_state[KEY_PROCESSED_TEXT] = " ".join(doc.filtered)
                st.session_state[KEY_SENTENCES]      = doc.sentences
                stats = get_basic_stats(raw_text)
                stats["vocab_size"]    = len(doc.vocab)
                stats["token_count"]   = len(doc.tokens)
                stats["filtered_count"]= len(doc.filtered)
                st.session_state[KEY_NLP_STATS] = stats
            except Exception as e:
                st.warning(f"Preprocessing warning: {e}. Basic stats still available.")
                st.session_state[KEY_NLP_STATS] = get_basic_stats(raw_text)

# ─────────────────────────────────────────────────────────────────────────────
# Active Document Inspection & Metrics View
# ─────────────────────────────────────────────────────────────────────────────
raw = st.session_state.get(KEY_RAW_TEXT)

if raw:
    section_divider("Ingested Document Intelligence")

    stats = st.session_state.get(KEY_NLP_STATS) or get_basic_stats(raw)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Words",     f"{stats.get('words', 0):,}")
    c2.metric("Sentences",       f"{stats.get('sentences', 0):,}")
    c3.metric("Characters",      f"{stats.get('characters', 0):,}")
    c4.metric("Paragraphs",      f"{stats.get('paragraphs', 0):,}")
    c5.metric("Unique Vocab",    f"{stats.get('vocab_size', '—')}")
    c6.metric("Est. Reading",    f"{stats.get('reading_time', 1)} min")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Text Comparison / Preview ─────────────────────────────────────────────
    processed = st.session_state.get(KEY_PROCESSED_TEXT)
    show_compare = st.toggle(
        "🔍 Compare Raw Source vs. Filtered Content Tokens",
        value=False,
        key="toggle_compare",
    )

    if show_compare and processed:
        col_raw, col_proc = st.columns(2)
        with col_raw:
            st.markdown("**Original Source Text**")
            st.markdown(
                f'<div class="nlp-card" style="font-size:0.84rem; line-height:1.65; '
                f'font-family:\'JetBrains Mono\',monospace; white-space:pre-wrap; '
                f'max-height:340px; overflow-y:auto; border-color:rgba(99,102,241,0.3);">'
                f'{raw[:2500]}{"…" if len(raw) > 2500 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_proc:
            st.markdown("**Cleaned Content Tokens** (Stopwords filtered, lower-cased)")
            st.markdown(
                f'<div class="nlp-card" style="font-size:0.84rem; line-height:1.65; '
                f'font-family:\'JetBrains Mono\',monospace; white-space:pre-wrap; '
                f'max-height:340px; overflow-y:auto; border-color:rgba(6,182,212,0.3);">'
                f'{processed[:2500]}{"…" if len(processed) > 2500 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )
            tok_count  = stats.get("token_count", len(raw.split()))
            filt_count = stats.get("filtered_count", tok_count)
            removed    = tok_count - filt_count
            st.caption(
                f"📊 {tok_count:,} raw tokens → {filt_count:,} content tokens "
                f"({removed:,} stopwords/punctuation elements eliminated)"
            )
    else:
        with st.expander("📖 View Document Excerpt", expanded=True):
            preview_len = min(2500, len(raw))
            st.markdown(
                f'<div class="nlp-card" style="font-size:0.86rem; line-height:1.7; '
                f'font-family:\'JetBrains Mono\',monospace; white-space:pre-wrap; '
                f'max-height:360px; overflow-y:auto;">'
                f'{raw[:preview_len]}{"…" if len(raw) > 2500 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

    sentences = st.session_state.get(KEY_SENTENCES, [])
    if sentences:
        with st.expander(f"📝 Segmented Sentences ({len(sentences):,} identified)", expanded=False):
            for i, sent in enumerate(sentences[:20], 1):
                st.markdown(
                    f'<div style="padding:0.5rem 0.85rem; margin-bottom:0.4rem; '
                    f'background:rgba(15,23,42,0.6); border-radius:8px; border-left:3px solid #6366f1; font-size:0.86rem; color:#f1f5f9;">'
                    f'<span style="color:#818cf8; font-weight:700; margin-right:0.6rem;">{i}.</span>'
                    f'{sent}</div>',
                    unsafe_allow_html=True,
                )
            if len(sentences) > 20:
                st.caption(f"… and {len(sentences) - 20} more sentences in the document.")

    # ── Next Action Stepper ───────────────────────────────────────────────────
    section_divider("Recommended Next Steps")
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button("📋 Generate Summary →", key="nav_summary", use_container_width=True):
            st.switch_page("pages/02_summary.py")
    with nav_col2:
        if st.button("🔑 Extract Keywords →", key="nav_keywords", use_container_width=True):
            st.switch_page("pages/03_keywords.py")
    with nav_col3:
        if st.button("🧠 Practice Quiz & Flashcards →", key="nav_quiz", use_container_width=True):
            st.switch_page("pages/04_quiz.py")

else:
    st.markdown("""
    <div class="nlp-empty nlp-animate">
        <div class="icon">📂</div>
        <h3 style="color:#ffffff; font-weight:700; font-size:1.3rem;">No Document Loaded Yet</h3>
        <p style="color:#94a3b8; font-size:0.92rem; max-width:440px; line-height:1.6;">
            Choose a study notes file above (.PDF, .DOCX, .PPTX, .TXT) or paste text directly. 
            Once ingested, the system extracts sentences, builds vocabulary frequencies, 
            and prepares your practice suite.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_divider("Supported Document Formats")
    fmt_c1, fmt_c2, fmt_c3, fmt_c4 = st.columns(4)
    formats = [
        ("📕", ".PDF Documents", "PyMuPDF text extraction with layout parsing", "#ef4444"),
        ("📘", ".DOCX Word", "python-docx parser with paragraph and table extraction", "#3b82f6"),
        ("📊", ".PPTX Slides", "python-pptx extractor across all slide frames", "#f59e0b"),
        ("📝", ".TXT Plain", "Multi-encoding UTF-8 & Latin-1 stream decoder", "#10b981"),
    ]
    for col, (icon, title, desc, color) in zip([fmt_c1, fmt_c2, fmt_c3, fmt_c4], formats):
        col.markdown(f"""
        <div class="nlp-card" style="text-align:center; padding:1.25rem 1rem;">
            <div style="font-size:2rem; margin-bottom:0.4rem;">{icon}</div>
            <div style="font-weight:700; color:#ffffff; font-size:0.95rem;">{title}</div>
            <div style="font-size:0.78rem; color:#94a3b8; margin-top:0.35rem; line-height:1.45;">{desc}</div>
        </div>""", unsafe_allow_html=True)
