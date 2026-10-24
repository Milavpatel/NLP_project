"""
pages/01_upload.py — Upload Notes
Phase 2: Connected to real file parser and preprocessing pipeline.
Preserves all Phase 1 UI, CSS, session-state keys.
"""

import streamlit as st

from utils.session    import (init_session, KEY_RAW_TEXT, KEY_FILE_NAME,
                               KEY_UPLOADED_FILE, KEY_PROCESSED_TEXT,
                               KEY_SENTENCES, KEY_NLP_STATS,
                               reset_all, clear_derived_data)
from utils.ui_helpers import (load_css, render_sidebar, page_header,
                               section_divider, coming_soon_banner)
from utils.file_parser import parse_file, ParseError, SUPPORTED_EXTENSIONS
from core.text_processor import (get_basic_stats, preprocess,
                                  split_sentences, spacy_status)

# ── Page config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="Upload Notes · NLP Summarizer",
    page_icon="📄",
    layout="wide",
)
init_session()
load_css()
render_sidebar(active_page="upload")

page_header(
    "📄",
    "Upload Notes",
    "Import your study notes in TXT, PDF, DOCX, or PPTX format.",
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
# Section 1 — Upload Widget
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### Choose a file")

uploaded = st.file_uploader(
    label="Drag & drop your notes file here, or click to browse",
    type=["txt", "pdf", "docx", "doc", "pptx"],
    key="file_uploader_widget",
    help="Supported: Plain Text (.txt), PDF (.pdf), Word (.docx), PowerPoint (.pptx)",
)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Paste alternative
# ─────────────────────────────────────────────────────────────────────────────
section_divider("or paste directly")

pasted_text = st.text_area(
    label="Paste your notes here",
    placeholder="Paste your notes directly into this box if you don't have a file …",
    height=180,
    key="paste_text_widget",
)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Preprocessing options
# ─────────────────────────────────────────────────────────────────────────────
section_divider("Preprocessing Options")

prep_col1, prep_col2, prep_col3 = st.columns(3)
with prep_col1:
    opt_stopwords = st.toggle(
        "Remove stopwords",
        value=True,
        key="opt_stopwords",
        help="Filter out common English words (the, is, at …)",
    )
with prep_col2:
    opt_stemming = st.toggle(
        "Porter stemming",
        value=False,
        key="opt_stemming",
        help="Reduce words to their root stem (running → run)",
    )
with prep_col3:
    opt_lemma = st.toggle(
        "spaCy lemmatisation",
        value=False,
        key="opt_lemma",
        help="Morphologically reduce words using spaCy (requires en_core_web_sm)",
        disabled=not _spacy_ok,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Action buttons
# ─────────────────────────────────────────────────────────────────────────────
process_col1, process_col2, _ = st.columns([1, 1, 3])
with process_col1:
    process_btn = st.button(
        "⚡ Process Notes", key="process_btn", use_container_width=True
    )
with process_col2:
    clear_btn = st.button(
        "🗑️ Clear", key="clear_btn", use_container_width=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# Clear handler
# ─────────────────────────────────────────────────────────────────────────────
if clear_btn:
    reset_all()
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Process handler
# ─────────────────────────────────────────────────────────────────────────────
if process_btn:
    raw_text  = None
    file_name = None

    # --- File upload path ---
    if uploaded is not None:
        with st.spinner(f"Parsing **{uploaded.name}** …"):
            try:
                content   = uploaded.read()
                raw_text, ftype = parse_file(uploaded.name, content)
                file_name = uploaded.name
                st.session_state[KEY_UPLOADED_FILE] = content
                st.success(f"✅ Loaded **{file_name}** ({ftype.upper()})")
            except ParseError as e:
                st.error(f"❌ **Parse error:** {e}")
                raw_text = None
            except ValueError as e:
                st.error(f"❌ **Unsupported format:** {e}")
                raw_text = None
            except Exception as e:
                st.error(f"❌ **Unexpected error:** {e}")
                raw_text = None

    # --- Paste path ---
    elif pasted_text.strip():
        raw_text  = pasted_text.strip()
        file_name = "pasted_text.txt"
        st.success("✅ Text loaded from paste.")

    else:
        st.warning("Please upload a file or paste some text first.")

    # --- Store raw text ---
    if raw_text:
        # Clear any downstream results from previous documents to prevent stale state
        clear_derived_data()

        st.session_state[KEY_RAW_TEXT]  = raw_text
        st.session_state[KEY_FILE_NAME] = file_name

        # --- Run preprocessing pipeline ---
        with st.spinner("Running NLP preprocessing …"):
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
# Preview section (shown whenever raw text is in session)
# ─────────────────────────────────────────────────────────────────────────────
raw = st.session_state.get(KEY_RAW_TEXT)

if raw:
    section_divider("Document Preview")

    # ── Metrics row ──────────────────────────────────────────────────────────
    stats = st.session_state.get(KEY_NLP_STATS) or get_basic_stats(raw)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Words",         f"{stats.get('words', 0):,}")
    c2.metric("Sentences",     f"{stats.get('sentences', 0):,}")
    c3.metric("Characters",    f"{stats.get('characters', 0):,}")
    c4.metric("Paragraphs",    f"{stats.get('paragraphs', 0):,}")
    c5.metric("Vocab Size",    f"{stats.get('vocab_size', '—')}")
    c6.metric("Reading Time",  f"{stats.get('reading_time', 1)} min")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Side-by-side compare: raw vs processed ────────────────────────────────
    processed = st.session_state.get(KEY_PROCESSED_TEXT)
    show_compare = st.toggle(
        "🔍 Compare original vs. processed text",
        value=False,
        key="toggle_compare",
    )

    if show_compare and processed:
        col_raw, col_proc = st.columns(2)
        with col_raw:
            st.markdown("**Original text**")
            st.markdown(
                f'<div class="nlp-card" style="font-size:0.82rem; line-height:1.65; '
                f'font-family:\'JetBrains Mono\',monospace; white-space:pre-wrap; '
                f'max-height:320px; overflow-y:auto;">'
                f'{raw[:2000]}{"…" if len(raw) > 2000 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_proc:
            st.markdown("**Processed tokens** (stopwords removed, lower-cased)")
            preview_tokens = processed[:2000]
            st.markdown(
                f'<div class="nlp-card" style="font-size:0.82rem; line-height:1.65; '
                f'font-family:\'JetBrains Mono\',monospace; white-space:pre-wrap; '
                f'max-height:320px; overflow-y:auto;">'
                f'{preview_tokens}{"…" if len(processed) > 2000 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Token stats
            tok_count  = stats.get("token_count",    len(raw.split()))
            filt_count = stats.get("filtered_count", tok_count)
            removed    = tok_count - filt_count
            st.caption(
                f"📊 {tok_count:,} raw tokens → {filt_count:,} after filtering "
                f"({removed:,} stopwords/punct removed)"
            )
    else:
        # Single original preview
        with st.expander("📖 Show full text", expanded=True):
            preview_len = min(2000, len(raw))
            st.markdown(
                f'<div class="nlp-card" style="font-size:0.88rem; line-height:1.7; '
                f'font-family:\'JetBrains Mono\',monospace; white-space:pre-wrap; '
                f'max-height:360px; overflow-y:auto;">'
                f'{raw[:preview_len]}{"…" if len(raw) > 2000 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Sentence preview ──────────────────────────────────────────────────────
    sentences = st.session_state.get(KEY_SENTENCES, [])
    if sentences:
        with st.expander(f"📝 Sentences ({len(sentences):,} detected)", expanded=False):
            for i, sent in enumerate(sentences[:20], 1):
                st.markdown(
                    f'<div style="padding:0.4rem 0.75rem; margin-bottom:0.35rem; '
                    f'border-left:3px solid #7c3aed; font-size:0.85rem; color:#e6edf3;">'
                    f'<span style="color:#7c3aed; font-weight:700; margin-right:0.5rem;">{i}.</span>'
                    f'{sent}</div>',
                    unsafe_allow_html=True,
                )
            if len(sentences) > 20:
                st.caption(f"… and {len(sentences) - 20} more sentences.")

    # ── Next Steps ────────────────────────────────────────────────────────────
    section_divider("Next Steps")
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button("📋 Generate Summary →", key="nav_summary", use_container_width=True):
            st.switch_page("pages/02_summary.py")
    with nav_col2:
        if st.button("🔑 Extract Keywords →", key="nav_keywords", use_container_width=True):
            st.switch_page("pages/03_keywords.py")
    with nav_col3:
        if st.button("📈 NLP Analysis →", key="nav_nlp", use_container_width=True):
            st.switch_page("pages/06_nlp_analysis.py")

# ─────────────────────────────────────────────────────────────────────────────
# Empty state (no document loaded yet)
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div class="nlp-empty nlp-animate">
        <div class="icon">📂</div>
        <h3>No Document Loaded</h3>
        <p>Upload a TXT, PDF, DOCX, or PPTX file above, or paste your notes
           into the text box. Once loaded, you'll see a live word count,
           sentence count, vocabulary size, and a full text preview.</p>
    </div>
    """, unsafe_allow_html=True)

    section_divider("Supported Formats")
    fmt_c1, fmt_c2, fmt_c3, fmt_c4 = st.columns(4)
    fmt_c1.markdown("""
    <div class="nlp-card" style="text-align:center;">
        <div style="font-size:2rem;">📝</div>
        <div style="font-weight:600; color:#e6edf3; margin-top:0.4rem;">.TXT</div>
        <div style="font-size:0.78rem; color:#8b949e; margin-top:0.3rem;">Plain text — best compatibility</div>
    </div>""", unsafe_allow_html=True)
    fmt_c2.markdown("""
    <div class="nlp-card" style="text-align:center;">
        <div style="font-size:2rem;">📕</div>
        <div style="font-weight:600; color:#e6edf3; margin-top:0.4rem;">.PDF</div>
        <div style="font-size:0.78rem; color:#10b981; margin-top:0.3rem;">✅ Real extraction via PyMuPDF</div>
    </div>""", unsafe_allow_html=True)
    fmt_c3.markdown("""
    <div class="nlp-card" style="text-align:center;">
        <div style="font-size:2rem;">📘</div>
        <div style="font-weight:600; color:#e6edf3; margin-top:0.4rem;">.DOCX</div>
        <div style="font-size:0.78rem; color:#10b981; margin-top:0.3rem;">✅ Real extraction via python-docx</div>
    </div>""", unsafe_allow_html=True)
    fmt_c4.markdown("""
    <div class="nlp-card" style="text-align:center;">
        <div style="font-size:2rem;">📊</div>
        <div style="font-weight:600; color:#e6edf3; margin-top:0.4rem;">.PPTX</div>
        <div style="font-size:0.78rem; color:#10b981; margin-top:0.3rem;">✅ Real extraction via python-pptx</div>
    </div>""", unsafe_allow_html=True)
