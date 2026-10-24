"""
utils/ui_helpers.py
────────────────────
Reusable Streamlit UI components used across all pages.
No NLP logic lives here — only presentation helpers.
"""

import streamlit as st
import os
from pathlib import Path


# ── CSS Loader ────────────────────────────────────────────────────────────────
def load_css() -> None:
    """Inject the global stylesheet into the app."""
    css_path = Path(__file__).parent.parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ── Sidebar Builder ───────────────────────────────────────────────────────────
def render_sidebar(active_page: str = "") -> None:
    """Render the sidebar with branding, pipeline status, and navigation."""
    from utils.session import (
        has_text, has_summary, has_keywords, has_quiz, has_revision,
        KEY_FILE_NAME, KEY_RAW_TEXT, reset_all
    )

    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="nlp-sidebar-brand">
            <div class="logo">📚</div>
            <h3>NLP Summarizer</h3>
            <p>Quiz Generator</p>
        </div>
        """, unsafe_allow_html=True)

        # Pipeline progress
        st.markdown("**Pipeline Status**")
        _pipeline_step("1", "Upload Notes",    has_text(),    active_page == "upload")
        _pipeline_step("2", "Summarization",   has_summary(), active_page == "summary")
        _pipeline_step("3", "Keywords",        has_keywords(), active_page == "keywords")
        _pipeline_step("4", "Quiz",            has_quiz(),    active_page == "quiz")
        _pipeline_step("5", "Revision Sheet",  has_revision(), active_page == "revision")
        _pipeline_step("6", "NLP Analysis",    False,         active_page == "nlp_analysis")

        st.divider()

        # Current file info
        fname = st.session_state.get(KEY_FILE_NAME)
        raw   = st.session_state.get(KEY_RAW_TEXT)

        st.markdown("**Current Document**")
        if fname:
            st.markdown(f"""
            <div class="nlp-sidebar-status">
                <div class="status-label">File</div>
                <div style="font-size:0.82rem; color:#e6edf3; word-break:break-all;">{fname}</div>
            </div>
            """, unsafe_allow_html=True)
            if raw:
                words = len(raw.split())
                chars = len(raw)
                col1, col2 = st.columns(2)
                col1.metric("Words", f"{words:,}")
                col2.metric("Chars", f"{chars:,}")
        else:
            st.markdown("""
            <div class="nlp-sidebar-status">
                <div class="status-label">Status</div>
                <div style="font-size:0.82rem; color:#8b949e;">No document loaded</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Actions
        if st.button("🗑️ Reset All Data", key="sidebar_reset", use_container_width=True):
            reset_all()
            st.success("All data cleared.")
            st.rerun()

        st.markdown("""
        <div style="text-align:center; padding-top:1rem;">
            <div style="font-size:0.7rem; color:#484f58;">NLP Project · Phase 5</div>
            <div style="font-size:0.7rem; color:#484f58;">Academic Dashboard</div>
        </div>
        """, unsafe_allow_html=True)


def _pipeline_step(num: str, label: str, done: bool, active: bool) -> None:
    done_class   = "done"   if done   else ""
    active_class = "active" if active else ""
    icon = "✓" if done else num
    st.markdown(f"""
    <div class="nlp-step {done_class} {active_class}">
        <div class="step-num">{icon}</div>
        <div style="font-size:0.85rem; font-weight:500;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Page Header ───────────────────────────────────────────────────────────────
def page_header(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div class="nlp-page-header nlp-animate">
        <h2>{icon} {title}</h2>
        {"<p>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


# ── Empty State ───────────────────────────────────────────────────────────────
def empty_state(
    icon: str,
    title: str,
    description: str,
    cta_label: str = "",
    cta_page: str = "",
) -> None:
    """Show a centred empty-state placeholder."""
    html = f"""
    <div class="nlp-empty nlp-animate">
        <div class="icon">{icon}</div>
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    if cta_label and cta_page:
        cols = st.columns([1, 2, 1])
        with cols[1]:
            if st.button(cta_label, key=f"cta_{cta_page}", use_container_width=True):
                st.switch_page(f"pages/{cta_page}.py")


# ── Info Card ─────────────────────────────────────────────────────────────────
def info_card(content: str, card_class: str = "nlp-card") -> None:
    st.markdown(
        f'<div class="{card_class} nlp-animate">{content}</div>',
        unsafe_allow_html=True,
    )


# ── Badge ─────────────────────────────────────────────────────────────────────
def badge(label: str, kind: str = "info") -> str:
    """Return HTML for a status badge. kind: info|ready|pending|error"""
    icons = {"info": "ℹ️", "ready": "✅", "pending": "⏳", "error": "❌"}
    icon  = icons.get(kind, "ℹ️")
    return f'<span class="nlp-badge nlp-badge-{kind}">{icon} {label}</span>'


# ── Chip Row ──────────────────────────────────────────────────────────────────
def chip_row(items: list[str]) -> None:
    """Render a horizontal list of keyword chips."""
    chips_html = "".join(f'<span class="nlp-chip">{item}</span>' for item in items)
    st.markdown(
        f'<div style="display:flex; flex-wrap:wrap; gap:0.25rem;">{chips_html}</div>',
        unsafe_allow_html=True,
    )


# ── Stat Row ──────────────────────────────────────────────────────────────────
def stat_row(stats: dict) -> None:
    """
    Render a row of metric pills.
    stats: {"Label": value, ...}
    """
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats.items()):
        col.metric(label, value)


# ── Divider with label ────────────────────────────────────────────────────────
def section_divider(label: str) -> None:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.75rem; margin:1.25rem 0;">
        <div style="flex:1; height:1px; background:#30363d;"></div>
        <div style="font-size:0.75rem; color:#8b949e; font-weight:600; letter-spacing:0.06em; text-transform:uppercase;">{label}</div>
        <div style="flex:1; height:1px; background:#30363d;"></div>
    </div>
    """, unsafe_allow_html=True)


# ── Coming Soon Banner ────────────────────────────────────────────────────────
def coming_soon_banner(feature: str) -> None:
    st.markdown(f"""
    <div class="nlp-card nlp-animate" style="border-color:rgba(124,58,237,0.4);
         background:rgba(124,58,237,0.06); text-align:center; padding:1rem;">
        <span style="font-size:1.4rem;">🔜</span>
        <div style="font-size:0.9rem; font-weight:600; color:#e6edf3; margin-top:0.4rem;">
            {feature} — Coming in Phase 3
        </div>
        <div style="font-size:0.8rem; color:#8b949e; margin-top:0.25rem;">
            Upload &amp; preprocessing is ready (Phase 2). NLP generation coming next.
        </div>
    </div>
    """, unsafe_allow_html=True)
