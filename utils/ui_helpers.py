"""
utils/ui_helpers.py
────────────────────
Reusable Streamlit UI components used across all pages.
Modern SaaS Academic Design System.
"""

from __future__ import annotations
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


# ── Step Button Helper ────────────────────────────────────────────────────────
def _pipeline_step(num: str, label: str, done: bool = False, active: bool = False, page: str = "") -> None:
    """Render a clickable pipeline step button using native Streamlit page_link for state preservation."""
    page_files = {
        "1": "pages/01_upload.py",
        "2": "pages/02_summary.py",
        "3": "pages/03_keywords.py",
        "4": "pages/04_quiz.py",
        "5": "pages/05_revision.py",
        "6": "pages/06_nlp_analysis.py",
    }
    target = page if (page and page.endswith(".py")) else page_files.get(num, "app.py")
    num_icons = {
        "1": "1️⃣",
        "2": "2️⃣",
        "3": "3️⃣",
        "4": "4️⃣",
        "5": "5️⃣",
        "6": "6️⃣",
    }
    icon = "✅" if done else num_icons.get(num, "🔹")
    st.page_link(target, label=label, icon=icon)


# ── Modern Sidebar Builder ────────────────────────────────────────────────────
def render_sidebar(active_page: str = "") -> None:
    """Render the sidebar with branding, clickable pipeline buttons, and document stats."""
    from utils.session import (
        has_text, has_summary, has_keywords, has_quiz, has_tf, has_flashcards, has_revision,
        KEY_FILE_NAME, KEY_RAW_TEXT, reset_all
    )

    with st.sidebar:
        # Brand Header
        st.markdown("""
        <div class="nlp-sidebar-brand">
            <div class="logo">⚡</div>
            <h3>NLP Studio</h3>
            <p>Academic Suite</p>
        </div>
        """, unsafe_allow_html=True)

        # Overview link
        st.page_link("app.py", label="Overview Dashboard", icon="🏠")

        st.markdown('<div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#64748b; letter-spacing:0.06em; margin:0.8rem 0 0.5rem 0.2rem;">Pipeline Status</div>', unsafe_allow_html=True)

        # 6 Clickable Pipeline Step Buttons (SPA routing preserves session state)
        _pipeline_step("1", "Upload Notes",    has_text(),                                  active_page == "upload",       page="pages/01_upload.py")
        _pipeline_step("2", "Summarization",   has_summary(),                               active_page == "summary",      page="pages/02_summary.py")
        _pipeline_step("3", "Keywords",        has_keywords(),                              active_page == "keywords",     page="pages/03_keywords.py")
        _pipeline_step("4", "Quiz & Practice", has_quiz() or has_tf() or has_flashcards(), active_page == "quiz",         page="pages/04_quiz.py")
        _pipeline_step("5", "Revision Sheet",  has_revision(),                              active_page == "revision",     page="pages/05_revision.py")
        _pipeline_step("6", "NLP Analysis",    False,                                       active_page == "nlp_analysis", page="pages/06_nlp_analysis.py")

        st.divider()

        # Current Document Info
        fname = st.session_state.get(KEY_FILE_NAME)
        raw   = st.session_state.get(KEY_RAW_TEXT)

        st.markdown('<div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#64748b; letter-spacing:0.06em; margin-bottom:0.5rem;">Current Document</div>', unsafe_allow_html=True)
        if fname and raw:
            words = len(raw.split())
            read_time = max(1, round(words / 200))
            st.markdown(f"""
            <div class="nlp-sidebar-status">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                    <span class="status-label">Active File</span>
                    <span class="nlp-pulse"></span>
                </div>
                <div style="font-size:0.84rem; color:#f1f5f9; font-weight:600; word-break:break-all;">{fname}</div>
                <div style="display:flex; gap:0.6rem; margin-top:0.4rem; font-size:0.76rem; color:#94a3b8;">
                    <span><b>{words:,}</b> words</span>
                    <span>·</span>
                    <span>~<b>{read_time}</b> min read</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="nlp-sidebar-status">
                <div class="status-label">Status</div>
                <div style="font-size:0.82rem; color:#64748b;">No document loaded</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Actions
        if st.button("🗑️ Reset All Data", key="sidebar_reset", use_container_width=True, type="secondary"):
            reset_all()
            st.success("All session data cleared.")
            st.rerun()

        st.markdown("""
        <div style="text-align:center; padding-top:0.8rem;">
            <div style="font-size:0.7rem; color:#64748b; font-weight:600;">✦ 100% Offline & Grounded</div>
            <div style="font-size:0.65rem; color:#475569; margin-top:0.15rem;">Academic Intelligence Dashboard</div>
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
        <h3 style="color:#ffffff; font-weight:700; font-size:1.25rem; margin-bottom:0.4rem;">{title}</h3>
        <p style="color:#94a3b8; font-size:0.92rem; max-width:380px; margin:0 auto 1.25rem auto;">{description}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    if cta_label and cta_page:
        cols = st.columns([1, 1.5, 1])
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
        f'<div style="display:flex; flex-wrap:wrap; gap:0.35rem; margin:0.5rem 0;">{chips_html}</div>',
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
    <div style="display:flex; align-items:center; gap:0.85rem; margin:1.75rem 0 1.25rem 0;">
        <div style="flex:1; height:1px; background:rgba(255,255,255,0.08);"></div>
        <div style="font-size:0.75rem; color:#64748b; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;">{label}</div>
        <div style="flex:1; height:1px; background:rgba(255,255,255,0.08);"></div>
    </div>
    """, unsafe_allow_html=True)


# ── Coming Soon Banner ────────────────────────────────────────────────────────
def coming_soon_banner(feature: str) -> None:
    st.markdown(f"""
    <div class="nlp-card nlp-animate" style="border-color:rgba(99,102,241,0.35);
         background:rgba(99,102,241,0.08); text-align:center; padding:1.25rem;">
        <span style="font-size:1.6rem;">🔜</span>
        <div style="font-size:0.95rem; font-weight:700; color:#f1f5f9; margin-top:0.4rem;">
            {feature} — Processing
        </div>
        <div style="font-size:0.82rem; color:#94a3b8; margin-top:0.25rem;">
            Feature is actively integrated into the pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)
