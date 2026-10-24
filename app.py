"""
app.py — NLP Student Notes Summarizer & Quiz Generator
════════════════════════════════════════════════════════
Main Streamlit entry point.
Run with:  streamlit run app.py

Architecture:
  • This file configures global page settings and renders the Home / Dashboard.
  • Individual feature pages live in pages/ and are auto-discovered by Streamlit.
  • Shared state is managed via utils/session.py
  • Reusable UI pieces live in utils/ui_helpers.py
"""

import streamlit as st
from utils.session   import init_session, has_text, has_summary, has_keywords, has_quiz
from utils.ui_helpers import load_css, render_sidebar, page_header, section_divider

# ── Page Configuration (must be FIRST Streamlit call) ────────────────────────
st.set_page_config(
    page_title="NLP Notes Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com/",
        "Report a bug": None,
        "About": "Student NLP Project — Notes Summarizer & Quiz Generator",
    },
)

# ── Global Init ───────────────────────────────────────────────────────────────
init_session()
load_css()
render_sidebar(active_page="home")

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="nlp-hero nlp-animate">
    <h1>📚 NLP Notes Summarizer</h1>
    <p>
        Transform your study notes into intelligent summaries, keyword maps,
        auto-generated quizzes, and revision sheets — powered by NLP.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Quick-Start Steps ─────────────────────────────────────────────────────────
st.markdown("### 🚀 How It Works")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="nlp-card nlp-animate" style="text-align:center; padding:1.5rem 1rem;">
        <div style="font-size:2.2rem;">📄</div>
        <div style="font-size:0.85rem; font-weight:600; margin-top:0.5rem; color:#e6edf3;">Step 1</div>
        <div style="font-size:1rem; font-weight:700; color:#9f67ff; margin-top:0.25rem;">Upload Notes</div>
        <div style="font-size:0.78rem; color:#8b949e; margin-top:0.4rem;">TXT · PDF · DOCX</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="nlp-card nlp-animate" style="text-align:center; padding:1.5rem 1rem; animation-delay:0.08s;">
        <div style="font-size:2.2rem;">🔍</div>
        <div style="font-size:0.85rem; font-weight:600; margin-top:0.5rem; color:#e6edf3;">Step 2</div>
        <div style="font-size:1rem; font-weight:700; color:#9f67ff; margin-top:0.25rem;">NLP Analysis</div>
        <div style="font-size:0.78rem; color:#8b949e; margin-top:0.4rem;">Summarise · Extract Keywords</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="nlp-card nlp-animate" style="text-align:center; padding:1.5rem 1rem; animation-delay:0.16s;">
        <div style="font-size:2.2rem;">📝</div>
        <div style="font-size:0.85rem; font-weight:600; margin-top:0.5rem; color:#e6edf3;">Step 3</div>
        <div style="font-size:1rem; font-weight:700; color:#9f67ff; margin-top:0.25rem;">Take the Quiz</div>
        <div style="font-size:0.78rem; color:#8b949e; margin-top:0.4rem;">MCQ · True/False</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="nlp-card nlp-animate" style="text-align:center; padding:1.5rem 1rem; animation-delay:0.24s;">
        <div style="font-size:2.2rem;">🎯</div>
        <div style="font-size:0.85rem; font-weight:600; margin-top:0.5rem; color:#e6edf3;">Step 4</div>
        <div style="font-size:1rem; font-weight:700; color:#9f67ff; margin-top:0.25rem;">Revision Sheet</div>
        <div style="font-size:0.78rem; color:#8b949e; margin-top:0.4rem;">Flashcards · Export</div>
    </div>""", unsafe_allow_html=True)

section_divider("Pipeline Status")

# ── Status Dashboard ──────────────────────────────────────────────────────────
st.markdown("### 📊 Session Overview")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Document Loaded",  "✅ Yes" if has_text()     else "❌ None",   help="Upload a file on the Upload page")
m2.metric("Summary Ready",    "✅ Yes" if has_summary()  else "⏳ Pending", help="Generate on the Summary page")
m3.metric("Keywords Ready",   "✅ Yes" if has_keywords() else "⏳ Pending", help="Extract on the Keywords page")
m4.metric("Quiz Ready",       "✅ Yes" if has_quiz()     else "⏳ Pending", help="Generate on the Quiz page")

section_divider("Features")

# ── Feature Cards Grid ────────────────────────────────────────────────────────
st.markdown("### 🧰 Features")

features = [
    ("📄", "Upload Notes",   "Supports TXT, PDF, DOCX, PPTX. Multi-encoding upload with preprocessing.",       "01_upload",      "ready"),
    ("📋", "Summary",        "Extractive TF-IDF & Normalized Word Frequency scoring with ROUGE metrics.",       "02_summary",     "ready"),
    ("🔑", "Keywords",       "TF-IDF, Normalized Frequency, RAKE keyphrases, and separate spaCy NER.",          "03_keywords",    "ready"),
    ("🧠", "Quiz Generator", "100% note-grounded MCQ questions with answer keys and source sentence citations.","04_quiz",        "ready"),
    ("🎯", "Revision Sheet", "Consolidated study guide with multi-page ReportLab PDF, Markdown, and TXT export.","05_revision",    "ready"),
    ("📈", "NLP Analysis",   "POS tagging, word frequency, Flesch readability, sentence histograms, word cloud.","06_nlp_analysis","ready"),
]

badge_map = {
    "ready":   ('<span class="nlp-badge nlp-badge-ready">✅ Ready</span>',   ""),
    "pending": ('<span class="nlp-badge nlp-badge-pending">⏳ In Progress</span>', "opacity:0.75;"),
}

row1 = st.columns(3)
row2 = st.columns(3)
rows = row1 + row2

for col, (icon, title, desc, page, status) in zip(rows, features):
    bdg, opacity = badge_map[status]
    with col:
        st.markdown(f"""
        <div class="nlp-card nlp-animate" style="{opacity}">
            <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
            <div style="font-size:1rem; font-weight:700; color:#e6edf3; margin-bottom:0.3rem;">{title}</div>
            {bdg}
            <div style="font-size:0.8rem; color:#8b949e; margin-top:0.6rem; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

section_divider("Tech Stack")

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Technology Stack")

tech_cols = st.columns(6)
tech_stack = [
    ("🐍", "Python 3.13"),
    ("🌊", "Streamlit"),
    ("📖", "NLTK"),
    ("🔬", "spaCy"),
    ("📐", "scikit-learn"),
    ("🤗", "Transformers"),
]
for col, (icon, name) in zip(tech_cols, tech_stack):
    col.markdown(f"""
    <div style="text-align:center; padding:0.75rem 0.5rem;
                background:#1c2333; border:1px solid #30363d;
                border-radius:10px;">
        <div style="font-size:1.5rem;">{icon}</div>
        <div style="font-size:0.75rem; color:#8b949e; margin-top:0.3rem; font-weight:500;">{name}</div>
    </div>
    """, unsafe_allow_html=True)

# ── CTA ───────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_divider("Get Started")

cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
with cta_col2:
    if st.button("📄 Upload Your First Notes →", key="home_cta", use_container_width=True):
        st.switch_page("pages/01_upload.py")
