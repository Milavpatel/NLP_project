"""
app.py — NLP Student Notes Summarizer & Quiz Generator
════════════════════════════════════════════════════════
Main Streamlit entry point.
Run with:  streamlit run app.py
"""

import streamlit as st
from utils.session   import init_session, has_text, has_summary, has_keywords, has_quiz, has_tf, has_flashcards, has_revision
from utils.ui_helpers import load_css, render_sidebar, page_header, section_divider

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Studio · Intelligent Study Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com/",
        "Report a bug": None,
        "About": "NLP Studio — Academic Notes Summarizer & Practice Suite",
    },
)

# ── Global Init ───────────────────────────────────────────────────────────────
init_session()
load_css()
render_sidebar(active_page="home")

# ── Modern Hero Section ───────────────────────────────────────────────────────
st.markdown("""
<div class="nlp-hero nlp-animate">
    <div class="nlp-hero-pill">
        <span class="nlp-pulse"></span>
        <span>100% Offline · Deterministic · Zero Hallucination</span>
    </div>
    <h1>Transform Study Notes into<br>Intelligent Practice Suites</h1>
    <p>
        An enterprise-grade NLP academic environment that converts raw lecture notes into 
        extractive summaries, keyphrase rankings, verified True/False assertions, and interactive 
        3D flashcards — backed by rigorous linguistic algorithms.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Quick-Start Workflow Steps ────────────────────────────────────────────────
st.markdown("### ⚡ Guided Study Workflow")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="nlp-workflow-step nlp-animate">
        <div class="nlp-step-number">Step 01</div>
        <div style="font-size:2.2rem; margin:0.3rem 0;">📄</div>
        <div style="font-size:1.05rem; font-weight:700; color:#ffffff;">Upload & Clean</div>
        <div style="font-size:0.82rem; color:#94a3b8; margin-top:0.35rem; line-height:1.5;">
            Import PDF, DOCX, PPTX, or TXT notes with automatic tokenization.
        </div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="nlp-workflow-step nlp-animate" style="animation-delay:0.06s;">
        <div class="nlp-step-number">Step 02</div>
        <div style="font-size:2.2rem; margin:0.3rem 0;">🔍</div>
        <div style="font-size:1.05rem; font-weight:700; color:#ffffff;">Analyze & Extract</div>
        <div style="font-size:0.82rem; color:#94a3b8; margin-top:0.35rem; line-height:1.5;">
            Score sentences via TF-IDF & Luhn frequency, plus RAKE keyphrases.
        </div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="nlp-workflow-step nlp-animate" style="animation-delay:0.12s;">
        <div class="nlp-step-number">Step 03</div>
        <div style="font-size:2.2rem; margin:0.3rem 0;">🧠</div>
        <div style="font-size:1.05rem; font-weight:700; color:#ffffff;">Practice & Master</div>
        <div style="font-size:0.82rem; color:#94a3b8; margin-top:0.35rem; line-height:1.5;">
            Test knowledge with note-grounded MCQs, True/False & 3D flashcards.
        </div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="nlp-workflow-step nlp-animate" style="animation-delay:0.18s;">
        <div class="nlp-step-number">Step 04</div>
        <div style="font-size:2.2rem; margin:0.3rem 0;">🎯</div>
        <div style="font-size:1.05rem; font-weight:700; color:#ffffff;">Export Revision</div>
        <div style="font-size:0.82rem; color:#94a3b8; margin-top:0.35rem; line-height:1.5;">
            Download multi-page ReportLab PDF, Markdown, or clean TXT sheets.
        </div>
    </div>""", unsafe_allow_html=True)

section_divider("Active Session Status")

# ── Live Session Status Metrics ───────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("1. Document Ingestion", "✅ Ready" if has_text() else "⏳ Awaiting Notes", help="Upload a document in Module 01")
m2.metric("2. Extractive Summary",  "✅ Ready" if has_summary() else "⏳ Pending", help="Generated in Module 02")
m3.metric("3. Keywords & Entities", "✅ Ready" if has_keywords() else "⏳ Pending", help="Generated in Module 03")
m4.metric("4. Practice Suite",      "✅ Ready" if (has_quiz() or has_tf() or has_flashcards()) else "⏳ Pending", help="Generated in Module 04")

section_divider("Feature Modules")

# ── Interactive Feature Modules Showcase ──────────────────────────────────────
st.markdown("### 🧰 Core NLP Intelligence Modules")

features = [
    ("📄", "Upload & Parse",    "Multi-format ingestion for PDF, Word, PowerPoint, and TXT with NLTK & spaCy preprocessing.", "01_upload", has_text()),
    ("📋", "Extractive Summary", "TF-IDF sentence scoring & Luhn word clustering with verbatim guarantee and ROUGE metrics.",  "02_summary", has_summary()),
    ("🔑", "Keywords & NER",     "Statistical TF-IDF unigrams/bigrams, RAKE keyphrases, and spaCy Named Entity classification.", "03_keywords", has_keywords()),
    ("🧠", "Quiz & Practice",    "Interactive study suite featuring note-grounded MCQs, True/False assertions, and 3D flashcards.", "04_quiz", (has_quiz() or has_tf() or has_flashcards())),
    ("📑", "Revision Sheet",     "Consolidated study guide generator with 2-pass ReportLab PDF, Markdown, and TXT exports.",   "05_revision", has_revision()),
    ("📊", "Linguistic Analysis", "Deep text analytics: POS tag distributions, Flesch readability dials, sentence histograms.", "06_nlp_analysis", True),
]

row1 = st.columns(3)
row2 = st.columns(3)
all_cols = row1 + row2

for col, (icon, title, desc, page_id, is_ready) in zip(all_cols, features):
    with col:
        badge_html = '<span class="nlp-badge nlp-badge-ready">✅ Ready</span>' if is_ready else '<span class="nlp-badge nlp-badge-pending">⏳ Available</span>'
        st.markdown(f"""
        <div class="nlp-card nlp-animate" style="height:100%; display:flex; flex-direction:column; justify-content:space-between;">
            <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                    <span style="font-size:2rem;">{icon}</span>
                    {badge_html}
                </div>
                <div style="font-size:1.1rem; font-weight:700; color:#ffffff; margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.84rem; color:#94a3b8; line-height:1.55; margin-bottom:1rem;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Open {title} ➔", key=f"btn_nav_{page_id}", use_container_width=True, type="secondary"):
            st.switch_page(f"pages/{page_id}.py")

section_divider("Algorithmic Engine")

# ── Technology & Algorithmic Foundation ───────────────────────────────────────
tech_cols = st.columns(6)
tech_stack = [
    ("🐍", "Python 3.13", "Core Engine"),
    ("🌊", "Streamlit 1.54+", "Reactive UI"),
    ("📖", "NLTK 3.9+", "Token & POS"),
    ("🔬", "spaCy 3.8+", "Lemmas & NER"),
    ("📐", "scikit-learn", "TF-IDF Vectorizer"),
    ("📄", "ReportLab 4.0+", "Dynamic PDF"),
]
for col, (icon, name, role) in zip(tech_cols, tech_stack):
    col.markdown(f"""
    <div style="text-align:center; padding:1rem 0.6rem;
                background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.06);
                border-radius:12px; backdrop-filter:blur(12px);">
        <div style="font-size:1.6rem;">{icon}</div>
        <div style="font-size:0.82rem; color:#f1f5f9; font-weight:700; margin-top:0.4rem;">{name}</div>
        <div style="font-size:0.72rem; color:#64748b; margin-top:0.15rem;">{role}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Primary CTA Banner ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
with cta_col2:
    if st.button("🚀 Upload Notes & Begin Study Session →", key="home_cta_primary", use_container_width=True):
        st.switch_page("pages/01_upload.py")
