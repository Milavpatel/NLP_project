"""
pages/03_keywords.py — Keyword & Keyphrase Extraction
Phase 3: Connected to Real NLP Algorithms (TF-IDF, Frequency, RAKE, and Separate spaCy NER).
Preserves UI theme, styling, session-state keys, and responsive layout.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.session import (
    init_session,
    KEY_RAW_TEXT,
    KEY_KEYWORDS,
    KEY_KEYWORD_METHOD,
    KEY_ENTITIES,
)
from utils.ui_helpers import (
    load_css,
    render_sidebar,
    page_header,
    section_divider,
    empty_state,
    chip_row,
)
from core.keyword_extractor import (
    tfidf_keywords,
    frequency_keywords,
    rake_keywords,
    spacy_entities,
)
from core.text_processor import spacy_status

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Keywords · NLP Summarizer",
    page_icon="🔑",
    layout="wide",
)
init_session()
load_css()
render_sidebar(active_page="keywords")

page_header(
    "🔑",
    "Keyword & Keyphrase Extraction",
    "Discover the most salient terms, keyphrases, and named entities using TF-IDF, Normalized Frequency, RAKE, and spaCy.",
)

raw = st.session_state.get(KEY_RAW_TEXT)

if not raw or not raw.strip():
    empty_state(
        icon="🔑",
        title="No Document Loaded",
        description="Please upload or paste your study notes on the Upload page first, then return here to extract keywords.",
        cta_label="📄 Go to Upload →",
        cta_page="01_upload",
    )
else:
    # ── Settings Panel ────────────────────────────────────────────────────────
    with st.expander("⚙️ Extraction Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox(
                "Keyword Extraction Algorithm",
                options=["tfidf", "frequency", "rake"],
                format_func=lambda x: {
                    "tfidf": "TF-IDF (Salient words & bigrams)",
                    "frequency": "Normalized Word Frequency (Content density)",
                    "rake": "RAKE (Rapid Automatic Multi-word Keyphrases)",
                }[x],
                index=0 if st.session_state.get(KEY_KEYWORD_METHOD, "tfidf") == "tfidf" else (1 if st.session_state.get(KEY_KEYWORD_METHOD) == "frequency" else 2),
                key="kw_method_select",
                help="TF-IDF weights terms against sentence mini-corpus; Frequency tracks content repetition; RAKE identifies multi-word candidate phrases.",
            )
        with col2:
            n_keywords = st.slider(
                "Number of Keywords to Extract",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                key="kw_n_slider",
            )

    gen_col1, gen_col2, _ = st.columns([1, 1, 3])
    with gen_col1:
        extract_btn = st.button("🔍 Extract Keywords", key="extract_kw_btn", use_container_width=True)
    with gen_col2:
        clear_btn = st.button("🗑️ Clear", key="clear_kw_btn", use_container_width=True)

    if clear_btn:
        st.session_state[KEY_KEYWORDS] = []
        st.session_state[KEY_ENTITIES] = []
        st.rerun()

    if extract_btn:
        with st.spinner(f"Extracting keywords using {method.upper()} …"):
            try:
                if method == "tfidf":
                    kws = tfidf_keywords(raw, n=n_keywords)
                elif method == "frequency":
                    kws = frequency_keywords(raw, n=n_keywords)
                else:
                    kws = rake_keywords(raw, n=n_keywords)

                st.session_state[KEY_KEYWORDS] = kws
                st.session_state[KEY_KEYWORD_METHOD] = method

                # Also pre-extract entities for the separate NER view
                ents = spacy_entities(raw)
                st.session_state[KEY_ENTITIES] = ents

                st.success(f"✅ Extracted {len(kws)} terms!")
            except Exception as e:
                st.error(f"❌ Keyword extraction error: {e}")

    # ── Results Presentation ──────────────────────────────────────────────────
    keywords = st.session_state.get(KEY_KEYWORDS, [])
    entities = st.session_state.get(KEY_ENTITIES, [])
    method_used = st.session_state.get(KEY_KEYWORD_METHOD, "tfidf")

    meth_label = {
        "tfidf": "TF-IDF",
        "frequency": "Normalized Frequency",
        "rake": "RAKE Keyphrases",
    }.get(method_used, "TF-IDF")

    if keywords:
        section_divider("Extracted Keywords & Visual Analytics")

        st.markdown(
            f'<span class="nlp-badge nlp-badge-ready">Algorithm: {meth_label}</span>&nbsp;'
            f'<span class="nlp-badge nlp-badge-info">{len(keywords)} Keywords</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Keyword Chips
        st.markdown("##### 🏷️ Keyword Chips")
        chip_row([kw for kw, _ in keywords])
        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for Visual Chart and Structured Table
        tab_chart, tab_table, tab_ner = st.tabs([
            "📊 Keyword Score Chart",
            "📋 Ranked Score Table",
            "🏢 Named Entity Recognition (NER)",
        ])

        with tab_chart:
            # Prepare dataframe for Plotly
            df_chart = pd.DataFrame(keywords, columns=["Keyword", "Score"])
            # Reverse for top-to-bottom bar chart order
            df_chart_sorted = df_chart.iloc[::-1]

            fig = px.bar(
                df_chart_sorted,
                x="Score",
                y="Keyword",
                orientation="h",
                title=f"Top Keywords Ranked by {meth_label} Score",
                color="Score",
                color_continuous_scale=["#6366f1", "#a855f7", "#06b6d4"],
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.5)",
                font_color="#f1f5f9",
                font_family="Plus Jakarta Sans, Inter, sans-serif",
                title_font_size=14,
                coloraxis_showscale=False,
                margin=dict(l=10, r=20, t=40, b=10),
                height=max(360, len(keywords) * 26),
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", range=[0, 1.05])
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab_table:
            df_table = pd.DataFrame(keywords, columns=["Keyword", "Score"])
            df_table["Rank"] = range(1, len(df_table) + 1)
            df_table["Score"] = df_table["Score"].round(4)
            df_table = df_table[["Rank", "Keyword", "Score"]]

            st.dataframe(
                df_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Keyword": st.column_config.TextColumn("Extracted Term / Phrase", width="medium"),
                    "Score": st.column_config.ProgressColumn(
                        "Importance Score",
                        format="%.4f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                },
            )

            csv_data = df_table.to_csv(index=False)
            st.download_button(
                "⬇️ Download Keywords (.csv)",
                data=csv_data,
                file_name="keywords.csv",
                mime="text/csv",
                key="dl_kw_btn",
            )

        with tab_ner:
            st.caption(
                "Named Entity Recognition identifies real-world entities (people, organizations, "
                "locations, dates) using spaCy. These are kept strictly distinct from general topic keywords."
            )
            spacy_ok, spacy_msg = spacy_status()

            if not spacy_ok:
                st.warning(f"⚠️ {spacy_msg}")
            elif not entities:
                st.info("No named entities detected in the current text.", icon="ℹ️")
            else:
                st.markdown(f"**Found {len(entities)} Named Entities:**")
                ent_df = pd.DataFrame(entities, columns=["Entity Text", "Entity Type"])
                ent_df.index = ent_df.index + 1

                # Group by entity type for clean view
                col_e1, col_e2 = st.columns([1.5, 1])
                with col_e1:
                    st.dataframe(ent_df, use_container_width=True, height=280)
                with col_e2:
                    type_counts = ent_df["Entity Type"].value_counts().reset_index()
                    type_counts.columns = ["Entity Type", "Count"]
                    fig_ent = px.pie(
                        type_counts,
                        names="Entity Type",
                        values="Count",
                        title="Entity Distribution",
                        template="plotly_dark",
                        color_discrete_sequence=["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#f43f5e"],
                    )
                    fig_ent.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,23,42,0.5)",
                        font_color="#f1f5f9",
                        font_family="Plus Jakarta Sans, Inter, sans-serif",
                        height=290,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_ent, use_container_width=True)

    else:
        st.markdown(
            """
            <div class="nlp-empty nlp-animate">
                <div class="icon">🏷️</div>
                <h3>No Keywords Extracted Yet</h3>
                <p>Select an extraction method and the number of keywords above, then click
                   <strong>🔍 Extract Keywords</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Educational Method Comparison ─────────────────────────────────────────
    section_divider("How These Keyword Algorithms Work")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(
            """
            <div class="nlp-card">
                <div style="font-weight:600; color:#9f67ff; font-size:0.95rem;">📐 TF-IDF</div>
                <div style="font-size:0.78rem; color:#8b949e; margin-top:0.4rem; line-height:1.6;">
                    Term Frequency–Inverse Document Frequency. Evaluates unigrams and bigrams by
                    penalizing terms ubiquitous across all sentences while highlighting terms concentrated
                    in key conceptual sections.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            """
            <div class="nlp-card">
                <div style="font-weight:600; color:#9f67ff; font-size:0.95rem;">📊 Normalized Frequency</div>
                <div style="font-size:0.78rem; color:#8b949e; margin-top:0.4rem; line-height:1.6;">
                    Calculates occurrence counts of non-stopword tokens and normalizes by the maximum
                    term frequency. Reveals the primary subject matter of study notes directly.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            """
            <div class="nlp-card">
                <div style="font-weight:600; color:#9f67ff; font-size:0.95rem;">🌿 RAKE Keyphrases</div>
                <div style="font-size:0.78rem; color:#8b949e; margin-top:0.4rem; line-height:1.6;">
                    Rapid Automatic Keyword Extraction. Identifies multi-word phrases by splitting on
                    stopwords and punctuation, then scoring phrases via word degree and co-occurrence graphs.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
