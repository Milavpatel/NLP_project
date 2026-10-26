"""
pages/06_nlp_analysis.py — NLP Analysis Dashboard
Phase 6: Quality-Assurance — Full Linguistic Analytics Suite.
Preserves UI theme, styling, session-state keys, and responsive layout.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.session import (
    init_session,
    KEY_RAW_TEXT,
    KEY_PROCESSED_TEXT,
    KEY_SUMMARY_RESULT,
    KEY_KEYWORDS,
    KEY_ENTITIES,
)
from utils.ui_helpers import (
    load_css,
    render_sidebar,
    page_header,
    section_divider,
    empty_state,
)
from core.text_processor import (
    get_basic_stats,
    split_sentences,
    tokenise_words,
    spacy_status,
)
from core.keyword_extractor import spacy_entities
from core.nlp_analyzer import (
    pos_tag_distribution,
    word_frequency,
    generate_wordcloud,
    sentence_length_distribution,
    compute_readability,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Analysis · NLP Summarizer",
    page_icon="📈",
    layout="wide",
)
init_session()
load_css()
render_sidebar(active_page="nlp_analysis")

page_header(
    "📈",
    "NLP Linguistic Analysis",
    "Deep-dive into linguistic patterns, POS tag distributions, term frequencies, readability, and entities.",
)

raw = st.session_state.get(KEY_RAW_TEXT)

if not raw or not raw.strip():
    empty_state(
        icon="📊",
        title="No Document Loaded",
        description="Please upload or paste your study notes on the Upload page first to see a full linguistic analysis here.",
        cta_label="📄 Go to Upload →",
        cta_page="01_upload",
    )
else:
    sentences = split_sentences(raw)
    stats = get_basic_stats(raw)
    readability = compute_readability(raw)

    # ── Section 1: Document Overview & Readability ────────────────────────────
    section_divider("1. Document Metrics & Readability")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Words", f"{stats.get('words', 0):,}")
    m2.metric("Sentences", f"{len(sentences):,}")
    m3.metric("Vocabulary", f"{stats.get('vocab_size', 0):,}")
    m4.metric("Reading Time", f"{stats.get('reading_time', 1)} min")
    m5.metric("Reading Ease", f"{readability['reading_ease']} / 100")
    m6.metric("Grade Level", f"Grade {readability['grade_level']}")

    st.caption(
        f"📖 **Readability Assessment (Flesch-Kincaid):** *{readability['interpretation']}* "
        f"(Avg sentence length: **{readability['avg_sentence_len']}** words | Avg syllables/word: **{readability['avg_syllables_per_word']}**)"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Sentence Length Distribution ───────────────────────────────
    section_divider("2. Sentence Length Distribution")

    sent_lengths = sentence_length_distribution(sentences)
    if sent_lengths:
        df_sent = pd.DataFrame({
            "Sentence #": [f"S{i+1}" for i in range(len(sent_lengths))],
            "Word Count": sent_lengths,
        })

        fig_sent = px.bar(
            df_sent,
            x="Sentence #",
            y="Word Count",
            title="Word Count per Sentence Across Notes",
            color="Word Count",
            color_continuous_scale=["#6366f1", "#a855f7", "#06b6d4"],
            template="plotly_dark",
        )
        fig_sent.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font_color="#f1f5f9",
            font_family="Plus Jakarta Sans, Inter, sans-serif",
            title_font_size=13,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=35, b=10),
            height=260,
        )
        fig_sent.update_xaxes(showgrid=False)
        fig_sent.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig_sent, use_container_width=True)

    # ── Section 3: In-Depth Linguistic Tabs ────────────────────────────────────
    section_divider("3. In-Depth Linguistic Layers")

    tab_pos, tab_freq, tab_wc, tab_ner, tab_pipe = st.tabs([
        "🏷️ POS Tags",
        "🔤 Word Frequency",
        "☁️ Word Cloud",
        "🏢 Named Entities",
        "⚡ Preprocessing Pipeline Comparison",
    ])

    # ── Tab 1: POS Tagging ───────────────────────────────────────────────────
    with tab_pos:
        st.markdown("##### Part-of-Speech Category Distribution (NLTK Averaged Perceptron)")
        pos_counts = pos_tag_distribution(raw)

        if pos_counts and sum(pos_counts.values()) > 0:
            df_pos = pd.DataFrame({
                "Category": list(pos_counts.keys()),
                "Count": list(pos_counts.values()),
            })

            fig_pos = px.bar(
                df_pos,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_sequence=["#6366f1", "#818cf8", "#06b6d4", "#10b981", "#f59e0b", "#f43f5e", "#64748b"],
                template="plotly_dark",
            )
            fig_pos.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.5)",
                font_color="#f1f5f9",
                font_family="Plus Jakarta Sans, Inter, sans-serif",
                showlegend=False,
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig_pos.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_pos, use_container_width=True)

            total_pos = sum(pos_counts.values())
            noun_pct = round((pos_counts.get("Nouns", 0) / total_pos) * 100, 1) if total_pos > 0 else 0
            verb_pct = round((pos_counts.get("Verbs", 0) / total_pos) * 100, 1) if total_pos > 0 else 0
            st.caption(f"Linguistic Takeaway: Document is **{noun_pct}%** Nouns and **{verb_pct}%** Verbs. Informational density is high.")
        else:
            st.info("POS tagger could not find enough alphabetic tokens.")

    # ── Tab 2: Word Frequency ─────────────────────────────────────────────────
    with tab_freq:
        st.markdown("##### Top 15 Most Frequent Content Words (Stopwords Filtered)")
        wf = word_frequency(raw, n=15)

        if wf:
            df_wf = pd.DataFrame(wf, columns=["Term", "Occurrences"])
            # Invert for horizontal chart
            df_wf_sorted = df_wf.iloc[::-1]

            fig_wf = px.bar(
                df_wf_sorted,
                x="Occurrences",
                y="Term",
                orientation="h",
                color="Occurrences",
                color_continuous_scale=["#6366f1", "#a855f7", "#06b6d4"],
                template="plotly_dark",
            )
            fig_wf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.5)",
                font_color="#f1f5f9",
                font_family="Plus Jakarta Sans, Inter, sans-serif",
                coloraxis_showscale=False,
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig_wf.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_wf, use_container_width=True)
        else:
            st.info("Not enough content words to build frequency distribution.")

    # ── Tab 3: Word Cloud ─────────────────────────────────────────────────────
    with tab_wc:
        st.markdown("##### Visual Salience Word Cloud")
        wc = generate_wordcloud(raw)

        if wc is not None:
            import io
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5), facecolor="#080c15")
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Could not generate WordCloud from current text (empty or all stopwords).")

    # ── Tab 4: Named Entities ─────────────────────────────────────────────────
    with tab_ner:
        st.markdown("##### Named Entity Recognition (spaCy en_core_web_sm)")
        spacy_ok, spacy_msg = spacy_status()

        if not spacy_ok:
            st.warning(f"⚠️ {spacy_msg}")
        else:
            ents = st.session_state.get(KEY_ENTITIES)
            if not ents:
                ents = spacy_entities(raw)
                st.session_state[KEY_ENTITIES] = ents

            if ents:
                df_ent = pd.DataFrame(ents, columns=["Entity", "Type"])
                df_ent.index = df_ent.index + 1

                ecol1, ecol2 = st.columns([1.5, 1])
                with ecol1:
                    st.dataframe(df_ent, use_container_width=True, height=280)
                with ecol2:
                    ent_counts = df_ent["Type"].value_counts().reset_index()
                    ent_counts.columns = ["Type", "Count"]
                    fig_e = px.pie(
                        ent_counts,
                        names="Type",
                        values="Count",
                        title="Entity Type Proportions",
                        template="plotly_dark",
                        color_discrete_sequence=["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#f43f5e"],
                    )
                    fig_e.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,23,42,0.5)",
                        font_color="#f1f5f9",
                        font_family="Plus Jakarta Sans, Inter, sans-serif",
                        height=290,
                        margin=dict(l=10, r=10, t=30, b=10),
                    )
                    st.plotly_chart(fig_e, use_container_width=True)
            else:
                st.info("No named entities detected in this document.")

    # ── Tab 5: Preprocessing Pipeline Comparison ──────────────────────────────
    with tab_pipe:
        st.markdown("##### Raw Text vs. Preprocessed Tokens")

        raw_tokens = tokenise_words(raw, remove_stopwords=False, remove_punct=False)
        clean_tokens = tokenise_words(raw, remove_stopwords=True, remove_punct=True)

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Raw Word Tokens", f"{len(raw_tokens):,}")
        pc2.metric("Filtered Content Tokens", f"{len(clean_tokens):,}")
        pct_reduced = round((1 - len(clean_tokens) / max(len(raw_tokens), 1)) * 100, 1)
        pc3.metric("Noise & Stopwords Removed", f"{pct_reduced}%")

        st.markdown("<br>", unsafe_allow_html=True)
        col_raw, col_proc = st.columns(2)
        with col_raw:
            st.markdown("**Original Raw Text Snippet**")
            st.text_area("raw_snippet", raw[:600] + (" …" if len(raw) > 600 else ""), height=180, disabled=True, label_visibility="collapsed")
        with col_proc:
            st.markdown("**Normalized & Filtered Tokens Snippet**")
            st.text_area("proc_snippet", " ".join(clean_tokens[:100]) + (" …" if len(clean_tokens) > 100 else ""), height=180, disabled=True, label_visibility="collapsed")
