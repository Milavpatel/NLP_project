"""
pages/02_summary.py — Summarization
Phase 3: Connected to Real Extractive Summarization Algorithms.
Preserves UI theme, styling, session-state keys, and responsive layout.
"""

import streamlit as st
import pandas as pd

from utils.session import (
    init_session,
    KEY_RAW_TEXT,
    KEY_SUMMARY,
    KEY_SUMMARY_METHOD,
    KEY_SUMMARY_RESULT,
)
from utils.ui_helpers import (
    load_css,
    render_sidebar,
    page_header,
    section_divider,
    empty_state,
)
from core.text_processor import split_sentences
from core.summarizer import (
    summarize_text,
    evaluate_rouge,
    SummaryResult,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Summary · NLP Summarizer",
    page_icon="📋",
    layout="wide",
)
init_session()
load_css()
render_sidebar(active_page="summary")

page_header(
    "📋",
    "Extractive Summarization",
    "Extract key sentences from your notes using TF-IDF or Normalized Frequency sentence scoring.",
)

raw = st.session_state.get(KEY_RAW_TEXT)

if not raw or not raw.strip():
    empty_state(
        icon="📋",
        title="No Document Loaded",
        description="Please upload or paste your study notes on the Upload page first, then return here to generate a summary.",
        cta_label="📄 Go to Upload →",
        cta_page="01_upload",
    )
else:
    # Pre-calculate sentences to guide length controls
    sentences = split_sentences(raw)
    total_sents = len(sentences)

    # ── Settings Panel ────────────────────────────────────────────────────────
    with st.expander("⚙️ Summarization Settings", expanded=True):
        col1, col2, col3 = st.columns([1.2, 1.2, 1.0])

        with col1:
            method_choice = st.selectbox(
                "Scoring Algorithm",
                options=["tfidf", "frequency"],
                index=0 if st.session_state.get(KEY_SUMMARY_METHOD, "tfidf") == "tfidf" else 1,
                format_func=lambda x: "TF-IDF Sentence Scoring" if x == "tfidf" else "Normalized Word Frequency",
                key="summary_algo_select",
                help="TF-IDF weights terms by uniqueness across sentences; Normalized Frequency scores by core content term density.",
            )

        with col2:
            length_mode = st.radio(
                "Length Control Mode",
                options=["ratio", "count"],
                format_func=lambda x: "Percentage / Ratio" if x == "ratio" else "Exact Sentence Count",
                horizontal=True,
                key="length_mode_radio",
            )

        with col3:
            if length_mode == "ratio":
                ratio_val = st.slider(
                    "Summary Ratio",
                    min_value=0.1,
                    max_value=0.8,
                    value=0.3,
                    step=0.05,
                    key="summary_ratio_slider",
                    help="Target percentage of original sentences to retain (e.g. 0.3 = 30%).",
                )
                sent_count_val = None
                target_count_preview = max(1, min(total_sents, int(round(total_sents * ratio_val))))
                st.caption(f"Estimated output: ~**{target_count_preview}** of **{total_sents}** sentences")
            else:
                sent_count_val = st.number_input(
                    "Sentence Count",
                    min_value=1,
                    max_value=max(1, total_sents),
                    value=min(3, max(1, total_sents)),
                    step=1,
                    key="summary_count_input",
                    help=f"Select between 1 and {total_sents} sentences.",
                )
                ratio_val = None
                st.caption(f"Will extract exactly **{sent_count_val}** sentences")

    # Short document notice
    if total_sents <= 2:
        st.info(
            f"ℹ️ **Short Document:** Your notes contain only **{total_sents}** sentence(s). "
            "Extractive summarization works best on longer notes with multiple paragraphs.",
            icon="ℹ️",
        )

    # ── Action Buttons ────────────────────────────────────────────────────────
    section_divider("Generate")

    gen_col1, gen_col2, _ = st.columns([1, 1, 3])
    with gen_col1:
        generate_btn = st.button("🚀 Generate Summary", key="gen_summary_btn", use_container_width=True)
    with gen_col2:
        clear_btn = st.button("🗑️ Clear Summary", key="clear_summary_btn", use_container_width=True)

    if clear_btn:
        st.session_state[KEY_SUMMARY] = None
        st.session_state[KEY_SUMMARY_RESULT] = None
        st.rerun()

    if generate_btn:
        with st.spinner(f"Computing sentence scores using {method_choice.upper()} …"):
            try:
                res: SummaryResult = summarize_text(
                    text=raw,
                    method=method_choice,
                    ratio=ratio_val,
                    num_sentences=sent_count_val,
                )
                st.session_state[KEY_SUMMARY] = res.summary_text
                st.session_state[KEY_SUMMARY_RESULT] = res
                st.session_state[KEY_SUMMARY_METHOD] = res.method
                st.success("✅ Summary generated successfully!")
            except Exception as e:
                st.error(f"❌ Summarization failed: {e}")

    # ── Summary Results Display ───────────────────────────────────────────────
    summary_text = st.session_state.get(KEY_SUMMARY)
    summary_res: SummaryResult = st.session_state.get(KEY_SUMMARY_RESULT)

    if summary_text and summary_res:
        section_divider("Generated Summary")

        meth_title = "TF-IDF Extractive" if summary_res.method == "tfidf" else "Normalized Word Frequency"
        st.markdown(
            f'<span class="nlp-badge nlp-badge-ready">Algorithm: {meth_title}</span>&nbsp;'
            f'<span class="nlp-badge nlp-badge-info">{summary_res.num_sentences_selected} of {summary_res.total_sentences} Sentences</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for Summary Text and Side-by-Side Comparison
        view_tab1, view_tab2 = st.tabs(["📋 Summary View", "⚖️ Original vs. Summary Comparison"])

        with view_tab1:
            st.markdown(
                f'<div class="nlp-card nlp-animate" style="font-size:0.95rem; line-height:1.8; color:#e6edf3;">'
                f'{summary_text}</div>',
                unsafe_allow_html=True,
            )

        with view_tab2:
            comp_c1, comp_c2 = st.columns(2)
            with comp_c1:
                st.markdown("**Original Notes**")
                st.markdown(
                    f'<div class="nlp-card" style="font-size:0.85rem; max-height:350px; overflow-y:auto; line-height:1.6; color:#8b949e;">'
                    f'{raw}</div>',
                    unsafe_allow_html=True,
                )
            with comp_c2:
                st.markdown("**Generated Extractive Summary**")
                st.markdown(
                    f'<div class="nlp-card" style="font-size:0.85rem; max-height:350px; overflow-y:auto; line-height:1.6; color:#e6edf3;">'
                    f'{summary_text}</div>',
                    unsafe_allow_html=True,
                )

        # Metrics Row
        st.markdown("<br>", unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Original Words", f"{summary_res.original_word_count:,}")
        mc2.metric("Summary Words", f"{summary_res.summary_word_count:,}")
        mc3.metric("Compression", f"{summary_res.compression_ratio}%")
        mc4.metric(
            "Sentence Retention",
            f"{summary_res.num_sentences_selected} / {summary_res.total_sentences}",
        )

        # ── Sentence Ranking Breakdown (Explainability) ─────────────────────────
        section_divider("Sentence Ranking & Score Breakdown (Viva Explainability)")

        with st.expander("🔍 View All Sentences with Calculated Scores & Ranks", expanded=False):
            st.caption(
                "Every sentence is independently scored by the selected algorithm. "
                "Sentences marked with ✅ **Selected** were included in the summary and "
                "re-ordered to preserve the original narrative sequence."
            )

            table_rows = []
            for item in summary_res.sentence_scores:
                table_rows.append({
                    "Rank": item.rank,
                    "Selected": "✅ Yes" if item.selected else "❌ No",
                    "Score": item.score,
                    "Sentence #": item.index + 1,
                    "Sentence Text": item.sentence,
                })

            df_scores = pd.DataFrame(table_rows)
            # Sort by rank for easy analysis
            df_scores = df_scores.sort_values(by="Rank")
            st.dataframe(
                df_scores,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Selected": st.column_config.TextColumn("In Summary", width="small"),
                    "Score": st.column_config.NumberColumn("Score", format="%.4f", width="small"),
                    "Sentence #": st.column_config.NumberColumn("Orig #", width="small"),
                    "Sentence Text": st.column_config.TextColumn("Sentence Text", width="large"),
                },
            )

        # ── Quality Evaluation (ROUGE Score) ───────────────────────────────────
        section_divider("Quality Evaluation (ROUGE Overlap)")
        with st.expander("📊 View ROUGE Overlap vs Original Document", expanded=False):
            st.caption(
                "ROUGE evaluates n-gram recall and longest common subsequence (LCS) overlap "
                "between the generated summary and the original source text. "
                "**Note on Methodology:** Comparing against the source document measures content "
                "retention / source coverage. In academic benchmarking, ROUGE evaluates against "
                "human-authored reference summaries."
            )
            rouge_scores = evaluate_rouge(raw, summary_text)
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("ROUGE-1 F1 (Unigram Overlap)", f"{rouge_scores['rouge1']:.4f}")
            rc2.metric("ROUGE-2 F1 (Bigram Overlap)", f"{rouge_scores['rouge2']:.4f}")
            rc3.metric("ROUGE-L F1 (Longest Common Subseq)", f"{rouge_scores['rougeL']:.4f}")

        # ── Export ────────────────────────────────────────────────────────────
        section_divider("Export")
        st.download_button(
            label="⬇️ Download Summary (.txt)",
            data=summary_text,
            file_name="summary.txt",
            mime="text/plain",
            key="dl_summary_btn",
        )

    else:
        st.markdown(
            """
            <div class="nlp-empty nlp-animate">
                <div class="icon">✨</div>
                <h3>Summary Not Yet Generated</h3>
                <p>Select your preferred algorithm and length above, then click
                   <strong>🚀 Generate Summary</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Educational Viva Section ──────────────────────────────────────────────
    section_divider("How These Extractive Algorithms Work")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="nlp-card">
                <div style="font-weight:600; color:#9f67ff; font-size:0.95rem;">📐 TF-IDF Sentence Scoring</div>
                <div style="font-size:0.8rem; color:#8b949e; margin-top:0.4rem; line-height:1.6;">
                    Treats each sentence as a mini-document. Computes term frequency and penalizes
                    words that appear uniformly across all sentences. Sentences containing distinctive,
                    high-weight topical terms receive top rank.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="nlp-card">
                <div style="font-weight:600; color:#9f67ff; font-size:0.95rem;">📊 Normalized Word Frequency (Luhn)</div>
                <div style="font-size:0.8rem; color:#8b949e; margin-top:0.4rem; line-height:1.6;">
                    Counts all content words across the document, normalizes by the maximum term frequency,
                    and scores each sentence by the average normalized weight of its words.
                    Penalizes verbose empty sentences while rewarding content-dense statements.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
