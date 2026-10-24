"""
pages/04_quiz.py — Quiz Generator
Phase 4: Connected to Note-Grounded MCQ Quiz Generation.
Preserves UI theme, styling, session-state keys, and responsive layout.
"""

import streamlit as st

from utils.session import (
    init_session,
    KEY_RAW_TEXT,
    KEY_SENTENCES,
    KEY_KEYWORDS,
    KEY_QUIZ_ITEMS,
    KEY_QUIZ_ANSWERS,
    KEY_QUIZ_SCORE,
)
from utils.ui_helpers import (
    load_css,
    render_sidebar,
    page_header,
    section_divider,
    empty_state,
)
from core.quiz_generator import (
    generate_mcq,
    score_quiz,
    evaluate_quiz_detailed,
)
from core.text_processor import split_sentences

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quiz · NLP Summarizer",
    page_icon="🧠",
    layout="wide",
)
init_session()
load_css()
render_sidebar(active_page="quiz")

page_header(
    "🧠",
    "Quiz Generator",
    "Test your understanding with 100% note-grounded multiple choice questions generated directly from your notes.",
)

raw = st.session_state.get(KEY_RAW_TEXT)

if not raw or not raw.strip():
    empty_state(
        icon="🧠",
        title="No Document Loaded",
        description="Please upload or paste your study notes on the Upload page first, then return here to generate a quiz.",
        cta_label="📄 Go to Upload →",
        cta_page="01_upload",
    )
else:
    # ── Settings Panel ────────────────────────────────────────────────────────
    with st.expander("⚙️ Quiz Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            quiz_type = st.selectbox(
                "Question Format",
                ["Mixed (Fill-in-the-blank & Definition)", "Fill-in-the-blank Only", "Definition-based Only"],
                index=0,
                key="quiz_type_select",
                help="Fill-in-the-blank tests key concepts in source statements; Definition tests conceptual recognition.",
            )
        with col2:
            n_questions = st.slider(
                "Number of Questions",
                min_value=2,
                max_value=10,
                value=5,
                step=1,
                key="quiz_n_slider",
                help="Desired number of multiple-choice questions.",
            )

    gen_col1, gen_col2, _ = st.columns([1, 1, 3])
    with gen_col1:
        gen_btn = st.button("🚀 Generate Quiz", key="gen_quiz_btn", use_container_width=True)
    with gen_col2:
        clear_btn = st.button("🗑️ Clear Quiz", key="clear_quiz_btn", use_container_width=True)

    if clear_btn:
        st.session_state[KEY_QUIZ_ITEMS] = []
        st.session_state[KEY_QUIZ_ANSWERS] = {}
        st.session_state[KEY_QUIZ_SCORE] = None
        st.rerun()

    if gen_btn:
        # Determine internal question type mapping
        q_type_map = {
            "Mixed (Fill-in-the-blank & Definition)": "mixed",
            "Fill-in-the-blank Only": "fill_in_the_blank",
            "Definition-based Only": "definition",
        }
        chosen_type = q_type_map.get(quiz_type, "mixed")

        with st.spinner("Extracting concepts and generating grounded questions …"):
            sentences = st.session_state.get(KEY_SENTENCES)
            if not sentences:
                sentences = split_sentences(raw)
                st.session_state[KEY_SENTENCES] = sentences

            keywords = st.session_state.get(KEY_KEYWORDS, [])

            items = generate_mcq(
                sentences=sentences,
                keywords=keywords if keywords else None,
                n=n_questions,
                raw_text=raw,
                question_type=chosen_type,
            )

            st.session_state[KEY_QUIZ_ITEMS] = items
            st.session_state[KEY_QUIZ_ANSWERS] = {}
            st.session_state[KEY_QUIZ_SCORE] = None

            if not items:
                st.warning(
                    "⚠️ Could not generate enough grounded questions with distinct distractors from this text. "
                    "Try uploading longer notes with more vocabulary and definitions.",
                    icon="⚠️",
                )
            elif len(items) < n_questions:
                st.info(
                    f"Generated **{len(items)}** quality questions (fewer than requested to ensure "
                    "every question has 3 distinct, non-trivial distractors without fabricating facts).",
                    icon="ℹ️",
                )
            else:
                st.success(f"✅ Generated {len(items)} multiple choice questions!")

    # ── Quiz Display & Interaction ────────────────────────────────────────────
    items = st.session_state.get(KEY_QUIZ_ITEMS, [])
    user_answers = st.session_state.get(KEY_QUIZ_ANSWERS, {})
    quiz_score = st.session_state.get(KEY_QUIZ_SCORE)
    is_submitted = quiz_score is not None

    if items:
        section_divider("Quiz Questions")

        # Before submission or for active attempt
        with st.form("quiz_interactive_form"):
            temp_answers = {}
            for i, item in enumerate(items):
                st.markdown(
                    f"""
                    <div class="nlp-card nlp-animate" style="margin-bottom:0.8rem; border-left:3px solid #7c3aed;">
                        <div style="font-size:0.75rem; color:#9f67ff; font-weight:600; margin-bottom:0.25rem;">
                            QUESTION {i+1} OF {len(items)} · {item.get('question_type', 'mcq').replace('_', ' ').upper()}
                        </div>
                        <div style="font-weight:600; color:#e6edf3; font-size:0.95rem; line-height:1.6;">
                            {item['question']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Radio options: note index=None so no option is pre-selected before user clicks!
                current_val = user_answers.get(i)
                current_idx = item["options"].index(current_val) if (current_val and current_val in item["options"]) else None

                temp_answers[i] = st.radio(
                    f"Select answer for Q{i+1}:",
                    options=item["options"],
                    index=current_idx,
                    key=f"quiz_opt_{i}",
                    label_visibility="collapsed",
                    disabled=is_submitted,
                )

            submit_col1, submit_col2 = st.columns([1, 4])
            with submit_col1:
                submitted = st.form_submit_button(
                    "✅ Submit Quiz",
                    use_container_width=True,
                    disabled=is_submitted,
                )

        if submitted:
            correct, total = score_quiz(items, temp_answers)
            st.session_state[KEY_QUIZ_SCORE] = correct
            st.session_state[KEY_QUIZ_ANSWERS] = temp_answers
            st.rerun()

        # ── Post-Submission Results & Detailed Feedback ───────────────────────
        if is_submitted:
            section_divider("Quiz Results & Detailed Feedback")

            correct_count = st.session_state[KEY_QUIZ_SCORE]
            total_count = len(items)
            pct = round((correct_count / max(total_count, 1)) * 100)

            # Score Banner
            score_col1, score_col2, score_col3 = st.columns(3)
            score_col1.metric("Score", f"{correct_count} / {total_count}")
            score_col2.metric("Accuracy", f"{pct}%")
            score_col3.metric("Status", "Completed 🎉" if pct >= 70 else "Needs Review 📚")

            if pct >= 80:
                st.success(f"🎉 **Outstanding!** You scored **{correct_count}/{total_count}** ({pct}%). You have mastered these notes!")
            elif pct >= 50:
                st.warning(f"📚 **Good effort!** You scored **{correct_count}/{total_count}** ({pct}%). Review the highlighted source sentences below.")
            else:
                st.error(f"🔄 **Keep practicing!** You scored **{correct_count}/{total_count}** ({pct}%). Check the explanations and re-read your notes.")

            # Detailed review for each question
            detailed = evaluate_quiz_detailed(items, user_answers)

            st.markdown("#### Question-by-Question Review")
            for res in detailed:
                q_num = res["id"]
                is_correct = res["is_correct"]
                is_ans = res["is_answered"]

                card_border = "#10b981" if is_correct else "#f43f5e"
                badge_html = (
                    '<span class="nlp-badge nlp-badge-ready">✅ Correct</span>'
                    if is_correct
                    else ('<span class="nlp-badge nlp-badge-error">❌ Incorrect</span>' if is_ans else '<span class="nlp-badge nlp-badge-warning">⚠️ Unanswered</span>')
                )

                st.markdown(
                    f"""
                    <div class="nlp-card" style="border-left:4px solid {card_border}; margin-bottom:1rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                            <span style="font-weight:600; color:#e6edf3;">Question {q_num}</span>
                            {badge_html}
                        </div>
                        <div style="color:#e6edf3; font-size:0.92rem; margin-bottom:0.6rem; line-height:1.6;">
                            {res['question']}
                        </div>
                        <div style="font-size:0.85rem; line-height:1.6;">
                            <span style="color:#8b949e;">Your Choice:</span> <strong style="color:{'#10b981' if is_correct else '#f43f5e'};">{res['user_answer'] if is_ans else 'None selected'}</strong><br>
                            <span style="color:#8b949e;">Correct Answer:</span> <strong style="color:#10b981;">{res['correct_answer']}</strong>
                        </div>
                        <div style="margin-top:0.75rem; padding-top:0.6rem; border-top:1px solid #30363d; font-size:0.82rem; color:#8b949e;">
                            📖 <strong style="color:#9f67ff;">Source from Notes:</strong> <em>"{res['source_sentence']}"</em>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Option to re-take the quiz
            if st.button("🔄 Retake This Quiz", key="retake_quiz_btn"):
                st.session_state[KEY_QUIZ_SCORE] = None
                st.session_state[KEY_QUIZ_ANSWERS] = {}
                st.rerun()

    else:
        # ── Empty state when no quiz generated yet ────────────────────────────
        st.markdown(
            """
            <div class="nlp-empty nlp-animate">
                <div class="icon">❓</div>
                <h3>Quiz Not Yet Generated</h3>
                <p>Select your question format and desired count above, then click
                   <strong>🚀 Generate Quiz</strong> to build questions from your notes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        section_divider("Quiz Generation Architecture")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                <div class="nlp-card">
                    <div style="font-weight:600; color:#9f67ff; font-size:0.95rem;">✏️ Fill-in-the-Blank Questions</div>
                    <div style="font-size:0.8rem; color:#8b949e; margin-top:0.4rem; line-height:1.6;">
                        Extracts key conceptual statements and blanks out the central keyword.
                        Distractors are selected from other real concepts in the text that share similar
                        linguistic structure but are demonstrably different.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="nlp-card">
                    <div style="font-weight:600; color:#9f67ff; font-size:0.95rem;">📖 Definition-Based Questions</div>
                    <div style="font-size:0.8rem; color:#8b949e; margin-top:0.4rem; line-height:1.6;">
                        Identifies definition patterns (<em>"X is the process of..."</em>, <em>"X refers to..."</em>)
                        and asks students to identify the correct definiendum among candidate terms
                        from their notes.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
