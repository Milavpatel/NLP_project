"""
pages/04_quiz.py — Quiz & Study Practice Suite
Phase 4 & 5 Extension: Note-Grounded MCQ, True/False Practice, and Interactive Flashcards.
Preserves UI theme, styling, session-state keys, and responsive layout.
"""

import random
import streamlit as st

from utils.session import (
    init_session,
    KEY_RAW_TEXT,
    KEY_SENTENCES,
    KEY_KEYWORDS,
    KEY_QUIZ_ITEMS,
    KEY_QUIZ_ANSWERS,
    KEY_QUIZ_SCORE,
    KEY_TF_ITEMS,
    KEY_TF_ANSWERS,
    KEY_TF_SCORE,
    KEY_FLASHCARDS,
    KEY_FLASHCARD_INDEX,
    KEY_FLASHCARD_MASTERED,
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
    generate_true_false,
    score_true_false,
    evaluate_true_false_detailed,
    generate_flashcards,
)
from core.text_processor import split_sentences

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quiz & Practice · NLP Summarizer",
    page_icon="🧠",
    layout="wide",
)
init_session()
load_css()
render_sidebar(active_page="quiz")

page_header(
    "🧠",
    "Quiz & Practice Suite",
    "Reinforce your knowledge with 100% note-grounded Multiple Choice questions, True/False assertions, and Interactive Flashcards.",
)

raw = st.session_state.get(KEY_RAW_TEXT)

if not raw or not raw.strip():
    empty_state(
        icon="🧠",
        title="No Document Loaded",
        description="Please upload or paste your study notes on the Upload page first, then return here to generate quizzes and flashcards.",
        cta_label="📄 Go to Upload →",
        cta_page="01_upload",
    )
else:
    sentences = st.session_state.get(KEY_SENTENCES)
    if not sentences:
        sentences = split_sentences(raw)
        st.session_state[KEY_SENTENCES] = sentences

    keywords = st.session_state.get(KEY_KEYWORDS, [])

    # ── Master Study Navigation Tabs ──────────────────────────────────────────
    tab_mcq, tab_tf, tab_flash = st.tabs([
        "📝 Multiple Choice (MCQ)",
        "⚖️ True / False Practice",
        "📇 Interactive Flashcards",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1: Multiple Choice Questions (MCQ)
    # ═════════════════════════════════════════════════════════════════════════
    with tab_mcq:
        with st.expander("⚙️ MCQ Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                quiz_type = st.selectbox(
                    "Question Format",
                    ["Mixed (Fill-in-the-blank & Definition)", "Fill-in-the-blank Only", "Definition-based Only"],
                    index=0,
                    key="mcq_type_select",
                    help="Fill-in-the-blank tests key concepts in source statements; Definition tests conceptual recognition.",
                )
            with col2:
                n_questions = st.slider(
                    "Number of Questions",
                    min_value=2,
                    max_value=10,
                    value=5,
                    step=1,
                    key="mcq_n_slider",
                )

        gen_col1, gen_col2, _ = st.columns([1, 1, 3])
        with gen_col1:
            gen_btn = st.button("🚀 Generate MCQs", key="gen_mcq_btn", use_container_width=True)
        with gen_col2:
            clear_btn = st.button("🗑️ Clear MCQs", key="clear_mcq_btn", use_container_width=True)

        if clear_btn:
            st.session_state[KEY_QUIZ_ITEMS] = []
            st.session_state[KEY_QUIZ_ANSWERS] = {}
            st.session_state[KEY_QUIZ_SCORE] = None
            st.rerun()

        if gen_btn:
            q_type_map = {
                "Mixed (Fill-in-the-blank & Definition)": "mixed",
                "Fill-in-the-blank Only": "fill_in_the_blank",
                "Definition-based Only": "definition",
            }
            chosen_type = q_type_map.get(quiz_type, "mixed")

            with st.spinner("Extracting concepts and generating grounded questions …"):
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

        # Display MCQs
        mcq_items = st.session_state.get(KEY_QUIZ_ITEMS, [])
        mcq_user_answers = st.session_state.get(KEY_QUIZ_ANSWERS, {})
        mcq_quiz_score = st.session_state.get(KEY_QUIZ_SCORE)
        mcq_submitted = mcq_quiz_score is not None

        if mcq_items:
            section_divider("MCQ Questions")
            with st.form("mcq_interactive_form"):
                temp_answers = {}
                for i, item in enumerate(mcq_items):
                    q_badge_text = f"QUESTION {i+1} OF {len(mcq_items)} · {item.get('question_type', 'mcq').replace('_', ' ').upper()}"
                    card_mcq_html = (
                        f'<div class="nlp-card nlp-animate" style="margin-bottom:0.8rem; border-left:3px solid #7c3aed;">'
                        f'<div style="font-size:0.75rem; color:#9f67ff; font-weight:600; margin-bottom:0.25rem;">{q_badge_text}</div>'
                        f'<div style="font-weight:600; color:#e6edf3; font-size:0.95rem; line-height:1.6;">{item["question"]}</div>'
                        f'</div>'
                    )
                    st.markdown(card_mcq_html, unsafe_allow_html=True)
                    current_val = mcq_user_answers.get(i)
                    current_idx = item["options"].index(current_val) if (current_val and current_val in item["options"]) else None

                    temp_answers[i] = st.radio(
                        f"Select answer for Q{i+1}:",
                        options=item["options"],
                        index=current_idx,
                        key=f"mcq_opt_{i}",
                        label_visibility="collapsed",
                        disabled=mcq_submitted,
                    )

                submit_mcq_btn = st.form_submit_button(
                    "✅ Submit MCQ Answers",
                    use_container_width=True,
                    disabled=mcq_submitted,
                )

            if submit_mcq_btn:
                correct, total = score_quiz(mcq_items, temp_answers)
                st.session_state[KEY_QUIZ_SCORE] = correct
                st.session_state[KEY_QUIZ_ANSWERS] = temp_answers
                st.rerun()

            if mcq_submitted:
                section_divider("MCQ Results & Detailed Feedback")
                c_cnt = st.session_state[KEY_QUIZ_SCORE]
                t_cnt = len(mcq_items)
                pct = round((c_cnt / max(t_cnt, 1)) * 100)

                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Score", f"{c_cnt} / {t_cnt}")
                sc2.metric("Accuracy", f"{pct}%")
                sc3.metric("Status", "Mastered 🎉" if pct >= 70 else "Review Needed 📚")

                detailed = evaluate_quiz_detailed(mcq_items, mcq_user_answers)
                for res in detailed:
                    card_border = "#10b981" if res["is_correct"] else "#f43f5e"
                    badge_html = (
                        '<span class="nlp-badge nlp-badge-ready">✅ Correct</span>'
                        if res["is_correct"]
                        else ('<span class="nlp-badge nlp-badge-error">❌ Incorrect</span>' if res["is_answered"] else '<span class="nlp-badge nlp-badge-warning">⚠️ Unanswered</span>')
                    )
                    choice_val = res['user_answer'] if res['is_answered'] else 'None selected'
                    card_feedback_html = (
                        f'<div class="nlp-card" style="border-left:4px solid {card_border}; margin-bottom:1rem;">'
                        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">'
                        f'<span style="font-weight:600; color:#e6edf3;">Question {res["id"]}</span>{badge_html}'
                        f'</div>'
                        f'<div style="color:#e6edf3; font-size:0.92rem; margin-bottom:0.6rem; line-height:1.6;">{res["question"]}</div>'
                        f'<div style="font-size:0.85rem; line-height:1.6;">'
                        f'<span style="color:#8b949e;">Your Choice:</span> <strong style="color:{"#10b981" if res["is_correct"] else "#f43f5e"};">{choice_val}</strong><br>'
                        f'<span style="color:#8b949e;">Correct Answer:</span> <strong style="color:#10b981;">{res["correct_answer"]}</strong>'
                        f'</div>'
                        f'<div style="margin-top:0.75rem; padding-top:0.6rem; border-top:1px solid #30363d; font-size:0.82rem; color:#8b949e;">'
                        f'📖 <strong style="color:#9f67ff;">Source from Notes:</strong> <em>"{res["source_sentence"]}"</em>'
                        f'</div>'
                        f'</div>'
                    )
                    st.markdown(card_feedback_html, unsafe_allow_html=True)

                if st.button("🔄 Retake MCQ Quiz", key="retake_mcq_btn"):
                    st.session_state[KEY_QUIZ_SCORE] = None
                    st.session_state[KEY_QUIZ_ANSWERS] = {}
                    st.rerun()
        else:
            st.info("Click **🚀 Generate MCQs** above to formulate multiple choice questions from your notes.")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2: True / False Practice Questions
    # ═════════════════════════════════════════════════════════════════════════
    with tab_tf:
        with st.expander("⚙️ True/False Settings", expanded=False):
            n_tf = st.slider(
                "Number of True/False Statements",
                min_value=2,
                max_value=10,
                value=5,
                step=1,
                key="tf_n_slider",
            )

        tf_col1, tf_col2, _ = st.columns([1, 1, 3])
        with tf_col1:
            gen_tf_btn = st.button("🚀 Generate True/False", key="gen_tf_btn", use_container_width=True)
        with tf_col2:
            clear_tf_btn = st.button("🗑️ Clear True/False", key="clear_tf_btn", use_container_width=True)

        if clear_tf_btn:
            st.session_state[KEY_TF_ITEMS] = []
            st.session_state[KEY_TF_ANSWERS] = {}
            st.session_state[KEY_TF_SCORE] = None
            st.rerun()

        if gen_tf_btn:
            with st.spinner("Extracting factual assertions and building True/False questions …"):
                tf_items = generate_true_false(
                    sentences=sentences,
                    keywords=keywords if keywords else None,
                    n=n_tf,
                    raw_text=raw,
                )
                st.session_state[KEY_TF_ITEMS] = tf_items
                st.session_state[KEY_TF_ANSWERS] = {}
                st.session_state[KEY_TF_SCORE] = None
                if tf_items:
                    st.success(f"✅ Generated {len(tf_items)} True/False practice assertions!")
                else:
                    st.warning("Could not generate enough assertions. Try uploading longer notes.")

        tf_items = st.session_state.get(KEY_TF_ITEMS, [])
        tf_user_answers = st.session_state.get(KEY_TF_ANSWERS, {})
        tf_score = st.session_state.get(KEY_TF_SCORE)
        tf_submitted = tf_score is not None

        if tf_items:
            section_divider("True or False Statements")
            with st.form("tf_interactive_form"):
                temp_tf = {}
                for i, item in enumerate(tf_items):
                    card_tf_html = (
                        f'<div class="tf-card nlp-animate">'
                        f'<div style="font-size:0.75rem; color:#22d3ee; font-weight:600; margin-bottom:0.25rem;">STATEMENT {i+1} OF {len(tf_items)}</div>'
                        f'<div style="font-weight:600; color:#e6edf3; font-size:0.95rem; line-height:1.6;">"{item["statement"]}"</div>'
                        f'</div>'
                    )
                    st.markdown(card_tf_html, unsafe_allow_html=True)
                    curr_choice = tf_user_answers.get(i)
                    opts = ["Select answer…", "True", "False"]
                    curr_idx = 1 if curr_choice is True else (2 if curr_choice is False else 0)

                    user_sel = st.selectbox(
                        f"Answer for Statement {i+1}:",
                        options=opts,
                        index=curr_idx,
                        key=f"tf_sel_{i}",
                        label_visibility="collapsed",
                        disabled=tf_submitted,
                    )
                    if user_sel == "True":
                        temp_tf[i] = True
                    elif user_sel == "False":
                        temp_tf[i] = False
                    else:
                        temp_tf[i] = None

                submit_tf_btn = st.form_submit_button(
                    "✅ Submit True/False Answers",
                    use_container_width=True,
                    disabled=tf_submitted,
                )

            if submit_tf_btn:
                correct, total = score_true_false(tf_items, temp_tf)
                st.session_state[KEY_TF_SCORE] = correct
                st.session_state[KEY_TF_ANSWERS] = temp_tf
                st.rerun()

            if tf_submitted:
                section_divider("True/False Review & Citations")
                tf_correct = st.session_state[KEY_TF_SCORE]
                tf_total = len(tf_items)
                tf_pct = round((tf_correct / max(tf_total, 1)) * 100)

                tfc1, tfc2 = st.columns(2)
                tfc1.metric("Score", f"{tf_correct} / {tf_total}")
                tfc2.metric("Accuracy", f"{tf_pct}%")

                detailed_tf = evaluate_true_false_detailed(tf_items, tf_user_answers)
                for res in detailed_tf:
                    border = "#10b981" if res["is_correct"] else "#f43f5e"
                    ans_text = "True" if res["correct_answer"] else "False"
                    badge = (
                        '<span class="nlp-badge nlp-badge-ready">✅ Correct</span>'
                        if res["is_correct"]
                        else ('<span class="nlp-badge nlp-badge-error">❌ Incorrect</span>' if res["is_answered"] else '<span class="nlp-badge nlp-badge-warning">⚠️ Unanswered</span>')
                    )
                    ans_color = "#10b981" if res["is_correct"] else "#f43f5e"
                    user_ans_str = str(res["user_answer"]) if res["is_answered"] else "Unanswered"
                    card_tf_feedback_html = (
                        f'<div class="nlp-card" style="border-left:4px solid {border}; margin-bottom:1rem;">'
                        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">'
                        f'<span style="font-weight:600; color:#e6edf3;">Statement {res["id"]}</span>{badge}'
                        f'</div>'
                        f'<div style="color:#e6edf3; font-size:0.92rem; margin-bottom:0.5rem;">"{res["statement"]}"</div>'
                        f'<div style="font-size:0.85rem; color:#8b949e; line-height:1.6;">'
                        f'Expected: <strong style="color:#22d3ee;">{ans_text}</strong> | '
                        f'Your Choice: <strong style="color:{ans_color};">{user_ans_str}</strong>'
                        f'</div>'
                        f'<div style="margin-top:0.6rem; padding-top:0.5rem; border-top:1px solid #30363d; font-size:0.82rem; color:#8b949e;">'
                        f'📖 <strong style="color:#9f67ff;">Explanation & Source Note:</strong> <em>{res["explanation"]}</em>'
                        f'</div>'
                        f'</div>'
                    )
                    st.markdown(card_tf_feedback_html, unsafe_allow_html=True)

                if st.button("🔄 Retake True/False", key="retake_tf_btn"):
                    st.session_state[KEY_TF_SCORE] = None
                    st.session_state[KEY_TF_ANSWERS] = {}
                    st.rerun()
        else:
            st.info("Click **🚀 Generate True/False** above to test your factual grasp of the notes.")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3: Interactive Study Flashcards
    # ═════════════════════════════════════════════════════════════════════════
    with tab_flash:
        with st.expander("⚙️ Flashcard Deck Settings", expanded=False):
            n_cards = st.slider(
                "Deck Size",
                min_value=3,
                max_value=20,
                value=10,
                step=1,
                key="flashcard_n_slider",
            )

        f_col1, f_col2, _ = st.columns([1, 1, 3])
        with f_col1:
            gen_fc_btn = st.button("📇 Build Flashcard Deck", key="gen_fc_btn", use_container_width=True)
        with f_col2:
            clear_fc_btn = st.button("🗑️ Clear Deck", key="clear_fc_btn", use_container_width=True)

        if clear_fc_btn:
            st.session_state[KEY_FLASHCARDS] = []
            st.session_state[KEY_FLASHCARD_INDEX] = 0
            st.session_state[KEY_FLASHCARD_MASTERED] = []
            st.session_state["fc_is_flipped"] = False
            st.rerun()

        if gen_fc_btn:
            with st.spinner("Extracting conceptual flashcards from notes …"):
                cards = generate_flashcards(
                    sentences=sentences,
                    keywords=keywords if keywords else None,
                    n=n_cards,
                    raw_text=raw,
                )
                st.session_state[KEY_FLASHCARDS] = cards
                st.session_state[KEY_FLASHCARD_INDEX] = 0
                st.session_state[KEY_FLASHCARD_MASTERED] = []
                st.session_state["fc_is_flipped"] = False
                if cards:
                    st.success(f"✅ Generated {len(cards)} conceptual flashcards!")
                else:
                    st.warning("Could not extract enough concepts for flashcards.")

        cards = st.session_state.get(KEY_FLASHCARDS, [])
        if cards:
            section_divider("Flashcard Deck")

            idx = st.session_state.get(KEY_FLASHCARD_INDEX, 0)
            if idx >= len(cards):
                idx = 0
                st.session_state[KEY_FLASHCARD_INDEX] = 0

            current_card = cards[idx]
            mastered_list = set(st.session_state.get(KEY_FLASHCARD_MASTERED, []))
            is_mastered = current_card["id"] in mastered_list

            # Deck Progress
            mastered_count = len(mastered_list)
            total_cards = len(cards)
            progress_pct = round((mastered_count / max(total_cards, 1)) * 100)

            prog_c1, prog_c2 = st.columns([3, 1])
            with prog_c1:
                prog_html = (
                    f'<div style="font-size:0.85rem; color:#8b949e; margin-bottom:0.25rem;">'
                    f'<strong>Study Mastery:</strong> {mastered_count} of {total_cards} Mastered ({progress_pct}%)'
                    f'</div>'
                    f'<div class="nlp-progress-wrap">'
                    f'<div class="nlp-progress-bar" style="width:{progress_pct}%;"></div>'
                    f'</div>'
                )
                st.markdown(prog_html, unsafe_allow_html=True)
            with prog_c2:
                st.markdown(f"<div style='text-align:right; font-size:0.9rem; font-weight:600; color:#9f67ff; padding-top:0.4rem;'>Card {idx+1} / {total_cards}</div>", unsafe_allow_html=True)

            # Flip State
            if "fc_is_flipped" not in st.session_state:
                st.session_state["fc_is_flipped"] = False
            is_flipped = st.session_state["fc_is_flipped"]

            # Render Card
            cat_badge_class = "badge-def" if current_card.get("category") == "Definition" else "badge-concept"
            mastered_badge = ' <span class="flashcard-badge badge-mastered">⭐ Mastered</span>' if is_mastered else ''

            if not is_flipped:
                # Front of Card
                card_front_html = (
                    f'<div class="flashcard-container nlp-animate">'
                    f'<div><span class="flashcard-badge {cat_badge_class}">{current_card.get("category", "Concept")}</span>{mastered_badge}</div>'
                    f'<div style="font-size:2.2rem; font-weight:800; color:#f1f5f9; margin:1.35rem 0; letter-spacing:-0.02em;">{current_card["term"]}</div>'
                    f'<div style="font-size:0.85rem; color:#94a3b8; letter-spacing:0.04em;">👆 Click "Flip to Reveal Definition" below to test yourself</div>'
                    f'</div>'
                )
                st.markdown(card_front_html, unsafe_allow_html=True)
            else:
                # Back of Card
                source_sentence = current_card.get("source_sentence", "")
                source_html = (
                    f'<div style="font-size:0.82rem; color:#94a3b8; border-top:1px solid rgba(255,255,255,0.08); padding-top:0.65rem; margin-top:0.85rem;">'
                    f'📖 <em>"{source_sentence}"</em>'
                    f'</div>'
                ) if source_sentence else ''

                card_category = current_card.get("category", "Key Concept")
                back_badge_class = "badge-def" if card_category == "Definition" else "badge-concept"
                card_back_html = (
                    f'<div class="flashcard-back-container nlp-animate">'
                    f'<div><span class="flashcard-badge {back_badge_class}">{card_category}</span>{mastered_badge}</div>'
                    f'<div style="font-size:1.35rem; font-weight:700; color:#38bdf8; margin-bottom:0.75rem;">{current_card["term"]}</div>'
                    f'<div style="font-size:1.02rem; line-height:1.75; color:#f1f5f9; margin-bottom:0.85rem;">{current_card["definition"]}</div>'
                    f'{source_html}'
                    f'</div>'
                )
                st.markdown(card_back_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Interactive Controls
            nav_c1, nav_c2, nav_c3, nav_c4 = st.columns([1, 1.5, 1, 1.2])

            with nav_c1:
                if st.button("⬅️ Previous", key="fc_prev_btn", use_container_width=True, disabled=idx == 0):
                    st.session_state[KEY_FLASHCARD_INDEX] = max(0, idx - 1)
                    st.session_state["fc_is_flipped"] = False
                    st.rerun()

            with nav_c2:
                flip_label = "🔄 Flip to Term (Front)" if is_flipped else "🔄 Flip to Definition (Back)"
                if st.button(flip_label, key="fc_flip_btn", use_container_width=True):
                    st.session_state["fc_is_flipped"] = not is_flipped
                    st.rerun()

            with nav_c3:
                if st.button("Next ➡️", key="fc_next_btn", use_container_width=True, disabled=idx >= total_cards - 1):
                    st.session_state[KEY_FLASHCARD_INDEX] = min(total_cards - 1, idx + 1)
                    st.session_state["fc_is_flipped"] = False
                    st.rerun()

            with nav_c4:
                master_label = "Unmark Mastered" if is_mastered else "⭐ Mark Mastered"
                if st.button(master_label, key="fc_master_btn", use_container_width=True):
                    curr_mastered = list(st.session_state.get(KEY_FLASHCARD_MASTERED, []))
                    if current_card["id"] in curr_mastered:
                        curr_mastered.remove(current_card["id"])
                    else:
                        curr_mastered.append(current_card["id"])
                    st.session_state[KEY_FLASHCARD_MASTERED] = curr_mastered
                    st.rerun()

        else:
            st.info("Click **📇 Build Flashcard Deck** above to generate interactive study cards from your notes.")
