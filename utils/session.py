"""
utils/session.py
─────────────────
Centralised Streamlit session-state management.
All session keys are defined here as constants to avoid typos across pages.
"""

import streamlit as st

# ── Session Key Constants ─────────────────────────────────────────────────────
KEY_UPLOADED_FILE    = "uploaded_file"       # Raw uploaded file bytes
KEY_FILE_NAME        = "file_name"           # Original filename
KEY_RAW_TEXT         = "raw_text"            # Extracted plain text
KEY_PROCESSED_TEXT   = "processed_text"      # Cleaned / tokenised text
KEY_SENTENCES        = "sentences"           # List[str] sentence list
KEY_SUMMARY          = "summary"             # Generated summary string
KEY_SUMMARY_RESULT   = "summary_result"      # SummaryResult dataclass object
KEY_SUMMARY_METHOD   = "summary_method"      # "tfidf" | "frequency"
KEY_KEYWORDS         = "keywords"            # List[Tuple[str, float]]
KEY_KEYWORD_METHOD   = "keyword_method"      # "tfidf" | "frequency" | "rake"
KEY_ENTITIES         = "entities"            # List[Tuple[str, str]] (NER)
KEY_QUIZ_ITEMS       = "quiz_items"          # List[Dict] quiz questions
KEY_QUIZ_ANSWERS     = "quiz_answers"        # Dict[int, str] user answers
KEY_QUIZ_SCORE       = "quiz_score"          # int
KEY_TF_ITEMS         = "tf_items"            # List[Dict] True/False questions
KEY_TF_ANSWERS       = "tf_answers"          # Dict[int, bool] user answers
KEY_TF_SCORE         = "tf_score"            # Dict or int score for True/False
KEY_FLASHCARDS       = "flashcards"          # List[Dict] flashcards
KEY_FLASHCARD_INDEX  = "flashcard_index"     # int current active card index
KEY_FLASHCARD_MASTERED = "flashcard_mastered" # List[int] IDs of mastered cards
KEY_REVISION_CARDS   = "revision_cards"      # List[Dict] flashcards (legacy compatibility)
KEY_NLP_STATS        = "nlp_stats"           # Dict with word/sentence counts
KEY_SETTINGS         = "settings"            # Dict with user preferences

# ── Default Settings ──────────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "summary_ratio":      0.3,    # fraction of sentences to include
    "max_keywords":       15,     # number of keywords
    "quiz_count":         5,      # number of quiz questions
    "summary_method":     "tfidf",
    "keyword_method":     "tfidf",
    "difficulty":         "medium",
    "theme":              "dark",
}


def init_session() -> None:
    """Initialise all session-state keys with safe defaults on first run."""
    defaults = {
        KEY_UPLOADED_FILE:      None,
        KEY_FILE_NAME:          None,
        KEY_RAW_TEXT:           None,
        KEY_PROCESSED_TEXT:     None,
        KEY_SENTENCES:          [],
        KEY_SUMMARY:            None,
        KEY_SUMMARY_RESULT:     None,
        KEY_SUMMARY_METHOD:     "tfidf",
        KEY_KEYWORDS:           [],
        KEY_KEYWORD_METHOD:     "tfidf",
        KEY_ENTITIES:           [],
        KEY_QUIZ_ITEMS:         [],
        KEY_QUIZ_ANSWERS:       {},
        KEY_QUIZ_SCORE:         None,
        KEY_TF_ITEMS:           [],
        KEY_TF_ANSWERS:         {},
        KEY_TF_SCORE:           None,
        KEY_FLASHCARDS:         [],
        KEY_FLASHCARD_INDEX:    0,
        KEY_FLASHCARD_MASTERED: [],
        KEY_REVISION_CARDS:     [],
        KEY_NLP_STATS:          {},
        KEY_SETTINGS:           DEFAULT_SETTINGS.copy(),
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def has_text() -> bool:
    """Return True if the user has uploaded and extracted text."""
    return bool(st.session_state.get(KEY_RAW_TEXT))


def has_summary() -> bool:
    return bool(st.session_state.get(KEY_SUMMARY))


def has_keywords() -> bool:
    return bool(st.session_state.get(KEY_KEYWORDS))


def has_quiz() -> bool:
    return bool(st.session_state.get(KEY_QUIZ_ITEMS))


def has_tf() -> bool:
    return bool(st.session_state.get(KEY_TF_ITEMS))


def has_flashcards() -> bool:
    return bool(st.session_state.get(KEY_FLASHCARDS))


def has_revision() -> bool:
    return bool(st.session_state.get(KEY_REVISION_CARDS) or st.session_state.get(KEY_FLASHCARDS))


def reset_all() -> None:
    """Clear all uploaded and derived data (keep settings)."""
    keys_to_clear = [
        KEY_UPLOADED_FILE, KEY_FILE_NAME, KEY_RAW_TEXT,
        KEY_PROCESSED_TEXT, KEY_SENTENCES, KEY_SUMMARY,
        KEY_SUMMARY_RESULT, KEY_KEYWORDS, KEY_ENTITIES,
        KEY_QUIZ_ITEMS, KEY_QUIZ_ANSWERS, KEY_QUIZ_SCORE,
        KEY_TF_ITEMS, KEY_TF_ANSWERS, KEY_TF_SCORE,
        KEY_FLASHCARDS, KEY_FLASHCARD_INDEX, KEY_FLASHCARD_MASTERED,
        KEY_REVISION_CARDS, KEY_NLP_STATS,
    ]
    for key in keys_to_clear:
        if key in (
            KEY_SENTENCES, KEY_KEYWORDS, KEY_ENTITIES,
            KEY_QUIZ_ITEMS, KEY_TF_ITEMS, KEY_FLASHCARDS,
            KEY_FLASHCARD_MASTERED, KEY_REVISION_CARDS
        ):
            st.session_state[key] = []
        elif key in (KEY_QUIZ_ANSWERS, KEY_TF_ANSWERS, KEY_NLP_STATS):
            st.session_state[key] = {}
        elif key == KEY_FLASHCARD_INDEX:
            st.session_state[key] = 0
        else:
            st.session_state[key] = None


def clear_derived_data() -> None:
    """Clear downstream NLP results (summary, keywords, quiz, flashcards) when a new document is loaded."""
    keys_to_clear = [
        KEY_SUMMARY, KEY_SUMMARY_RESULT, KEY_KEYWORDS, KEY_ENTITIES,
        KEY_QUIZ_ITEMS, KEY_QUIZ_ANSWERS, KEY_QUIZ_SCORE,
        KEY_TF_ITEMS, KEY_TF_ANSWERS, KEY_TF_SCORE,
        KEY_FLASHCARDS, KEY_FLASHCARD_INDEX, KEY_FLASHCARD_MASTERED,
        KEY_REVISION_CARDS,
    ]
    for key in keys_to_clear:
        if key in (
            KEY_KEYWORDS, KEY_ENTITIES, KEY_QUIZ_ITEMS,
            KEY_TF_ITEMS, KEY_FLASHCARDS, KEY_FLASHCARD_MASTERED,
            KEY_REVISION_CARDS
        ):
            st.session_state[key] = []
        elif key in (KEY_QUIZ_ANSWERS, KEY_TF_ANSWERS):
            st.session_state[key] = {}
        elif key == KEY_FLASHCARD_INDEX:
            st.session_state[key] = 0
        else:
            st.session_state[key] = None


def get_setting(key: str):
    return st.session_state.get(KEY_SETTINGS, DEFAULT_SETTINGS).get(
        key, DEFAULT_SETTINGS.get(key)
    )


def set_setting(key: str, value) -> None:
    if KEY_SETTINGS not in st.session_state:
        st.session_state[KEY_SETTINGS] = DEFAULT_SETTINGS.copy()
    st.session_state[KEY_SETTINGS][key] = value
