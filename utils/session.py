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
KEY_REVISION_CARDS   = "revision_cards"      # List[Dict] flashcards
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
        KEY_UPLOADED_FILE:  None,
        KEY_FILE_NAME:      None,
        KEY_RAW_TEXT:       None,
        KEY_PROCESSED_TEXT: None,
        KEY_SENTENCES:      [],
        KEY_SUMMARY:        None,
        KEY_SUMMARY_RESULT: None,
        KEY_SUMMARY_METHOD: "tfidf",
        KEY_KEYWORDS:       [],
        KEY_KEYWORD_METHOD: "tfidf",
        KEY_ENTITIES:       [],
        KEY_QUIZ_ITEMS:     [],
        KEY_QUIZ_ANSWERS:   {},
        KEY_QUIZ_SCORE:     None,
        KEY_REVISION_CARDS: [],
        KEY_NLP_STATS:      {},
        KEY_SETTINGS:       DEFAULT_SETTINGS.copy(),
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


def has_revision() -> bool:
    return bool(st.session_state.get(KEY_REVISION_CARDS))


def reset_all() -> None:
    """Clear all uploaded and derived data (keep settings)."""
    keys_to_clear = [
        KEY_UPLOADED_FILE, KEY_FILE_NAME, KEY_RAW_TEXT,
        KEY_PROCESSED_TEXT, KEY_SENTENCES, KEY_SUMMARY,
        KEY_SUMMARY_RESULT, KEY_KEYWORDS, KEY_ENTITIES,
        KEY_QUIZ_ITEMS, KEY_QUIZ_ANSWERS, KEY_QUIZ_SCORE,
        KEY_REVISION_CARDS, KEY_NLP_STATS,
    ]
    for key in keys_to_clear:
        st.session_state[key] = None if key not in (
            KEY_SENTENCES, KEY_KEYWORDS, KEY_ENTITIES,
            KEY_QUIZ_ITEMS, KEY_QUIZ_ANSWERS, KEY_REVISION_CARDS
        ) else []


def clear_derived_data() -> None:
    """Clear downstream NLP results (summary, keywords, quiz, revision cards) when a new document is loaded."""
    keys_to_clear = [
        KEY_SUMMARY, KEY_SUMMARY_RESULT, KEY_KEYWORDS, KEY_ENTITIES,
        KEY_QUIZ_ITEMS, KEY_QUIZ_ANSWERS, KEY_QUIZ_SCORE, KEY_REVISION_CARDS,
    ]
    for key in keys_to_clear:
        st.session_state[key] = None if key not in (
            KEY_KEYWORDS, KEY_ENTITIES, KEY_QUIZ_ITEMS,
            KEY_QUIZ_ANSWERS, KEY_REVISION_CARDS
        ) else []


def get_setting(key: str):
    return st.session_state.get(KEY_SETTINGS, DEFAULT_SETTINGS).get(
        key, DEFAULT_SETTINGS.get(key)
    )


def set_setting(key: str, value) -> None:
    if KEY_SETTINGS not in st.session_state:
        st.session_state[KEY_SETTINGS] = DEFAULT_SETTINGS.copy()
    st.session_state[KEY_SETTINGS][key] = value
