"""
core/text_processor.py
────────────────────────
Phase 2: Real text preprocessing pipeline.

Provides:
  - clean_text()          — normalise whitespace and encoding artifacts
  - split_sentences()     — NLTK punkt sentence tokeniser
  - tokenise_words()      — NLTK word tokeniser with optional stopword removal
  - stem_tokens()         — Porter stemmer
  - lemmatise_tokens()    — spaCy lemmatiser (optional, degrades gracefully)
  - preprocess()          — full pipeline returning ProcessedDoc dataclass
  - get_basic_stats()     — word/sentence/char/vocab counts (kept from Phase 1)

All functions degrade gracefully:
  - NLTK data auto-downloaded on first use (punkt, stopwords, etc.)
  - spaCy model absence is caught and reported without crashing
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
import nltk

def _ensure_nltk_data() -> None:
    """Download required NLTK corpora if not already present."""
    needed = [
        ("tokenizers/punkt_tab",       "punkt_tab"),
        ("tokenizers/punkt",           "punkt"),
        ("corpora/stopwords",          "stopwords"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/averaged_perceptron_tagger",     "averaged_perceptron_tagger"),
    ]
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass  # offline — caller will surface error

_ensure_nltk_data()


# ── spaCy bootstrap ───────────────────────────────────────────────────────────
_SPACY_NLP = None          # lazy-loaded
_SPACY_AVAILABLE = None    # tri-state: None=unchecked, True/False

def _get_spacy_nlp():
    """Return a loaded spaCy model, or None if unavailable."""
    global _SPACY_NLP, _SPACY_AVAILABLE
    if _SPACY_AVAILABLE is not None:
        return _SPACY_NLP

    try:
        import spacy
        _SPACY_NLP = spacy.load("en_core_web_sm")
        _SPACY_AVAILABLE = True
    except OSError:
        # Model not downloaded
        _SPACY_AVAILABLE = False
        _SPACY_NLP = None
    except ImportError:
        _SPACY_AVAILABLE = False
        _SPACY_NLP = None

    return _SPACY_NLP


def spacy_status() -> tuple[bool, str]:
    """
    Returns (available: bool, message: str).
    Safe to call from UI.
    """
    nlp = _get_spacy_nlp()
    if nlp:
        return True, "spaCy en_core_web_sm loaded ✅"
    try:
        import spacy
        return False, "spaCy installed but model missing. Run: python -m spacy download en_core_web_sm"
    except ImportError:
        return False, "spaCy not installed. Run: pip install spacy"


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class ProcessedDoc:
    """Full preprocessing result returned by preprocess()."""
    raw_text:       str
    cleaned_text:   str
    sentences:      list[str]          = field(default_factory=list)
    tokens:         list[str]          = field(default_factory=list)   # lowercased words
    filtered:       list[str]          = field(default_factory=list)   # stopwords removed
    stemmed:        list[str]          = field(default_factory=list)
    lemmatised:     list[str]          = field(default_factory=list)
    vocab:          set[str]           = field(default_factory=set)
    stopwords_used: bool               = False
    stemming_used:  bool               = False
    lemma_used:     bool               = False


# ── 1. clean_text ─────────────────────────────────────────────────────────────
def clean_text(raw: str) -> str:
    """
    Normalise whitespace and remove common encoding artifacts.
    Does NOT strip words — keeps original vocabulary intact.
    """
    if not raw:
        return ""
    text = raw
    # Replace common unicode quotes / dashes with ASCII equivalents
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")         # non-breaking space
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of spaces/tabs (but keep newlines for paragraph detection)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── 2. split_sentences ────────────────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using NLTK's punkt tokeniser.
    Falls back to regex-based splitting if NLTK data unavailable.
    """
    if not text.strip():
        return []
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except LookupError:
        # Fallback: split on .  !  ?
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]


# ── 3. tokenise_words ─────────────────────────────────────────────────────────
def tokenise_words(
    text: str,
    remove_stopwords: bool = False,
    remove_punct: bool = True,
) -> list[str]:
    """
    Tokenise text into lowercase word tokens using NLTK word_tokenize.

    Args:
        text:             Input text.
        remove_stopwords: If True, filter out English stopwords.
        remove_punct:     If True, discard pure-punctuation tokens.

    Returns:
        List of token strings.
    """
    if not text.strip():
        return []
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text.lower())
    except LookupError:
        tokens = text.lower().split()

    if remove_punct:
        tokens = [t for t in tokens if re.search(r"[a-z0-9]", t)]

    if remove_stopwords:
        stop = _get_stopwords()
        tokens = [t for t in tokens if t not in stop]

    return tokens


@lru_cache(maxsize=1)
def _get_stopwords() -> frozenset[str]:
    """Return NLTK English stopwords as a frozenset (cached)."""
    try:
        from nltk.corpus import stopwords
        return frozenset(stopwords.words("english"))
    except LookupError:
        # Minimal hard-coded fallback
        return frozenset(
            "i me my myself we our ours ourselves you your yours "
            "yourself he him his himself she her hers herself it its "
            "itself they them their theirs themselves what which who "
            "whom this that these those am is are was were be been "
            "being have has had having do does did doing a an the "
            "and but if or because as until while of at by for with "
            "about against between into through during before after "
            "above below to from up down in out on off over under "
            "again further then once here there when where why how "
            "all both each few more most other some such no nor not "
            "only own same so than too very s t can will just don "
            "should now d ll m o re ve y ain aren couldn didn doesn "
            "hadn hasn haven isn ma mightn mustn needn shan shouldn "
            "wasn weren won wouldn".split()
        )


# ── 4. stem_tokens ────────────────────────────────────────────────────────────
def stem_tokens(tokens: list[str]) -> list[str]:
    """
    Apply Porter stemmer to a list of tokens.
    Returns stemmed tokens (same length as input).
    """
    if not tokens:
        return []
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


# ── 5. lemmatise_tokens ───────────────────────────────────────────────────────
def lemmatise_tokens(tokens: list[str]) -> list[str]:
    """
    Lemmatise tokens using spaCy (en_core_web_sm).

    Returns the same token list (unchanged) if spaCy is unavailable.
    The caller can check spacy_status() to inform the user.
    """
    if not tokens:
        return []
    nlp = _get_spacy_nlp()
    if nlp is None:
        return tokens[:]    # graceful degradation

    # spaCy processes a space-joined string, then reads lemma_ per token
    text = " ".join(tokens)
    doc = nlp(text)
    return [token.lemma_ for token in doc]


# ── 6. get_basic_stats ────────────────────────────────────────────────────────
def get_basic_stats(text: str) -> dict:
    """
    Return word / sentence / character / vocabulary counts.
    Uses NLTK tokenisers for accuracy (falls back to naive split).
    Kept API-compatible with Phase 1.
    """
    if not text.strip():
        return {
            "words": 0, "sentences": 0, "characters": 0,
            "paragraphs": 0, "avg_word_len": 0.0,
            "reading_time": 0, "vocab_size": 0,
        }

    sentences  = split_sentences(text)
    tokens     = tokenise_words(text, remove_stopwords=False, remove_punct=True)
    words      = len(tokens)
    chars      = len(text)
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])
    vocab      = len(set(tokens))
    avg_wl     = round(sum(len(t) for t in tokens) / max(words, 1), 2)

    return {
        "words":        words,
        "sentences":    len(sentences),
        "characters":   chars,
        "paragraphs":   paragraphs,
        "avg_word_len": avg_wl,
        "reading_time": max(1, round(words / 200)),
        "vocab_size":   vocab,
    }


# ── 7. preprocess (full pipeline) ─────────────────────────────────────────────
def preprocess(
    raw_text: str,
    remove_stopwords: bool = True,
    use_stemming: bool = False,
    use_lemma: bool = False,
) -> ProcessedDoc:
    """
    Full preprocessing pipeline.

    Steps:
        1. clean_text     — normalise whitespace / encoding
        2. split_sentences — NLTK punkt
        3. tokenise_words  — NLTK word_tokenize, lowercase
        4. filter stopwords (optional)
        5. stem (optional) — Porter
        6. lemmatise (optional) — spaCy

    Returns a ProcessedDoc with all intermediate results.
    """
    cleaned    = clean_text(raw_text)
    sentences  = split_sentences(cleaned)
    tokens     = tokenise_words(cleaned, remove_stopwords=False, remove_punct=True)
    filtered   = tokenise_words(cleaned, remove_stopwords=remove_stopwords, remove_punct=True)
    vocab      = set(filtered)

    stemmed    = stem_tokens(filtered)      if use_stemming else []
    lemmatised = lemmatise_tokens(filtered) if use_lemma    else []

    return ProcessedDoc(
        raw_text       = raw_text,
        cleaned_text   = cleaned,
        sentences      = sentences,
        tokens         = tokens,
        filtered       = filtered,
        stemmed        = stemmed,
        lemmatised     = lemmatised,
        vocab          = vocab,
        stopwords_used = remove_stopwords,
        stemming_used  = use_stemming,
        lemma_used     = use_lemma,
    )
