"""
core/keyword_extractor.py
──────────────────────────
Phase 3: Real Keyword & Entity Extraction.

Provides:
  - tfidf_keywords()     — TF-IDF keyword extraction using scikit-learn
  - frequency_keywords() — Normalized word frequency ranking
  - rake_keywords()      — RAKE (Rapid Automatic Keyword Extraction) multi-word keyphrases
  - spacy_entities()     — Named Entity Recognition via spaCy (kept strictly separate from keywords)
  - extract_keywords()   — Unified dispatcher for keyword extraction methods

Design Principles:
  1. Explainability: TF-IDF and frequency calculations are straightforward and viva-ready.
  2. Multi-algorithm: Supports TF-IDF, Normalized Frequency, and RAKE keyphrases.
  3. Clean Separation: Named entities (people, orgs, places) are NEVER conflated with
     general topical keywords.
  4. Robustness: Gracefully handles empty strings, stop-word-only text, and short notes.
"""

from __future__ import annotations

import collections
import re
from typing import Optional

from core.text_processor import (
    clean_text,
    split_sentences,
    tokenise_words,
    _get_spacy_nlp,
    _get_stopwords,
)


# ── Method 1: TF-IDF Keyword Extraction ───────────────────────────────────────
def tfidf_keywords(text: str, n: int = 15) -> list[tuple[str, float]]:
    """
    Extract top-N keywords using TF-IDF across sentences as documents.

    Algorithm:
      1. Treat each sentence as an individual document in a mini-corpus.
      2. Compute TF-IDF matrix using scikit-learn's TfidfVectorizer with English stopwords.
      3. For each vocabulary term, compute its max TF-IDF weight across any sentence.
      4. Normalize scores by the highest term weight into the range [0.0, 1.0].
      5. Return top-N terms sorted descending by score.

    Returns:
      List of (term, score) tuples sorted descending by score.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    sentences = split_sentences(cleaned)
    if not sentences:
        sentences = [cleaned]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Support unigrams and informative bigrams
        vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{1,}\b",
            ngram_range=(1, 2),
            min_df=1,
            norm="l2",
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()

        if len(feature_names) == 0:
            return []

        # Compute max TF-IDF score for each term across all sentences
        max_scores = tfidf_matrix.max(axis=0).toarray().ravel()
        highest_score = float(max_scores.max()) if len(max_scores) > 0 else 1.0
        if highest_score <= 0.0:
            highest_score = 1.0

        scored_terms: list[tuple[str, float]] = []
        for term, score in zip(feature_names, max_scores):
            # Normalize to 0.0 - 1.0 range
            norm_score = round(float(score) / highest_score, 4)
            if norm_score > 0.0:
                scored_terms.append((term, norm_score))

        # Sort descending by score, tie-break alphabetically
        scored_terms.sort(key=lambda item: (-item[1], item[0]))
        return scored_terms[: max(1, n)]

    except (ValueError, ImportError):
        # Fallback to normalized frequency if TF-IDF cannot build vocabulary
        return frequency_keywords(cleaned, n=n)


# ── Method 2: Normalized Word Frequency Ranking ───────────────────────────────
def frequency_keywords(text: str, n: int = 15) -> list[tuple[str, float]]:
    """
    Extract top-N keywords using normalized term frequency.

    Algorithm:
      1. Tokenise words, convert to lowercase, remove stopwords and punctuation.
      2. Filter out single-character noise.
      3. Count term occurrences using collections.Counter.
      4. Normalize scores by the highest frequency: Score(w) = count(w) / max_count.
      5. Return top-N terms sorted descending by score.

    Returns:
      List of (term, score) tuples sorted descending by score.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    tokens = tokenise_words(cleaned, remove_stopwords=True, remove_punct=True)
    # Filter single-character or purely numeric tokens for quality
    valid_tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]

    if not valid_tokens:
        return []

    counter = collections.Counter(valid_tokens)
    max_count = max(counter.values())

    scored_terms: list[tuple[str, float]] = [
        (term, round(count / max_count, 4))
        for term, count in counter.most_common()
    ]

    return scored_terms[: max(1, n)]


# ── Method 3: RAKE (Rapid Automatic Keyword Extraction) ───────────────────────
def rake_keywords(text: str, n: int = 15) -> list[tuple[str, float]]:
    """
    Extract keyphrases using RAKE (Rapid Automatic Keyword Extraction).

    RAKE splits text on stopwords and phrase delimiters, builds a word co-occurrence
    graph, and scores candidate keyphrases based on word degree divided by word frequency.

    Returns:
      List of (keyphrase, normalized_score) tuples sorted descending by score.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    try:
        from rake_nltk import Rake

        stopwords_set = _get_stopwords()
        r = Rake(
            stopwords=stopwords_set,
            punctuations=r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""",
            min_length=1,
            max_length=3,
        )
        r.extract_keywords_from_text(cleaned)
        ranked_phrases = r.get_ranked_phrases_with_scores()

        if not ranked_phrases:
            return frequency_keywords(cleaned, n=n)

        # Filter empty or noisy keyphrases
        valid_items: list[tuple[str, float]] = []
        seen: set[str] = set()

        for score, phrase in ranked_phrases:
            phrase_clean = phrase.strip().lower()
            if len(phrase_clean) > 1 and phrase_clean not in seen and not phrase_clean.isdigit():
                seen.add(phrase_clean)
                valid_items.append((phrase_clean, float(score)))

        if not valid_items:
            return []

        # Normalize RAKE scores into [0.0, 1.0] for consistent UI presentation
        max_score = max(s for _, s in valid_items)
        if max_score <= 0.0:
            max_score = 1.0

        normalized = [
            (phrase, round(score / max_score, 4))
            for phrase, score in valid_items
        ]

        normalized.sort(key=lambda item: (-item[1], item[0]))
        return normalized[: max(1, n)]

    except Exception:
        # Fallback to TF-IDF or frequency
        return tfidf_keywords(cleaned, n=n)


# ── Method 4: spaCy Named Entity Recognition (NER) ────────────────────────────
def spacy_entities(text: str) -> list[tuple[str, str]]:
    """
    Extract named entities using spaCy's en_core_web_sm model.

    NOTE: Named entities are fundamentally distinct from topical keywords.
    Entities represent specific real-world objects (PERSON, ORG, GPE, DATE, etc.)
    and are kept in their own dedicated data structure.

    Returns:
      List of unique (entity_text, entity_label) tuples in appearance order.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    nlp = _get_spacy_nlp()
    if nlp is None:
        return []

    doc = nlp(cleaned)
    entities: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for ent in doc.ents:
        ent_text = ent.text.strip()
        # Filter single punctuation or trivial strings
        if len(ent_text) > 1:
            item = (ent_text, ent.label_)
            if item not in seen:
                seen.add(item)
                entities.append(item)

    return entities


# ── Unified Dispatcher ────────────────────────────────────────────────────────
def extract_keywords(
    text: str,
    method: str = "tfidf",
    n: int = 15,
) -> list[tuple[str, float]]:
    """
    Unified entry point for keyword extraction.

    Args:
      text: Source text.
      method: 'tfidf', 'frequency' (or 'freq'), or 'rake'.
      n: Number of keywords to return.

    Returns:
      List of (term, score) tuples.
    """
    method_norm = method.lower().strip()
    if method_norm in ("frequency", "freq", "word_frequency"):
        return frequency_keywords(text, n=n)
    elif method_norm in ("rake", "keyphrases"):
        return rake_keywords(text, n=n)
    else:
        return tfidf_keywords(text, n=n)
