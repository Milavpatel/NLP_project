"""
core/nlp_analyzer.py
──────────────────────
Phase 6: Quality-Assurance — Complete NLP Linguistic Analysis Suite.

Provides:
  - pos_tag_distribution()       — NLTK POS tagging mapped to high-level grammatical categories
  - word_frequency()             — Term frequency distribution for top non-stopword tokens
  - generate_wordcloud()         — Visual WordCloud generator (PIL image / WordCloud object)
  - sentence_length_distribution() — Word count distribution across document sentences
  - compute_readability()        — Explainable Flesch Reading Ease & Flesch-Kincaid Grade metrics
"""

from __future__ import annotations

import collections
import re
from typing import Optional

from core.text_processor import (
    clean_text,
    split_sentences,
    tokenise_words,
    _get_stopwords,
)


# ── 1. Part-of-Speech Tagging Distribution ────────────────────────────────────
def pos_tag_distribution(text: str) -> dict[str, int]:
    """
    Compute distribution of grammatical categories using NLTK's averaged perceptron tagger.
    Maps Penn Treebank tags to 7 interpretable linguistic categories:
      - Nouns (NN, NNS, NNP, NNPS)
      - Verbs (VB, VBD, VBG, VBN, VBP, VBZ)
      - Adjectives (JJ, JJR, JJS)
      - Adverbs (RB, RBR, RBS)
      - Pronouns (PRP, PRP$, WP, WP$)
      - Prepositions & Conjunctions (IN, CC)
      - Other (determiners, particles, punctuation)

    Returns:
      Dictionary mapping category name to count.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {}

    try:
        import nltk
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(cleaned)
        # Filter pure punctuation for grammatical POS count
        tokens = [t for t in tokens if re.search(r"[a-zA-Z0-9]", t)]
        if not tokens:
            return {}

        tagged = nltk.pos_tag(tokens)

        category_counts: dict[str, int] = {
            "Nouns": 0,
            "Verbs": 0,
            "Adjectives": 0,
            "Adverbs": 0,
            "Pronouns": 0,
            "Prep & Conj": 0,
            "Other": 0,
        }

        for _, tag in tagged:
            if tag.startswith("NN"):
                category_counts["Nouns"] += 1
            elif tag.startswith("VB"):
                category_counts["Verbs"] += 1
            elif tag.startswith("JJ"):
                category_counts["Adjectives"] += 1
            elif tag.startswith("RB"):
                category_counts["Adverbs"] += 1
            elif tag in ("PRP", "PRP$", "WP", "WP$"):
                category_counts["Pronouns"] += 1
            elif tag in ("IN", "CC", "TO"):
                category_counts["Prep & Conj"] += 1
            else:
                category_counts["Other"] += 1

        return category_counts

    except Exception:
        # Fallback if NLTK tagger is unavailable
        return {}


# ── 2. Word Frequency Analysis ────────────────────────────────────────────────
def word_frequency(text: str, n: int = 20) -> list[tuple[str, int]]:
    """
    Extract the top-N most frequent content tokens (excluding stopwords and punctuation).

    Returns:
      List of (term, count) tuples sorted descending by frequency.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    tokens = tokenise_words(cleaned, remove_stopwords=True, remove_punct=True)
    # Filter single-character noise or numbers
    valid_tokens = [t for t in tokens if len(t) >= 2 and not t.isdigit()]

    if not valid_tokens:
        return []

    counter = collections.Counter(valid_tokens)
    return counter.most_common(n)


# ── 3. Word Cloud Generation ──────────────────────────────────────────────────
def generate_wordcloud(
    text: str,
    background_color: str = "#0d1117",
    width: int = 800,
    height: int = 380,
) -> Optional[object]:
    """
    Generate a WordCloud object from the input text.

    Returns:
      WordCloud object or None if text is empty or wordcloud is not installed.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return None

    try:
        from wordcloud import WordCloud

        stopwords_set = set(_get_stopwords())
        wc = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            colormap="Purples",
            stopwords=stopwords_set,
            min_font_size=10,
            max_words=120,
            collocations=False,
        )
        return wc.generate(cleaned)
    except Exception:
        return None


# ── 4. Sentence Length Distribution ───────────────────────────────────────────
def sentence_length_distribution(sentences: list[str]) -> list[int]:
    """Return list of word counts per sentence."""
    if not sentences:
        return []
    return [len(s.split()) for s in sentences if s.strip()]


# ── 5. Explainable Readability Metrics ─────────────────────────────────────────
def _count_syllables_in_word(word: str) -> int:
    """Heuristic rule-based syllable counter for English words."""
    w = word.lower().strip()
    if len(w) <= 3:
        return 1
    # Remove trailing 'e', 'es', 'ed' if preceded by consonant
    if w.endswith("e") and not w.endswith("le"):
        w = w[:-1]
    # Count vowel groups
    vowels = "aeiouy"
    count = len(re.findall(r"[aeiouy]+", w))
    return max(1, count)


def compute_readability(text: str) -> dict[str, float | str]:
    """
    Compute standard Flesch Reading Ease and Flesch-Kincaid Grade Level metrics.

    Formulas:
      Flesch Reading Ease = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
      Flesch-Kincaid Grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

    Returns:
      Dictionary with 'reading_ease', 'grade_level', and 'interpretation'.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "reading_ease": 0.0,
            "grade_level": 0.0,
            "interpretation": "N/A (Empty Document)",
        }

    sentences = split_sentences(cleaned)
    n_sentences = max(1, len(sentences))

    tokens = tokenise_words(cleaned, remove_stopwords=False, remove_punct=True)
    words = [t for t in tokens if re.search(r"[a-z]", t)]
    n_words = max(1, len(words))

    total_syllables = sum(_count_syllables_in_word(w) for w in words)

    asl = n_words / n_sentences               # Average sentence length
    asw = total_syllables / n_words           # Average syllables per word

    # Standard formulas
    fre = 206.835 - (1.015 * asl) - (84.6 * asw)
    fkgl = (0.39 * asl) + (11.8 * asw) - 15.59

    fre = round(max(0.0, min(100.0, fre)), 1)
    fkgl = round(max(1.0, fkgl), 1)

    # Interpreted reading difficulty level
    if fre >= 80:
        interp = "Easy (School level)"
    elif fre >= 60:
        interp = "Standard (High School level)"
    elif fre >= 40:
        interp = "Fairly Difficult (College level)"
    else:
        interp = "Difficult / Academic (Graduate level)"

    return {
        "reading_ease": fre,
        "grade_level": fkgl,
        "interpretation": interp,
        "avg_sentence_len": round(asl, 1),
        "avg_syllables_per_word": round(asw, 2),
    }
