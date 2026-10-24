"""
core/quiz_generator.py
────────────────────────
Phase 4: Reliable, Note-Grounded MCQ Quiz Generator.

Provides:
  - generate_mcq()        — Generates multiple-choice questions grounded in uploaded notes
  - score_quiz()          — Evaluates user answers and computes score percentage
  - generate_true_false() — Kept for API compatibility

Design Principles (Academic Integrity & Zero Hallucination):
  1. 100% Note-Grounded: Every question, answer, and distractor originates directly
     from the student's uploaded notes. No external facts or LLM APIs are used.
  2. Four Unique Options: Exactly 1 defensible correct answer and 3 distinct distractors.
  3. Quality Distractors: Distractors are selected from other key terms in the same document.
     Trivial variations (same stem, substrings, duplicates) are strictly rejected.
  4. Skip Rather Than Invent: If 3 valid distractors cannot be found for a question,
     the question is skipped rather than inventing fake options.
  5. Complete Traceability: Every question includes its source sentence and explanation.
  6. Shuffled Options: Option order is randomized without breaking validation.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, asdict
from typing import Optional

from core.text_processor import (
    clean_text,
    split_sentences,
    tokenise_words,
)
from core.keyword_extractor import (
    tfidf_keywords,
    frequency_keywords,
    rake_keywords,
)


@dataclass
class QuizQuestion:
    """Structure for a single multiple-choice question."""
    id: int
    question: str
    options: list[str]          # Exactly 4 options, shuffled
    answer: str                 # Exact text of the correct option
    source_sentence: str        # Verbatim sentence from source notes
    explanation: str            # Traceable explanation of why this answer is correct
    question_type: str          # 'fill_in_the_blank' or 'definition'
    keyword: str                # The targeted key concept

    def to_dict(self) -> dict:
        return asdict(self)


# ── Distractor Quality Helpers ────────────────────────────────────────────────
def _get_stem(word: str) -> str:
    """Return Porter stem for a word/phrase to detect morphological duplicates."""
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        # Stem each token in a multi-word phrase
        tokens = word.lower().split()
        return " ".join(stemmer.stem(t) for t in tokens if t)
    except Exception:
        return word.lower().strip()


def _is_valid_distractor(
    candidate: str,
    target_keyword: str,
    source_sentence: str,
    selected_distractors: list[str],
) -> bool:
    """
    Validate that candidate is a plausible, distinct, non-ambiguous distractor.

    Rules:
      1. Not empty and length >= 2.
      2. Not identical to target_keyword (case-insensitive).
      3. Stem must differ from target_keyword (prevents 'token' vs 'tokens').
      4. Neither is a complete substring of the other (prevents 'NLP' vs 'Modern NLP').
      5. Does not appear as an independent word in the source sentence (avoids ambiguity).
      6. Does not duplicate any already selected distractor (by text or stem).
    """
    cand_clean = candidate.strip()
    target_clean = target_keyword.strip()

    if len(cand_clean) < 2 or cand_clean.isdigit():
        return False

    cand_lower = cand_clean.lower()
    target_lower = target_clean.lower()

    # Rule 2: Cannot be identical
    if cand_lower == target_lower:
        return False

    # Rule 3: Different stem
    if _get_stem(cand_clean) == _get_stem(target_clean):
        return False

    # Rule 4: Substring check
    if cand_lower in target_lower or target_lower in cand_lower:
        return False

    # Rule 5: Should not appear in source sentence as a word
    pattern = r"\b" + re.escape(cand_clean) + r"\b"
    if re.search(pattern, source_sentence, flags=re.IGNORECASE):
        return False

    # Rule 6: Distinct from existing distractors
    cand_stem = _get_stem(cand_clean)
    for existing in selected_distractors:
        if cand_lower == existing.lower():
            return False
        if cand_stem == _get_stem(existing):
            return False

    return True


def _match_case(sample: str, template: str) -> str:
    """Format sample to match the casing style of template (Title Case or lower)."""
    if template.istitle():
        return sample.title()
    elif template.isupper():
        return sample.upper()
    elif template[0].isupper() and len(template) > 1 and template[1:].islower():
        # Capitalized first letter
        return sample[:1].upper() + sample[1:]
    return sample


# ── Definition Pattern Matcher ────────────────────────────────────────────────
_DEF_PATTERNS = [
    # "X is defined as Y"
    r"^(?P<term>[A-Za-z][A-Za-z0-9_\s-]{1,40}?)\s+is\s+defined\s+as\s+(?P<def>.+)$",
    # "X refers to Y"
    r"^(?P<term>[A-Za-z][A-Za-z0-9_\s-]{1,40}?)\s+refers\s+to\s+(?P<def>.+)$",
    # "X is the process of Y"
    r"^(?P<term>[A-Za-z][A-Za-z0-9_\s-]{1,40}?)\s+is\s+the\s+process\s+of\s+(?P<def>.+)$",
    # "X is a branch of Y"
    r"^(?P<term>[A-Za-z][A-Za-z0-9_\s-]{1,40}?)\s+is\s+a\s+branch\s+of\s+(?P<def>.+)$",
    # "X is an algorithm for Y"
    r"^(?P<term>[A-Za-z][A-Za-z0-9_\s-]{1,40}?)\s+is\s+an?\s+(?:algorithm|method|technique|model|tool|framework)\s+(?:for|to|that)\s+(?P<def>.+)$",
    # "X means Y"
    r"^(?P<term>[A-Za-z][A-Za-z0-9_\s-]{1,40}?)\s+means\s+(?P<def>.+)$",
]


def _try_create_definition_question(
    sentence: str,
    keyword: str,
) -> Optional[tuple[str, str]]:
    """
    Attempt to extract a definition question from sentence where keyword is the definiendum.
    Returns (question_text, correct_answer) or None.
    """
    for pat in _DEF_PATTERNS:
        m = re.match(pat, sentence, flags=re.IGNORECASE)
        if m:
            extracted_term = m.group("term").strip()
            def_part = m.group("def").strip().rstrip(".")
            # Check if extracted term matches our keyword
            if extracted_term.lower() == keyword.lower() or keyword.lower() in extracted_term.lower():
                # Format clean question
                q_text = (
                    f"According to your notes, which term is described as: "
                    f"\"{def_part}\"?"
                )
                return q_text, extracted_term

    return None


def _create_fill_in_the_blank_question(
    sentence: str,
    keyword: str,
) -> tuple[str, str]:
    """
    Create a fill-in-the-blank question by replacing the keyword with '_______'.
    Returns (question_text, correct_answer).
    """
    # Find exact match with word boundaries
    pattern = r"\b" + re.escape(keyword) + r"\b"
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    if match:
        actual_word = match.group(0)
        blanked = sentence[:match.start()] + "_______" + sentence[match.end():]
        q_text = f"Complete the statement from your notes:\n\"{blanked}\""
        return q_text, actual_word

    # Fallback: simple replace
    blanked = sentence.replace(keyword, "_______")
    q_text = f"Complete the statement from your notes:\n\"{blanked}\""
    return q_text, keyword


# ── Main Question Generation Function ─────────────────────────────────────────
def generate_mcq(
    sentences: list[str],
    keywords: Optional[list[tuple[str, float]]] = None,
    n: int = 5,
    raw_text: Optional[str] = None,
    question_type: str = "mixed",
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Generate note-grounded multiple choice questions.

    Args:
      sentences: List of source sentences from the document.
      keywords: List of (keyword, score) tuples. If None or empty, extracted automatically.
      n: Desired number of questions (default 5).
      raw_text: Full raw text of the document (optional, used if sentences/keywords are empty).
      question_type: 'mixed', 'fill_in_the_blank', or 'definition'.
      seed: Optional random seed for reproducible tests.

    Returns:
      List of dicts, each containing:
        - 'id': int
        - 'question': str
        - 'options': list[str] (4 items)
        - 'answer': str (correct option string)
        - 'source_sentence': str
        - 'explanation': str
        - 'question_type': str
        - 'keyword': str
    """
    rng = random.Random(seed)

    # 1. Ensure sentences and text are available
    if not sentences and raw_text:
        cleaned = clean_text(raw_text)
        sentences = split_sentences(cleaned)

    if not sentences:
        return []

    # Clean sentences
    valid_sentences = [
        s.strip() for s in sentences
        if len(s.strip().split()) >= 5  # Require at least 5 words for a meaningful question
    ]

    if not valid_sentences:
        return []

    # 2. Ensure candidate keywords are available
    full_text = raw_text if raw_text else " ".join(sentences)
    if not keywords:
        # Extract keywords across TF-IDF, frequency, and RAKE to form rich candidate pool
        tfidf_kws = tfidf_keywords(full_text, n=25)
        freq_kws = frequency_keywords(full_text, n=25)
        keywords = tfidf_kws if tfidf_kws else freq_kws

    if not keywords:
        return []

    # 3. Build candidate distractor pool
    # Combine extracted keywords, RAKE keyphrases, and prominent nouns from the notes
    distractor_pool: list[str] = []
    seen_distractors: set[str] = set()

    # Collect from keywords
    for kw, _ in keywords:
        k_clean = kw.strip()
        if len(k_clean) >= 2 and k_clean.lower() not in seen_distractors:
            seen_distractors.add(k_clean.lower())
            distractor_pool.append(k_clean)

    # Supplement with frequency keywords
    for kw, _ in frequency_keywords(full_text, n=30):
        k_clean = kw.strip()
        if len(k_clean) >= 2 and k_clean.lower() not in seen_distractors:
            seen_distractors.add(k_clean.lower())
            distractor_pool.append(k_clean)

    # Supplement with RAKE phrases
    for kw, _ in rake_keywords(full_text, n=20):
        k_clean = kw.strip()
        if len(k_clean) >= 2 and k_clean.lower() not in seen_distractors:
            seen_distractors.add(k_clean.lower())
            distractor_pool.append(k_clean)

    # 4. Generate candidate questions
    generated_questions: list[QuizQuestion] = []
    used_sentences: set[str] = set()
    used_keywords: set[str] = set()

    # Iterate through sentences to find keyword matches
    for sent in valid_sentences:
        if sent in used_sentences:
            continue

        # Look for keywords present in this sentence
        matching_keywords: list[str] = []
        for kw, _ in keywords:
            if kw.lower() in used_keywords:
                continue
            # Whole word boundary check
            pat = r"\b" + re.escape(kw.strip()) + r"\b"
            if re.search(pat, sent, flags=re.IGNORECASE):
                matching_keywords.append(kw.strip())

        if not matching_keywords:
            continue

        # Prefer longer, more specific keywords in this sentence
        matching_keywords.sort(key=lambda k: len(k), reverse=True)
        chosen_keyword = matching_keywords[0]

        # Determine question type
        q_type_req = question_type.lower().strip()
        def_candidate = _try_create_definition_question(sent, chosen_keyword)

        if q_type_req == "definition":
            if not def_candidate:
                continue
            q_text, correct_answer = def_candidate
            actual_q_type = "definition"
        elif q_type_req == "fill_in_the_blank":
            q_text, correct_answer = _create_fill_in_the_blank_question(sent, chosen_keyword)
            actual_q_type = "fill_in_the_blank"
        else:  # "mixed"
            if def_candidate and len(generated_questions) % 2 == 1:
                q_text, correct_answer = def_candidate
                actual_q_type = "definition"
            else:
                q_text, correct_answer = _create_fill_in_the_blank_question(sent, chosen_keyword)
                actual_q_type = "fill_in_the_blank"

        # 5. Select 3 distinct, valid distractors from the distractor pool
        selected_distractors: list[str] = []

        # Shuffle candidates for variety
        pool_shuffled = list(distractor_pool)
        rng.shuffle(pool_shuffled)

        for cand in pool_shuffled:
            if _is_valid_distractor(cand, correct_answer, sent, selected_distractors):
                # Match casing to correct answer
                formatted_distractor = _match_case(cand, correct_answer)
                selected_distractors.append(formatted_distractor)
                if len(selected_distractors) == 3:
                    break

        # Constraint check: If fewer than 3 valid distractors found, SKIP!
        if len(selected_distractors) < 3:
            continue

        # 6. Build the 4 options and shuffle
        options = [correct_answer] + selected_distractors
        rng.shuffle(options)

        # Build traceable explanation
        explanation = (
            f"According to your notes: \"{sent}\""
        )

        q_obj = QuizQuestion(
            id=len(generated_questions) + 1,
            question=q_text,
            options=options,
            answer=correct_answer,
            source_sentence=sent,
            explanation=explanation,
            question_type=actual_q_type,
            keyword=chosen_keyword,
        )

        generated_questions.append(q_obj)
        used_sentences.add(sent)
        used_keywords.add(chosen_keyword.lower())

        if len(generated_questions) >= n:
            break

    return [q.to_dict() for q in generated_questions]


# ── Quiz Scoring Function ─────────────────────────────────────────────────────
def score_quiz(
    items: list[dict],
    user_answers: dict[int, str],
) -> tuple[int, int]:
    """
    Score the completed quiz.

    Args:
      items: List of question dicts from generate_mcq().
      user_answers: Mapping of question index (0-based) to selected option string.

    Returns:
      (correct_count, total_count)
    """
    if not items:
        return 0, 0

    correct = 0
    total = len(items)

    for idx, item in enumerate(items):
        expected_answer = item.get("answer", "").strip().lower()
        user_choice = user_answers.get(idx)

        if user_choice is not None and isinstance(user_choice, str):
            if user_choice.strip().lower() == expected_answer:
                correct += 1

    return correct, total


def evaluate_quiz_detailed(
    items: list[dict],
    user_answers: dict[int, str],
) -> list[dict]:
    """
    Return detailed item-by-item breakdown for post-submission feedback.
    """
    results: list[dict] = []
    for idx, item in enumerate(items):
        user_choice = user_answers.get(idx)
        expected = item.get("answer", "")
        is_correct = (
            user_choice is not None
            and isinstance(user_choice, str)
            and user_choice.strip().lower() == expected.strip().lower()
        )
        results.append({
            "id": item.get("id", idx + 1),
            "question": item.get("question", ""),
            "options": item.get("options", []),
            "user_answer": user_choice,
            "correct_answer": expected,
            "is_correct": is_correct,
            "is_answered": user_choice is not None and bool(str(user_choice).strip()),
            "source_sentence": item.get("source_sentence", ""),
            "explanation": item.get("explanation", ""),
            "question_type": item.get("question_type", "mcq"),
        })
    return results


# ── True/False Stub for API Compatibility ─────────────────────────────────────
def generate_true_false(sentences: list[str], n: int = 5) -> list[dict]:
    """
    API compatibility stub for True/False questions.
    """
    return []
