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
    spacy_entities,
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


@dataclass
class TrueFalseQuestion:
    """Structure for a note-grounded True/False statement."""
    id: int
    statement: str              # The assertion presented to the student
    correct_answer: bool        # True or False
    source_sentence: str        # Verbatim sentence from source notes
    explanation: str            # Traceable explanation why it is True or False
    altered: bool               # Whether the statement was syntactically altered
    target_keyword: str         # The concept examined

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Flashcard:
    """Structure for a conceptual study flashcard."""
    id: int
    term: str                   # Front: Concept / Keyword
    definition: str             # Back: Definition or core explanatory statement
    source_sentence: str        # Verbatim sentence from notes
    category: str               # "Definition", "Key Concept", etc.
    salience_score: float       # Relevance/salience score (0.0 - 1.0)

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


# ── Study Sentence Validation ──────────────────────────────────────────────────
def is_study_sentence(sentence: str) -> bool:
    """
    Validate whether a sentence is an informative study proposition rather than
    document metadata, course syllabus lines, instructor titles, or raw headers.
    """
    if not sentence or not sentence.strip():
        return False
    s_clean = sentence.strip()
    words = s_clean.split()
    if len(words) < 5:
        return False

    lower = s_clean.lower()
    # Reject syllabus/document headers
    if lower.startswith((
        "course:", "lecture notes", "comprehensive study notes",
        "chapter ", "table of contents", "instructor:", "dr.", "prof.",
        "syllabus:", "page ", "overview:"
    )):
        return False
    if "comprehensive study notes:" in lower or re.match(r"^course\s*:\s*", lower):
        return False
    if lower.endswith((":", "takeaways:", "overview:", "pipelines", "vectorization", "modeling")):
        return False

    return True


def _clean_extracted_term(term: str) -> str:
    """Strip leading transitional preambles and articles from an extracted concept term."""
    t = term.strip()
    # Strip transitional preambles, introductory phrases, and adverbial markers
    t = re.sub(
        r"^(?:in\s+[a-z\s]+,|at\s+its\s+core,|fundamentally,|technically,|broadly\s+speaking,|by\s+definition,|according\s+to\s+[^,]+,|specifically,|furthermore,|additionally,|moreover,|historically,|traditionally,|formally,|namely,|essentially,|here,|in\s+contrast,)\s*",
        "",
        t,
        flags=re.IGNORECASE,
    ).strip()
    # Strip leading 'the ' or 'a ' if term is a concept
    t = re.sub(r"^(?:the|an?)\s+", "", t, flags=re.IGNORECASE).strip()
    return t


def _clean_definition_text(def_text: str) -> str:
    """
    Clean and prune trailing non-restrictive relative clauses, citations,
    or sentence-ending fragments from an extracted definition clause.
    """
    d = def_text.strip().rstrip(".")
    # If the definition has a non-restrictive relative clause like ", which was..." or ", which means...",
    # if the core predicate before the comma is at least 4 words, prune the trailing clause
    parts = re.split(
        r",\s*(?:which\s+|although\s+|whereas\s+|whereby\s+|while\s+|as\s+seen\s+in\s+)",
        d,
        flags=re.IGNORECASE,
    )
    if len(parts) > 1 and len(parts[0].split()) >= 4:
        d = parts[0].strip()
    return d


# ── Definition Pattern Matcher ────────────────────────────────────────────────
_BOUNDARY_PREFIX = (
    r"(?:^|(?<=[.!?])\s+|[;\n]\s*)"
    r"(?:(?:in\s+[a-z\s]+|at\s+its\s+core|fundamentally|technically|specifically|"
    r"furthermore|additionally|moreover|historically|traditionally|formally|"
    r"broadly\s+speaking|by\s+definition|here|as\s+such),\s*)?"
)

_DEF_PATTERNS = [
    # "X is defined as Y"
    _BOUNDARY_PREFIX + r"(?P<term>[A-Za-z0-9_./\s-]{1,45}?)\s+is\s+defined\s+as\s+(?P<def>[^;]+)",
    # "X refers to Y"
    _BOUNDARY_PREFIX + r"(?P<term>[A-Za-z0-9_./\s-]{1,45}?)\s+refers\s+to\s+(?P<def>[^;]+)",
    # "X is the process of Y"
    _BOUNDARY_PREFIX + r"(?P<term>[A-Za-z0-9_./\s-]{1,45}?)\s+is\s+(?P<def>(?:the|a)\s+(?:heuristic\s+)?process\s+of\s+[^;]+)",
    # "X is an algorithm/method for Y"
    _BOUNDARY_PREFIX + r"(?P<term>[A-Za-z0-9_./\s-]{1,45}?)\s+is\s+(?P<def>an?\s+(?:extractive\s+)?(?:algorithm|method|technique|model|tool|framework|measure|metric)\s+[^;.]+)",
    # "X is a branch of Y"
    _BOUNDARY_PREFIX + r"(?P<term>[A-Za-z0-9_./\s-]{1,45}?)\s+is\s+(?P<def>a\s+(?:branch|form|subfield|discipline|type)\s+of\s+[^;]+)",
    # "X means Y"
    _BOUNDARY_PREFIX + r"(?P<term>[A-Za-z0-9_./\s-]{1,45}?)\s+means\s+(?P<def>[^;]+)",
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
        m = re.search(pat, sentence, flags=re.IGNORECASE)
        if m:
            raw_term = m.group("term").strip()
            extracted_term = _clean_extracted_term(raw_term)
            def_part = _clean_definition_text(m.group("def"))
            # Check if extracted term matches our keyword
            if (
                extracted_term.lower() == keyword.lower()
                or keyword.lower() in extracted_term.lower()
                or extracted_term.lower() in keyword.lower()
            ):
                q_text = (
                    f"According to your notes, which term is described as: "
                    f"\"{def_part}\"?"
                )
                return q_text, extracted_term.title()

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

    # Clean sentences, filtering out metadata headers, title banners, and syllabus lines
    valid_sentences = [s.strip() for s in sentences if is_study_sentence(s)]
    if not valid_sentences:
        valid_sentences = [
            s.strip() for s in sentences
            if len(s.strip().split()) >= 5
        ]

    if not valid_sentences:
        return []

    full_text = raw_text if raw_text else " ".join(valid_sentences)
    if len(full_text.split()) < 6:
        return []

    # 2. Extract definitions across valid sentences
    def_map: dict[str, tuple[str, str]] = {}
    for sent in valid_sentences:
        for pat in _DEF_PATTERNS:
            m = re.search(pat, sent, flags=re.IGNORECASE)
            if m:
                raw_term = m.group("term").strip()
                term = _clean_extracted_term(raw_term)
                def_part = _clean_definition_text(m.group("def"))
                if len(term) >= 2 and not term.lower().startswith(("the ", "a ", "an ")):
                    def_map[sent] = (term.title(), def_part)
                break

    # 3. Extract named entities
    raw_entities = spacy_entities(full_text)
    person_pool: list[str] = []
    org_pool: list[str] = []
    date_pool: list[str] = []
    entity_terms: list[str] = []
    for ent_text, ent_type in raw_entities:
        clean_ent = ent_text.strip().lstrip("• ").strip()
        if len(clean_ent) < 2 or clean_ent.isdigit() or "\n" in clean_ent:
            if clean_ent.isdigit() and len(clean_ent) == 4:
                date_pool.append(clean_ent)
            continue
        if clean_ent.lower() in ("corpora", "nlp", "gpe", "organization", "course", "100%"):
            continue
        if ent_type == "PERSON":
            person_pool.append(clean_ent)
        elif ent_type in ("ORG", "GPE"):
            org_pool.append(clean_ent)
        elif ent_type == "DATE":
            date_pool.append(clean_ent)
        entity_terms.append(clean_ent)

    # 4. Build comprehensive unique concept pool for targets and distractors
    STOP_ENDS = {'of', 'in', 'at', 'by', 'for', 'with', 'from', 'to', 'and', 'or', 'the', 'a', 'an', 'non', 'as', 'like', 'than', 'into'}
    STOP_WORDS = {'exam', 'notes', 'study', 'takeaways', 'core', 'introduction', 'date', 'words', 'word', 'sentences', 'sentence', 'like', 'become', 'published', 'text', 'including'}

    concept_candidates = [t for t, _ in def_map.values()] + entity_terms

    if keywords:
        for kw, _ in keywords:
            k = kw.strip()
            if len(k) >= 3 and not k.isdigit():
                concept_candidates.append(k.title())

    for phrase, _ in rake_keywords(full_text, n=60):
        p = phrase.strip()
        w = p.lower().split()
        if len(w) >= 2 and w[-1] not in STOP_ENDS and w[0] not in STOP_ENDS:
            if not any(x in STOP_WORDS for x in w):
                concept_candidates.append(p.title())

    for kw, _ in tfidf_keywords(full_text, n=40):
        k = kw.strip()
        w = k.lower().split()
        if w[-1] not in STOP_ENDS and not any(x in STOP_WORDS for x in w) and len(k) >= 3:
            concept_candidates.append(k.title())

    for kw, _ in frequency_keywords(full_text, n=40):
        k = kw.strip()
        w = k.lower().split()
        if w[-1] not in STOP_ENDS and not any(x in STOP_WORDS for x in w) and len(k) >= 3:
            concept_candidates.append(k.title())

    unique_concepts: list[str] = []
    seen_concepts: set[str] = set()
    for c in concept_candidates:
        c_clean = c.strip()
        if c_clean.lower() not in seen_concepts and len(c_clean) >= 2:
            seen_concepts.add(c_clean.lower())
            unique_concepts.append(c_clean)

    # Need at least 4 distinct terms to form a 4-option question without inventing fake options
    if len(unique_concepts) < 4:
        return []

    # 5. Generate candidate questions
    questions: list[QuizQuestion] = []
    used_sentences: set[str] = set()
    used_answers: set[str] = set()
    q_type_req = question_type.lower().strip()

    # Pass 1: Definition questions (used in definition mode and balanced in mixed mode)
    if q_type_req in ("definition", "mixed"):
        for sent in valid_sentences:
            if sent in used_sentences or sent not in def_map:
                continue

            term, def_part = def_map[sent]
            if term.lower() in used_answers:
                continue

            q_text = f"According to your notes, which term is described as: \"{def_part}\"?"
            target_word = term
            actual_type = "definition"

            c_pool = list(unique_concepts)
            rng.shuffle(c_pool)
            selected_distractors: list[str] = []
            for cand in c_pool:
                if _is_valid_distractor(cand, target_word, sent, selected_distractors):
                    selected_distractors.append(_match_case(cand, target_word))
                    if len(selected_distractors) == 3:
                        break

            if len(selected_distractors) == 3:
                options = [target_word] + selected_distractors
                rng.shuffle(options)
                explanation = f"According to your notes: \"{sent}\""
                q_obj = QuizQuestion(
                    id=len(questions) + 1,
                    question=q_text,
                    options=options,
                    answer=target_word,
                    source_sentence=sent,
                    explanation=explanation,
                    question_type=actual_type,
                    keyword=target_word,
                )
                questions.append(q_obj)
                used_sentences.add(sent)
                used_answers.add(target_word.lower())

                if q_type_req == "mixed" and len(questions) >= max(1, n // 2):
                    break
                if len(questions) >= n:
                    break

    # Pass 2: Fill-in-the-blank questions
    if len(questions) < n and q_type_req != "definition":
        for sent in valid_sentences:
            if sent in used_sentences:
                continue

            target_word = None
            if sent in def_map and def_map[sent][0].lower() not in used_answers:
                target_word = def_map[sent][0]
            else:
                matched: list[str] = []
                for c in unique_concepts:
                    if c.lower() in used_answers:
                        continue
                    if re.search(r"\b" + re.escape(c) + r"\b", sent, flags=re.IGNORECASE):
                        matched.append(c)
                if matched:
                    matched.sort(key=lambda x: len(x), reverse=True)
                    target_word = matched[0]

            if not target_word:
                continue

            pattern = r"\b" + re.escape(target_word) + r"\b"
            blanked, count = re.subn(pattern, "_______", sent, count=1, flags=re.IGNORECASE)
            if count == 0:
                idx = sent.lower().find(target_word.lower())
                if idx != -1:
                    blanked = sent[:idx] + "_______" + sent[idx + len(target_word):]
                else:
                    continue

            q_text = f"Complete the statement from your notes:\n\"{blanked}\""
            actual_type = "fill_in_the_blank"

            ans_clean = target_word.strip().lower()
            if any(ans_clean in p.lower() or p.lower() in ans_clean for p in person_pool):
                c_pool = list(person_pool) + [c for c in unique_concepts if c not in person_pool]
            elif any(ans_clean in o.lower() or o.lower() in ans_clean for o in org_pool):
                c_pool = list(org_pool) + [c for c in unique_concepts if c not in org_pool]
            else:
                c_pool = list(unique_concepts)

            rng.shuffle(c_pool)
            selected_distractors = []
            for cand in c_pool:
                if _is_valid_distractor(cand, target_word, sent, selected_distractors):
                    selected_distractors.append(_match_case(cand, target_word))
                    if len(selected_distractors) == 3:
                        break

            if len(selected_distractors) == 3:
                options = [target_word] + selected_distractors
                rng.shuffle(options)
                explanation = f"According to your notes: \"{sent}\""
                q_obj = QuizQuestion(
                    id=len(questions) + 1,
                    question=q_text,
                    options=options,
                    answer=target_word,
                    source_sentence=sent,
                    explanation=explanation,
                    question_type=actual_type,
                    keyword=target_word,
                )
                questions.append(q_obj)
                used_sentences.add(sent)
                used_answers.add(target_word.lower())

                if len(questions) >= n:
                    break

    # Pass 3: If still under n (e.g. in definition mode when document has fewer definitions than requested),
    # supplement with fill-in-the-blank on remaining sentences so user receives the requested quantity
    if len(questions) < n:
        for sent in valid_sentences:
            if sent in used_sentences:
                continue

            matched = []
            for c in unique_concepts:
                if c.lower() in used_answers:
                    continue
                if re.search(r"\b" + re.escape(c) + r"\b", sent, flags=re.IGNORECASE):
                    matched.append(c)
            if not matched:
                continue
            matched.sort(key=lambda x: len(x), reverse=True)
            target_word = matched[0]

            pattern = r"\b" + re.escape(target_word) + r"\b"
            blanked, count = re.subn(pattern, "_______", sent, count=1, flags=re.IGNORECASE)
            if count == 0:
                idx = sent.lower().find(target_word.lower())
                if idx != -1:
                    blanked = sent[:idx] + "_______" + sent[idx + len(target_word):]
                else:
                    continue

            if q_type_req == "definition":
                q_text = f"According to your notes, which term is described in this statement:\n\"{blanked}\"?"
                actual_type = "definition"
            else:
                q_text = f"Complete the statement from your notes:\n\"{blanked}\""
                actual_type = "fill_in_the_blank"

            c_pool = list(unique_concepts)
            rng.shuffle(c_pool)
            selected_distractors = []
            for cand in c_pool:
                if _is_valid_distractor(cand, target_word, sent, selected_distractors):
                    selected_distractors.append(_match_case(cand, target_word))
                    if len(selected_distractors) == 3:
                        break

            if len(selected_distractors) == 3:
                options = [target_word] + selected_distractors
                rng.shuffle(options)
                questions.append(QuizQuestion(
                    id=len(questions) + 1,
                    question=q_text,
                    options=options,
                    answer=target_word,
                    source_sentence=sent,
                    explanation=f"According to your notes: \"{sent}\"",
                    question_type=actual_type,
                    keyword=target_word,
                ))
                used_sentences.add(sent)
                used_answers.add(target_word.lower())

                if len(questions) >= n:
                    break

    return [q.to_dict() for q in questions]


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


# ── True / False Generation & Evaluation ──────────────────────────────────────
# ── True / False Generation & Evaluation ──────────────────────────────────────
_OPPOSING_PAIRS: list[tuple[str, str]] = [
    ("supervised", "unsupervised"),
    ("unsupervised", "supervised"),
    ("extractive", "abstractive"),
    ("abstractive", "extractive"),
    ("discrete", "continuous"),
    ("continuous", "discrete"),
    ("deterministic", "probabilistic"),
    ("probabilistic", "deterministic"),
    ("linear", "non-linear"),
    ("non-linear", "linear"),
    ("symmetric", "asymmetric"),
    ("asymmetric", "symmetric"),
    ("high", "low"),
    ("low", "high"),
    ("maximum", "minimum"),
    ("minimum", "maximum"),
    ("increases", "decreases"),
    ("decreases", "increases"),
    ("increase", "decrease"),
    ("decrease", "increase"),
    ("before", "after"),
    ("after", "before"),
    ("loss", "reward"),
    ("reward", "loss"),
    ("overfitting", "underfitting"),
    ("underfitting", "overfitting"),
]

_NEGATION_REPLACEMENTS = [
    (r"\bis defined as\b", "is not defined as"),
    (r"\bis not defined as\b", "is defined as"),
    (r"\bis the process of\b", "is not the process of"),
    (r"\bis a branch of\b", "is not a branch of"),
    (r"\brefers to\b", "does not refer to"),
    (r"\bdoes not refer to\b", "refers to"),
    (r"\bis an?\b", "is not a"),
    (r"\bis not an?\b", "is a"),
    (r"\bare\b", "are not"),
    (r"\bare not\b", "are"),
    (r"\bcan be\b", "cannot be"),
    (r"\bcannot be\b", "can be"),
    (r"\buses\b", "does not use"),
    (r"\bdoes not use\b", "uses"),
    (r"\brequires\b", "does not require"),
    (r"\bdoes not require\b", "requires"),
    (r"\bwas\b", "was not"),
    (r"\bwere\b", "were not"),
]


def generate_true_false(
    sentences: list[str],
    keywords: Optional[list[tuple[str, float]]] = None,
    n: int = 5,
    raw_text: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Generate note-grounded True/False statements with verified factual integrity.
    
    Principles:
      1. Intended distribution is tracked explicitly: targets ~50% True and ~50% False items.
      2. True statements are verbatim declarative sentences from notes.
      3. False statements are generated ONLY via verified transformations:
         - Domain opposing concept pairs (e.g., supervised <-> unsupervised).
         - Syntactic negation (inverting assertions).
         - Semantic entity peer swaps (PERSON <-> PERSON, ORG <-> ORG) where the replacement
           is verified not to appear with that predicate in the document.
         - Definition term substitutions where the replacement term is distinct and verified.
      4. Strict validation: If an alteration cannot be verified or is uncertain,
         the candidate question is OMITTED rather than silently reverting to True.
      5. Every question includes verbatim source notes and traceable explanation.
    """
    rng = random.Random(seed)

    if not sentences and raw_text:
        cleaned = clean_text(raw_text)
        sentences = split_sentences(cleaned)

    if not sentences:
        return []

    # Clean sentences, filtering out metadata headers, syllabus lines, and raw headings
    valid_sentences = [s.strip() for s in sentences if is_study_sentence(s)]
    if not valid_sentences:
        valid_sentences = [
            s.strip() for s in sentences
            if len(s.strip().split()) >= 5
        ]
    if not valid_sentences:
        return []

    full_text = raw_text if raw_text else " ".join(valid_sentences)
    full_text_lower = full_text.lower()
    if not keywords:
        keywords = tfidf_keywords(full_text, n=25) or frequency_keywords(full_text, n=25)

    # Collect entity pools for semantic peer mutation
    raw_entities = spacy_entities(full_text)
    person_pool = [
        e[0].strip() for e in raw_entities
        if e[1] == "PERSON" and len(e[0].strip()) >= 3 and e[0].strip()[0].isupper() and not e[0].strip().islower() and not e[0].strip().startswith("•")
    ]
    org_pool = [
        e[0].strip() for e in raw_entities
        if e[1] in ("ORG", "GPE") and len(e[0].strip()) >= 2 and e[0].strip()[0].isupper() and not e[0].strip().startswith("•")
    ]

    # Concept pool from keywords & RAKE
    concept_pool: list[str] = []
    seen_concepts: set[str] = set()
    for kw, _ in (keywords or []):
        k_clean = kw.strip()
        if len(k_clean) >= 3 and not k_clean.isdigit() and k_clean.lower() not in seen_concepts:
            seen_concepts.add(k_clean.lower())
            concept_pool.append(k_clean)

    for phrase, _ in rake_keywords(full_text, n=15):
        p_clean = phrase.strip()
        if len(p_clean) >= 3 and not p_clean.isdigit() and p_clean.lower() not in seen_concepts:
            seen_concepts.add(p_clean.lower())
            concept_pool.append(p_clean)

    distractor_pool = list(concept_pool) + list(person_pool) + list(org_pool)

    # Explicit target distribution tracking
    target_false = n // 2
    target_true = n - target_false

    true_items: list[TrueFalseQuestion] = []
    false_items: list[TrueFalseQuestion] = []
    used_sentences: set[str] = set()

    # Pass 1: Attempt to construct verified False questions up to target_false
    for sent in valid_sentences:
        if len(false_items) >= target_false:
            break
        if sent in used_sentences:
            continue

        matched_kw = None
        for kw, _ in (keywords or []):
            pat = r"\b" + re.escape(kw.strip()) + r"\b"
            if re.search(pat, sent, flags=re.IGNORECASE):
                matched_kw = kw.strip()
                break

        altered_sent = None
        swap_explanation = None

        # A. Opposing pair swap (high confidence factual contradiction)
        for orig_term, opp_term in _OPPOSING_PAIRS:
            pat = r"\b" + re.escape(orig_term) + r"\b"
            if re.search(pat, sent, flags=re.IGNORECASE):
                cand = _match_case(opp_term, orig_term)
                candidate_altered, count = re.subn(pat, cand, sent, flags=re.IGNORECASE, count=1)
                # Verify alteration produces a statement not present in notes
                if count > 0 and candidate_altered != sent and candidate_altered.lower() not in full_text_lower:
                    altered_sent = candidate_altered
                    swap_explanation = (
                        f"False. The statement claims '{cand}', whereas according to your notes: \"{sent}\""
                    )
                    matched_kw = orig_term
                    break

        # B. Semantic entity peer swap (Person with Person, Org with Org)
        if not altered_sent:
            matched_person = next((p for p in person_pool if re.search(r"\b" + re.escape(p) + r"\b", sent, re.IGNORECASE)), None)
            matched_org = next((o for o in org_pool if re.search(r"\b" + re.escape(o) + r"\b", sent, re.IGNORECASE)), None)

            if matched_person and len(person_pool) >= 2:
                other_persons = [
                    p for p in person_pool 
                    if p.lower() != matched_person.lower() 
                    and _get_stem(p) != _get_stem(matched_person)
                    and p.lower() not in sent.lower()
                ]
                if other_persons:
                    rng.shuffle(other_persons)
                    cand = other_persons[0]
                    pat = r"\b" + re.escape(matched_person) + r"\b"
                    candidate_altered, count = re.subn(pat, cand, sent, flags=re.IGNORECASE, count=1)
                    if count > 0 and candidate_altered != sent and candidate_altered.lower() not in full_text_lower:
                        altered_sent = candidate_altered
                        swap_explanation = (
                            f"False. The statement incorrectly attributes this to '{cand}'. "
                            f"According to your notes: \"{sent}\""
                        )
                        matched_kw = matched_person

            elif matched_org and len(org_pool) >= 2 and not altered_sent:
                other_orgs = [
                    o for o in org_pool 
                    if o.lower() != matched_org.lower() 
                    and _get_stem(o) != _get_stem(matched_org)
                    and o.lower() not in sent.lower()
                ]
                if other_orgs:
                    rng.shuffle(other_orgs)
                    cand = other_orgs[0]
                    pat = r"\b" + re.escape(matched_org) + r"\b"
                    candidate_altered, count = re.subn(pat, cand, sent, flags=re.IGNORECASE, count=1)
                    if count > 0 and candidate_altered != sent and candidate_altered.lower() not in full_text_lower:
                        altered_sent = candidate_altered
                        swap_explanation = (
                            f"False. The statement incorrectly claims this took place at or was developed by '{cand}'. "
                            f"According to your notes: \"{sent}\""
                        )
                        matched_kw = matched_org

        # C. Definition term swap
        if not altered_sent:
            for pat in _DEF_PATTERNS:
                m = re.search(pat, sent, flags=re.IGNORECASE)
                if m:
                    raw_term = m.group("term").strip()
                    extracted_term = _clean_extracted_term(raw_term)
                    alt_concepts = [
                        c for c in concept_pool 
                        if _is_valid_distractor(c, extracted_term, sent, [])
                        and c.lower() not in sent.lower()
                    ]
                    if alt_concepts:
                        rng.shuffle(alt_concepts)
                        cand = _match_case(alt_concepts[0], extracted_term)
                        pat_term = r"\b" + re.escape(extracted_term) + r"\b"
                        candidate_altered, count = re.subn(pat_term, cand, sent, flags=re.IGNORECASE, count=1)
                        if count > 0 and candidate_altered != sent and candidate_altered.lower() not in full_text_lower:
                            altered_sent = candidate_altered
                            swap_explanation = f"False. This describes '{extracted_term}', not '{cand}'. According to your notes: \"{sent}\""
                            matched_kw = extracted_term
                            break

        # D. Syntactic negation
        if not altered_sent:
            for pat, rep in _NEGATION_REPLACEMENTS:
                if re.search(pat, sent, flags=re.IGNORECASE):
                    candidate_altered, count = re.subn(pat, rep, sent, flags=re.IGNORECASE, count=1)
                    if count > 0 and candidate_altered != sent and candidate_altered.lower() not in full_text_lower:
                        altered_sent = candidate_altered
                        swap_explanation = f"False. According to your notes: \"{sent}\""
                        break

        # Verification check: Only add if alteration is verified!
        # When verification is uncertain, OMIT the question rather than silently reverting to True!
        if altered_sent and swap_explanation:
            q_obj = TrueFalseQuestion(
                id=len(false_items) + 1,
                statement=altered_sent,
                correct_answer=False,
                source_sentence=sent,
                explanation=swap_explanation,
                altered=True,
                target_keyword=matched_kw or "Key Concept",
            )
            false_items.append(q_obj)
            used_sentences.add(sent)

    # Pass 2: Select verified True questions up to target_true
    for sent in valid_sentences:
        if len(true_items) >= target_true:
            break
        if sent in used_sentences:
            continue

        matched_kw = None
        for kw, _ in (keywords or []):
            pat = r"\b" + re.escape(kw.strip()) + r"\b"
            if re.search(pat, sent, flags=re.IGNORECASE):
                matched_kw = kw.strip()
                break

        q_obj = TrueFalseQuestion(
            id=len(true_items) + 1,
            statement=sent,
            correct_answer=True,
            source_sentence=sent,
            explanation=f"True. According to your notes: \"{sent}\"",
            altered=False,
            target_keyword=matched_kw or "Key Concept",
        )
        true_items.append(q_obj)
        used_sentences.add(sent)

    # Pass 3: If target_false or target_true were not met (e.g. notes are very short or lacked alterable sentences),
    # fill remaining slots up to n with available clean True sentences, keeping answers verified
    if len(false_items) + len(true_items) < n:
        for sent in valid_sentences:
            if len(false_items) + len(true_items) >= n:
                break
            if sent in used_sentences:
                continue
            matched_kw = None
            for kw, _ in (keywords or []):
                pat = r"\b" + re.escape(kw.strip()) + r"\b"
                if re.search(pat, sent, flags=re.IGNORECASE):
                    matched_kw = kw.strip()
                    break
            q_obj = TrueFalseQuestion(
                id=len(true_items) + 1,
                statement=sent,
                correct_answer=True,
                source_sentence=sent,
                explanation=f"True. According to your notes: \"{sent}\"",
                altered=False,
                target_keyword=matched_kw or "Key Concept",
            )
            true_items.append(q_obj)
            used_sentences.add(sent)

    # Interleave True and False items to balance the sequence
    combined: list[TrueFalseQuestion] = []
    t_idx = 0
    f_idx = 0
    pick_true = True
    while t_idx < len(true_items) or f_idx < len(false_items):
        if pick_true and t_idx < len(true_items):
            combined.append(true_items[t_idx])
            t_idx += 1
            if f_idx < len(false_items):
                pick_true = False
        elif f_idx < len(false_items):
            combined.append(false_items[f_idx])
            f_idx += 1
            if t_idx < len(true_items):
                pick_true = True
        elif t_idx < len(true_items):
            combined.append(true_items[t_idx])
            t_idx += 1

    # Re-index question IDs 1..len(combined)
    for idx, q in enumerate(combined):
        q.id = idx + 1

    return [q.to_dict() for q in combined]


def score_true_false(
    items: list[dict],
    user_answers: dict[int, bool],
) -> tuple[int, int]:
    """Score completed True/False quiz."""
    if not items:
        return 0, 0
    correct = 0
    total = len(items)
    for idx, item in enumerate(items):
        expected = item.get("correct_answer")
        user_choice = user_answers.get(idx)
        if user_choice is not None and user_choice == expected:
            correct += 1
    return correct, total


def evaluate_true_false_detailed(
    items: list[dict],
    user_answers: dict[int, bool],
) -> list[dict]:
    """Detailed item breakdown for post-submission True/False feedback."""
    results: list[dict] = []
    for idx, item in enumerate(items):
        user_choice = user_answers.get(idx)
        expected = item.get("correct_answer")
        is_correct = user_choice is not None and user_choice == expected
        results.append({
            "id": item.get("id", idx + 1),
            "statement": item.get("statement", ""),
            "user_answer": user_choice,
            "correct_answer": expected,
            "is_correct": is_correct,
            "is_answered": user_choice is not None,
            "source_sentence": item.get("source_sentence", ""),
            "explanation": item.get("explanation", ""),
            "target_keyword": item.get("target_keyword", ""),
        })
    return results


# ── Interactive Flashcard Generation ──────────────────────────────────────────
def generate_flashcards(
    sentences: list[str],
    keywords: Optional[list[tuple[str, float]]] = None,
    n: int = 10,
    raw_text: Optional[str] = None,
) -> list[dict]:
    """
    Generate note-grounded conceptual study flashcards.
    
    Structure:
      - Front: Term/concept title + category chip
      - Back: Grounded definition or core explanatory insight + source citation
    """
    if not sentences and raw_text:
        cleaned = clean_text(raw_text)
        sentences = split_sentences(cleaned)

    if not sentences:
        return []

    # Filter out syllabus headers and title banners
    valid_sentences = [s.strip() for s in sentences if is_study_sentence(s)]
    if not valid_sentences:
        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 4]

    full_text = raw_text if raw_text else " ".join(valid_sentences)

    if not keywords:
        keywords = tfidf_keywords(full_text, n=25) or frequency_keywords(full_text, n=25)

    flashcards: list[Flashcard] = []
    seen_terms: set[str] = set()

    # Step 1: Extract syntactically explicit definitions first
    for sent in valid_sentences:
        for pat in _DEF_PATTERNS:
            m = re.search(pat, sent, flags=re.IGNORECASE)
            if m:
                raw_term = m.group("term").strip()
                term = _clean_extracted_term(raw_term)
                def_part = _clean_definition_text(m.group("def"))
                stem = _get_stem(term)
                if stem not in seen_terms and len(term) >= 2:
                    seen_terms.add(stem)
                    clean_def = def_part[:1].upper() + def_part[1:] + "."
                    flashcards.append(Flashcard(
                        id=len(flashcards) + 1,
                        term=term.title(),
                        definition=clean_def,
                        source_sentence=sent,
                        category="Definition",
                        salience_score=1.0,
                    ))
                    break
        if len(flashcards) >= n:
            break

    # Step 2: Extract historical milestones, key authors, and recognized algorithms
    if len(flashcards) < n:
        def _format_milestone_item(m, s):
            author = m.group("author")
            item = m.group("item").strip().title()
            action = m.group("action").capitalize()
            after_item = s[m.end():]
            year_match = re.search(r"\b(19\d\d|20\d\d)\b", s)
            year_str = f" in {year_match.group(0)}" if year_match else ""

            purpose_clause = ""
            pm = re.search(r"\b(establishing|to|based\s+on)\s+([^.;]+)", after_item)
            if pm:
                kw_verb = pm.group(1).lower()
                predicate = pm.group(2).strip()
                if kw_verb == "establishing":
                    purpose_clause = f", establishing {predicate}"
                elif kw_verb == "to":
                    purpose_clause = f" to {predicate}"
                elif kw_verb.startswith("based"):
                    purpose_clause = f", based on {predicate}"

            return item, f"{action} by {author}{year_str}{purpose_clause}.", "Historical Milestone"

        _CONTRIB_PATTERNS = [
            (
                r"(?:when\s+|in\s+\d{4},\s*)?(?P<author>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?P<action>proposed|introduced|developed|published|created|invented|formulated)\s+(?:the\s+)?(?P<item>[A-Za-z0-9_\s/-]+?)(?:\s+in\s+|\s+at\s+|\s+to\s+|$)(?P<rest>.*)",
                _format_milestone_item
            ),
            (
                r"(?P<item>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+widely\s+used\s+in\s+(?P<usage>[^,]+),\s*(?P<note>.*)",
                lambda m, s: (
                    m.group("item").strip().title(),
                    f"Widely used in {m.group('usage')}; {m.group('note').strip().rstrip('.')}.",
                    "Key Algorithm"
                )
            ),
            (
                r"(?:in\s+\d{4},\s*)?(?:researchers\s+at\s+)?(?P<org>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:in\s+[A-Za-z\s]+,\s*)?published\s+(?:the\s+)?(?P<item>[A-Za-z0-9_\s/-]+?)\s+(?:based\s+on|to|for)\s+(?P<desc>.*)",
                lambda m, s: (
                    m.group("item").strip().title(),
                    f"Introduced by {m.group('org')}; based on {m.group('desc').strip().rstrip('.')}.",
                    "Key Architecture"
                )
            ),
        ]

        for sent in valid_sentences:
            for pat, handler in _CONTRIB_PATTERNS:
                m = re.search(pat, sent)
                if m:
                    t, d, c = handler(m, sent)
                    t_stem = _get_stem(t)
                    if t_stem not in seen_terms and len(t) >= 2:
                        seen_terms.add(t_stem)
                        flashcards.append(Flashcard(
                            id=len(flashcards) + 1,
                            term=t,
                            definition=d,
                            source_sentence=sent,
                            category=c,
                            salience_score=0.9,
                        ))
                    break
            if len(flashcards) >= n:
                break

    # Step 3: Extract top keyword concepts from key sentences
    if len(flashcards) < n and keywords:
        multiword_candidates = [
            phrase.title() for phrase, _ in rake_keywords(full_text, n=15)
            if len(phrase.split()) >= 2
        ]

        for kw, score in keywords:
            kw_clean = kw.strip()
            kw_stem = _get_stem(kw_clean)
            if kw_stem in seen_terms or len(kw_clean) < 2:
                continue

            display_term = kw_clean.title()
            for mw in multiword_candidates:
                if kw_clean.lower() in mw.lower() and _get_stem(mw) not in seen_terms:
                    display_term = mw
                    break

            best_sent = None
            pat = r"\b" + re.escape(kw_clean) + r"\b"
            for sent in valid_sentences:
                if re.search(pat, sent, flags=re.IGNORECASE):
                    best_sent = sent
                    break

            if best_sent:
                seen_terms.add(_get_stem(display_term))
                # Extract explanatory predicate clause where available
                exp_text = best_sent
                for verb_pat in [r"\b(?:measures|evaluates|identifies|transforms|reduces|selects|divides|preserves)\s+(.+)"]:
                    vm = re.search(verb_pat, best_sent, flags=re.IGNORECASE)
                    if vm:
                        clause = vm.group(0).strip().rstrip(".")
                        exp_text = clause[:1].upper() + clause[1:] + "."
                        break

                flashcards.append(Flashcard(
                    id=len(flashcards) + 1,
                    term=display_term,
                    definition=exp_text,
                    source_sentence=best_sent,
                    category="Key Concept",
                    salience_score=round(float(score), 4),
                ))

            if len(flashcards) >= n:
                break

    return [f.to_dict() for f in flashcards]
