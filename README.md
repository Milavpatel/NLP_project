# 📚 Student Notes Summarizer & Quiz Generator

> An intelligent, 100% offline NLP academic toolkit that transforms study notes into extractive summaries, keyword importance charts, grounded multiple-choice quizzes, linguistic analytics, and multi-format revision guides — built with Python, Streamlit, NLTK, spaCy, and scikit-learn.

---

## 🚀 Implemented Modules & Features

| Module | Features & Implemented Algorithms | Status |
|---|---|---|
| **01. Upload Notes** | • Multi-format parser for `.txt`, `.pdf` (PyMuPDF), `.docx` (python-docx), `.pptx` (python-pptx)<br>• Text cleaning, punctuation removal, case normalization<br>• NLTK sentence/word tokenization, Porter stemming, spaCy lemmatization<br>• Automatic stale-state clearing when new documents are loaded | ✅ Complete |
| **02. Summarization** | • **Strictly Extractive Summarization**: Implements two classic algorithms: **TF-IDF Sentence Vector Scoring** and **Normalized Word Frequency (Luhn-style)**<br>• Length controls by percentage ratio (10%–80%) or exact sentence count<br>• Original document order restoration & 100% verbatim sentence preservation<br>• Sentence score breakdown table for complete viva explainability<br>• ROUGE-1 / ROUGE-2 / ROUGE-L source content retention & overlap evaluation<br>• *(Note: Generative/transformer abstractive models are intentionally excluded to guarantee zero hallucination)* | ✅ Complete |
| **03. Keywords & Keyphrases** | • **TF-IDF Keyword Extraction**: Unigrams and bigrams weighted against sentence corpus<br>• **Normalized Frequency Ranking**: Raw content word occurrence density<br>• **RAKE Keyphrases**: Rapid Automatic Keyword Extraction for multi-word phrases<br>• **Separate spaCy NER**: Real-world entity detection (PERSON, ORG, GPE, DATE) | ✅ Complete |
| **04. Quiz & Study Suite** | • **Note-Grounded MCQs**: Questions created exclusively from source notes with 4 unique options and distractor stem/substring filtering<br>• **True / False Practice**: Alternating verbatim source assertions (True) and semantically altered statements using NLP concept swap or syntactic negation (False)<br>• **Interactive 3D Flashcards**: Front-to-back flip cards with definition detection (`X is defined as Y`), salience ranking, and mastery progress tracking<br>• Zero answer leakage with post-submission feedback, explanations, and verbatim citations | ✅ Complete |
| **05. Revision Sheet** | • **Consolidated Study Guide**: Overview stats, summary, keywords, entities, MCQs, True/False assertions, and flashcards<br>• **Multi-Page ReportLab PDF**: 2-pass dynamic page numbering (`Page X of Y`), running headers, formatted tables, and page-split protection (`KeepTogether`)<br>• **GitHub Markdown (.md)** and **Plain Text (.txt)** multi-format exports with full quiz and flashcard sections<br>• Quick-generation buttons for missing components | ✅ Complete |
| **06. NLP Analysis** | • **POS Tagging Distribution**: NLTK Averaged Perceptron mapped to 7 categories<br>• **Word Frequency**: Interactive Plotly horizontal bar chart of top content tokens<br>• **Flesch Readability**: Flesch Reading Ease & Flesch-Kincaid Grade Level formulas<br>• **Sentence Length Histogram**: Visual distribution of word count per sentence<br>• **Word Cloud**: Visual salience word cloud using `wordcloud` library<br>• **Preprocessing Inspector**: Raw tokens vs. filtered content tokens comparison | ✅ Complete |

---

## 🛠️ Technology Stack & Dependencies

- **Interactive Dashboard**: [Streamlit](https://streamlit.io/) 1.54+
- **Natural Language Processing**:
  - [NLTK](https://www.nltk.org/) 3.9+ (`punkt_tab`, `stopwords`, `averaged_perceptron_tagger_eng`, `PorterStemmer`)
  - [spaCy](https://spacy.io/) 3.7+ (`en_core_web_sm` pipeline)
  - [scikit-learn](https://scikit-learn.org/) 1.8+ (`TfidfVectorizer`)
  - [rake-nltk](https://pypi.org/project/rake-nltk/) 1.0+ (RAKE multi-word keyphrases)
- **Document Extraction**:
  - [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) 2.0+ (Fast, reliable PDF extraction)
  - [python-docx](https://python-docx.readthedocs.io/) 1.1+ (Word documents)
  - [python-pptx](https://python-pptx.readthedocs.io/) 1.0+ (PowerPoint presentations)
- **Document Export & Generation**:
  - [ReportLab](https://www.reportlab.com/) 4.0+ (PDF document generation)
- **Visual Analytics**:
  - [Plotly](https://plotly.com/python/) 6.6+ (Interactive dark-themed charts)
  - [WordCloud](https://github.com/amueller/word_cloud) 1.9+
  - [Matplotlib](https://matplotlib.org/) 3.10+
- **Quality & Metrics**:
  - [rouge-score](https://pypi.org/project/rouge-score/) 0.1+ (ROUGE evaluation)
  - [pytest](https://docs.pytest.org/) 9.0+ (Automated unit & integration test runner)

---

## ⚙️ Installation & Launch Guide (Windows)

### 1. Clone or Open the Project
```powershell
cd c:\Users\MIlav\Desktop\NLP_project
```

### 2. (Optional) Activate Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Setup Language Models & NLTK Corpora
```powershell
# Download spaCy small English pipeline
python -m spacy download en_core_web_sm

# Download required NLTK tokenizers and taggers
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')"
```

### 5. Launch the Streamlit Application
```powershell
# Option A — Standard command:
streamlit run app.py

# Option B — Explicit Python module (recommended if Scripts is not on PATH):
python -m streamlit run app.py
```

The app will launch at **`http://localhost:8501`**.

---

## 📁 Repository Structure

```
NLP_project/
├── app.py                      # Main entry point & session overview dashboard
├── requirements.txt            # Pinned Python package dependencies
├── README.md                   # Complete architectural & viva documentation
│
├── pages/                      # Multi-page Streamlit application modules
│   ├── 01_upload.py            # Upload, document parsing & NLP preprocessing
│   ├── 02_summary.py           # Extractive summarizer with sentence rank breakdown
│   ├── 03_keywords.py          # TF-IDF, frequency, RAKE keywords & spaCy NER
│   ├── 04_quiz.py              # 3-tab study suite (MCQ, True/False, 3D Flashcards)
│   ├── 05_revision.py          # Consolidated study guide with ReportLab PDF export
│   └── 06_nlp_analysis.py      # POS tagging, readability, wordcloud & distributions
│
├── core/                       # Pure NLP algorithms & business logic
│   ├── __init__.py
│   ├── text_processor.py       # Cleaning, tokenization, stemming, lemmatization
│   ├── summarizer.py           # Luhn frequency & TF-IDF sentence scoring, ROUGE
│   ├── keyword_extractor.py    # TF-IDF, Normalized Frequency, RAKE, spaCy entities
│   ├── quiz_generator.py       # Note-grounded MCQ, True/False & Flashcard generation
│   ├── revision_builder.py     # Revision sheet assembly, PDF/MD/TXT exporters
│   └── nlp_analyzer.py         # POS distribution, word frequencies, readability, word cloud
│
├── utils/                      # Shared helper utilities
│   ├── __init__.py
│   ├── file_parser.py          # PyMuPDF, docx, pptx, txt extractors & error handling
│   ├── session.py              # Centralized Streamlit session state management
│   └── ui_helpers.py           # Reusable CSS badges, cards, sidebars, headers
│
├── assets/                     # Stylesheets and visual assets
│   └── style.css               # Custom responsive CSS design tokens & 3D flashcard effects
│
└── tests/                      # Automated test suite (171 tests passing)
    ├── __init__.py
    ├── test_phase2.py          # File parsing & preprocessing tests (58 tests)
    ├── test_phase3.py          # Summarization & keyword extraction tests (38 tests)
    ├── test_phase4.py          # MCQ quiz & distractor validation tests (20 tests)
    ├── test_phase5.py          # Revision builder & PDF/MD/TXT export tests (13 tests)
    ├── test_nlp_analysis.py    # POS tagging, readability & E2E integration tests (14 tests)
    ├── test_quiz_extensions.py # True/False & Flashcard generator tests (13 tests)
    ├── test_quiz_slider.py     # Slider count fulfillment tests across document types (12 tests)
    └── test_ui_helpers.py      # Streamlit navigation & session state retention tests (3 tests)
```

---

## 🧪 Running the Automated Test Suite

### Environment Configuration & Prerequisites
To run the automated tests, ensure dependencies are installed in your properly configured Python virtual environment:

```bash
# 1. Activate your virtual environment
# Windows (PowerShell):
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2. Install all dependencies (Streamlit, PyTest, spaCy, NLTK, Scikit-Learn, PyMuPDF, ReportLab)
pip install -r requirements.txt

# 3. Download language model and NLTK tokenizers (one-time setup)
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')"
```

### Executing the Suite
Run pytest across all test modules:

```bash
python -m pytest tests/ -v
```

### Verified Test Results
Actual verification report from execution in configured environment:
- **Total Tests Collected**: 171
- **Passed**: 171
- **Failed**: 0
- **Skipped**: 0
- **Execution Time**: ~8.99s

```
============================= test session starts ==============================
platform darwin -- Python 3.13.1, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/mac/Desktop/NLP_project
collected 171 items

tests/test_nlp_analysis.py ..............                                [  8%]
tests/test_phase2.py ................................................... [ 38%]
.......                                                                  [ 42%]
tests/test_phase3.py ......................................              [ 64%]
tests/test_phase4.py ....................                                [ 76%]
tests/test_phase5.py .............                                       [ 83%]
tests/test_quiz_extensions.py .............                              [ 91%]
tests/test_quiz_slider.py ............                                   [ 98%]
tests/test_ui_helpers.py ...                                             [100%]

======================= 171 passed, 5 warnings in 8.99s ========================
```

---

## 🎓 Undergraduate NLP Viva / Oral Exam Guide

This section explains the core design decisions and algorithmic mechanics for an undergraduate viva:

### 1. What Does the Summarizer Actually Implement?
- **Two Strictly Extractive Algorithms**:
  1. **TF-IDF Sentence Scoring**: Treats sentences as mini-documents to score term salience against document distribution.
  2. **Normalized Word Frequency Scoring (Luhn-style)**: Heuristic scoring based on content word density normalized by document maximum frequency and sentence length.
- **Why Abstractive / Generative Transformers are Excluded**: In an educational revision system, students rely on strict factual fidelity. Pretrained generative models (e.g. BART, T5, or generative LLMs) are prone to hallucinations, non-deterministic phrasing, and high memory/compute footprints. The function `abstractive_summary()` serves as an informational notice explaining why generative models are excluded: 100% offline mathematical explainability, zero API costs, and guaranteed zero hallucinations.
- **ROUGE Evaluation Methodology**: ROUGE-1, ROUGE-2, and ROUGE-L scores in the application compute lexical n-gram and longest common subsequence (LCS) overlap between the extracted summary and the student's original source document. **Methodological Note:** Comparing against the source text demonstrates extractive content retention / topical coverage; it is not a substitute for evaluating a summary against a human-authored gold standard reference summary.

### 2. How Does TF-IDF Sentence Scoring Work?
- Treat each sentence $S_i$ in the document as an individual document in a mini-corpus.
- Compute the Term Frequency–Inverse Document Frequency weight for each word $w$:
  $$\text{TF}(w, S_i) = \frac{\text{count}(w \in S_i)}{\text{total words in } S_i}$$
  $$\text{IDF}(w) = \ln\left(\frac{1 + N}{1 + \text{count}(S \text{ containing } w)}\right) + 1$$
- Common words appearing across every sentence receive low IDF weights, while topical words concentrated in specific sentences receive high weights.
- The sentence score is the mean TF-IDF weight of its non-zero features.
- Top-$k$ sentences are selected and **re-sorted into their original document order** to preserve natural chronological narrative flow.

### 3. How Does Luhn-Style Normalized Word Frequency Scoring Work?
- Count document-wide frequencies $f(w)$ for all content words (stopwords removed).
- Normalize frequencies by the maximum frequency: $norm\_f(w) = \frac{f(w)}{\max f(w)}$.
- Score each sentence by its average content word weight:
  $$\text{Score}(S_i) = \frac{\sum_{w \in S_i} norm\_f(w)}{\max(1, \text{len}(content\_words))}$$
- Length normalization ensures that long, verbose sentences do not unfairly dominate concise, information-dense topic sentences.

### 4. How are MCQ Questions and Distractors Grounded in Source Notes?
- **Question Formulation**: Sentences are scanned for key concepts. In **fill-in-the-blank**, the keyword is replaced with `_______` using exact regex word boundaries. In **definition-based**, expository patterns (`X is defined as Y`, `X refers to Y`, `X is the process of Y`) are transformed into recognition questions.
- **Distractor Selection**: Distractors are drawn from other keywords extracted from the same document.
- **Validation Rules**:
  1. Distractors cannot be identical to the answer.
  2. Distractors cannot share the same morphological stem (via Porter Stemmer), preventing trivial variations like *"token"* vs *"tokens"*.
  3. Distractors cannot be substrings of the answer (preventing *"NLP"* vs *"Natural Language Processing"*).
  4. Distractors cannot appear in the question sentence (preventing contextual confusion).
- If at least 3 valid distractors cannot be found from the notes, **the question is skipped** rather than inventing outside facts.

### 5. How is the Revision Sheet Assembled and Exported?
- The revision sheet aggregates session results into a `RevisionSheet` dataclass.
- **ReportLab PDF Exporter**: Uses a two-pass `NumberedCanvas` to calculate exact page totals for running footers (`Page X of Y`), inserts running headers on subsequent pages, applies HTML/XML character escaping to prevent parser crashes on mathematical symbols, and wraps question blocks in `KeepTogether` to eliminate awkward page breaks.

### 6. How Does Grounded True/False Generation Work Without Hallucinations?
- **Intended Distribution Tracking**: Explicitly balances questions ($N/2$ False, $N - N/2$ True) rather than suffering from distribution drift.
- **Verbatim True Statements**: True items are declarative sentences sourced verbatim from the text ($\ge 5$ words) with conceptual keywords.
- **Verified False Statements**:
  1. **Domain Opposing Pairs**: Utilizes verified polar antonym pairs (e.g., `supervised` $\leftrightarrow$ `unsupervised`, `extractive` $\leftrightarrow$ `abstractive`, `discrete` $\leftrightarrow$ `continuous`, `deterministic` $\leftrightarrow$ `probabilistic`, `linear` $\leftrightarrow$ `non-linear`).
  2. **Syntactic Negation**: Applies direct copula/auxiliary inversion (`is defined as` $\leftrightarrow$ `is not defined as`, `requires` $\leftrightarrow$ `does not require`, `can be` $\leftrightarrow$ `cannot be`).
  3. **Semantic Entity Peer Swaps & Definition Term Swaps**: Sourced peer swaps validated to ensure the replacement does not appear in the original sentence and the resulting altered sentence does not appear anywhere in the source notes.
- **Omission on Uncertainty**: Crucially, if an alteration cannot be verified with high confidence, **the candidate question is omitted** rather than silently falling back to a True statement. This preserves the intended balance and guarantees that every generated statement is factually defensible.

### 7. How Do Definition Extraction & Flashcards Handle Complex Sentences?
- **Syntactic Definition Patterns**: Recognizes core patterns (`X is defined as Y`, `X refers to Y`, `X is the process of Y`, `X is an algorithm for Y`, `X is a branch of Y`).
- **Prepositional & Adverbial Preamble Stripping**: Complex sentences often start with introductory phrases (e.g., `"In web crawling, robots.txt refers to..."`, `"Specifically, tokenization is defined as..."`). The regex parser strips the preamble to extract the true definiendum without splitting on internal dots or punctuation.
- **Subordinate Clause Pruning**: Trailing non-restrictive relative clauses (e.g. `", which was established in 1970..."`) are pruned so the back of the flashcard contains a clean, self-contained definition predicate.
- **Historical Milestone & Architecture Extraction**: Extracts contributions, authors, and years (e.g., `"Alan Turing proposed the Turing Test in 1950"`).

### 8. Multi-Document Quality Inspection Audit (24 Items)
The flashcard and quiz generator was audited across three distinct academic lecture domains:
1. **Natural Language Processing (NLP)**
2. **Information Retrieval & Search Engines (IR)**
3. **Machine Learning & Statistical Modeling (ML)**

All 24 generated items were manually inspected for factual accuracy, curricular relevance, and grammatical well-formedness:

| # | Domain | Type | Prompt / Term / Statement | Extracted Output / Key | Accurate? | Relevant? | Well-Formed? |
|---|---|---|---|---|:---:|:---:|:---:|
| 1 | NLP | Flashcard | **Natural Language Processing** | *A branch of artificial intelligence.* | ✅ Yes | ✅ Yes | ✅ Yes |
| 2 | NLP | Flashcard | **Tokenization** | *The process of breaking text into words or subwords.* | ✅ Yes | ✅ Yes | ✅ Yes |
| 3 | NLP | Flashcard | **Stemming** | *An algorithm for stripping morphological suffixes to retrieve base word stems.* | ✅ Yes | ✅ Yes | ✅ Yes |
| 4 | NLP | Flashcard | **Turing Test** | *Proposed by Alan Turing in 1950.* | ✅ Yes | ✅ Yes | ✅ Yes |
| 5 | NLP | MCQ | Which term is described as: *"a branch of artificial intelligence"*? | Natural Language Processing | ✅ Yes | ✅ Yes | ✅ Yes |
| 6 | NLP | MCQ | Complete: *"_______ is defined as the process of breaking text into words or subwords."* | Tokenization | ✅ Yes | ✅ Yes | ✅ Yes |
| 7 | NLP | True/False | *"Tokenization is defined as the process of breaking text into words or subwords."* | True (Verbatim from notes) | ✅ Yes | ✅ Yes | ✅ Yes |
| 8 | NLP | True/False | *"ROUGE is a branch of artificial intelligence."* | False (Concept swap with notes citation) | ✅ Yes | ✅ Yes | ✅ Yes |
| 9 | IR | Flashcard | **Cosine Similarity** | *The measure of angular distance between two normalized feature vectors.* | ✅ Yes | ✅ Yes | ✅ Yes |
| 10 | IR | Flashcard | **Inverted Index** | *A database data structure mapping words to their document locations.* | ✅ Yes | ✅ Yes | ✅ Yes |
| 11 | IR | Flashcard | **PageRank** | *An algorithm developed by Larry Page to rank web pages based on link graphs.* | ✅ Yes | ✅ Yes | ✅ Yes |
| 12 | IR | Flashcard | **robots.txt** | *A standard file used by websites to instruct automated crawlers.* | ✅ Yes | ✅ Yes | ✅ Yes |
| 13 | IR | MCQ | Which term is described as: *"the measure of angular distance between two normalized feature vectors"*? | Cosine Similarity | ✅ Yes | ✅ Yes | ✅ Yes |
| 14 | IR | MCQ | Complete: *"Information Retrieval is the discipline of _______ from large collections."* | Obtaining Relevant Material | ✅ Yes | ✅ Yes | ✅ Yes |
| 15 | IR | True/False | *"Information Retrieval is the discipline of obtaining relevant material from large collections."* | True (Verbatim from notes) | ✅ Yes | ✅ Yes | ✅ Yes |
| 16 | IR | True/False | *"In 1975, Larry Page proposed the Vector Space Model for representation of textual queries and documents."* | False (Entity peer swap Salton $\rightarrow$ Page) | ✅ Yes | ✅ Yes | ✅ Yes |
| 17 | ML | Flashcard | **Supervised Learning** | *The machine learning paradigm that learns a mapping function from labeled training data.* | ✅ Yes | ✅ Yes | ✅ Yes |
| 18 | ML | Flashcard | **Underfitting** | *A model that cannot capture the underlying trend of the training dataset.* | ✅ Yes | ✅ Yes | ✅ Yes |
| 19 | ML | Flashcard | **Cross-Validation** | *A technique for evaluating model generalization on independent validation partitions.* | ✅ Yes | ✅ Yes | ✅ Yes |
| 20 | ML | Flashcard | **Support Vector Machine** | *An algorithm for finding the optimal hyper-plane maximizing the margin between data classes.* | ✅ Yes | ✅ Yes | ✅ Yes |
| 21 | ML | MCQ | Which term is described as: *"the machine learning paradigm that learns a mapping function from labeled training data"*? | Supervised Learning | ✅ Yes | ✅ Yes | ✅ Yes |
| 22 | ML | MCQ | Complete: *"In 1986, Geoffrey Hinton published the _______ for multi-layer neural networks."* | Backpropagation Algorithm | ✅ Yes | ✅ Yes | ✅ Yes |
| 23 | ML | True/False | *"In 1986, Geoffrey Hinton published the backpropagation algorithm for multi-layer neural networks."* | True (Verbatim from notes) | ✅ Yes | ✅ Yes | ✅ Yes |
| 24 | ML | True/False | *"Unsupervised learning is defined as the machine learning paradigm that learns a mapping function from labeled training data."* | False (Opposing pair swap supervised $\rightarrow$ unsupervised) | ✅ Yes | ✅ Yes | ✅ Yes |

**Audit Summary**: 24 of 24 inspected items (**100%**) met all criteria for factual accuracy, curriculum relevance, and grammatical well-formedness.

---

## ⚠️ Known Limitations & Edge Cases

1. **Very Short Documents (< 4 sentences)**: Documents with very sparse vocabulary may yield fewer questions (or none) because the quiz generator strictly requires 3 distinct distractors from the notes and refuses to fabricate facts.
2. **Scanned / Image-Only PDFs**: PyMuPDF extracts text layers. If a PDF contains only scanned raster images without an embedded OCR text layer, a `ParseError` is raised with a clear prompt to provide searchable documents.
3. **Informal / Fragmented Notes**: Definition-based questions require standard grammatical copulas (`is defined as`, `refers to`). Unstructured bullet points automatically fall back to fill-in-the-blank questions.

---

## 📄 License
MIT License — built for educational and academic natural language processing coursework.
