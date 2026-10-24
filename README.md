# 📚 Student Notes Summarizer & Quiz Generator

> An intelligent, 100% offline NLP academic toolkit that transforms study notes into extractive summaries, keyword importance charts, grounded multiple-choice quizzes, linguistic analytics, and multi-format revision guides — built with Python, Streamlit, NLTK, spaCy, and scikit-learn.

---

## 🚀 Implemented Modules & Features

| Module | Features & Implemented Algorithms | Status |
|---|---|---|
| **01. Upload Notes** | • Multi-format parser for `.txt`, `.pdf` (PyMuPDF), `.docx` (python-docx), `.pptx` (python-pptx)<br>• Text cleaning, punctuation removal, case normalization<br>• NLTK sentence/word tokenization, Porter stemming, spaCy lemmatization<br>• Automatic stale-state clearing when new documents are loaded | ✅ Complete |
| **02. Summarization** | • **TF-IDF Sentence Scoring**: Treats sentences as mini-documents to score term salience<br>• **Normalized Word Frequency (Luhn)**: Scores sentences by content word density<br>• Length controls by percentage ratio (10%–80%) or exact sentence count<br>• Original document order restoration & 100% verbatim sentence preservation<br>• Sentence score breakdown table & ROUGE-1 / ROUGE-2 / ROUGE-L overlap evaluation | ✅ Complete |
| **03. Keywords & Keyphrases** | • **TF-IDF Keyword Extraction**: Unigrams and bigrams weighted against sentence corpus<br>• **Normalized Frequency Ranking**: Raw content word occurrence density<br>• **RAKE Keyphrases**: Rapid Automatic Keyword Extraction for multi-word phrases<br>• **Separate spaCy NER**: Real-world entity detection (PERSON, ORG, GPE, DATE) | ✅ Complete |
| **04. Quiz Generator** | • **Note-Grounded MCQs**: Questions created exclusively from source notes<br>• **Fill-in-the-Blank**: Key concept blanking with word boundaries (`_______`)<br>• **Definition-Based**: Syntactic definition recognition (`X is the process of...`)<br>• **4 Unique Options**: 1 correct answer + 3 distinct distractors from document vocabulary<br>• **Distractor Quality Filters**: Stem uniqueness, substring exclusion, case matching<br>• **Zero Answer Leakage**: Radio buttons unselected; answers revealed only after submission<br>• Post-submission review with accuracy score, explanations, and verbatim source citations | ✅ Complete |
| **05. Revision Sheet** | • **Consolidated Study Guide**: Overview stats, summary, keywords, entities, practice quiz<br>• **Multi-Page ReportLab PDF**: 2-pass dynamic page numbering (`Page X of Y`), running headers, formatted tables, and page-split protection (`KeepTogether`)<br>• **GitHub Markdown (.md)** and **Plain Text (.txt)** multi-format exports<br>• Quick-generation buttons for missing components | ✅ Complete |
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
│   ├── 04_quiz.py              # Note-grounded MCQ quiz & post-submission feedback
│   ├── 05_revision.py          # Consolidated study guide with ReportLab PDF export
│   └── 06_nlp_analysis.py      # POS tagging, readability, wordcloud & distributions
│
├── core/                       # Pure NLP algorithms & business logic
│   ├── __init__.py
│   ├── text_processor.py       # Cleaning, tokenization, stemming, lemmatization
│   ├── summarizer.py           # Luhn frequency & TF-IDF sentence scoring, ROUGE
│   ├── keyword_extractor.py    # TF-IDF, Normalized Frequency, RAKE, spaCy entities
│   ├── quiz_generator.py       # Note-grounded MCQ generation & distractor validation
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
│   └── style.css               # Custom responsive CSS design tokens
│
└── tests/                      # Automated test suite (143 tests passing)
    ├── __init__.py
    ├── test_phase2.py          # File parsing & preprocessing tests (58 tests)
    ├── test_phase3.py          # Summarization & keyword extraction tests (38 tests)
    ├── test_phase4.py          # MCQ quiz & distractor validation tests (20 tests)
    ├── test_phase5.py          # Revision builder & PDF/MD/TXT export tests (13 tests)
    └── test_nlp_analysis.py    # POS tagging, readability & E2E integration tests (14 tests)
```

---

## 🧪 Running the Automated Test Suite

To execute all 143 tests across the entire application:

```powershell
python -m pytest tests/ -v
```

Expected output:
```
============================ 143 passed in 12.20s =============================
```

---

## 🎓 Undergraduate NLP Viva / Oral Exam Guide

This section explains the core design decisions and algorithmic mechanics for an undergraduate viva:

### 1. Why Extractive Summarization instead of Transformer Abstractive Models?
- **Academic Grounding & Zero Hallucination**: In educational note revision, factual accuracy is paramount. Pre-trained abstractive models (BART/T5/LLMs) frequently hallucinate facts or conflate numbers not present in the lecture notes. Extractive summarization guarantees 100% precision because every sentence is directly quoted from the student's text.
- **Explainability**: Every sentence in our summarizer has a mathematical score derived from term frequencies or TF-IDF weights. Students and examiners can inspect exactly why sentence $A$ was selected over sentence $B$.
- **Resource Efficiency**: Runs 100% locally and offline in seconds without requiring GPU hardware or expensive cloud APIs.

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

---

## ⚠️ Known Limitations & Edge Cases

1. **Very Short Documents (< 4 sentences)**: Documents with very sparse vocabulary may yield fewer questions (or none) because the quiz generator strictly requires 3 distinct distractors from the notes and refuses to fabricate facts.
2. **Scanned / Image-Only PDFs**: PyMuPDF extracts text layers. If a PDF contains only scanned raster images without an embedded OCR text layer, a `ParseError` is raised with a clear prompt to provide searchable documents.
3. **Informal / Fragmented Notes**: Definition-based questions require standard grammatical copulas (`is defined as`, `refers to`). Unstructured bullet points automatically fall back to fill-in-the-blank questions.

---

## 📄 License
MIT License — built for educational and academic natural language processing coursework.
