# Agentic Document Extraction & Summarization

A GCP-native prototype that ingests a news corpus, extracts structured entities, generates multi-style summaries, and exposes these capabilities to an autonomous AI agent that answers natural-language analytical questions with cited sources.

**Stack:** Vertex AI (Gemini 1.5 Pro) · Cloud Natural Language API · BigQuery · Cloud Storage · Firestore · Python 3.11+

---

## Table of Contents

1. [GCP Setup](#1-gcp-setup)
2. [Local Setup](#2-local-setup)
3. [Running the Pipeline](#3-running-the-pipeline)
4. [Agent Demo](#4-agent-demo)
5. [Development](#5-development)
6. [Project Structure](#6-project-structure)

---

## 1. GCP Setup

Complete these steps once before running anything locally.

### 1.1 Create or select a GCP project

Go to [console.cloud.google.com](https://console.cloud.google.com), create a project (e.g. `agentic-nlp-demo`), and enable billing.

### 1.2 Enable required APIs

In **APIs & Services → Enable APIs**, enable the following (or run the `gcloud` command below):

| API | Purpose |
|---|---|
| Vertex AI API | Gemini models, embeddings |
| Cloud Natural Language API | Entity extraction, sentiment |
| Cloud Storage API | Raw data landing zone |
| BigQuery API | Structured document/entity/summary store |
| Cloud Firestore API | Agent session memory and doc cache |
| Cloud Build API | CI/CD (productionization) |
| Cloud Run API | Agent serving (productionization) |
| Cloud Pub/Sub API | Event-driven ingestion (productionization) |

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  language.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  firestore.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  pubsub.googleapis.com \
  --project=YOUR_PROJECT_ID
```

### 1.3 Create a service account

In **IAM & Admin → Service Accounts**, create `agentic-nlp-sa` and grant it these roles:

| Role | Why |
|---|---|
| Storage Object Admin | Read/write GCS bucket |
| BigQuery Data Editor | Read/write BQ tables |
| BigQuery Job User | Run BQ queries |
| Vertex AI User | Gemini, embeddings, Vertex AI APIs |
| Cloud Datastore User | Read/write Firestore |

> **Note:** Service account key downloads are blocked in many organisations. This project uses Application Default Credentials — no JSON key file needed (see §1.5).

### 1.4 Create GCP resources

**GCS bucket** (Console: Cloud Storage → Create, or):
```bash
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-nlp-corpus/
```

**BigQuery dataset** (Console: BigQuery → Create dataset, or):
```bash
bq --location=us-central1 mk --dataset YOUR_PROJECT_ID:nlp_demo
```

**Firestore** (Console: Firestore → Create database):
- Mode: **Native**
- Location: `us-central1`

> All three resources must be in the same region (`us-central1`) to avoid cross-region egress charges.

### 1.5 Authenticate locally

This project uses **Application Default Credentials (ADC)** — no service account key file required:

```bash
# Authenticate your personal Google account
gcloud auth login

# Set your active project
gcloud config set project YOUR_PROJECT_ID

# Generate ADC credentials used by Google client libraries
gcloud auth application-default login

# Set the quota project (required for Vertex AI billing attribution)
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

To run as the service account without a key file (optional, for testing exact SA permissions):

```bash
gcloud auth application-default login \
  --impersonate-service-account=agentic-nlp-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

---

## 2. Local Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd agentic-document-extraction-and-summarization

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Install spaCy language model (used by the optional baseline extractor)
python -m spacy download en_core_web_sm

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

Or use the Makefile shortcut:
```bash
make setup
```

### Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your real values:

```env
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GCS_BUCKET=your-project-id-nlp-corpus
BQ_DATASET=nlp_demo

# Authentication is handled by ADC — no key file needed.
# Run once: gcloud auth application-default login
```

> `.env` is git-ignored. Never commit it. Never commit `*.json` key files.

---

## 3. Running the Pipeline

All commands assume the virtualenv is active and you are in the repo root.
Run scripts in order — each step depends on the previous one.

### Step 1 — Ingest & Preprocess

Downloads [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) from HuggingFace, uploads raw JSONL to GCS, loads into BigQuery, then cleans and deduplicates.

```bash
python -m scripts.01_ingest_data              # default: 1500 docs
python -m scripts.01_ingest_data --limit 200  # quick smoke test
```

**What happens:**
1. HuggingFace → `gs://<GCS_BUCKET>/raw/documents.jsonl`
2. GCS → BigQuery `nlp_demo.documents` (raw)
3. Clean (English only, 100–5000 tokens, dedupe by SHA-256) → BigQuery `nlp_demo.documents_clean`

**Output:**
```
Ingested  : 1,500 documents
After clean: 1,423 documents
Dropped   : 77 documents
```

---

### Step 2 — Extract Entities

Runs Cloud Natural Language API + Gemini structured extraction on cleaned documents in parallel. Optionally runs spaCy as a baseline.

```bash
python -m scripts.02_run_extraction                          # default: 500 docs, 4 workers
python -m scripts.02_run_extraction --limit 200 --workers 4  # faster run
python -m scripts.02_run_extraction --limit 500 --skip-spacy # skip spaCy baseline
```

| Flag | Default | Description |
|---|---|---|
| `--limit` | 500 | Max documents to process |
| `--workers` | 4 | ThreadPoolExecutor parallelism (max 8 recommended) |
| `--skip-spacy` | False | Skip the optional spaCy NER baseline |

**What happens:**
- Cloud NL API → entities + salience + sentiment + IAB categories → BigQuery `nlp_demo.entities`
- Gemini 1.5 Pro → `core_issue`, `actors`, `key_metrics`, `dates`, `sentiment_label` → BigQuery `nlp_demo.extractions_gemini`
- (Optional) spaCy `en_core_web_sm` → NER baseline rows appended to `nlp_demo.entities`

---

### Step 3 — Summarize

Generates TL;DR, bullet points, and abstract summaries for each document using Gemini 1.5 Pro.

```bash
python -m scripts.03_run_summarization                          # default: 500 docs, 4 workers
python -m scripts.03_run_summarization --limit 200 --workers 4
```

| Flag | Default | Description |
|---|---|---|
| `--limit` | 500 | Max documents to summarize |
| `--workers` | 4 | ThreadPoolExecutor parallelism |

**What happens:**
- Gemini 1.5 Pro produces `tldr` (1 sentence), `bullets` (3 points), `abstract` (75 words) per doc
- Long documents (>16k chars) go through map-reduce: chunk → summarize each → reduce
- Results → BigQuery `nlp_demo.summaries`

---

### Step 4 — Evaluate

Computes ROUGE-1/2/L scores comparing Gemini (abstractive) vs TextRank (extractive) against CNN/DM reference summaries.

```bash
python -m scripts.04_run_evaluation             # default: 100-doc sample
python -m scripts.04_run_evaluation --sample 50
```

**What happens:**
- Runs both summarizers on a sample of documents
- Computes macro-averaged ROUGE scores
- Prints a comparison table
- Writes per-doc scores → BigQuery `nlp_demo.evaluation_results`

**Example output:**
```
=======================================================
 ROUGE Results  (n=100 documents)
=======================================================
  Model           ROUGE-1   ROUGE-2   ROUGE-L
  ------------------------------------------
  Gemini            0.382     0.162     0.251
  TextRank          0.441     0.208     0.389
=======================================================
```

---

### Step 5 — EDA Notebook

Open the exploratory analysis notebook after Step 1 completes:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Reads from BigQuery `documents_clean` and produces:
- Token/sentence length distributions
- Top 20 unigrams and bigrams
- Vocabulary size and type-token ratio
- Reference summary length distribution
- 5 sample document previews

Charts are saved to `docs/` for use in the report.

---

## 4. Agent Demo

Run the interactive Research Insight Agent REPL. Requires Steps 1–3 to have completed so the corpus, entities, and summaries are available in BigQuery and Firestore.

```bash
python -m scripts.05_agent_demo
```

Resume a previous session:
```bash
python -m scripts.05_agent_demo --session <session-id>
```

**Sample queries to try:**
```
Find articles about climate change and list the top 5 entities mentioned.
Summarise the most relevant article about the US economy.
Compare coverage of healthcare across 3 articles and highlight common themes.
Find articles about elections and show the sentiment distribution of entities.
```

**Example trace output:**
```
You: Find articles about climate change and list the top entities

──────────────────────────────────────────────────────────────
  Agent reasoning:
──────────────────────────────────────────────────────────────

  ┌─ Step 1 ── search_documents
  │  Args        : {"query": "climate change", "top_k": 5}
  │  Observation : {"results": [{"doc_id": "abc-123", "snippet": "..."}, ...]}
  └──────────────────────────────────────────────────────────

  ┌─ Step 2 ── extract_entities
  │  Args        : {"doc_id": "abc-123"}
  │  Observation : {"entities": [{"name": "UN", "type": "ORGANIZATION", ...}]}
  └──────────────────────────────────────────────────────────

  ┌─ Step 3 ── aggregate_entities
  │  Args        : {"doc_ids": ["abc-123", ...], "top_n": 5}
  │  Observation : {"top_entities": [...], "sentiment_distribution": {...}}
  └──────────────────────────────────────────────────────────

──────────────────────────────────────────────────────────────
  Agent:
──────────────────────────────────────────────────────────────

Across 5 articles about climate change, the most frequently
mentioned entities are: ...

Sources: abc-123, def-456, ghi-789
```

---

## 5. Development

```bash
make lint    # ruff check (zero issues enforced)
make fmt     # black formatting
make test    # pytest tests/
```

**Pre-commit hooks** (installed via `make setup`) run ruff, black, and a private-key detector on every commit.

### Running the test suite

The tests use mocked GCP clients — no live GCP credentials required.

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test module
pytest tests/test_preprocess.py -v

# Run a specific test class or case
pytest tests/test_agent_tools.py::TestAggregateEntities -v
pytest tests/test_evaluator.py::TestEvaluator::test_identical_text_scores_perfect -v

# Stop on first failure
pytest tests/ -x

# Show coverage summary (requires pytest-cov)
pytest tests/ --cov=src --cov-report=term-missing
```

**Test layout:**

| Module | What it covers |
|---|---|
| `test_preprocess.py` | Whitespace normalisation, token counting, SHA-256, language detection, `_clean_record` boundary cases |
| `test_extraction_models.py` | Pydantic model defaults, validation, nested construction |
| `test_schema_helper.py` | `_pydantic_to_vertex_schema` — `$ref` inlining, title stripping, mutation safety |
| `test_evaluator.py` | ROUGE-1/2/L scoring, corpus averaging, edge cases |
| `test_agent_schemas.py` | Input schema validation — bounds on `top_k`, `top_n`, `style` enum |
| `test_bq_client.py` | `table_ref`, `insert_rows` error propagation, `query` result mapping |
| `test_gcs_client.py` | JSONL upload format, `blob_exists` |
| `test_agent_memory.py` | Delegation to `FirestoreClient`, cache hit/miss |
| `test_agent_tools.py` | All 5 agent tools — cache paths, BQ error paths, aggregation logic |
| `test_planner.py` | `_history_to_contents` for all role types, `Planner` constructor |

All 138 tests pass with no live GCP calls.

**Cost tips:**
- Use `--limit 50` on first runs to verify GCP connectivity before full-scale processing
- Gemini Flash is used for bulk extraction/summarization; Pro is reserved for the agent planner
- Firestore and BQ caching mean re-running the agent on the same documents costs zero incremental API calls

---

## 6. Project Structure

```
├── config/
│   ├── settings.py              # Pydantic settings loaded from .env
│   └── prompts/
│       ├── entity_extraction.yaml
│       ├── summarization.yaml
│       └── agent_system.yaml
├── notebooks/
│   └── 01_eda.ipynb             # Corpus EDA — run after Step 1
├── src/
│   ├── data/                    # Ingest, preprocess, BQ schemas
│   ├── extraction/              # Cloud NL API + Gemini + spaCy extractors
│   ├── summarization/           # Gemini summarizer + TextRank + ROUGE evaluator
│   ├── agent/                   # ReAct agent, 5 tools, Firestore memory, planner
│   ├── gcp/                     # GCS, BQ, Firestore, Vertex AI client wrappers
│   └── utils/                   # Structured logging, tenacity retry decorator
├── scripts/
│   ├── 01_ingest_data.py        # HuggingFace → GCS → BigQuery
│   ├── 02_run_extraction.py     # Cloud NL API + Gemini extraction (parallel)
│   ├── 03_run_summarization.py  # Gemini summarization (parallel)
│   ├── 04_run_evaluation.py     # ROUGE evaluation: Gemini vs TextRank
│   └── 05_agent_demo.py         # Interactive agent REPL
├── infra/
│   ├── terraform/               # GCP infrastructure as code (stubs)
│   ├── pipelines/               # Vertex AI Pipelines KFP DAG (stub)
│   └── cloud_run/               # Dockerfile + FastAPI app for production serving
├── docs/
│   ├── REPORT.md / REPORT.pdf   # Architecture & Agent Design Report
│   └── eda_*.png                # EDA charts (generated by notebook)
├── tests/                       # Unit tests with mocked GCP clients
├── .env.example                 # Config template — copy to .env and fill in values
├── requirements.txt
├── Makefile
└── pyproject.toml               # ruff + black + pytest config
```

---

## GCP Resources Created

| Resource | Name | Purpose |
|---|---|---|
| GCS Bucket | `<project-id>-nlp-corpus` | Raw JSONL landing zone |
| BQ Dataset | `nlp_demo` | All structured data |
| BQ Table | `nlp_demo.documents` | Raw ingested articles |
| BQ Table | `nlp_demo.documents_clean` | Filtered/deduped articles |
| BQ Table | `nlp_demo.entities` | Extracted entities (Cloud NL + spaCy) |
| BQ Table | `nlp_demo.extractions_gemini` | Structured Gemini extraction results |
| BQ Table | `nlp_demo.summaries` | TL;DR / bullets / abstract per doc |
| BQ Table | `nlp_demo.evaluation_results` | Per-doc ROUGE scores |
| Firestore | `sessions/{id}/turns` | Agent conversation history |
| Firestore | `doc_cache/{doc_id}` | Extraction/summary cache |
