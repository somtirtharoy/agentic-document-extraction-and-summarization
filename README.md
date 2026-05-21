# Agentic Document Extraction and Summarization

Prototype pipeline for ingesting news articles, storing them on GCP, and (in later steps) extracting entities and generating summaries with Vertex AI.

## Prerequisites

- Python 3.11+
- A GCP project with billing enabled
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud` CLI)
- GCS bucket and BigQuery dataset created in your project (see below)

### GCP resources

Create these once in your project (Console or `gcloud`):

| Resource | Example name |
|----------|----------------|
| GCS bucket | `your-project-id-nlp-corpus` |
| BigQuery dataset | `nlp_demo` |

Your identity (user account or service account via impersonation) needs at least:

- Storage Object Admin
- BigQuery Data Editor
- BigQuery Job User

### Authentication (no JSON key file)

Many organizations block service account key downloads. Use **Application Default Credentials** instead:

```bash
gcloud auth application-default login
```

To run as a service account without a key file:

```bash
gcloud auth application-default login \
  --impersonate-service-account=agentic-nlp-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

Do **not** set `GOOGLE_APPLICATION_CREDENTIALS` unless you have a local key file.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pre-commit install            # optional
```

Copy the environment template and fill in your project values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GCS_BUCKET=your-project-id-nlp-corpus
BQ_DATASET=nlp_demo
```

Or run the Makefile setup target (also installs spaCy for later baseline scripts):

```bash
make setup
```

## Run

All commands assume the virtualenv is active and you are in the repo root.

### Step 1 — Ingest and preprocess (implemented)

Downloads [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) from HuggingFace, uploads JSONL to GCS, loads the `documents` table in BigQuery, then cleans and deduplicates into `documents_clean`.

```bash
python -m scripts.01_ingest_data
```

Smaller test run (faster, fewer API calls):

```bash
python -m scripts.01_ingest_data --limit 50
```

Same via Make (default limit 1500):

```bash
make ingest
```

**What it does**

1. **Ingest** — HuggingFace → `gs://<GCS_BUCKET>/raw/documents.jsonl` → BigQuery `documents`
2. **Preprocess** — filter English, token length 100–5000, dedupe by hash → BigQuery `documents_clean`

The first run downloads the dataset from HuggingFace (network required; may take several minutes).

### Later pipeline steps (not yet implemented)

These Makefile targets will be added as scripts land:

```bash
make extract      # scripts/02_run_extraction
make summarize    # scripts/03_run_summarization
make evaluate     # scripts/04_run_evaluation
make agent-demo   # scripts/05_agent_demo
```

## Development

```bash
make lint    # ruff
make fmt     # black
make test    # pytest (when tests/ exists)
```

Never commit `.env` or `*-key.json` files.
