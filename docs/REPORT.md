# Architecture & Agent Design Report
## Data Extraction & Summarization with Agentic Workflows on GCP

**Author:** Somtirtha Roy &nbsp;|&nbsp; **Role:** Lead AI/ML Engineer &nbsp;|&nbsp; **Client:** TEKSystems Global Services

---

## 1. Technical Approach & GCP Services

The system is a fully GCP-native prototype that ingests a public text corpus, extracts structured information, generates multi-style summaries, and exposes these capabilities as tools to an autonomous AI agent. All components are built on managed GCP services — no self-hosted models or external vector databases.

**Dataset:** CNN/DailyMail v3.0.0 (1,500-article slice via HuggingFace `datasets`). Selected because it ships with human-written reference summaries, enabling quantitative ROUGE evaluation, and contains rich named entities across diverse news domains.

| GCP Service | Role in the System |
|---|---|
| **Cloud Storage** | Raw JSONL landing zone (`raw/documents.jsonl`) before BQ load |
| **BigQuery** | Central queryable store for documents, entities, summaries, and evaluation results |
| **Cloud Natural Language API** | Managed NER — entity salience, per-entity sentiment, IAB content categories |
| **Vertex AI — Gemini 1.5 Pro** | Structured extraction (`response_schema`), multi-style summarization, agent planner, cross-doc synthesis |
| **Vertex AI — text-embedding-004** | Document embeddings for semantic search (production upgrade path) |
| **Firestore (Native mode)** | Agent session memory (turn history) and doc-level extraction/summary cache |
| **Cloud Run** | Production serving target for the agent as a containerised FastAPI service |
| **Vertex AI Pipelines (KFP)** | Batch orchestration DAG: ingest → preprocess → extract → summarize → load |
| **Pub/Sub** | Event-driven real-time trigger for new document ingestion |

Authentication uses Application Default Credentials (ADC) via `gcloud auth application-default login` — no service account key files. All GCP clients are initialised once via singleton wrappers in `src/gcp/`.

---

## 2. High-Level GCP Architecture

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         DATA PIPELINE                               │
  │                                                                     │
  │  HuggingFace ──► Cloud Storage ──► BigQuery                        │
  │  (CNN/DM)         (raw JSONL)      (documents, documents_clean)     │
  └──────────────────────────┬──────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
  ┌───────────────────────┐     ┌───────────────────────┐
  │   EXTRACTION          │     │   SUMMARIZATION        │
  │                       │     │                        │
  │  Cloud NL API         │     │  Vertex AI             │
  │  (entities, salience, │     │  Gemini 1.5 Pro        │
  │   sentiment, cats)    │     │  (tldr / bullets /     │
  │          +            │     │   abstract)            │
  │  Vertex AI            │     │          +             │
  │  Gemini 1.5 Pro       │     │  TextRank baseline     │
  │  (structured schema   │     │  (ROUGE comparison)    │
  │   extraction)         │     │                        │
  └──────────┬────────────┘     └──────────┬─────────────┘
             │                             │
             ▼                             ▼
  ┌──────────────────────────────────────────────────────┐
  │              BigQuery  (nlp_demo dataset)             │
  │   entities  |  extractions_gemini  |  summaries  |   │
  │                  evaluation_results                   │
  └───────────────────────────┬──────────────────────────┘
                              │
                              ▼
  ┌───────────────────────────────────────────────────────────────┐
  │                    AGENTIC LAYER                               │
  │                                                               │
  │   User Query                                                  │
  │       │                                                       │
  │       ▼                                                       │
  │   Agent (Gemini 1.5 Pro — native function calling)            │
  │       │                                                       │
  │       ├──► search_documents   ──► BigQuery (keyword search)   │
  │       ├──► extract_entities   ──► Cloud NL API + Firestore ◄─┐│
  │       ├──► summarize_document ──► BigQuery / Gemini        ◄─┤│
  │       ├──► aggregate_entities ──► BigQuery (entities table) │ ││
  │       └──► compare_summaries  ──► Gemini 1.5 Pro           │ ││
  │                                                             │ ││
  │   Firestore ◄─── session memory + doc cache ───────────────┘ ││
  │       (turns/{session_id}, doc_cache/{doc_id})               ││
  │                                                               │
  │   Final Answer (with doc_id citations)                        │
  └───────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │         PRODUCTION SERVING     │
              │                               │
              │   Cloud Run (FastAPI)          │
              │       ├── POST /chat           │
              │       ├── POST /session        │
              │       └── GET  /health         │
              └───────────────────────────────┘
```

---

## 3. Agentic Workflow

### Scenario: Research Insight Agent

An analyst asks natural-language questions about a news corpus. Rather than manually querying databases or reading dozens of articles, the agent autonomously searches the corpus, extracts entities, summarises relevant documents, aggregates findings across multiple articles, and delivers an evidence-backed answer with `doc_id` citations.

**Example query:** *"What are the most frequently mentioned entities in articles about climate change, and what is the overall sentiment?"*

**Agent response flow:**
1. `search_documents("climate change", top_k=5)` → returns 5 relevant doc_ids
2. `extract_entities(doc_id)` × 5 → entities extracted (Firestore-cached after first call)
3. `aggregate_entities([doc_id_1...5])` → top entities ranked by frequency + sentiment
4. Final answer synthesised with citations

### Tools

The agent has five modular tools, each implemented as a typed Python function and registered as a Vertex AI `FunctionDeclaration`:

| Tool | Backing Service | Cache |
|---|---|---|
| `search_documents(query, top_k)` | BigQuery `REGEXP_CONTAINS` (→ VECTOR_SEARCH in prod) | None |
| `extract_entities(doc_id)` | Cloud NL API + Gemini extractor | Firestore `doc_cache` |
| `summarize_document(doc_id, style)` | BigQuery `summaries` table / Gemini live | BQ + Firestore |
| `aggregate_entities(doc_ids, top_n)` | BigQuery `entities` table | None |
| `compare_summaries(doc_ids, focus)` | Gemini 1.5 Pro synthesis | None |

### Planning & Reasoning (ReAct Loop)

The agent uses **Gemini 1.5 Pro's native function calling** in a ReAct (Reason + Act) loop — no external frameworks (LangChain, LangGraph) are introduced. This keeps the dependency surface minimal and the entire control flow visible and debuggable.

```
for step in 1..8:
    response = gemini.generate(history, tools=TOOL_DECLARATIONS)

    if response has function_call:
        args = validate(function_call.args)      # Pydantic validation
        observation = tools.dispatch(name, args) # execute tool
        history += [function_call, observation]  # append to context
        firestore.save(session_id, turn)         # persist turn

    else:
        return response.text                     # final answer
```

**Ambiguity handling:** the system prompt instructs the agent to ask at most one clarifying question before proceeding, preventing infinite clarification loops.

**Error resilience:** every tool wraps its body in `try/except` and returns `{"error": "...", "hint": "..."}` rather than raising. The agent reads the error as an observation and retries with adjusted arguments.

### Memory Architecture

| Layer | Store | Scope |
|---|---|---|
| Short-term (session) | Firestore `sessions/{id}/turns` | Per-conversation turn history; rehydrated each call |
| Long-term cache | Firestore `doc_cache/{doc_id}` | Extraction/summary results; avoids repeat API calls |
| Persistent results | BigQuery `summaries`, `entities` | Batch-precomputed; queried by tools at runtime |

---

## 4. Results & Evaluation

ROUGE scores were computed on a 100-document eval slice from CNN/DailyMail, comparing Gemini (abstractive) against TextRank (extractive) using the dataset's human-written highlights as reference.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Gemini 1.5 Pro (abstract) | ~0.38 | ~0.16 | ~0.25 |
| TextRank (extractive) | ~0.44 | ~0.21 | ~0.39 |

**Interpretation:** TextRank scores higher on ROUGE because it reuses exact original sentences, maximising lexical overlap with the reference. Gemini produces paraphrased, coherent abstracts that are more readable but score lower on surface-level n-gram overlap metrics. Qualitative review of 20 documents confirmed Gemini summaries were consistently more fluent and informative.

**Extraction quality:** Gemini's structured extractor (`response_schema`) captured domain-specific fields (key metrics, core issue, actor roles) that the Cloud NL API's fixed taxonomy cannot — e.g., "GDP fell 3.2%" as a `key_metric`, or "accused of fraud" as an actor's `role`. The NL API excelled at salience scoring, which Gemini does not provide.

---

## 5. Challenges & Trade-offs

**Gemini `response_schema` and `$ref` resolution:** Vertex AI's `GenerationConfig` does not support JSON Schema `$ref` / `$defs` produced by Pydantic's `model_json_schema()`. Resolved by implementing a `_pydantic_to_vertex_schema()` helper that recursively inlines all `$ref` references before passing the schema to the API.

**No LangChain / agent framework:** The ReAct loop is implemented directly against Vertex AI's function calling API. This choice reduces the dependency surface and keeps the control flow fully transparent, but requires manually managing `Content` object construction for multi-turn history. The trade-off is more boilerplate vs. full visibility — appropriate for a production system where debugging agent behaviour matters.

**ROUGE as the sole quantitative metric:** ROUGE measures lexical overlap, not semantic quality. A stronger evaluation would combine ROUGE with BERTScore (semantic similarity) and human ratings. Scoped out for the 48-hour prototype; noted as a future-work item.

**Keyword search vs. semantic search:** `search_documents` currently uses BQ `REGEXP_CONTAINS` for simplicity. In production, this would be replaced by `BQ VECTOR_SEARCH` with `text-embedding-004` embeddings, enabling semantic retrieval of conceptually related documents even when exact keywords don't match.

**Firestore cold-start for cache:** On first run against a new corpus, every tool call is a cache miss. The batch extraction and summarization scripts (Phases 2–3) are designed to pre-populate both BQ and Firestore so the agent operates primarily from cache during interactive use.

---

## 6. Productionization Approach

**Scalability & Orchestration:** Batch processing is orchestrated via a Vertex AI Pipelines (KFP) DAG (`infra/pipelines/vertex_pipeline.py`) covering ingest → preprocess → extract → summarize → load, scheduled via Cloud Scheduler. Real-time ingestion of new documents is handled via Pub/Sub: a `documents.new` topic triggers a Cloud Function that runs extraction and summarization, then writes results to BQ and Firestore. The agent is served as a containerised FastAPI service on Cloud Run (`infra/cloud_run/`) with `min-instances=1` to eliminate cold starts.

**Security & Privacy:** Each pipeline component runs under a dedicated service account (`ingest-sa`, `extract-sa`, `agent-sa`) with least-privilege IAM roles. No service account key files are used — authentication is via Workload Identity Federation in Cloud Run and ADC locally. Customer-managed encryption keys (CMEK) are applied to GCS, BQ, and Firestore for sensitive corpora. Cloud DLP pre-extraction scan is applied to flag or redact PII before text enters the pipeline.

**Monitoring & Reliability:** All components emit structured JSON logs to Cloud Logging with a consistent `doc_id` trace field for end-to-end request tracing. Cloud Monitoring dashboards track Gemini token consumption, tool error rates, and BQ query latency. Dead-letter queues on Pub/Sub subscriptions capture ingestion failures. Tenacity retry decorators with exponential backoff handle transient `ResourceExhausted` errors from the Gemini and NL APIs.

**Cost Management:** Gemini 1.5 Flash is used for batch extraction and summarization (high volume, lower quality requirement). Gemini 1.5 Pro is reserved for the agent planner and cross-document synthesis (low volume, high quality requirement). Firestore and BQ caching ensure that every re-queried document costs zero incremental API calls. GCS lifecycle policies move raw data to Nearline at 30 days and Coldline at 90 days.

**CI/CD:** GitHub Actions runs lint (ruff), type checks (mypy), and unit tests on every push. Cloud Build triggers on merge to `main`, builds the container, pushes to Artifact Registry, and deploys to Cloud Run using a 10% → 100% canary rollout. Prompt YAML files are versioned in the repository and loaded at runtime — prompt changes can be rolled back without a container redeploy.

---

*Built with Vertex AI · BigQuery · Cloud Natural Language API · Firestore · Cloud Run*
*Python 3.11 · Pydantic v2 · google-cloud-aiplatform 1.91.0*
