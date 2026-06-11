# VoiceDoc Intelligence

**Voice-Commanded Document Intelligence Multi-Agent System**  
Built for the Google Cloud Rapid Agent Hackathon

---

## What It Does

VoiceDoc Intelligence lets you submit a research query (text or voice) and automatically dispatches a parallel multi-agent pipeline to scan the web, evaluate sources, extract content, chunk and embed documents, and answer questions — all backed by Gemini 2.0 Flash, LangGraph, MongoDB Atlas, and Celery.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT (Browser)                               │
│   index.html + app.js  ──  WebSocket (WS /ws/{session_id})             │
│                        ──  REST (POST /api/process, POST /api/query)    │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ HTTP / WS
┌──────────────────────────────────▼──────────────────────────────────────┐
│                        FastAPI  (app/main.py)                           │
│                                                                         │
│   POST /api/process  ──►  SupervisorAgent  (LangGraph StateGraph)      │
│                               │                                         │
│                    ┌──────────▼──────────┐                             │
│                    │  parse_query node   │  validate + gen job_id      │
│                    └──────────┬──────────┘                             │
│                    ┌──────────▼──────────┐                             │
│                    │  plan_tasks node    │  Gemini 2.0 Flash           │
│                    │  (LLM planning)     │  → subtasks + search queries│
│                    └──────────┬──────────┘                             │
│                    ┌──────────▼──────────┐                             │
│                    │  save_job node      │  MongoDB → job record       │
│                    └──────────┬──────────┘                             │
│                    ┌──────────▼──────────┐                             │
│                    │ dispatch_pipeline   │  fire N Celery chains       │
│                    │      node           │  (1 per search query)       │
│                    └─────────────────────┘                             │
│                                                                         │
│   POST /api/query  ──►  QueryAgent  (vector search → Gemini synthesis) │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ Celery tasks (Redis broker)
┌─────────────────────────────────▼───────────────────────────────────────┐
│                      Celery Worker(s)                                   │
│                                                                         │
│   ScannerAgent   → search the web (DuckDuckGo) for relevant URLs       │
│        │                                                                │
│   EvaluatorAgent → score & filter URLs by relevance (Gemini)           │
│        │                                                                │
│   ExtractorAgent → fetch + parse page content (trafilatura / BS4)      │
│        │                                                                │
│   ProcessorAgent → chunk → embed (gemini-embedding-001) → MongoDB      │
│                                                                         │
│   Each step broadcasts progress via WebSocket Manager                  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────┐
          │                       │                   │
   ┌──────▼──────┐       ┌────────▼───────┐   ┌──────▼──────┐
   │  MongoDB     │       │     Redis      │   │  Gemini API │
   │  Atlas       │       │  (broker +     │   │  (LLM +     │
   │  (docs,      │       │   result       │   │  embeddings)│
   │   chunks,    │       │   backend)     │   │             │
   │   jobs,      │       └────────────────┘   └─────────────┘
   │   vectors)   │
   └─────────────-┘
         ▲
         │  optional stdio transport
   ┌─────┴──────────────┐
   │  MongoDB MCP Server │
   │  (mongodb-mcp-     │
   │   server@1.12.0)   │
   └────────────────────┘
```

### Agent Pipeline Summary

| Agent | Role |
|---|---|
| **SupervisorAgent** | LangGraph orchestrator — plans tasks with Gemini, saves job, dispatches Celery chains |
| **ScannerAgent** | Web search via DuckDuckGo, returns ranked URL list |
| **EvaluatorAgent** | Scores URLs for relevance using Gemini |
| **ExtractorAgent** | Fetches and parses page content (trafilatura, BeautifulSoup) |
| **ProcessorAgent** | Chunks text, generates embeddings, upserts to MongoDB Atlas |
| **QueryAgent** | Vector search over stored chunks, synthesises answer via Gemini |

---

## Prerequisites

- Python 3.12 (not 3.14 — wheel support is incomplete for numpy/scipy)
- Docker and Docker Compose
- Node.js (only if you need to install the MCP server globally)
- A [Google AI Studio](https://aistudio.google.com) API key (Gemini)
- A [MongoDB Atlas](https://cloud.mongodb.com) cluster with a **vector index** configured
- Redis (provided via Docker Compose)

---

## Setup Instructions

### 1. Clone and initialise

```bash
git clone <repo-url>
cd hackathon
```

### 2. Run the setup script

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This creates a `venv` with Python 3.12, installs all dependencies, and copies `.env.example` → `.env`.

### 3. Configure environment variables

Edit `.env` with your real credentials:

```dotenv
# Required
GOOGLE_API_KEY=your_google_ai_studio_key
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
SECRET_KEY=at_least_32_random_characters_here

# Defaults that usually work as-is
GEMINI_MODEL=gemini-2.0-flash
EMBEDDING_MODEL=models/gemini-embedding-001
VECTOR_DIMENSIONS=3072
MONGODB_DATABASE=voicedoc_intelligence
```

### 4. Create the MongoDB Atlas vector index

In Atlas UI → your cluster → **Search** → **Create Search Index**, use the following JSON definition on the `chunks` collection:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 3072,
      "similarity": "cosine"
    }
  ]
}
```

Name the index `vector_index` (matches `VECTOR_INDEX_NAME` in `.env`).

---

## Running Locally

### Option A — Docker Compose (recommended)

Starts the API, Celery worker, Celery Beat, Flower dashboard, and Redis in one command:

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |
| Frontend | open `frontend/index.html` in browser |
| Flower (Celery monitor) | http://localhost:5555 |

### Option B — Manual (without Docker)

Start Redis first (requires Docker for the Redis container, or a local install):

```bash
docker run -p 6379:6379 redis:7-alpine
```

In three separate terminals:

```bash
# Terminal 1 — FastAPI
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Celery worker
source venv/bin/activate
celery -A app.celery_app.celery_app worker --loglevel=info --concurrency=4

# Terminal 3 — Celery Beat (optional, for scheduled tasks)
source venv/bin/activate
celery -A app.celery_app.celery_app beat --loglevel=info
```

Then open `frontend/index.html` in your browser.

### Health check

```bash
curl http://localhost:8000/api/health
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/api/health` | Dependency health (MongoDB, Redis, LLM) |
| `POST` | `/api/process` | Submit a research query → returns `job_id` |
| `GET` | `/api/job/{job_id}` | Poll job status and per-agent progress |
| `POST` | `/api/query` | Ask a question against stored documents |
| `WS` | `/ws/{session_id}` | Real-time progress stream |

---

## Deploying to Cloud Run

### Prerequisites

- Google Cloud project with Cloud Run, Cloud Build, and Container Registry APIs enabled
- `gcloud` CLI authenticated: `gcloud auth login`

### Step 1 — Set your project

```bash
export PROJECT_ID=your-gcp-project-id
gcloud config set project $PROJECT_ID
```

### Step 2 — Store secrets in Secret Manager (recommended)

```bash
echo -n "your_google_api_key" | gcloud secrets create GOOGLE_API_KEY --data-file=-
echo -n "mongodb+srv://..." | gcloud secrets create MONGODB_URI --data-file=-
echo -n "your_secret_key"    | gcloud secrets create SECRET_KEY --data-file=-
```

Grant the Cloud Run service account access to each secret.

### Step 3 — Build and deploy via Cloud Build

```bash
gcloud builds submit --config cloudbuild.yaml
```

`cloudbuild.yaml` builds the Docker image, pushes it to Container Registry, and deploys to Cloud Run with:

- 2 vCPU, 2 GiB RAM
- 0–10 instances (scale to zero)
- 300 s request timeout
- 80 concurrent requests per instance

### Step 4 — Set environment variables on the Cloud Run service

```bash
gcloud run services update voicedoc-intelligence \
  --region us-central1 \
  --set-env-vars "APP_ENV=production" \
  --set-env-vars "GEMINI_MODEL=gemini-2.0-flash" \
  --set-env-vars "MONGODB_DATABASE=voicedoc_intelligence" \
  --set-secrets "GOOGLE_API_KEY=GOOGLE_API_KEY:latest" \
  --set-secrets "MONGODB_URI=MONGODB_URI:latest" \
  --set-secrets "SECRET_KEY=SECRET_KEY:latest"
```

> **Note:** Celery workers cannot run inside Cloud Run (no long-lived background processes). For production, run workers on Cloud Run Jobs, GKE, or a Compute Engine VM. For the hackathon demo, run workers locally pointed at the deployed API.

### Step 5 — Verify

```bash
SERVICE_URL=$(gcloud run services describe voicedoc-intelligence \
  --region us-central1 --format 'value(status.url)')

curl $SERVICE_URL/api/health
```

---

## MCP Server Setup

VoiceDoc uses the [MongoDB MCP Server](https://github.com/mongodb-js/mongodb-mcp-server) to give agents direct, structured access to MongoDB via the Model Context Protocol over `stdio`.

### Install

```bash
npm install -g mongodb-mcp-server
```

Verify:

```bash
mongodb-mcp-server --version
```

### Configure

The MCP config lives in `mcp_config.json`. The server reads your connection string from the environment:

```bash
export MDB_MCP_CONNECTION_STRING="mongodb+srv://user:pass@cluster.mongodb.net/"
```

Or add it to `.env`:

```dotenv
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
```

### Auto-approved tools

The following read-only tools are pre-approved in `mcp_config.json` (no per-call confirmation needed):

- `find` — query documents with filter + projection
- `aggregate` — run aggregation pipelines including `$vectorSearch`
- `count` — count documents matching a filter
- `list-collections` / `list-databases` — schema discovery
- `collection-schema` / `collection-indexes` / `db-stats` — metadata

Write tools (`insert-many`, `update-one`, etc.) require explicit approval.

### Run the demo script

```bash
source venv/bin/activate
python mcp_demo.py
```

This exercises the full MCP integration — listing collections, running a vector search aggregation, and inserting sample documents — without starting the full API.

### How agents use MCP

| File | Role |
|---|---|
| `app/mcp/mcp_client.py` | Python wrapper that spawns the MCP server as a subprocess and communicates over `stdio` (JSON-RPC) |
| `app/agents/query_agent.py` | Optionally routes vector search through MCP instead of direct PyMongo |
| `app/agents/processor_agent.py` | Optionally routes chunk insertion through MCP |

---

## Project Structure

```
hackathon/
├── app/
│   ├── agents/
│   │   ├── supervisor_agent.py   # LangGraph orchestrator
│   │   ├── scanner_agent.py      # Web search (DuckDuckGo)
│   │   ├── evaluator_agent.py    # URL relevance scoring (Gemini)
│   │   ├── extractor_agent.py    # Content extraction
│   │   ├── processor_agent.py    # Chunking + embedding + upsert
│   │   └── query_agent.py        # Vector search + answer synthesis
│   ├── api/
│   │   ├── routes_process.py     # POST /api/process, GET /api/job/{id}
│   │   ├── routes_query.py       # POST /api/query
│   │   └── routes_websocket.py   # WS /ws/{session_id}
│   ├── database/db.py            # Motor (async MongoDB) connection pool
│   ├── mcp/mcp_client.py         # MCP stdio client
│   ├── models/                   # Pydantic + Motor document models
│   ├── utils/
│   │   ├── llm.py                # Gemini / Groq LLM factory
│   │   ├── job_manager.py        # Job CRUD + WebSocket broadcast
│   │   └── websocket_manager.py  # Connection registry
│   ├── celery_app.py             # Celery application instance
│   ├── config.py                 # Pydantic Settings (env-driven)
│   └── main.py                   # FastAPI app + lifespan
├── frontend/
│   ├── index.html                # Single-page UI
│   └── app.js                    # Fetch + WebSocket client
├── scripts/
│   ├── setup.sh                  # One-shot dev environment setup
│   └── deploy.sh                 # Cloud Run deploy helper
├── mcp_config.json               # MongoDB MCP server config
├── mcp_demo.py                   # Standalone MCP integration demo
├── cloudbuild.yaml               # Cloud Build CI/CD pipeline
├── docker-compose.yml            # Local multi-service stack
├── Dockerfile                    # Production container image
├── requirements.txt
└── .env.example                  # Environment variable template
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 2.0 Flash (`gemini-2.0-flash`) |
| Embeddings | `gemini-embedding-001` (3072 dims) |
| Orchestration | LangGraph `StateGraph` |
| API | FastAPI + Uvicorn |
| Task queue | Celery 5 + Redis 7 |
| Database | MongoDB Atlas (vector + document store) |
| MCP | `mongodb-mcp-server@1.12.0` via stdio |
| Web scraping | trafilatura, BeautifulSoup4 |
| Web search | DuckDuckGo Search |
| Container | Docker, Google Cloud Run |
| CI/CD | Google Cloud Build |

---

## License

MIT
