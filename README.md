# DP Research Assistant

A production-grade AI research assistant for differential privacy (DP)
and electronic health records (EHR) research.

Built with: Anthropic Claude, pgvector, voyage-3, LangGraph, FastAPI.

## What it does

- **Ingest** research papers from PDF or ArXiv
- **Ask questions** across your entire paper corpus with citations
- **Extract concepts** — ε, δ, mechanisms, datasets from every paper
- **Map relationships** between papers (cites, extends, contradicts)
- **Research agent** — synthesises findings, identifies gaps, drafts literature reviews
- **Evals + tracing** — quality checks, Langfuse observability, cost tracking

## Quickstart

```bash
# 1. Start the database
docker compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Set up schema
python -m src.db

# 5. Start the API
uvicorn src.api.main:app --reload --port 8000
```

## Ingest papers

```bash
# From ArXiv
curl -X POST http://localhost:8000/ingest/arxiv \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "1607.00133"}'

# All PDFs in data/papers/
curl -X POST http://localhost:8000/ingest/all

# Extract concepts + map relationships
curl -X POST http://localhost:8000/extract/all
curl -X POST http://localhost:8000/relationships/map
```

## Query

```bash
# Direct Q&A with citations
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What epsilon values are used in DP-SGD?"}'

# Research agent (synthesis, gaps, literature review)
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are open research gaps in DP for EHR transformers?"}'

# Filter papers by concepts
curl -X POST http://localhost:8000/concepts/search \
  -H "Content-Type: application/json" \
  -d '{"epsilon_max": 1.0, "mechanisms": ["gaussian"]}'
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest/arxiv` | Ingest paper by ArXiv ID |
| POST | `/ingest/local` | Ingest local PDF |
| POST | `/ingest/all` | Ingest all PDFs in data/papers/ |
| POST | `/ask` | RAG Q&A with citations |
| POST | `/research` | Full research agent |
| POST | `/extract/all` | Extract DP concepts from all papers |
| POST | `/concepts/search` | Filter papers by ε, mechanism, dataset |
| GET  | `/concepts/summary` | Aggregate view across corpus |
| POST | `/relationships/map` | Map inter-paper relationships |
| GET  | `/relationships/{id}` | Get related papers |
| GET  | `/papers` | List all ingested papers |
| POST | `/evals/run` | Run eval suite |
| POST | `/evals/baseline` | Save eval baseline |
| GET  | `/evals/regression` | Check for regressions |
| GET  | `/metrics` | Cost and usage metrics |
| GET  | `/health` | System health |

## Environment variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
DATABASE_URL=postgresql://postgres:password@localhost:5432/dp_research

# Optional — Langfuse tracing
LANGFUSE_PUBLIC_KEY=lf_pk_...
LANGFUSE_SECRET_KEY=lf_sk_...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Project structure

```
src/
├── db.py                    # Schema setup
├── observability.py         # Langfuse tracing + cost tracking
├── ingestion/
│   ├── pdf_parser.py        # PDF → structured sections
│   ├── arxiv_loader.py      # ArXiv download
│   ├── chunker.py           # Section-aware chunking
│   ├── embedder.py          # voyage-3 embedding + pgvector storage
│   └── ingest.py            # Orchestrates ingestion
├── retrieval/
│   ├── search.py            # ANN search + reranking
│   └── qa.py                # RAG Q&A with citations
├── extraction/
│   ├── concepts.py          # DP concept extraction
│   ├── relationships.py     # Paper relationship mapping
│   └── query.py             # Concept filtering queries
├── agents/
│   ├── tools.py             # Tool definitions + implementations
│   └── research_agent.py   # LangGraph research agent
├── evals/
│   └── eval_suite.py        # Eval cases + runners
└── api/
    └── main.py              # FastAPI service
```
