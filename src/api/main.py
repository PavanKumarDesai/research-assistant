import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rich.console import Console

load_dotenv()

from ..db import setup_schema, get_conn
from ..ingestion.ingest import ingest_arxiv, ingest_local_pdf, ingest_all_local
from ..retrieval.qa import answer as qa_answer
from ..retrieval.search import search
from ..extraction.concepts import extract_concepts, extract_all_papers
from ..extraction.relationships import map_relationships, get_related_papers
from ..extraction.query import search_by_concepts, get_concept_summary
from ..agents.research_agent import run_research_query
from ..observability import tracker, score_response
from ..evals.eval_suite import run_evals, save_baseline, check_regression

console = Console()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_schema()
    console.print("[bold green]✓ dp-research-assistant ready[/bold green]")
    yield


app = FastAPI(
    title="DP Research Assistant",
    description="RAG system over differential privacy + EHR research papers",
    lifespan=lifespan,
)


# ── Request / Response models ──────────────────────────────────────────────

class ArxivRequest(BaseModel):
    arxiv_id: str   # e.g. "2301.12345"

class LocalPDFRequest(BaseModel):
    filepath: str   # e.g. "data/papers/my_paper.pdf"

class QuestionRequest(BaseModel):
    question:  str
    top_k:     int           = 4
    paper_ids: list[int] | None = None  # restrict to specific papers
    stream:    bool          = False

class SourceInfo(BaseModel):
    paper_title: str
    authors:     list[str]
    year:        Optional[int]
    section:     str
    score:       float

class AnswerResponse(BaseModel):
    answer:        str
    sources:       list[SourceInfo]
    input_tokens:  int
    output_tokens: int
    cost_usd:      float


# ── Ingestion endpoints ────────────────────────────────────────────────────

@app.post("/ingest/arxiv")
def ingest_from_arxiv(req: ArxivRequest):
    """Download a paper from ArXiv by ID and ingest it."""
    try:
        result = ingest_arxiv(req.arxiv_id)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ingest/local")
def ingest_from_local(req: LocalPDFRequest):
    """Ingest a local PDF file."""
    try:
        result = ingest_local_pdf(req.filepath)
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ingest/all")
def ingest_all():
    """Ingest every PDF in data/papers/ not yet ingested."""
    results = ingest_all_local()
    return {"ingested": len(results), "results": results}


# ── Query endpoints ────────────────────────────────────────────────────────

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    """Ask a research question. Returns a cited answer."""
    result = qa_answer(
        question=req.question,
        top_k=req.top_k,
        paper_ids=req.paper_ids,
        stream=False,
    )

    # Rough cost calculation (Sonnet pricing)
    cost = (result.input_tokens * 3.0 + result.output_tokens * 15.0) / 1_000_000

    return AnswerResponse(
        answer=result.answer,
        sources=[
            SourceInfo(
                paper_title=s.paper_title,
                authors=s.authors,
                year=s.year,
                section=s.section,
                score=round(s.score, 4),
            )
            for s in result.sources
        ],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=round(cost, 6),
    )


@app.get("/papers")
def list_papers():
    """List all ingested papers."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.id, p.title, p.authors, p.year, p.source,
                       COUNT(c.id) AS chunk_count
                FROM papers p
                LEFT JOIN chunks c ON c.paper_id = p.id
                GROUP BY p.id
                ORDER BY p.ingested_at DESC
            """)
            rows = cur.fetchall()
    return [
        {
            "id": r[0], "title": r[1], "authors": r[2],
            "year": r[3], "source": r[4], "chunks": r[5],
        }
        for r in rows
    ]


@app.get("/health")
def health():
    """System health including paper count and cost metrics."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM papers")
            paper_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM paper_concepts")
            concepts_count = cur.fetchone()[0]

    return {
        "status":          "ok",
        "papers_ingested": paper_count,
        "chunks_indexed":  chunk_count,
        "concepts_extracted": concepts_count,
        "session_cost_usd": round(tracker.total_cost_usd, 4),
        "session_calls":   tracker.total_calls,
    }

# ── Phase 2 endpoints — add these to the bottom of main.py ────────────────

class ConceptFilterRequest(BaseModel):
    epsilon_max:  float | None = None
    epsilon_min:  float | None = None
    mechanisms:   list[str] | None = None
    datasets:     list[str] | None = None
    is_federated: bool | None = None
    tasks:        list[str] | None = None

@app.post("/extract/all")
def extract_all():
    """Extract concepts from all papers not yet extracted."""
    results = extract_all_papers()
    return {"extracted": len(results), "results": results}

@app.post("/extract/{paper_id}")
def extract_paper_concepts(paper_id: int):
    """Extract DP concepts from a single paper."""
    try:
        result = extract_concepts(paper_id)
        return {"status": "ok", "paper_id": paper_id, **result}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))



@app.post("/relationships/map")
def map_all_relationships():
    """Map relationships between all paper pairs."""
    results = map_relationships()
    return {"mapped": len(results), "relationships": results}


@app.get("/relationships/{paper_id}")
def get_paper_relationships(paper_id: int):
    """Get all papers related to a given paper."""
    related = get_related_papers(paper_id)
    return {"paper_id": paper_id, "related": related}


@app.post("/concepts/search")
def concept_search(req: ConceptFilterRequest):
    """Filter papers by extracted technical concepts."""
    results = search_by_concepts(
        epsilon_max=req.epsilon_max,
        epsilon_min=req.epsilon_min,
        mechanisms=req.mechanisms,
        datasets=req.datasets,
        is_federated=req.is_federated,
        tasks=req.tasks,
    )
    return {"count": len(results), "papers": results}


@app.get("/concepts/summary")
def concepts_summary():
    """Aggregate summary across all extracted concepts."""
    return get_concept_summary()

class ResearchRequest(BaseModel):
    query:     str
    thread_id: str = "default"   # use per-user thread IDs in production

class ResearchResponse(BaseModel):
    response:        str
    query_type:      str
    tool_calls_made: int


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    """
    Run a research query through the full agent pipeline.

    Examples:
    - "What epsilon values are used with EHR data across these papers?"
    - "Synthesise what these papers say about the utility-privacy tradeoff"
    - "What are the open research gaps in DP for EHR transformers?"
    - "Draft a literature review section on DP mechanisms used in healthcare ML"
    """
    try:
        result = run_research_query(req.query, thread_id=req.thread_id)
        return ResearchResponse(**result)
    except Exception as e:
        raise HTTPException(500, str(e))

# ── Observability endpoints ────────────────────────────────────────────────

@app.get("/metrics")
def get_metrics():
    """Live cost and usage metrics for this server session."""
    return tracker.summary


@app.get("/metrics/log")
def get_metrics_log():
    """Per-call cost log for this session."""
    return {"calls": tracker.session_log[-50:]}   # last 50 calls


class FeedbackRequest(BaseModel):
    score:    float   # 0.0 = bad, 1.0 = good
    comment:  str = ""

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """
    Submit user feedback on a response.
    Attaches a score to the Langfuse trace if configured.
    """
    score_response(req.trace_id, req.score, req.comment)
    return {"status": "recorded"}


# ── Eval endpoints ─────────────────────────────────────────────────────────

@app.post("/evals/run")
def run_eval_suite():
    """
    Run the full eval suite.
    Returns pass rate, scores, and failures.
    Takes 1-3 minutes depending on corpus size.
    """
    summary = run_evals(save_results=True)
    return summary


@app.post("/evals/baseline")
def set_baseline():
    """Save current eval results as the regression baseline."""
    summary = run_evals(save_results=False)
    save_baseline(summary)
    return {"status": "baseline saved", **summary}

@app.get("/evals/regression")
def regression_check():
    """Check if current performance has regressed vs baseline."""
    summary = run_evals(save_results=False)
    passed  = check_regression(summary["pass_rate"])
    return {
        "regression_detected": not passed,
        "current_pass_rate":   summary["pass_rate"],
        **summary,
    }


