import os
from pathlib import Path

import psycopg2.extras
from rich.console import Console

from ..db import get_conn
from .arxiv_loader import download_arxiv_paper, list_local_pdfs
from .chunker import chunk_paper
from .embedder import embed_and_store
from .pdf_parser import parse_pdf

console = Console()


def _paper_already_ingested(arxiv_id: str | None, filename: str) -> int | None:
    """Return paper DB id if already ingested, else None."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            if arxiv_id:
                cur.execute("SELECT id FROM papers WHERE arxiv_id = %s", (arxiv_id,))
            else:
                cur.execute("SELECT id FROM papers WHERE filename = %s", (filename,))
            row = cur.fetchone()
            return row[0] if row else None


def _store_paper_metadata(meta: dict) -> int:
    """Insert paper metadata, return new paper id."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO papers
                    (arxiv_id, title, authors, year, abstract, source, filename)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    meta.get("arxiv_id"),
                    meta["title"][:500],
                    meta.get("authors", []),
                    meta.get("year"),
                    meta.get("abstract", "")[:2000],
                    meta["source"],
                    meta.get("filename"),
                ),
            )
            return cur.fetchone()[0]


def ingest_arxiv(arxiv_id: str) -> dict:
    """Download from ArXiv and ingest. Idempotent."""
    clean_id = arxiv_id.split("v")[0].strip()

    existing_id = _paper_already_ingested(arxiv_id=clean_id, filename="")
    if existing_id:
        console.print(f"[yellow]Already ingested: {clean_id}[/yellow]")
        return {"status": "already_ingested", "paper_id": existing_id}

    pdf_path, arxiv_meta = download_arxiv_paper(clean_id)
    return _ingest_pdf_file(str(pdf_path), extra_meta=arxiv_meta)


def ingest_local_pdf(filepath: str) -> dict:
    """Ingest a local PDF file. Idempotent."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    existing_id = _paper_already_ingested(arxiv_id=None, filename=path.name)
    if existing_id:
        console.print(f"[yellow]Already ingested: {path.name}[/yellow]")
        return {"status": "already_ingested", "paper_id": existing_id}

    return _ingest_pdf_file(filepath)


def _ingest_pdf_file(filepath: str, extra_meta: dict | None = None) -> dict:
    """Core ingestion flow — parse, chunk, embed, store."""
    path = Path(filepath)
    console.print(f"\n[bold]Ingesting:[/bold] {path.name}")

    # 1. Parse
    console.print("  [dim]→ Parsing PDF...[/dim]")
    paper = parse_pdf(filepath)
    console.print(f"  [dim]  title:    {paper.title[:70]}[/dim]")
    console.print(f"  [dim]  sections: {[s.name for s in paper.sections]}[/dim]")
    console.print(f"  [dim]  pages:    {paper.total_pages}[/dim]")

    # 2. Build metadata (ArXiv metadata takes priority over PDF-extracted)
    meta = {
        "title":    paper.title,
        "authors":  paper.authors,
        "year":     paper.year,
        "abstract": paper.abstract,
        "source":   "local",
        "filename": path.name,
    }
    if extra_meta:
        meta.update(extra_meta)

    # 3. Store paper record
    paper_id = _store_paper_metadata(meta)
    console.print(f"  [dim]→ Paper stored (id={paper_id})[/dim]")

    # 4. Chunk
    console.print("  [dim]→ Chunking...[/dim]")
    chunks = chunk_paper(paper)
    console.print(f"  [dim]  {len(chunks)} chunks across {len(paper.sections)} sections[/dim]")

    # 5. Embed + store
    embed_and_store(paper_id, chunks)

    return {
        "status":      "ingested",
        "paper_id":    paper_id,
        "title":       meta["title"],
        "chunks":      len(chunks),
        "sections":    [s.name for s in paper.sections],
    }


def ingest_all_local() -> list[dict]:
    """Ingest every PDF in data/papers/ that hasn't been ingested yet."""
    pdfs = list_local_pdfs()
    if not pdfs:
        console.print("[yellow]No PDFs found in data/papers/[/yellow]")
        return []

    console.print(f"[bold]Found {len(pdfs)} PDFs[/bold]")
    results = []
    for pdf in pdfs:
        result = ingest_local_pdf(str(pdf))
        results.append(result)
    return results
