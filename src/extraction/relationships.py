import json
import os
import re
from itertools import combinations

import anthropic
import numpy as np
from rich.console import Console

from ..db import get_conn

console = Console()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

RELATIONSHIP_SYSTEM = """You are a research analyst. Given two research paper abstracts, \
determine how they relate to each other.

Respond ONLY with valid JSON:
{
  "relationship": "cites" | "extends" | "contradicts" | "similar" | "unrelated",
  "evidence": "one sentence explaining why"
}

Definitions:
- cites: paper A explicitly references or builds on paper B's specific work
- extends: paper A takes paper B's method and improves or applies it differently
- contradicts: papers report conflicting results or make opposing claims
- similar: papers tackle the same problem with different approaches (no clear citation)
- unrelated: papers are in different subfields or address different problems
"""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _get_paper_abstract_embedding(paper_id: int) -> tuple[str, str, list[float] | None]:
    """Return (title, abstract, abstract_embedding) for a paper."""
    title, abstract, embedding = "", "", None

    with get_conn() as conn:
        with conn.cursor() as cur:

            # ── Step 1: fetch title and abstract ──────────────────────────
            cur.execute(
                "SELECT title, abstract FROM papers WHERE id = %s",
                (paper_id,)
            )
            row = cur.fetchone()

            if not row:
                console.print(f"[red]Paper {paper_id} not found[/red]")
                return title, abstract, embedding

            # Defensive: unpack only what's there
            title    = row[0] if len(row) > 0 and row[0] else ""
            abstract = row[1] if len(row) > 1 and row[1] else ""

            # ── Step 2: fetch any chunk embedding for this paper ───────────
            # Try abstract section first, fall back to first available chunk
            cur.execute(
                """
                SELECT embedding::text FROM chunks
                WHERE paper_id = %s
                ORDER BY
                    CASE WHEN section ILIKE '%abstract%' THEN 0 ELSE 1 END,
                    chunk_index
                LIMIT 1
                """,
                (paper_id,)
            )
            emb_row = cur.fetchone()

            if emb_row and len(emb_row) > 0 and emb_row[0]:
                try:
                    raw = emb_row[0].strip("[]")
                    embedding = [float(x) for x in raw.split(",") if x.strip()]
                except (ValueError, AttributeError) as e:
                    console.print(f"[yellow]Could not parse embedding for paper {paper_id}: {e}[/yellow]")
                    embedding = None

    return title, abstract, embedding

def map_relationships(min_similarity: float = 0.75) -> list[dict]:
    """
    Map relationships between all pairs of ingested papers.

    Strategy:
    1. Compute embedding similarity between all pairs
    2. For similar pairs (sim > threshold), use LLM to classify relationship type
    3. Store all relationships
    """
    # Get all papers
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, title FROM papers ORDER BY id")
            papers = cur.fetchall()

    if len(papers) < 2:
        console.print("[yellow]Need at least 2 papers for relationship mapping.[/yellow]")
        return []

    console.print(f"[bold]Mapping relationships between {len(papers)} papers...[/bold]")

    results = []
    paper_ids = [p[0] for p in papers]

    for id_a, id_b in combinations(paper_ids, 2):
        # Skip if already mapped
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id FROM paper_relationships
                    WHERE source_paper_id = %s AND target_paper_id = %s
                    """,
                    (id_a, id_b),
                )
                if cur.fetchone():
                    continue

        try:
            title_a, abstract_a, emb_a = _get_paper_abstract_embedding(id_a)
            title_b, abstract_b, emb_b = _get_paper_abstract_embedding(id_b)
        except Exception as e:
            console.print(f"[red]Skipping pair ({id_a}, {id_b}): {e}[/red]")
            continue

        if not title_a or not title_b:
            console.print(f"[yellow]Skipping pair ({id_a}, {id_b}): missing title[/yellow]")
            continue

        # Compute embedding similarity if both have embeddings
        sim_score = 0.0
        if emb_a and emb_b:
            sim_score = _cosine_similarity(emb_a, emb_b)

        console.print(
            f"  [dim]{title_a[:35]}... ↔ {title_b[:35]}... "
            f"(sim={sim_score:.2f})[/dim]"
        )

        # Only use LLM for pairs with meaningful similarity
        # Low-similarity pairs get marked 'unrelated' cheaply
        if sim_score < min_similarity:
            _store_relationship(
                id_a, id_b,
                relationship="unrelated",
                evidence="Low embedding similarity — likely different subfields.",
                similarity_score=sim_score,
            )
            continue

        # Use LLM to classify the relationship type
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            temperature=0.0,
            system=RELATIONSHIP_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"Paper A: {title_a}\n{abstract_a[:500]}\n\n"
                    f"Paper B: {title_b}\n{abstract_b[:500]}"
                ),
            }],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE)

        try:
            rel = json.loads(raw)
            relationship = rel.get("relationship", "similar")
            evidence = rel.get("evidence", "")
        except json.JSONDecodeError:
            relationship = "similar"
            evidence = "Could not parse LLM output."

        _store_relationship(id_a, id_b, relationship, evidence, sim_score)
        console.print(
            f"  [green]→ {relationship}: {evidence[:60]}[/green]"
        )
        results.append({
            "paper_a": title_a, "paper_b": title_b,
            "relationship": relationship, "similarity": sim_score,
        })

    return results


def _store_relationship(
    source_id: int,
    target_id: int,
    relationship: str,
    evidence: str,
    similarity_score: float,
):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO paper_relationships
                    (source_paper_id, target_paper_id, relationship,
                     evidence, similarity_score)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_paper_id, target_paper_id, relationship)
                DO NOTHING
                """,
                (source_id, target_id, relationship, evidence, similarity_score),
            )


def get_related_papers(paper_id: int, min_similarity: float = 0.0) -> list[dict]:
    """Get all papers related to a given paper, with relationship type."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    CASE WHEN r.source_paper_id = %s
                         THEN r.target_paper_id
                         ELSE r.source_paper_id END AS related_id,
                    p.title,
                    p.year,
                    r.relationship,
                    r.evidence,
                    r.similarity_score
                FROM paper_relationships r
                JOIN papers p ON p.id = CASE
                    WHEN r.source_paper_id = %s THEN r.target_paper_id
                    ELSE r.source_paper_id END
                WHERE (r.source_paper_id = %s OR r.target_paper_id = %s)
                  AND r.relationship != 'unrelated'
                  AND r.similarity_score >= %s
                ORDER BY r.similarity_score DESC
                """,
                (paper_id, paper_id, paper_id, paper_id, min_similarity),
            )
            rows = cur.fetchall()

    return [
        {
            "paper_id":     r[0],
            "title":        r[1],
            "year":         r[2],
            "relationship": r[3],
            "evidence":     r[4],
            "similarity":   round(r[5], 3),
        }
        for r in rows
    ]
