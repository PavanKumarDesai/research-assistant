import os
from dataclasses import dataclass

import voyageai

from ..db import get_conn
from ..ingestion.embedder import embed_query

vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))


@dataclass
class SearchResult:
    chunk_id:    int
    paper_id:    int
    paper_title: str
    authors:     list[str]
    year:        int | None
    section:     str
    text:        str
    score:       float        # cosine similarity (0–1)


def search(
    query: str,
    top_k_ann:   int = 10,   # retrieve more...
    top_k_final: int = 4,    # ...rerank down to fewer
    paper_ids:   list[int] | None = None,   # optional: restrict to specific papers
) -> list[SearchResult]:
    """
    Two-stage retrieval: ANN search → rerank.

    paper_ids: if provided, only search chunks from those papers.
    Useful for: "in this paper specifically, what does it say about..."
    """
    # 1. Embed query
    q_emb = embed_query(query)

    # 2. ANN search in pgvector
    with get_conn() as conn:
        with conn.cursor() as cur:
            if paper_ids:
                cur.execute(
                    """
                    SELECT c.id, c.paper_id, p.title, p.authors, p.year,
                           c.section, c.text,
                           1 - (c.embedding <=> %s::vector) AS score
                    FROM chunks c
                    JOIN papers p ON p.id = c.paper_id
                    WHERE c.paper_id = ANY(%s)
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q_emb, paper_ids, q_emb, top_k_ann),
                )
            else:
                cur.execute(
                    """
                    SELECT c.id, c.paper_id, p.title, p.authors, p.year,
                           c.section, c.text,
                           1 - (c.embedding <=> %s::vector) AS score
                    FROM chunks c
                    JOIN papers p ON p.id = c.paper_id
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q_emb, q_emb, top_k_ann),
                )
            rows = cur.fetchall()

    if not rows:
        return []

    candidates = [
        SearchResult(
            chunk_id=r[0], paper_id=r[1], paper_title=r[2],
            authors=r[3] or [], year=r[4], section=r[5],
            text=r[6], score=float(r[7]),
        )
        for r in rows
    ]

    # 3. Rerank — cross-encoder scores true query-document relevance
    rerank_result = vo.rerank(
        query=query,
        documents=[c.text for c in candidates],
        model="rerank-2",
        top_k=top_k_final,
    )

    return [
        SearchResult(
            **{k: v for k, v in vars(candidates[r.index]).items() if k != "score"},
            score=r.relevance_score,
        )
        for r in rerank_result.results
    ]
