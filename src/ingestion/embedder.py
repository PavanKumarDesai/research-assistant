import os
import time
from dataclasses import dataclass

import psycopg2.extras
import voyageai
from rich.console import Console
from rich.progress import track

from ..db import get_conn
from .chunker import Chunk

console = Console()
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

EMBED_MODEL  = "voyage-3"
EMBED_BATCH  = 64       # voyage-3 max batch
EMBED_DIM    = 1024     # voyage-3 output dimensions


def embed_and_store(
    paper_db_id: int,
    chunks: list[Chunk],
) -> int:
    """
    Embed all chunks for a paper and store them in pgvector.
    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    all_embeddings: list[list[float]] = []

    # Embed in batches — voyage-3 has a batch limit
    for i in track(
        range(0, len(texts), EMBED_BATCH),
        description=f"[cyan]Embedding {len(texts)} chunks...[/cyan]",
    ):
        batch = texts[i : i + EMBED_BATCH]
        result = vo.embed(batch, model=EMBED_MODEL, input_type="document")
        all_embeddings.extend(result.embeddings)
        if i + EMBED_BATCH < len(texts):
            time.sleep(0.1)   # rate limit buffer

    # Store in postgres
    with get_conn() as conn:
        with conn.cursor() as cur:
            records = [
                (
                    paper_db_id,
                    chunk.chunk_index,
                    chunk.section,
                    chunk.text,
                    chunk.token_estimate,
                    emb,               # list[float] — pgvector handles it
                )
                for chunk, emb in zip(chunks, all_embeddings)
            ]
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chunks
                    (paper_id, chunk_index, section, text, token_count, embedding)
                VALUES %s
                """,
                records,
                template="(%s, %s, %s, %s, %s, %s::vector)",
            )

    console.print(f"[green]✓ Stored {len(chunks)} chunks[/green]")
    return len(chunks)


def embed_query(query: str) -> list[float]:
    """Embed a single user query. Uses 'query' input_type — important."""
    result = vo.embed([query], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]
