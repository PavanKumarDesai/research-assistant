import os
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

@contextmanager
def get_conn():
    """Context manager — auto-commits and closes."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def setup_schema():
    """Create tables and indexes. Safe to run multiple times."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Papers table — one row per paper
            cur.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id          SERIAL PRIMARY KEY,
                    arxiv_id    TEXT UNIQUE,          -- null for local PDFs
                    title       TEXT NOT NULL,
                    authors     TEXT[],
                    year        INTEGER,
                    abstract    TEXT,
                    source      TEXT NOT NULL,        -- 'arxiv' or 'local'
                    filename    TEXT,                 -- original filename
                    ingested_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Chunks table — one row per chunk, FK to papers
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          SERIAL PRIMARY KEY,
                    paper_id    INTEGER REFERENCES papers(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    section     TEXT,                 -- 'abstract', 'introduction', etc.
                    text        TEXT NOT NULL,
                    token_count INTEGER,
                    embedding   vector(1024)          -- voyage-3 dimensions
                )
            """)

            # HNSW index on embeddings — fast approximate cosine search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

            # Index for filtering by paper
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_paper_id_idx
                ON chunks (paper_id)
            """)

            # ── Phase 2: concept extraction ───────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS paper_concepts (
                    id              SERIAL PRIMARY KEY,
                    paper_id        INTEGER REFERENCES papers(id) ON DELETE CASCADE,

                    -- Privacy budget parameters
                    epsilon_values  FLOAT[],          -- e.g. [0.1, 1.0, 8.0]
                    delta_values    FLOAT[],          -- e.g. [1e-5, 1e-6]
                    epsilon_min     FLOAT,            -- for easy range queries
                    epsilon_max     FLOAT,

                    -- Technical approach
                    dp_mechanisms   TEXT[],           -- ['gaussian', 'laplace', 'dp-sgd']
                    model_types     TEXT[],           -- ['transformer', 'lstm', 'logistic']
                    datasets        TEXT[],           -- ['mimic', 'eicu', 'physionet']
                    tasks           TEXT[],           -- ['classification', 'generation']

                    -- Federated / centralised
                    is_federated    BOOLEAN,
                    framework       TEXT,             -- 'tensorflow-privacy', 'opacus', etc.

                    -- Claims
                    utility_metric  TEXT,             -- e.g. 'AUC 0.82 at ε=1'
                    main_claim      TEXT,             -- one-sentence summary
                    limitations     TEXT[],

                    -- Raw LLM output for debugging
                    raw_extraction  JSONB,
                    extracted_at    TIMESTAMPTZ DEFAULT NOW(),

                    UNIQUE (paper_id)                 -- one extraction per paper
                )
            """)

            # ── Phase 2: paper relationships ──────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS paper_relationships (
                    id              SERIAL PRIMARY KEY,
                    source_paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
                    target_paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
                    relationship    TEXT NOT NULL,    -- 'cites', 'extends', 'contradicts', 'similar'
                    evidence        TEXT,             -- quote or reason for the relationship
                    similarity_score FLOAT,           -- embedding cosine sim (for 'similar')
                    extracted_at    TIMESTAMPTZ DEFAULT NOW(),

                    UNIQUE (source_paper_id, target_paper_id, relationship)
                )
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS concepts_epsilon_idx
                ON paper_concepts (epsilon_min, epsilon_max)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS relationships_source_idx
                ON paper_relationships (source_paper_id)
            """)

    print("✓ Schema ready")

if __name__ == "__main__":
    setup_schema()
