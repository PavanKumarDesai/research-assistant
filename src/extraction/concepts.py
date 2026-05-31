import json
import os
import re
from dataclasses import dataclass

import anthropic
import psycopg2.extras
from rich.console import Console

from ..db import get_conn

console = Console()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Extraction prompt ──────────────────────────────────────────────────────
# Highly specific to the DP + EHR domain — this is what makes it useful.
EXTRACTION_SYSTEM = """You are a research analyst specialising in differential privacy (DP) \
and machine learning on electronic health records (EHR).

Extract structured technical information from research paper text.
Respond ONLY with valid JSON — no markdown fences, no prose, no explanation.

JSON schema (use null for fields not mentioned in the paper):
{
  "epsilon_values":  [float],       // all ε values reported, e.g. [0.1, 1.0, 8.0]
  "delta_values":    [float],       // all δ values reported, e.g. [1e-5]
  "dp_mechanisms":   [string],      // e.g. ["gaussian", "laplace", "dp-sgd", "pate"]
  "model_types":     [string],      // e.g. ["transformer", "lstm", "logistic regression"]
  "datasets":        [string],      // e.g. ["mimic-iii", "eicu", "physionet", "synthea"]
  "tasks":           [string],      // e.g. ["mortality prediction", "text generation", "icd coding"]
  "is_federated":    boolean,       // true if federated learning is used
  "framework":       string | null, // e.g. "opacus", "tensorflow-privacy", "autodp"
  "utility_metric":  string | null, // e.g. "AUC 0.84 at ε=1.0" — best result reported
  "main_claim":      string | null, // one sentence: what does this paper claim to achieve?
  "limitations":     [string]       // explicitly stated limitations, max 3
}

Rules:
- epsilon_values: include ALL epsilon values mentioned, even in ablations
- datasets: normalise names (mimic-iii not MIMIC, physionet not PhysioNet)
- dp_mechanisms: use lowercase standard names (gaussian not Gaussian Mechanism)
- If a field is genuinely absent from the paper, use null or []
- Never hallucinate values not present in the text"""


def extract_concepts(paper_id: int) -> dict:
    """
    Extract structured DP concepts from a paper using its stored chunks.
    Returns the extracted concept dict.
    """
    # Check if already extracted
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM paper_concepts WHERE paper_id = %s", (paper_id,)
            )
            if cur.fetchone():
                console.print(f"[yellow]Already extracted: paper_id={paper_id}[/yellow]")
                return {}

    # Fetch paper metadata + key chunks
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT title, abstract, year FROM papers WHERE id = %s", (paper_id,)
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Paper not found: {paper_id}")
            title, abstract, year = row

            # Fetch the most useful sections for extraction
            # Abstract + intro + methods tell us the most
            cur.execute(
                """
                SELECT section, text FROM chunks
                WHERE paper_id = %s
                  AND section NOT IN ('references', 'appendix')
                ORDER BY chunk_index
                LIMIT 20
                """,
                (paper_id,),
            )
            chunks = cur.fetchall()

    # Build context — abstract first, then key chunks
    context_parts = [f"Title: {title}", f"Year: {year}", f"Abstract: {abstract}"]
    for section, text in chunks:
        context_parts.append(f"[{section}]\n{text}")

    # Cap context to avoid blowing the budget
    # ~6000 chars ≈ ~1500 tokens — enough to extract all key fields
    context = "\n\n".join(context_parts)[:6000]

    console.print(f"[cyan]Extracting concepts from: {title[:60]}...[/cyan]")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        temperature=0.0,        # extraction = deterministic
        system=EXTRACTION_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Extract structured information from this paper:\n\n{context}"
        }],
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown fences if the model added them despite instructions
    raw_text = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.MULTILINE)

    try:
        concepts = json.loads(raw_text)
    except json.JSONDecodeError as e:
        console.print(f"[red]JSON parse error: {e}[/red]")
        console.print(f"[dim]Raw: {raw_text[:200]}[/dim]")
        raise

    # Store in database
    _store_concepts(paper_id, concepts, raw_json=concepts)

    console.print(f"[green]✓ Extracted: ε={concepts.get('epsilon_values')}, "
                  f"mechanisms={concepts.get('dp_mechanisms')}, "
                  f"datasets={concepts.get('datasets')}[/green]")
    return concepts


def _store_concepts(paper_id: int, concepts: dict, raw_json: dict):
    """Persist extracted concepts to the database."""
    epsilon_values = concepts.get("epsilon_values") or []
    epsilon_min = min(epsilon_values) if epsilon_values else None
    epsilon_max = max(epsilon_values) if epsilon_values else None

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO paper_concepts (
                    paper_id, epsilon_values, delta_values,
                    epsilon_min, epsilon_max,
                    dp_mechanisms, model_types, datasets, tasks,
                    is_federated, framework,
                    utility_metric, main_claim, limitations,
                    raw_extraction
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (paper_id) DO UPDATE SET
                    epsilon_values  = EXCLUDED.epsilon_values,
                    epsilon_min     = EXCLUDED.epsilon_min,
                    epsilon_max     = EXCLUDED.epsilon_max,
                    dp_mechanisms   = EXCLUDED.dp_mechanisms,
                    datasets        = EXCLUDED.datasets,
                    raw_extraction  = EXCLUDED.raw_extraction,
                    extracted_at    = NOW()
                """,
                (
                    paper_id,
                    epsilon_values or [],
                    concepts.get("delta_values") or [],
                    epsilon_min,
                    epsilon_max,
                    concepts.get("dp_mechanisms") or [],
                    concepts.get("model_types") or [],
                    concepts.get("datasets") or [],
                    concepts.get("tasks") or [],
                    concepts.get("is_federated"),
                    concepts.get("framework"),
                    concepts.get("utility_metric"),
                    concepts.get("main_claim"),
                    concepts.get("limitations") or [],
                    psycopg2.extras.Json(raw_json),
                ),
            )


def extract_all_papers() -> list[dict]:
    """Extract concepts from every paper that hasn't been extracted yet."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, title FROM papers
                WHERE id NOT IN (SELECT paper_id FROM paper_concepts)
                ORDER BY id
            """)
            pending = cur.fetchall()

    if not pending:
        console.print("[yellow]All papers already extracted.[/yellow]")
        return []

    console.print(f"[bold]Extracting concepts from {len(pending)} papers...[/bold]")
    results = []
    for paper_id, title in pending:
        try:
            result = extract_concepts(paper_id)
            results.append({"paper_id": paper_id, "title": title, **result})
        except Exception as e:
            console.print(f"[red]Failed {paper_id} ({title[:40]}): {e}[/red]")
    return results
