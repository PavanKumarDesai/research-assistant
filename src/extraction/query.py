from ..db import get_conn


def search_by_concepts(
    epsilon_max: float | None = None,
    epsilon_min: float | None = None,
    mechanisms: list[str] | None = None,
    datasets: list[str] | None = None,
    is_federated: bool | None = None,
    tasks: list[str] | None = None,
) -> list[dict]:
    """
    Filter papers by their extracted technical concepts.

    Example: find all papers using ε ≤ 1 with MIMIC-III:
        search_by_concepts(epsilon_max=1.0, datasets=["mimic-iii"])
    """
    conditions = []
    params = []

    if epsilon_max is not None:
        conditions.append("c.epsilon_min <= %s")
        params.append(epsilon_max)

    if epsilon_min is not None:
        conditions.append("c.epsilon_max >= %s")
        params.append(epsilon_min)

    if mechanisms:
        # Match any of the provided mechanisms (case-insensitive overlap)
        conditions.append(
            "c.dp_mechanisms && %s::text[]"   # && = array overlap operator
        )
        params.append(mechanisms)

    if datasets:
        conditions.append("c.datasets && %s::text[]")
        params.append(datasets)

    if is_federated is not None:
        conditions.append("c.is_federated = %s")
        params.append(is_federated)

    if tasks:
        conditions.append("c.tasks && %s::text[]")
        params.append(tasks)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    query = f"""
        SELECT
            p.id, p.title, p.authors, p.year,
            c.epsilon_values, c.delta_values,
            c.dp_mechanisms, c.datasets, c.tasks,
            c.is_federated, c.framework,
            c.utility_metric, c.main_claim
        FROM papers p
        JOIN paper_concepts c ON c.paper_id = p.id
        {where}
        ORDER BY p.year DESC NULLS LAST, p.id
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    return [
        {
            "paper_id":      r[0],
            "title":         r[1],
            "authors":       r[2] or [],
            "year":          r[3],
            "epsilon_values": r[4] or [],
            "delta_values":  r[5] or [],
            "dp_mechanisms": r[6] or [],
            "datasets":      r[7] or [],
            "tasks":         r[8] or [],
            "is_federated":  r[9],
            "framework":     r[10],
            "utility_metric": r[11],
            "main_claim":    r[12],
        }
        for r in rows
    ]


def get_concept_summary() -> dict:
    """
    Aggregate view across all papers — useful for a dashboard or overview.
    Returns counts, ranges, and most common values.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Paper count with concepts extracted
            cur.execute("SELECT COUNT(*) FROM paper_concepts")
            total = cur.fetchone()[0]

            # Epsilon range across all papers
            cur.execute("""
                SELECT MIN(epsilon_min), MAX(epsilon_max),
                       AVG(epsilon_min), AVG(epsilon_max)
                FROM paper_concepts
                WHERE epsilon_min IS NOT NULL
            """)
            eps = cur.fetchone()

            # Most common mechanisms
            cur.execute("""
                SELECT unnest(dp_mechanisms) AS mech, COUNT(*) AS cnt
                FROM paper_concepts
                GROUP BY mech
                ORDER BY cnt DESC
                LIMIT 10
            """)
            mechanisms = [{"mechanism": r[0], "count": r[1]} for r in cur.fetchall()]

            # Most common datasets
            cur.execute("""
                SELECT unnest(datasets) AS ds, COUNT(*) AS cnt
                FROM paper_concepts
                GROUP BY ds
                ORDER BY cnt DESC
                LIMIT 10
            """)
            datasets = [{"dataset": r[0], "count": r[1]} for r in cur.fetchall()]

            # Federated vs centralised split
            cur.execute("""
                SELECT is_federated, COUNT(*) FROM paper_concepts
                GROUP BY is_federated
            """)
            fed = {str(r[0]): r[1] for r in cur.fetchall()}

    return {
        "total_papers_extracted": total,
        "epsilon_range": {
            "min":     eps[0],
            "max":     eps[1],
            "avg_min": round(eps[2], 4) if eps[2] else None,
            "avg_max": round(eps[3], 4) if eps[3] else None,
        } if eps and eps[0] is not None else {},
        "top_mechanisms": mechanisms,
        "top_datasets":   datasets,
        "federated_split": fed,
    }
