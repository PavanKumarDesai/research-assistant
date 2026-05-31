import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import anthropic
from rich.console import Console
from rich.table import Table

from ..retrieval.qa import answer
from ..agents.research_agent import run_research_query

console = Console()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── Eval case definitions ──────────────────────────────────────────────────

@dataclass
class EvalCase:
    id:           str
    query:        str
    eval_type:    str          # 'exact', 'contains', 'llm_judge'
    expected:     str          # for exact/contains; criteria for llm_judge
    use_agent:    bool = False # True = run through /research agent


@dataclass
class EvalResult:
    case_id:    str
    query:      str
    actual:     str
    passed:     bool
    score:      float          # 0.0 – 1.0
    reason:     str
    latency_s:  float
    cost_usd:   float = 0.0


# ── Domain-specific eval cases ─────────────────────────────────────────────
# These are grounded in what the papers actually contain.
# Add more as your colleague identifies important invariants.

EVAL_CASES = [
    # ── Exact / contains checks (fast, cheap) ─────────────────────────────
    EvalCase(
        id="dp-sgd-authors",
        query="Who proposed DP-SGD?",
        eval_type="contains",
        expected="abadi",   # Abadi et al. 2016 — must be cited
        use_agent=False,
    ),
    EvalCase(
        id="epsilon-definition",
        query="What does the epsilon parameter control in differential privacy?",
        eval_type="contains",
        expected="privacy",
        use_agent=False,
    ),
    EvalCase(
        id="gaussian-mechanism",
        query="What is the Gaussian mechanism in differential privacy?",
        eval_type="contains",
        expected="noise",
        use_agent=False,
    ),
    EvalCase(
        id="no-hallucination",
        query="What did the 2019 paper on quantum EHR blockchain report?",
        eval_type="contains",
        expected="don't have enough information",  # must refuse, not hallucinate
        use_agent=False,
    ),

    # ── LLM-as-judge checks (quality, depth, accuracy) ────────────────────
    EvalCase(
        id="epsilon-tradeoff-quality",
        query="What is the tradeoff between epsilon and model accuracy in DP training?",
        eval_type="llm_judge",
        expected=(
            "Answer must explain that lower epsilon = stronger privacy but lower accuracy. "
            "Must reference specific epsilon values or papers. "
            "Must not hallucinate results not in the corpus."
        ),
        use_agent=False,
    ),
    EvalCase(
        id="synthesis-quality",
        query="Synthesise the main approaches to differentially private EHR model training.",
        eval_type="llm_judge",
        expected=(
            "Must cover multiple papers and approaches. "
            "Must cite specific papers with years. "
            "Must be well-structured and accurate. "
            "Must not contradict itself."
        ),
        use_agent=True,
    ),
    EvalCase(
        id="gap-identification",
        query="What are open research gaps in DP for EHR transformers?",
        eval_type="llm_judge",
        expected=(
            "Must identify specific, concrete gaps — not vague statements. "
            "Gaps must be grounded in what the papers do and don't cover. "
            "Should mention at least 2 distinct research directions."
        ),
        use_agent=True,
    ),
    EvalCase(
        id="citation-accuracy",
        query="What privacy budget did Abadi et al. use in their experiments?",
        eval_type="llm_judge",
        expected=(
            "Must cite Abadi et al. correctly. "
            "Must mention specific epsilon values. "
            "Must not attribute results to wrong papers."
        ),
        use_agent=False,
    ),
]


# ── Evaluation runners ─────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating the quality of a research assistant's answer.

Criteria: {criteria}

Question asked: {question}
Answer given: {answer}

Score the answer from 1-5 on whether it meets ALL the criteria:
1 = fails criteria entirely
2 = partially meets criteria with significant gaps
3 = meets most criteria with minor gaps
4 = meets all criteria well
5 = meets all criteria excellently

Respond ONLY with JSON: {{"score": int, "reason": str}}"""


def _run_single_case(case: EvalCase) -> EvalResult:
    """Run one eval case and return the result."""
    start = time.time()
    cost = 0.0

    try:
        # Get the system's answer
        if case.use_agent:
            result = run_research_query(
                case.query,
                thread_id=f"eval-{case.id}"
            )
            actual = result["response"]
        else:
            result = answer(case.query, top_k=4)
            actual = result.answer
            # Estimate cost
            cost = (
                result.input_tokens * 3.0 +
                result.output_tokens * 15.0
            ) / 1_000_000

        latency = time.time() - start

        # Evaluate based on type
        if case.eval_type == "exact":
            passed = actual.strip().lower() == case.expected.strip().lower()
            score  = 1.0 if passed else 0.0
            reason = "exact match" if passed else f"expected '{case.expected}'"

        elif case.eval_type == "contains":
            passed = case.expected.lower() in actual.lower()
            score  = 1.0 if passed else 0.0
            reason = "keyword found" if passed else f"'{case.expected}' not in response"

        elif case.eval_type == "llm_judge":
            judgment = _llm_judge(case.query, actual, case.expected)
            score    = judgment["score"] / 5.0   # normalise to 0-1
            passed   = judgment["score"] >= 4    # 4+ = passing
            reason   = judgment["reason"]
            # Add judge cost
            cost += 0.001   # rough estimate for judge call

        else:
            passed, score, reason = False, 0.0, f"Unknown eval_type: {case.eval_type}"

    except Exception as e:
        latency = time.time() - start
        actual  = f"ERROR: {e}"
        passed  = False
        score   = 0.0
        reason  = str(e)

    return EvalResult(
        case_id=case.id,
        query=case.query,
        actual=actual[:300],   # truncate for display
        passed=passed,
        score=score,
        reason=reason,
        latency_s=round(latency, 2),
        cost_usd=round(cost, 5),
    )


def _llm_judge(question: str, answer_text: str, criteria: str) -> dict:
    """Use Claude to judge answer quality against criteria."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                criteria=criteria,
                question=question,
                answer=answer_text[:1500],   # cap to save tokens
            )
        }],
    )
    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 1, "reason": f"Could not parse judge output: {raw[:100]}"}


# ── Main eval runner ───────────────────────────────────────────────────────

def run_evals(
    cases:        list[EvalCase] = None,
    save_results: bool = True,
) -> dict:
    """
    Run the full eval suite and return a summary.
    Prints a rich table to the console.
    """
    cases = cases or EVAL_CASES
    console.print(f"\n[bold]Running {len(cases)} eval cases...[/bold]\n")

    results: list[EvalResult] = []
    for i, case in enumerate(cases, 1):
        console.print(f"[dim]({i}/{len(cases)}) {case.id}...[/dim]", end=" ")
        result = _run_single_case(case)
        results.append(result)
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(f"{status} ({result.latency_s}s)")

    # ── Summary table ──────────────────────────────────────────────────────
    table = Table(title="Eval Results", show_lines=True)
    table.add_column("ID",      style="dim",   width=22)
    table.add_column("Type",    width=10)
    table.add_column("Result",  width=6)
    table.add_column("Score",   width=7)
    table.add_column("Latency", width=8)
    table.add_column("Reason",  width=40)

    for r in results:
        table.add_row(
            r.case_id,
            next(c.eval_type for c in cases if c.id == r.case_id),
            "✓ PASS" if r.passed else "✗ FAIL",
            f"{r.score:.2f}",
            f"{r.latency_s}s",
            r.reason[:38],
        )

    console.print(table)

    passed     = sum(r.passed for r in results)
    total      = len(results)
    avg_score  = sum(r.score for r in results) / total
    total_cost = sum(r.cost_usd for r in results)
    total_time = sum(r.latency_s for r in results)

    summary = {
        "timestamp":   datetime.now().isoformat(),
        "passed":      passed,
        "total":       total,
        "pass_rate":   round(passed / total, 3),
        "avg_score":   round(avg_score, 3),
        "total_cost":  round(total_cost, 4),
        "total_time_s": round(total_time, 1),
        "failures":    [
            {"id": r.case_id, "reason": r.reason, "actual": r.actual}
            for r in results if not r.passed
        ],
    }

    console.print(
        f"\n[bold]Result: {passed}/{total} passed "
        f"({summary['pass_rate']:.0%}) | "
        f"avg score: {avg_score:.2f} | "
        f"cost: ${total_cost:.4f}[/bold]"
    )

    # ── Save to disk for regression tracking ──────────────────────────────
    if save_results:
        out_dir = Path("evals")
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_file.write_text(json.dumps(summary, indent=2))
        console.print(f"[dim]Saved to {out_file}[/dim]")

    return summary


def check_regression(
    current_pass_rate: float,
    baseline_path: str = "evals/baseline.json",
    tolerance: float = 0.10,   # allow up to 10% regression
) -> bool:
    """
    Compare current pass rate against a saved baseline.
    Returns True if no regression detected.
    """
    baseline_file = Path(baseline_path)
    if not baseline_file.exists():
        console.print(
            "[yellow]No baseline found. "
            "Run save_baseline() after a good eval run.[/yellow]"
        )
        return True

    baseline = json.loads(baseline_file.read_text())
    baseline_rate = baseline["pass_rate"]
    threshold = baseline_rate - tolerance

    passed = current_pass_rate >= threshold
    console.print(
        f"Regression check: current={current_pass_rate:.0%} "
        f"baseline={baseline_rate:.0%} "
        f"threshold={threshold:.0%} → "
        f"{'[green]OK[/green]' if passed else '[red]REGRESSION[/red]'}"
    )
    return passed


def save_baseline(summary: dict, path: str = "evals/baseline.json"):
    """Save current eval results as the regression baseline."""
    Path(path).write_text(json.dumps(summary, indent=2))
    console.print(f"[green]✓ Baseline saved to {path}[/green]")
