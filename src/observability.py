import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from rich.console import Console

console = Console()

COST_RATES = {
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":  {"input": 0.80,  "output": 4.00},
    "claude-opus-4-6":   {"input": 15.00, "output": 75.00},
}


# ── Cost tracker ───────────────────────────────────────────────────────────
@dataclass
class CostTracker:
    total_input_tokens:  int   = 0
    total_output_tokens: int   = 0
    total_calls:         int   = 0
    total_cost_usd:      float = 0.0
    session_log:         list  = field(default_factory=list)

    def record(
        self,
        input_tokens:  int,
        output_tokens: int,
        model:         str = "claude-sonnet-4-6",
        operation:     str = "",
    ) -> float:
        rates = COST_RATES.get(model, COST_RATES["claude-sonnet-4-6"])
        cost = (
            input_tokens  * rates["input"]  / 1_000_000 +
            output_tokens * rates["output"] / 1_000_000
        )
        self.total_input_tokens  += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls         += 1
        self.total_cost_usd      += cost
        self.session_log.append({
            "operation":     operation,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      round(cost, 6),
            "model":         model,
        })
        return cost

    @property
    def summary(self) -> dict:
        return {
            "total_calls":         self.total_calls,
            "total_input_tokens":  self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd":      round(self.total_cost_usd, 4),
            "avg_cost_per_call":   round(
                self.total_cost_usd / max(self.total_calls, 1), 5
            ),
        }


tracker = CostTracker()


# ── Langfuse v4 setup ──────────────────────────────────────────────────────
_langfuse_enabled = False
_langfuse_client  = None   # raw client for flush()
observe           = None
langfuse_context  = None   # get_client() in v4


def _noop_observe(name=None, **kwargs):
    """No-op when Langfuse is disabled — code using @observe still works."""
    def decorator(fn):
        return fn
    return decorator


def _setup_langfuse():
    global _langfuse_enabled, _langfuse_client, observe, langfuse_context

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        console.print("[dim]Langfuse: keys not set — tracing disabled[/dim]")
        observe = _noop_observe
        return

    # Set env vars — v4 reads these automatically
    os.environ["LANGFUSE_PUBLIC_KEY"]  = public_key
    os.environ["LANGFUSE_SECRET_KEY"]  = secret_key
    os.environ.setdefault(
        "LANGFUSE_HOST",
        os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )

    try:
        import langfuse as _lf_pkg
        console.print(f"[dim]Langfuse version: {_lf_pkg.__version__}[/dim]")

        # v4 API
        from langfuse import observe as _obs, get_client as _get_client

        observe          = _obs
        langfuse_context = _get_client   # callable — call it to get the client
        _langfuse_client = _get_client()
        _langfuse_enabled = True
        console.print("[green]✓ Langfuse v4 tracing enabled[/green]")

    except ImportError as e:
        console.print(f"[yellow]Langfuse import failed: {e}[/yellow]")
        observe = _noop_observe
    except Exception as e:
        console.print(f"[yellow]Langfuse setup failed: {e}[/yellow]")
        observe = _noop_observe


_setup_langfuse()


def flush():
    """Force-flush pending traces to Langfuse."""
    if _langfuse_client:
        try:
            _langfuse_client.flush()
        except Exception:
            pass


def send_test_trace():
    """Send a test trace on startup to verify the connection."""
    if not _langfuse_enabled:
        console.print("[dim]Langfuse disabled — skipping test trace[/dim]")
        return

    @observe(name="connection-test")
    def _test():
        return "ok"

    try:
        _test()
        flush()
        console.print("[green]✓ Langfuse test trace sent — check dashboard[/green]")
    except Exception as e:
        console.print(f"[yellow]Test trace failed: {e}[/yellow]")


# ── trace_operation ────────────────────────────────────────────────────────
@contextmanager
def trace_operation(
    name:       str,
    user_id:    str  = "anonymous",
    session_id: str  = "default",
    metadata:   dict = None,
):
    """
    Trace a block of code. Always tracks cost locally.
    When called inside an @observe function, creates a nested span in Langfuse.

    Usage:
        with trace_operation("llm_call") as ctx:
            response = client.messages.create(...)
            ctx["input_tokens"]  = response.usage.input_tokens
            ctx["output_tokens"] = response.usage.output_tokens
    """
    ctx = {"name": name, "start": time.time()}

    # Tag the parent trace with user/session info (v4 API)
    if _langfuse_enabled and _langfuse_client:
        try:
            lf = langfuse_context()   # get_client() returns active client
            lf.update_current_trace(
                user_id=user_id,
                session_id=session_id,
                tags=["dp-research-assistant"],
            )
        except Exception:
            pass   # not inside @observe — fine, just skip

    try:
        yield ctx
    finally:
        latency = round(time.time() - ctx["start"], 3)
        ctx["latency_s"] = latency
        cost = 0.0

        if "input_tokens" in ctx and "output_tokens" in ctx:
            cost = tracker.record(
                input_tokens=ctx["input_tokens"],
                output_tokens=ctx["output_tokens"],
                operation=name,
            )
            ctx["cost_usd"] = cost

            # Update the current span with token usage (v4 API)
            if _langfuse_enabled and _langfuse_client:
                try:
                    lf = langfuse_context()
                    lf.update_current_observation(
                        name=name,
                        usage={
                            "input":  ctx["input_tokens"],
                            "output": ctx["output_tokens"],
                            "unit":   "TOKENS",
                        },
                        metadata={
                            **(metadata or {}),
                            "cost_usd":  round(cost, 6),
                            "latency_s": latency,
                        },
                    )
                except Exception:
                    pass

        # Flush after every traced operation so traces appear immediately
        flush()

        console.print(
            f"[dim][trace] {name} | "
            f"{latency}s | "
            f"${cost:.5f}[/dim]"
        )


def score_response(score: float, comment: str = "", name: str = "quality"):
    """Score the current trace. Call inside an @observe function."""
    if _langfuse_enabled and _langfuse_client:
        try:
            lf = langfuse_context()
            lf.score_current_trace(
                name=name,
                value=score,
                comment=comment,
            )
            flush()
        except Exception as e:
            console.print(f"[yellow]Could not score trace: {e}[/yellow]")
