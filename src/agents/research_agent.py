import json
import os
from typing import Annotated, TypedDict
import operator

import anthropic
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from .tools import AGENT_TOOLS, execute_tool
from ..observability import trace_operation, observe, langfuse_context, _langfuse_enabled

console = Console()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

AGENT_SYSTEM = """You are an expert research assistant specialising in differential privacy (DP) \
and electronic health records (EHR).

You have access to a corpus of research papers on this topic. Your job is to help \
researchers understand the field deeply — synthesising findings, spotting contradictions, \
identifying gaps, and drafting research content.

Available tools:
- search_papers: semantic search across paper text
- filter_by_concepts: find papers by ε, mechanism, dataset
- get_related_papers: trace citations and relationships
- get_corpus_summary: overview of the entire corpus
- ask_question: get a cited answer to a specific question

Approach:
1. For synthesis tasks — start with get_corpus_summary, then search_papers for specifics
2. For gap analysis — search what exists, then reason about what's missing
3. For contradiction finding — filter_by_concepts to find comparable papers, then compare
4. Always ground claims in specific papers with year and author
5. Be precise about ε and δ values — these are critical in DP research
6. When you have enough information, produce a well-structured response"""


# ── State ──────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    messages:        Annotated[list, operator.add]
    query:           str
    query_type:      str   # 'qa', 'synthesis', 'gap', 'literature_review'
    tool_calls_made: int
    final_response:  str


# ── Nodes ──────────────────────────────────────────────────────────────────
def classify_query(state: ResearchState) -> dict:
    """
    Classify the research query to guide the agent's strategy.
    Simple LLM call — cheap, fast, no tools needed.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        temperature=0.0,
        system=(
            "Classify this research query into exactly one category. "
            "Reply with one word only: qa | synthesis | gap | literature_review\n\n"
            "qa = specific factual question about papers\n"
            "synthesis = summarise/compare findings across papers\n"
            "gap = identify missing research or open problems\n"
            "literature_review = draft a literature review section"
        ),
        messages=[{"role": "user", "content": state["query"]}],
    )
    query_type = response.content[0].text.strip().lower()
    if query_type not in ("qa", "synthesis", "gap", "literature_review"):
        query_type = "qa"   # safe default

    console.print(f"[dim][classify] query_type = {query_type}[/dim]")
    return {"query_type": query_type}


def _clean_messages_for_api(messages: list) -> list:
    """
    Strip internal bookkeeping keys before sending to the Anthropic API.
    Claude rejects unknown fields like _stop_reason with a 400 error.
    """
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict):
            cleaned.append({k: v for k, v in msg.items() if not k.startswith("_")})
        else:
            cleaned.append(msg)
    return cleaned

def call_agent(state: ResearchState) -> dict:
    """
    Main agent node — calls Claude with tools.
    Injects query-type-specific guidance into the first call.
    """
    messages = state["messages"]

    # On first call, inject strategy hint based on query type
    if state["tool_calls_made"] == 0:
        strategy_hints = {
            "qa": (
                "Strategy: use ask_question for a direct cited answer. "
                "If the question involves specific ε values or mechanisms, "
                "also use filter_by_concepts."
            ),
            "synthesis": (
                "Strategy: start with get_corpus_summary to understand the landscape. "
                "Then use search_papers and filter_by_concepts to gather evidence. "
                "Compare findings across papers, noting agreements and disagreements."
            ),
            "gap": (
                "Strategy: use get_corpus_summary and search_papers to map what exists. "
                "Then reason carefully about what combinations, settings, or problems "
                "are NOT addressed in the literature."
            ),
            "literature_review": (
                "Strategy: use get_corpus_summary first, then search_papers for each "
                "major theme. Draft a structured literature review with clear sections, "
                "citing papers as [Author et al., Year]."
            ),
        }
        hint = strategy_hints.get(state["query_type"], "")
        if hint and messages:
            # Prepend strategy hint to the first user message
            first_msg = messages[0]
            augmented = {
                **first_msg,
                "content": f"{hint}\n\nResearch task: {first_msg['content']}"
            }
            messages = [augmented] + messages[1:]
    api_messages = _clean_messages_for_api(messages)

    with trace_operation(
        f"agent_llm_call_{state['tool_calls_made']}",
        metadata={
            "tool_calls_so_far": state["tool_calls_made"],
            "query_type":        state["query_type"],
        }
    ) as ctx:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=AGENT_SYSTEM,
            tools=AGENT_TOOLS,
            messages=api_messages,
        )
        ctx["input_tokens"]  = response.usage.input_tokens
        ctx["output_tokens"] = response.usage.output_tokens
    console.print(
        f"[dim][agent] stop_reason={response.stop_reason} "
        f"tool_calls={state['tool_calls_made']}[/dim]"
    )

    return {
        "messages": [{
            "role":         "assistant",
            "content":      response.content,
            "_stop_reason": response.stop_reason,
        }],
        "tool_calls_made": state["tool_calls_made"],
    }


def run_tools(state: ResearchState) -> dict:
    """Execute all tool calls from the last assistant message."""
    last = state["messages"][-1]
    tool_results = []
    calls_made = 0

    for block in last["content"]:
        if block.type != "tool_use":
            continue

        console.print(f"[cyan]  → {block.name}({json.dumps(block.input)[:80]})[/cyan]")
        result = execute_tool(block.name, block.input)

        # Preview result in console (truncated)
        preview = result[:120].replace("\n", " ")
        console.print(f"[dim]    ← {preview}...[/dim]")

        tool_results.append({
            "type":        "tool_result",
            "tool_use_id": block.id,
            "content":     result,
        })
        calls_made += 1

    return {
        "messages":        [{"role": "user", "content": tool_results}],
        "tool_calls_made": state["tool_calls_made"] + calls_made,
    }


def extract_final_response(state: ResearchState) -> dict:
    """Pull the final text response from the last assistant message."""
    for msg in reversed(state["messages"]):
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if hasattr(block, "text") and block.text:
                return {"final_response": block.text}
    return {"final_response": "No response generated."}


# ── Router ─────────────────────────────────────────────────────────────────
def should_continue(state: ResearchState) -> str:
    last = state["messages"][-1]

    # Hard cap — prevent runaway loops
    if state["tool_calls_made"] >= 12:
        console.print("[yellow][router] tool cap reached → finishing[/yellow]")
        return "finish"

    stop_reason = last.get("_stop_reason", "")
    if stop_reason == "tool_use":
        return "run_tools"

    return "finish"


# ── Build the graph ────────────────────────────────────────────────────────
def build_agent() -> any:
    builder = StateGraph(ResearchState)

    builder.add_node("classify",  classify_query)
    builder.add_node("agent",     call_agent)
    builder.add_node("run_tools", run_tools)
    builder.add_node("finish",    extract_final_response)

    builder.set_entry_point("classify")
    builder.add_edge("classify",  "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"run_tools": "run_tools", "finish": "finish"}
    )
    builder.add_edge("run_tools", "agent")
    builder.add_edge("finish",    END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Singleton — built once, reused across requests
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent

def _make_observed_research_query():
    def _run_impl(query: str, thread_id: str = "default") -> dict:
        console.print(f"\n[bold]Research query:[/bold] {query}")

        if _langfuse_enabled and langfuse_context:
            try:
                langfuse_context.update_current_trace(
                    name="research_agent",
                    session_id=thread_id,
                    tags=["agent", "dp-research"],
                    metadata={"query": query[:200]},
                )
            except Exception:
                pass

        agent = get_agent()
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "messages":        [{"role": "user", "content": query}],
            "query":           query,
            "query_type":      "qa",
            "tool_calls_made": 0,
            "final_response":  "",
        }

        result = agent.invoke(initial_state, config=config)

        return {
            "response":        result["final_response"],
            "query_type":      result["query_type"],
            "tool_calls_made": result["tool_calls_made"],
        }

    if _langfuse_enabled and observe is not None:
        return observe(name="research_agent")(_run_impl)
    return _run_impl


run_research_query = _make_observed_research_query()
