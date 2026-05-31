# Full replacement for src/retrieval/qa.py

import os
from dataclasses import dataclass

import anthropic
from rich.console import Console

from .search import search, SearchResult
from ..observability import trace_operation, observe, _langfuse_enabled

console = Console()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a research assistant specialising in differential privacy (DP) and \
electronic health records (EHR).

Your answers are grounded strictly in the provided research paper excerpts.

Rules:
- Answer using ONLY the provided context. Never use outside knowledge.
- For every claim you make, cite the paper using [Author et al., Year] or [Paper Title] format.
- If the context doesn't contain enough information, say exactly:
  "The provided papers don't contain enough information to answer this."
- When papers disagree, highlight the disagreement explicitly.
- Use precise technical language — your colleague is a researcher, not a beginner.
- Structure longer answers with clear headings."""


@dataclass
class QAResult:
    answer:        str
    sources:       list[SearchResult]
    input_tokens:  int
    output_tokens: int


def _make_observed_answer():
    """
    Build the answer function with @observe if Langfuse is enabled,
    otherwise return a plain function. This avoids errors when Langfuse
    is not configured.
    """
    def _answer_impl(
        question:  str,
        top_k:     int = 4,
        paper_ids: list[int] | None = None,
        user_id:   str = "anonymous",
        session_id: str = "default",
        stream:     bool = False,
    ) -> QAResult:
        # 1. Retrieve
        results = search(question, top_k_final=top_k, paper_ids=paper_ids)

        if not results:
            return QAResult(
                answer="No relevant paper excerpts found. Please ingest papers first.",
                sources=[],
                input_tokens=0,
                output_tokens=0,
            )

        # 2. Build context
        context_parts = []
        for i, r in enumerate(results, 1):
            authors_short = _format_authors(r.authors)
            header = (
                f"[{i}] {r.paper_title} — "
                f"{authors_short}, {r.year or 'n.d.'} (§{r.section})"
            )
            context_parts.append(f"{header}\n{r.text}")

        context = "\n\n---\n\n".join(context_parts)
        user_message = (
            f"Research question: {question}\n\n"
            f"--- Paper excerpts ---\n\n{context}\n\n---\n\n"
            f"Answer based solely on the excerpts above. "
            f"Cite sources by number [1], [2], etc."
        )

        # 3. Generate — traced
        with trace_operation(
            "llm_generation",
            user_id=user_id,
            session_id=session_id,
            metadata={"question": question[:100], "chunks_retrieved": len(results)},
        ) as ctx:
            if stream:
                full_text = ""
                with client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    temperature=0.0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                ) as s:
                    for text in s.text_stream:
                        print(text, end="", flush=True)
                        full_text += text
                    final = s.get_final_message()
                    ctx["input_tokens"]  = final.usage.input_tokens
                    ctx["output_tokens"] = final.usage.output_tokens
                print()
                answer_text = full_text
            else:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    temperature=0.0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                ctx["input_tokens"]  = response.usage.input_tokens
                ctx["output_tokens"] = response.usage.output_tokens
                answer_text = response.content[0].text
        return QAResult(
            answer=response.content[0].text,
            sources=results,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    # Wrap with @observe only if Langfuse is active
    if _langfuse_enabled and observe is not None:
        return observe(name="rag_qa")(_answer_impl)
    return _answer_impl


answer = _make_observed_answer()


def _format_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0].split()[-1]
    if len(authors) == 2:
        return f"{authors[0].split()[-1]} & {authors[1].split()[-1]}"
    return f"{authors[0].split()[-1]} et al."
