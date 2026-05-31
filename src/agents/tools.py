import json
from ..retrieval.search import search
from ..retrieval.qa import answer
from ..extraction.query import search_by_concepts, get_concept_summary
from ..extraction.relationships import get_related_papers

# ── Tool definitions (passed to Claude) ───────────────────────────────────
AGENT_TOOLS = [
    {
        "name": "search_papers",
        "description": (
            "Semantically search across all ingested research papers. "
            "Use for finding relevant passages, quotes, or evidence on any topic. "
            "Returns ranked text chunks with paper titles and sections."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "filter_by_concepts",
        "description": (
            "Filter papers by their extracted technical concepts. "
            "Use when you need papers with specific privacy parameters or methods. "
            "Example: find all papers using epsilon <= 1 with MIMIC-III dataset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "epsilon_max": {
                    "type": "number",
                    "description": "Maximum epsilon value (privacy budget upper bound)"
                },
                "epsilon_min": {
                    "type": "number",
                    "description": "Minimum epsilon value"
                },
                "mechanisms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "DP mechanisms to filter by e.g. ['gaussian', 'dp-sgd']"
                },
                "datasets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Dataset names e.g. ['mimic-iii', 'eicu']"
                },
                "is_federated": {
                    "type": "boolean",
                    "description": "Filter to federated (true) or centralised (false) only"
                }
            }
        }
    },
    {
        "name": "get_related_papers",
        "description": (
            "Get papers related to a specific paper by ID. "
            "Returns relationship type (extends, contradicts, similar) and evidence. "
            "Use to trace how ideas evolved or find contradicting results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "integer",
                    "description": "Paper ID from the database"
                }
            },
            "required": ["paper_id"]
        }
    },
    {
        "name": "get_corpus_summary",
        "description": (
            "Get an aggregate summary of the entire paper corpus: "
            "epsilon ranges, most common mechanisms, datasets, federated vs centralised split. "
            "Use at the start of synthesis tasks to understand the landscape."
        ),
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "ask_question",
        "description": (
            "Ask a focused question and get a cited answer grounded in the papers. "
            "Use for specific factual questions where you need precise citations. "
            "Better than search_papers when you want a synthesised answer, not raw chunks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The specific question to answer"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of paper chunks to retrieve (default 4)",
                    "default": 4
                }
            },
            "required": ["question"]
        }
    }
]


# ── Tool implementations ───────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool by name and return JSON string result.
    This is called by the agent loop whenever Claude requests a tool.
    """
    try:
        if tool_name == "search_papers":
            results = search(
                query=tool_input["query"],
                top_k_final=tool_input.get("top_k", 5)
            )
            return json.dumps([
                {
                    "paper_title": r.paper_title,
                    "authors":     r.authors,
                    "year":        r.year,
                    "section":     r.section,
                    "text":        r.text,
                    "score":       round(r.score, 3),
                }
                for r in results
            ])

        elif tool_name == "filter_by_concepts":
            results = search_by_concepts(
                epsilon_max=tool_input.get("epsilon_max"),
                epsilon_min=tool_input.get("epsilon_min"),
                mechanisms=tool_input.get("mechanisms"),
                datasets=tool_input.get("datasets"),
                is_federated=tool_input.get("is_federated"),
            )
            return json.dumps(results)

        elif tool_name == "get_related_papers":
            results = get_related_papers(tool_input["paper_id"])
            return json.dumps(results)

        elif tool_name == "get_corpus_summary":
            return json.dumps(get_concept_summary())

        elif tool_name == "ask_question":
            result = answer(
                question=tool_input["question"],
                top_k=tool_input.get("top_k", 4),
            )
            return json.dumps({
                "answer":  result.answer,
                "sources": [
                    {
                        "title":   s.paper_title,
                        "year":    s.year,
                        "section": s.section,
                    }
                    for s in result.sources
                ],
            })

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})
