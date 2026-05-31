import time
import urllib.request
from pathlib import Path

import arxiv
from rich.console import Console

console = Console()
PAPERS_DIR = Path("data/papers")
PAPERS_DIR.mkdir(parents=True, exist_ok=True)


def download_arxiv_paper(arxiv_id: str) -> tuple[Path, dict]:
    """
    Download a paper from ArXiv by ID.
    Returns (local_pdf_path, metadata_dict).
    """
    clean_id = arxiv_id.split("v")[0].strip()

    # Check if already downloaded
    existing = list(PAPERS_DIR.glob(f"{clean_id}*.pdf"))
    if existing:
        console.print(f"[dim]Already downloaded: {existing[0].name}[/dim]")
        meta = _fetch_metadata(clean_id)
        return existing[0], meta

    console.print(f"[cyan]Fetching metadata for {clean_id}...[/cyan]")
    meta = _fetch_metadata(clean_id)

    # Build the direct PDF URL — always works regardless of library version
    pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
    output_path = PAPERS_DIR / f"{clean_id}.pdf"

    console.print(f"[cyan]Downloading PDF from {pdf_url}...[/cyan]")

    # Download with a browser-like User-Agent so ArXiv doesn't block us
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; research-tool/1.0; "
            "mailto:researcher@example.com)"
        )
    }
    req = urllib.request.Request(pdf_url, headers=headers)

    with urllib.request.urlopen(req, timeout=60) as response:
        pdf_bytes = response.read()

    output_path.write_bytes(pdf_bytes)
    console.print(f"[green]✓ Downloaded: {output_path.name} "
                  f"({len(pdf_bytes) / 1024:.0f} KB)[/green]")

    time.sleep(1)   # be polite to ArXiv
    return output_path, meta


def _fetch_metadata(arxiv_id: str) -> dict:
    """Fetch paper metadata via the arxiv library."""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])

    try:
        paper = next(client.results(search))
    except StopIteration:
        raise ValueError(f"ArXiv paper not found: {arxiv_id}")

    return {
        "arxiv_id": arxiv_id,
        "title":    paper.title,
        "authors":  [str(a) for a in paper.authors],
        "abstract": paper.summary,
        "year":     paper.published.year if paper.published else None,
        "source":   "arxiv",
    }


def list_local_pdfs() -> list[Path]:
    return sorted(PAPERS_DIR.glob("*.pdf"))
