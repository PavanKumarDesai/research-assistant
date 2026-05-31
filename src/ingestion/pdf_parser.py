import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf


@dataclass
class ParsedSection:
    name: str
    text: str
    page_start: int
    page_end: int


@dataclass
class ParsedPaper:
    title: str
    authors: list[str]
    abstract: str
    year: int | None
    sections: list[ParsedSection]
    full_text: str
    filename: str
    total_pages: int

    @property
    def all_text_by_section(self) -> dict[str, str]:
        return {s.name: s.text for s in self.sections}


# ── Section header detection ───────────────────────────────────────────────
# Covers numbered (1. Introduction) and unnumbered (Introduction) headers
SECTION_PATTERNS = [
    r"^abstract$",
    r"^1\.?\s+introduction",
    r"^2\.?\s+(related work|background|preliminaries|prior work)",
    r"^3\.?\s+(method|approach|model|framework|proposed|algorithm|system)",
    r"^4\.?\s+(experiment|result|evaluation|empirical)",
    r"^5\.?\s+(discussion|analysis|ablation)",
    r"^6\.?\s+conclusion",
    r"^\d+\.?\s+(privacy|differential privacy|federated|training|learning|mechanism)",
    r"^(introduction|related work|background|preliminaries)$",
    r"^(methodology|methods|approach|model|framework)$",
    r"^(experiments?|results?|evaluation|empirical study)$",
    r"^(discussion|analysis|conclusion|future work)$",
    r"^references$",
    r"^appendix",
]

SECTION_RE = re.compile(
    "|".join(SECTION_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)


def _clean_text(text: str) -> str:
    """
    Remove NUL bytes and other control characters that break Postgres.
    Postgres TEXT columns cannot contain 0x00 — this is the 'NUL' error.
    """
    # Remove NUL bytes (the main culprit)
    text = text.replace("\x00", "")
    # Remove other non-printable control chars except newline/tab
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def _detect_section(text: str) -> str | None:
    """Return normalised section name if line looks like a header."""
    line = text.strip()
    if len(line) > 80 or len(line) < 3:
        return None
    if SECTION_RE.match(line.lower()):
        # Normalise: strip leading number, lowercase, trim
        name = re.sub(r"^\d+\.?\s+", "", line).strip().lower()
        return name[:40]
    return None


def _is_likely_header(text: str, font_size: float, page_max_size: float) -> bool:
    """
    Secondary heuristic: large font relative to page = likely a section header.
    Catches headers that don't match our regex patterns.
    """
    if font_size < 1:
        return False
    ratio = font_size / page_max_size if page_max_size > 0 else 0
    # Header if font is >85% of the largest font on the page and text is short
    return ratio > 0.85 and len(text.strip()) < 80 and len(text.strip()) > 3


def parse_pdf(filepath: str) -> ParsedPaper:
    """
    Extract structured text from a research paper PDF.
    Handles NUL bytes and uses both regex + font-size heuristics for sections.
    """
    path = Path(filepath)
    doc = fitz.open(str(path))
    total_pages = len(doc)

    full_text_parts: list[str] = []
    sections: list[ParsedSection] = []
    current_section = "preamble"
    current_text: list[str] = []
    section_page_start = 0

    for page_num, page in enumerate(doc):
        # Use dict mode so we get font size information per span
        page_dict = page.get_text("dict")

        # Find the largest font size on this page (used for header detection)
        all_sizes = [
            span["size"]
            for block in page_dict["blocks"]
            if block.get("type") == 0
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]
        page_max_size = max(all_sizes) if all_sizes else 12.0

        for block in page_dict["blocks"]:
            if block.get("type") != 0:  # skip image blocks
                continue

            for line in block.get("lines", []):
                # Concatenate all spans in this line
                line_text = "".join(
                    span.get("text", "") for span in line.get("spans", [])
                )
                line_text = _clean_text(line_text)
                stripped = line_text.strip()

                if not stripped:
                    continue

                # Get font size of first span in line
                first_span_size = (
                    line["spans"][0]["size"] if line.get("spans") else 0
                )

                # Detect section by regex OR font-size heuristic
                detected = _detect_section(stripped)
                if detected is None and _is_likely_header(
                    stripped, first_span_size, page_max_size
                ):
                    # Only promote to section if it looks like a section name
                    words = stripped.split()
                    if 1 <= len(words) <= 6:
                        detected = stripped.lower()[:40]

                if detected:
                    # Save current section
                    if current_text:
                        sections.append(ParsedSection(
                            name=current_section,
                            text="\n".join(current_text).strip(),
                            page_start=section_page_start,
                            page_end=page_num,
                        ))
                    current_section = detected
                    current_text = []
                    section_page_start = page_num
                else:
                    current_text.append(stripped)
                    full_text_parts.append(stripped)

    # Flush last section
    if current_text:
        sections.append(ParsedSection(
            name=current_section,
            text="\n".join(current_text).strip(),
            page_start=section_page_start,
            page_end=total_pages - 1,
        ))

    full_text = _clean_text("\n".join(full_text_parts))

    # ── Metadata extraction ────────────────────────────────────────
    title = _extract_title(doc) or path.stem.replace("_", " ")
    title = _clean_text(title)

    abstract = ""
    for s in sections:
        if "abstract" in s.name.lower():
            abstract = s.text[:1000]
            break

    year = _extract_year(doc)
    authors = [_clean_text(a) for a in _extract_authors(doc)]

    doc.close()

    return ParsedPaper(
        title=title,
        authors=authors,
        abstract=abstract,
        year=year,
        sections=sections,
        full_text=full_text,
        filename=path.name,
        total_pages=total_pages,
    )


def _extract_title(doc: fitz.Document) -> str | None:
    """Extract title from first page using font size heuristic."""
    if not doc or len(doc) == 0:
        return None
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]
    candidates = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = _clean_text(span.get("text", "")).strip()
                size = span.get("size", 0)
                if text and size > 14:
                    candidates.append((size, text))
    if candidates:
        candidates.sort(reverse=True)
        for _, text in candidates:
            if len(text) > 10:
                return text[:200]
    return None


def _extract_year(doc: fitz.Document) -> int | None:
    """Find a 4-digit year (2000–2030) in the first two pages."""
    year_re = re.compile(r"\b(20[0-2]\d)\b")
    for page in doc[:2]:
        text = _clean_text(page.get_text())
        matches = year_re.findall(text)
        if matches:
            years = [int(m) for m in matches]
            return max(set(years), key=years.count)
    return None


def _extract_authors(doc: fitz.Document) -> list[str]:
    """Heuristic author extraction from page 1."""
    if not doc or len(doc) == 0:
        return []
    text = _clean_text(doc[0].get_text())
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    authors = []
    for line in lines[1:8]:
        if re.search(r"\band\b|,\s+[A-Z]|\d{4}", line):
            parts = re.split(r"\band\b|,", line)
            for part in parts:
                name = part.strip()
                if 3 < len(name) < 40 and not any(c.isdigit() for c in name):
                    authors.append(name)
    return authors[:10]
