from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .pdf_parser import ParsedPaper, ParsedSection

# Sections to skip entirely — they add noise without useful content
SKIP_SECTIONS = {"references", "appendix", "acknowledgements", "acknowledgments"}

# Sections to keep as single chunks regardless of length
KEEP_WHOLE = {"abstract"}


@dataclass
class Chunk:
    text: str
    section: str
    chunk_index: int       # global index within the paper
    section_chunk_index: int  # index within this section
    token_estimate: int    # rough estimate: chars / 4


def chunk_paper(paper: ParsedPaper, chunk_size: int = 400, overlap: int = 60) -> list[Chunk]:
    """
    Chunk a parsed paper into retrieval-ready pieces.

    Key rules:
    - Never cross section boundaries
    - Keep abstract as one chunk (it's gold for search)
    - Skip references / appendix
    - Attach section name to every chunk for metadata filtering
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Chunk] = []
    global_index = 0

    for section in paper.sections:
        section_name = section.name.lower()

        # Skip low-value sections
        if any(skip in section_name for skip in SKIP_SECTIONS):
            continue

        # Skip very short sections (page headers, footers, etc.)
        if len(section.text.strip()) < 100:
            continue

        # Abstract → single chunk always
        if any(keep in section_name for keep in KEEP_WHOLE):
            chunk = Chunk(
                text=section.text.strip(),
                section=section.name,
                chunk_index=global_index,
                section_chunk_index=0,
                token_estimate=len(section.text) // 4,
            )
            all_chunks.append(chunk)
            global_index += 1
            continue

        # All other sections → split by size
        texts = splitter.split_text(section.text)
        for sec_idx, text in enumerate(texts):
            if len(text.strip()) < 50:   # skip tiny fragments
                continue
            chunk = Chunk(
                text=text.strip(),
                section=section.name,
                chunk_index=global_index,
                section_chunk_index=sec_idx,
                token_estimate=len(text) // 4,
            )
            all_chunks.append(chunk)
            global_index += 1

    return all_chunks
