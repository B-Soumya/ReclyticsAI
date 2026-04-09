"""
Document Parser Agent.
Extracts text from PDF and Word documents, splits into chunks and sections.
"""

from __future__ import annotations

import re
from io import BytesIO


def extract_text_from_pdf(file: BytesIO) -> tuple[str, int]:
    """Extract all text from every page of a PDF file.
    Returns (text, page_count).  Tries PyMuPDF first (more robust
    with complex layouts & fonts) and falls back to pdfplumber."""

    # ── Primary: PyMuPDF (fitz) — handles most PDFs reliably ──
    try:
        import fitz  # PyMuPDF

        file.seek(0)
        doc = fitz.open(stream=file.read(), filetype="pdf")
        pages_text: list[str] = []
        for page in doc:
            pt = page.get_text("text")
            if pt and pt.strip():
                pages_text.append(pt)
        total_pages = len(doc)
        doc.close()
        combined = "\n".join(pages_text).strip()
        if combined and len(combined.split()) > 50:
            return combined, total_pages
    except ImportError:
        pass
    except Exception:
        pass

    # ── Fallback: pdfplumber ──
    import pdfplumber

    file.seek(0)
    text = ""
    total_pages = 0
    with pdfplumber.open(file) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip(), total_pages


def extract_text_from_docx(file: BytesIO) -> str:
    """Extract all text from a Word (.docx) file."""
    from docx import Document

    doc = Document(file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def extract_text(file: BytesIO, filename: str) -> tuple[str, int]:
    """Route to the correct parser based on file extension.
    Returns (text, page_count)."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file), 0
    else:
        raise ValueError(f"Unsupported file format: .{ext}  (use PDF or Word)")


def clean_text(text: str) -> str:
    """Whitespace cleanup — preserves mathematical symbols and Unicode."""
    text = re.sub(r"[ \t]+", " ", text)       # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse excessive newlines
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-level chunks (sentence-aware)."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        wc = len(sent.split())
        if current_len + wc > chunk_size and current:
            chunks.append(" ".join(current))
            # keep trailing sentences for overlap
            overlap_sents: list[str] = []
            ol = 0
            for s in reversed(current):
                sw = len(s.split())
                if ol + sw > overlap:
                    break
                overlap_sents.insert(0, s)
                ol += sw
            current = overlap_sents
            current_len = ol
        current.append(sent)
        current_len += wc

    if current:
        chunks.append(" ".join(current))
    return chunks


def extract_sections(text: str) -> list[str]:
    """Heuristically split document into logical sections."""
    patterns = [
        r"\n(?=\d+\.\s+[A-Z])",            # 1. Introduction
        r"\n(?=[A-Z][A-Z ]{3,}\n)",          # INTRODUCTION
        r"\n(?=#{1,3}\s)",                   # Markdown headings
        r"\n(?=(?:Abstract|Introduction|Background|Literature Review|"
        r"Methodology|Methods|Results|Discussion|Conclusion|"
        r"References|Acknowledgements|Appendix)\b)",
    ]
    for pat in patterns:
        parts = re.split(pat, text)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 50]
        if len(parts) > 1:
            return parts

    # fallback: double-newline paragraphs
    paras = text.split("\n\n")
    paras = [p.strip() for p in paras if p.strip() and len(p.strip()) > 50]
    if len(paras) > 1:
        return paras

    # final fallback: chunk
    return chunk_text(text, chunk_size=800, overlap=0)


def extract_references(text: str) -> list[str]:
    """Extract individual reference titles from the References / Bibliography section.

    Returns a list of paper titles found in the reference list.
    If no References section is detected, returns an empty list.
    """
    # ── Step 1: Find the References section ──
    ref_section = ""
    # Try several heading patterns
    for pat in [
        r"(?i)\n\s*(?:References|Bibliography|Works Cited|Literature Cited)\s*\n",
        r"\n\s*(?:REFERENCES|BIBLIOGRAPHY)\s*\n",
        r"\n\d+\.\s*(?:References|Bibliography)\s*\n",
    ]:
        m = re.search(pat, text)
        if m:
            ref_section = text[m.end():]
            # Cut off anything after the references (e.g. Appendix)
            for stop_pat in [
                r"\n\s*(?:Appendix|Appendices|APPENDIX)\b",
                r"\n\s*(?:Supplementary|SUPPLEMENTARY)\b",
            ]:
                stop = re.search(stop_pat, ref_section)
                if stop:
                    ref_section = ref_section[:stop.start()]
            break

    if not ref_section or len(ref_section.strip()) < 50:
        return []

    # ── Step 2: Split into individual reference entries ──
    entries: list[str] = []
    # Pattern: [1], [2] ... or 1. 2. ... or 1) 2) ...
    split_entries = re.split(r"\n\s*(?:\[\d+\]|\d+[\.\)])\s*", ref_section)
    if len(split_entries) < 3:
        # Fallback: split by blank lines or lines starting with author-like patterns
        split_entries = re.split(r"\n\s*\n", ref_section)
    entries = [e.strip() for e in split_entries if e.strip() and len(e.strip()) > 20]

    # ── Step 3: Extract title from each entry ──
    titles: list[str] = []
    for entry in entries:
        title = _extract_title_from_reference(entry)
        if title and len(title) > 10:
            titles.append(title)

    return titles


def _extract_title_from_reference(entry: str) -> str:
    """Heuristically extract a paper title from a single reference entry.

    Common formats:
      Author, A., Author, B. (2020). Title of the paper. Journal, ...
      Author, A., Author, B., "Title of the paper," Journal, ...
      Author A, Author B. Title of the paper. In Proceedings of ...
    """
    # Try quoted title first:  "Title" or \u2018Title\u2019
    m = re.search(r'["\u201c](.{15,}?)["\u201d]', entry)
    if m:
        return m.group(1).strip().rstrip(".")

    # Try pattern: (year). Title. (APA style)
    m = re.search(r'\(\d{4}[a-z]?\)\.\s*(.+?)\.(?:\s|$)', entry)
    if m:
        candidate = m.group(1).strip()
        # Make sure it's not just a journal name (usually short)
        if len(candidate) > 15:
            return candidate

    # Try pattern: (year) Title. (variant)
    m = re.search(r'\(\d{4}[a-z]?\)\s+(.+?)\.(?:\s|$)', entry)
    if m:
        candidate = m.group(1).strip()
        if len(candidate) > 15:
            return candidate

    # Try: after the first period following author names
    # Authors typically end with a period or comma before the title
    parts = entry.split(". ")
    if len(parts) >= 2:
        # First part is usually authors, second is title
        candidate = parts[1].strip().rstrip(".")
        if len(candidate) > 15:
            return candidate

    # Last resort: take the longest sentence-like segment
    segments = re.split(r"\.\s+", entry)
    segments = [s.strip() for s in segments if len(s.strip()) > 15]
    if segments:
        return max(segments, key=len).rstrip(".")

    return ""


class DocumentAgent:
    """Parses an uploaded file and produces structured output."""

    name = "Document Parser Agent"

    def process(self, file: BytesIO, filename: str) -> dict:
        raw_text, page_count = extract_text(file, filename)
        if not raw_text:
            raise ValueError("Could not extract any text from the uploaded file.")

        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)
        sections = extract_sections(raw_text)
        references = extract_references(raw_text)

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "chunks": chunks,
            "sections": sections,
            "references": references,
            "word_count": len(cleaned.split()),
            "num_chunks": len(chunks),
            "num_sections": len(sections),
            "num_pages": page_count,
            "num_references": len(references),
        }
