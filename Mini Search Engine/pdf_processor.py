"""
PDF processing utilities.
Extracts text from uploaded PDFs and splits into overlapping chunks.
"""

from __future__ import annotations
import io
import re
from typing import List

import pypdf


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract all text from a Streamlit-uploaded PDF file object.

    Parameters
    ----------
    pdf_file : UploadedFile
        The file-like object returned by st.file_uploader.

    Returns
    -------
    str
        Concatenated plain text from all pages.
    """
    reader = pypdf.PdfReader(io.BytesIO(pdf_file.read()))
    pages: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n".join(pages)


def _clean_text(text: str) -> str:
    """Collapse whitespace runs and strip leading/trailing space."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping character-level chunks.

    Parameters
    ----------
    text         : Full document text.
    chunk_size   : Maximum characters per chunk.
    chunk_overlap: Characters shared between adjacent chunks.

    Returns
    -------
    List[str]
        Non-empty text chunks.
    """
    text = _clean_text(text)
    if not text:
        return []

    step   = max(1, chunk_size - chunk_overlap)
    chunks = []
    start  = 0

    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks
