"""
Embedding utilities using OpenAI's text-embedding-3-small model.
"""

from __future__ import annotations
from typing import List
import openai


def get_embedding(text: str, api_key: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Generate an embedding vector for a single text string.

    Parameters
    ----------
    text    : The text to embed.
    api_key : OpenAI API key.
    model   : Embedding model to use (default: text-embedding-3-small, dim=1536).

    Returns
    -------
    List[float]
        The embedding vector.
    """
    client   = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def get_embeddings_batch(
    texts: List[str],
    api_key: str,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts in batches.

    Parameters
    ----------
    texts      : List of text strings.
    api_key    : OpenAI API key.
    model      : Embedding model.
    batch_size : Number of texts per API call.

    Returns
    -------
    List[List[float]]
        List of embedding vectors in the same order as input texts.
    """
    client     = openai.OpenAI(api_key=api_key)
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch    = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        # Sort by index to ensure ordering
        sorted_data = sorted(response.data, key=lambda x: x.index)
        embeddings.extend([item.embedding for item in sorted_data])

    return embeddings
