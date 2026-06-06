"""
Pinecone vector database utilities.
Handles index initialisation, upsert, and query operations.
"""

from __future__ import annotations
from typing import List, Dict, Any
import uuid

from pinecone import Pinecone, ServerlessSpec
from utils.embeddings import get_embeddings_batch, get_embedding


EMBEDDING_DIM = 1536


def init_pinecone(api_key: str, index_name: str) -> Any:
    """
    Initialise Pinecone and return a handle to the named index.
    Creates the index if it does not already exist.

    Parameters
    ----------
    api_key    : Pinecone API key.
    index_name : Name of the Pinecone index.

    Returns
    -------
    pinecone.Index
        A connected index object ready for upsert / query calls.
    """
    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name      = index_name,
            dimension = EMBEDDING_DIM,
            metric    = "cosine",
            spec      = ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(index_name)


def upsert_chunks(
    index,
    chunks: List[str],
    source_name: str,
    openai_api_key: str,
    batch_size: int = 100,
) -> None:
    """
    Embed a list of text chunks and upsert them into the Pinecone index.

    Parameters
    ----------
    index          : Pinecone Index object.
    chunks         : List of text chunk strings.
    source_name    : Name of the source PDF (stored as metadata).
    openai_api_key : OpenAI API key for embedding generation.
    batch_size     : Number of vectors to upsert per Pinecone call.
    """
    if not chunks:
        return

    embeddings = get_embeddings_batch(chunks, openai_api_key)

    # Build vector records
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id"      : str(uuid.uuid4()),
            "values"  : embedding,
            "metadata": {
                "text"        : chunk,
                "source"      : source_name,
                "chunk_index" : i,
            },
        })

    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])


def query_index(
    index,
    query_text: str,
    openai_api_key: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Embed a query and retrieve the Top-K most similar chunks.

    Parameters
    ----------
    index          : Pinecone Index object.
    query_text     : User's natural language query.
    openai_api_key : OpenAI API key.
    top_k          : Number of results to return.

    Returns
    -------
    List[dict]
        List of match dicts with keys: id, score, metadata.
    """
    query_embedding = get_embedding(query_text, openai_api_key)

    response = index.query(
        vector          = query_embedding,
        top_k           = top_k,
        include_metadata= True,
    )

    results = []
    for match in response.matches:
        results.append({
            "id"      : match.id,
            "score"   : match.score,
            "metadata": match.metadata,
        })

    return results
