"""Vector store module for embedding storage and retrieval."""

from .faiss_store import FAISSVectorStore, SearchResult

__all__ = ["FAISSVectorStore", "SearchResult"]

