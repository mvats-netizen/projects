"""
FAISS Vector Store for Transcript Chunk Retrieval

Provides efficient similarity search over embedded transcript chunks.
Supports:
- Adding chunks with metadata
- Semantic search with filtering
- Persistence (save/load index)

Usage:
    >>> store = FAISSVectorStore(dimensions=1536)
    >>> store.add_chunks(chunks_with_embeddings)
    >>> results = store.search("pivot table multiple sheets", top_k=5)
    >>> for result in results:
    ...     print(f"Score: {result.score:.3f}, Text: {result.chunk.center_sentence[:50]}...")
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Optional, Callable
import logging

import numpy as np

from ..chunking.sentence_chunker import ChunkResult

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with score and chunk data."""
    chunk: ChunkResult
    score: float  # Similarity score (higher = more similar)
    rank: int  # Rank in results (1-indexed)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "rank": self.rank,
            "score": self.score,
            "chunk_id": self.chunk.chunk_id,
            "center_sentence": self.chunk.center_sentence,
            "full_text": self.chunk.full_text,
            "item_id": self.chunk.item_id,
            "course_id": self.chunk.course_id,
            "item_type": self.chunk.item_type,
            "start_time": self.chunk.start_time,
            "end_time": self.chunk.end_time,
        }


class FAISSVectorStore:
    """
    FAISS-based vector store for semantic search over transcript chunks.
    
    Features:
    - Efficient approximate nearest neighbor search
    - Metadata filtering (by course, item type, etc.)
    - Persistence (save/load to disk)
    - Incremental updates
    
    Example:
        >>> store = FAISSVectorStore(dimensions=1536)
        >>> 
        >>> # Add chunks
        >>> store.add_chunks(chunks_with_embeddings)
        >>> 
        >>> # Search
        >>> query_embedding = embedding_pipeline.embed_query("pivot tables")
        >>> results = store.search_by_embedding(query_embedding, top_k=10)
        >>> 
        >>> # Or with text query (requires embedding function)
        >>> results = store.search("pivot tables", top_k=10, embed_fn=pipeline.embed_query)
        >>> 
        >>> # Save for later
        >>> store.save("./vector_store/")
        >>> 
        >>> # Load existing
        >>> store = FAISSVectorStore.load("./vector_store/")
    """
    
    def __init__(
        self,
        dimensions: int = 1536,
        index_type: str = "Flat",
        nlist: int = 100,
    ):
        """
        Initialize the vector store.
        
        Args:
            dimensions: Embedding dimensions (1536 for OpenAI, 768 for Gemini)
            index_type: FAISS index type ("Flat" for exact, "IVFFlat" for approximate)
            nlist: Number of clusters for IVF index
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("Please install faiss-cpu: pip install faiss-cpu")
        
        self.dimensions = dimensions
        self.index_type = index_type
        self.nlist = nlist
        
        # Initialize index
        if index_type == "Flat":
            # Exact search (best for < 100k vectors)
            self.index = faiss.IndexFlatIP(dimensions)  # Inner product (cosine sim for normalized)
        elif index_type == "IVFFlat":
            # Approximate search (better for large datasets)
            quantizer = faiss.IndexFlatIP(dimensions)
            self.index = faiss.IndexIVFFlat(quantizer, dimensions, nlist)
            self._needs_training = True
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Store metadata separately (FAISS only stores vectors)
        self.chunks: List[ChunkResult] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        
        # Track if IVF index is trained
        self._needs_training = index_type == "IVFFlat"
        
        logger.info(f"Initialized FAISS store with {dimensions}D {index_type} index")
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return vectors / norms
    
    def add_chunks(
        self,
        chunks: List[ChunkResult],
        normalize: bool = True,
    ) -> int:
        """
        Add chunks with embeddings to the store.
        
        Args:
            chunks: List of ChunkResult objects with embeddings
            normalize: Whether to normalize embeddings (recommended for cosine sim)
            
        Returns:
            Number of chunks added
        """
        # Filter chunks with valid embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]
        
        if not valid_chunks:
            logger.warning("No chunks with valid embeddings to add")
            return 0
        
        # Extract embeddings
        embeddings = np.array([c.embedding for c in valid_chunks], dtype=np.float32)
        
        if normalize:
            embeddings = self._normalize(embeddings)
        
        # Train IVF index if needed
        if self._needs_training and not self.index.is_trained:
            logger.info(f"Training IVF index with {len(embeddings)} vectors")
            self.index.train(embeddings)
            self._needs_training = False
        
        # Add to index
        start_idx = len(self.chunks)
        self.index.add(embeddings)
        
        # Store metadata
        for i, chunk in enumerate(valid_chunks):
            self.chunks.append(chunk)
            self.chunk_id_to_idx[chunk.chunk_id] = start_idx + i
        
        logger.info(f"Added {len(valid_chunks)} chunks to store (total: {len(self.chunks)})")
        return len(valid_chunks)
    
    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_fn: Optional[Callable[[ChunkResult], bool]] = None,
        normalize: bool = True,
    ) -> List[SearchResult]:
        """
        Search for similar chunks using a query embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_fn: Optional function to filter results (e.g., by course)
            normalize: Whether to normalize query embedding
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        if len(self.chunks) == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Prepare query
        query = np.array([query_embedding], dtype=np.float32)
        if normalize:
            query = self._normalize(query)
        
        # Search (fetch more if filtering)
        search_k = top_k * 3 if filter_fn else top_k
        search_k = min(search_k, len(self.chunks))
        
        scores, indices = self.index.search(query, search_k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            
            chunk = self.chunks[idx]
            
            # Apply filter if provided
            if filter_fn and not filter_fn(chunk):
                continue
            
            results.append(SearchResult(
                chunk=chunk,
                score=float(score),
                rank=len(results) + 1,
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        filter_fn: Optional[Callable[[ChunkResult], bool]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar chunks using a text query.
        
        Args:
            query: Text query
            top_k: Number of results
            embed_fn: Function to embed the query (required)
            filter_fn: Optional filter function
            
        Returns:
            List of SearchResult objects
        """
        if embed_fn is None:
            raise ValueError("embed_fn is required for text search")
        
        query_embedding = embed_fn(query)
        return self.search_by_embedding(query_embedding, top_k, filter_fn)
    
    def search_by_course(
        self,
        query_embedding: List[float],
        course_id: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search within a specific course."""
        return self.search_by_embedding(
            query_embedding,
            top_k,
            filter_fn=lambda c: c.course_id == course_id,
        )
    
    def search_by_item_type(
        self,
        query_embedding: List[float],
        item_type: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search within a specific item type (video, reading, lab)."""
        return self.search_by_embedding(
            query_embedding,
            top_k,
            filter_fn=lambda c: c.item_type == item_type,
        )
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkResult]:
        """Retrieve a chunk by its ID."""
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self.chunks[idx]
        return None
    
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save index and metadata
        """
        import faiss
        
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata = {
            "dimensions": self.dimensions,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        logger.info(f"Saved vector store to {directory} ({len(self.chunks)} chunks)")
    
    @classmethod
    def load(cls, directory: str) -> "FAISSVectorStore":
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory containing saved index and metadata
            
        Returns:
            Loaded FAISSVectorStore
        """
        import faiss
        
        path = Path(directory)
        
        # Load metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Create instance
        store = cls(
            dimensions=metadata["dimensions"],
            index_type=metadata["index_type"],
            nlist=metadata.get("nlist", 100),
        )
        
        # Load FAISS index
        index_path = path / "index.faiss"
        store.index = faiss.read_index(str(index_path))
        store._needs_training = False
        
        # Load chunks
        for chunk_dict in metadata["chunks"]:
            chunk = ChunkResult(
                chunk_id=chunk_dict["chunk_id"],
                center_sentence=chunk_dict["center_sentence"],
                full_text=chunk_dict["full_text"],
                sentence_index=chunk_dict["sentence_index"],
                start_char=chunk_dict["start_char"],
                end_char=chunk_dict["end_char"],
                start_time=chunk_dict.get("start_time"),
                end_time=chunk_dict.get("end_time"),
                item_id=chunk_dict.get("item_id"),
                course_id=chunk_dict.get("course_id"),
                item_type=chunk_dict.get("item_type"),
                context_before=chunk_dict.get("context_before", []),
                context_after=chunk_dict.get("context_after", []),
            )
            store.chunks.append(chunk)
            store.chunk_id_to_idx[chunk.chunk_id] = len(store.chunks) - 1
        
        logger.info(f"Loaded vector store from {directory} ({len(store.chunks)} chunks)")
        return store
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __repr__(self) -> str:
        return f"FAISSVectorStore(chunks={len(self.chunks)}, dimensions={self.dimensions})"

