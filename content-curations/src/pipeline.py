"""
Main Pipeline: Transcript → Chunks → Embeddings → Vector Store → Search

This module ties together all components for the AI-Led Curations system.

End-to-end usage:
    >>> pipeline = TranscriptSearchPipeline(provider="openai")
    >>> 
    >>> # Index transcripts
    >>> pipeline.index_items([
    ...     {"item_id": "video_123", "transcript": "...", "course_id": "course_1"},
    ...     {"item_id": "video_456", "transcript": "...", "course_id": "course_1"},
    ... ])
    >>> 
    >>> # Search
    >>> results = pipeline.search("How do I create a pivot table from multiple sheets?")
    >>> for r in results:
    ...     print(f"{r.score:.3f}: {r.chunk.center_sentence[:60]}...")
"""

import asyncio
from pathlib import Path
from typing import Union, List, Optional, Callable
import logging

from .chunking.sentence_chunker import SentenceChunker, ChunkResult
from .embeddings.embedding_pipeline import EmbeddingPipeline
from .vector_store.faiss_store import FAISSVectorStore, SearchResult

logger = logging.getLogger(__name__)


class TranscriptSearchPipeline:
    """
    End-to-end pipeline for transcript indexing and semantic search.
    
    Components:
    1. SentenceChunker: Splits transcripts into sentence-level windows
    2. EmbeddingPipeline: Generates embeddings for chunks
    3. FAISSVectorStore: Stores and searches embeddings
    
    Example:
        >>> # Initialize
        >>> pipeline = TranscriptSearchPipeline(
        ...     provider="openai",
        ...     context_size=2,  # 2 sentences before/after
        ... )
        >>> 
        >>> # Index a batch of items
        >>> pipeline.index_items([
        ...     {
        ...         "item_id": "video_001",
        ...         "course_id": "excel_basics",
        ...         "item_type": "video",
        ...         "transcript": "Welcome to this course. Today we'll learn about pivot tables...",
        ...     },
        ...     # ... more items
        ... ])
        >>> 
        >>> # Search
        >>> results = pipeline.search("pivot table from multiple sheets", top_k=5)
        >>> 
        >>> # Save for later
        >>> pipeline.save("./my_index/")
        >>> 
        >>> # Load existing index
        >>> pipeline = TranscriptSearchPipeline.load("./my_index/", provider="openai")
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        context_size: int = 2,
        embedding_dimensions: int = 1536,
        batch_size: int = 100,
    ):
        """
        Initialize the pipeline.
        
        Args:
            provider: Embedding provider ("openai" or "gemini")
            model: Embedding model name (optional, uses default for provider)
            context_size: Number of sentences before/after center in chunks
            embedding_dimensions: Embedding vector size (1536 for OpenAI, 768 for Gemini)
            batch_size: Number of chunks to embed at once
        """
        self.provider = provider
        self.model = model
        self.context_size = context_size
        
        # Initialize components
        self.chunker = SentenceChunker(
            context_before=context_size,
            context_after=context_size,
        )
        
        self.embedder = EmbeddingPipeline(
            provider=provider,
            model=model,
            batch_size=batch_size,
        )
        
        # Get actual dimensions from embedder (handles all providers correctly)
        actual_dimensions = self.embedder.dimensions
        
        self.vector_store = FAISSVectorStore(
            dimensions=actual_dimensions,
        )
        
        logger.info(f"Initialized pipeline with {provider} embeddings, context_size={context_size}")
    
    def index_transcript(
        self,
        transcript: Union[str, List[dict]],
        item_id: str,
        course_id: Optional[str] = None,
        item_type: Optional[str] = None,
        show_progress: bool = True,
    ) -> int:
        """
        Index a single transcript.
        
        Args:
            transcript: Text or timestamped segments
            item_id: Unique item identifier
            course_id: Parent course ID
            item_type: Type (video, reading, lab)
            show_progress: Show embedding progress
            
        Returns:
            Number of chunks indexed
        """
        # Chunk
        chunks = self.chunker.chunk_transcript(
            transcript=transcript,
            item_id=item_id,
            course_id=course_id,
            item_type=item_type,
        )
        
        if not chunks:
            logger.warning(f"No chunks generated for {item_id}")
            return 0
        
        # Embed
        chunks = self.embedder.embed_chunks_sync(chunks, show_progress=show_progress)
        
        # Store
        return self.vector_store.add_chunks(chunks)
    
    def index_items(
        self,
        items: List[dict],
        show_progress: bool = True,
    ) -> int:
        """
        Index multiple items in batch.
        
        Args:
            items: List of dicts with keys:
                - item_id: str (required)
                - transcript: str or List[dict] (required)
                - course_id: str (optional)
                - item_type: str (optional)
            show_progress: Show progress bar
            
        Returns:
            Total number of chunks indexed
        """
        # Chunk all items first
        all_chunks = self.chunker.chunk_batch(items)
        
        if not all_chunks:
            logger.warning("No chunks generated from items")
            return 0
        
        logger.info(f"Generated {len(all_chunks)} chunks from {len(items)} items")
        
        # Embed all chunks
        all_chunks = self.embedder.embed_chunks_sync(all_chunks, show_progress=show_progress)
        
        # Add to store
        return self.vector_store.add_chunks(all_chunks)
    
    async def index_items_async(
        self,
        items: List[dict],
        show_progress: bool = True,
    ) -> int:
        """Index multiple items asynchronously."""
        all_chunks = self.chunker.chunk_batch(items)
        
        if not all_chunks:
            return 0
        
        all_chunks = await self.embedder.embed_chunks(all_chunks, show_progress=show_progress)
        return self.vector_store.add_chunks(all_chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        course_id: Optional[str] = None,
        item_type: Optional[str] = None,
        deduplicate_by_item: bool = True,
    ) -> List[SearchResult]:
        """
        Search for relevant content.
        
        Args:
            query: Natural language query
            top_k: Number of results (unique items if deduplicate_by_item=True)
            course_id: Filter by course (optional)
            item_type: Filter by type (optional)
            deduplicate_by_item: If True, return only best chunk per item (default: True)
            
        Returns:
            List of SearchResult objects with matched chunks
        """
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Build filter function
        filter_fn = None
        if course_id or item_type:
            def filter_fn(chunk: ChunkResult) -> bool:
                if course_id and chunk.course_id != course_id:
                    return False
                if item_type and chunk.item_type != item_type:
                    return False
                return True
        
        # Search - fetch more results if deduplicating
        search_k = top_k * 5 if deduplicate_by_item else top_k
        
        results = self.vector_store.search_by_embedding(
            query_embedding=query_embedding,
            top_k=search_k,
            filter_fn=filter_fn,
        )
        
        # Deduplicate by item - keep only best chunk per item
        if deduplicate_by_item:
            seen_items = set()
            unique_results = []
            
            for result in results:
                item_id = result.chunk.item_id
                if item_id not in seen_items:
                    seen_items.add(item_id)
                    # Update rank to reflect deduplicated position
                    result.rank = len(unique_results) + 1
                    unique_results.append(result)
                    
                    if len(unique_results) >= top_k:
                        break
            
            return unique_results
        
        return results
    
    async def search_async(
        self,
        query: str,
        top_k: int = 10,
        **filters,
    ) -> List[SearchResult]:
        """Search asynchronously."""
        query_embedding = await self.embedder.embed_query_async(query)
        
        filter_fn = None
        if filters:
            def filter_fn(chunk: ChunkResult) -> bool:
                for key, value in filters.items():
                    if getattr(chunk, key, None) != value:
                        return False
                return True
        
        return self.vector_store.search_by_embedding(query_embedding, top_k, filter_fn)
    
    def save(self, directory: str) -> None:
        """
        Save the indexed data to disk.
        
        Args:
            directory: Directory to save to
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        self.vector_store.save(str(path / "vector_store"))
        
        # Save config
        import json
        config = {
            "provider": self.provider,
            "model": self.model,
            "context_size": self.context_size,
            "dimensions": self.vector_store.dimensions,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)
        
        logger.info(f"Saved pipeline to {directory}")
    
    @classmethod
    def load(
        cls,
        directory: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> "TranscriptSearchPipeline":
        """
        Load a saved pipeline.
        
        Args:
            directory: Directory containing saved data
            provider: Override embedding provider (optional)
            model: Override embedding model (optional)
            
        Returns:
            Loaded TranscriptSearchPipeline
        """
        import json
        
        path = Path(directory)
        
        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create instance
        pipeline = cls(
            provider=provider or config["provider"],
            model=model or config.get("model"),
            context_size=config["context_size"],
            embedding_dimensions=config["dimensions"],
        )
        
        # Load vector store
        pipeline.vector_store = FAISSVectorStore.load(str(path / "vector_store"))
        
        logger.info(f"Loaded pipeline from {directory} ({len(pipeline.vector_store)} chunks)")
        return pipeline
    
    def stats(self) -> dict:
        """Get pipeline statistics."""
        chunks = self.vector_store.chunks
        
        # Count by course
        courses = {}
        for chunk in chunks:
            course = chunk.course_id or "unknown"
            courses[course] = courses.get(course, 0) + 1
        
        # Count by item type
        item_types = {}
        for chunk in chunks:
            itype = chunk.item_type or "unknown"
            item_types[itype] = item_types.get(itype, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "unique_items": len(set(c.item_id for c in chunks)),
            "unique_courses": len(courses),
            "chunks_by_course": courses,
            "chunks_by_item_type": item_types,
        }
    
    def __repr__(self) -> str:
        return f"TranscriptSearchPipeline(provider={self.provider}, chunks={len(self.vector_store)})"

