"""
Embedding Pipeline for Transcript Chunks

Supports multiple embedding providers:
- OpenAI (text-embedding-3-small, text-embedding-3-large)
- Google Gemini (models/embedding-001)

Usage:
    >>> pipeline = EmbeddingPipeline(provider="openai", model="text-embedding-3-small")
    >>> chunks_with_embeddings = await pipeline.embed_chunks(chunks)
"""

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Optional
import logging

import numpy as np
from tqdm import tqdm

from ..chunking.sentence_chunker import ChunkResult

logger = logging.getLogger(__name__)


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    LOCAL = "local"  # Sentence Transformers (FREE)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    provider: EmbeddingProvider
    model: str
    dimensions: int
    batch_size: int = 100
    api_key: Optional[str] = None


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def embed_text_sync(self, text: str) -> List[float]:
        """Generate embedding for a single text (synchronous)."""
        pass


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding provider."""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Import here to avoid dependency issues
        from openai import OpenAI, AsyncOpenAI
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def embed_text_sync(self, text: str) -> List[float]:
        """Generate single embedding synchronously."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding


class GeminiEmbedder(BaseEmbedder):
    """Google Gemini embedding provider."""
    
    def __init__(self, model: str = "models/embedding-001", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        # Import here to avoid dependency issues
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.genai = genai
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Gemini API."""
        # Gemini doesn't have native async, run in executor
        loop = asyncio.get_event_loop()
        embeddings = []
        
        for text in texts:
            embedding = await loop.run_in_executor(
                None, 
                lambda t=text: self.genai.embed_content(
                    model=self.model,
                    content=t,
                    task_type="retrieval_document",
                )["embedding"]
            )
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_text_sync(self, text: str) -> List[float]:
        """Generate single embedding synchronously."""
        result = self.genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]


class LocalEmbedder(BaseEmbedder):
    """
    Local embedding provider using Sentence Transformers.
    
    FREE - No API keys needed!
    
    Recommended models:
    - 'all-MiniLM-L6-v2': Fast, good quality (384 dims)
    - 'all-mpnet-base-v2': Better quality (768 dims)
    - 'e5-large-v2': Best quality (1024 dims) - RECOMMENDED
    """
    
    def __init__(self, model: str = "all-mpnet-base-v2"):
        self.model_name = model
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local model: {model}")
            self.model = SentenceTransformer(model)
            self.dimensions = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded {model} with {self.dimensions} dimensions")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings locally (runs in thread pool for async compat)."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, show_progress_bar=False).tolist()
        )
        return embeddings
    
    def embed_text_sync(self, text: str) -> List[float]:
        """Generate single embedding synchronously."""
        return self.model.encode(text, show_progress_bar=False).tolist()
    
    def embed_batch_sync(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Embed a batch of texts efficiently."""
        return self.model.encode(texts, show_progress_bar=show_progress).tolist()


class EmbeddingPipeline:
    """
    Pipeline for generating embeddings for transcript chunks.
    
    Handles batching, rate limiting, and error recovery.
    
    Example:
        >>> pipeline = EmbeddingPipeline(provider="openai")
        >>> chunks = chunker.chunk_transcript(transcript, "video_123")
        >>> chunks_with_embeddings = await pipeline.embed_chunks(chunks)
        >>> 
        >>> # Or synchronously
        >>> chunks_with_embeddings = pipeline.embed_chunks_sync(chunks)
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        batch_size: int = 100,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the embedding pipeline.
        
        Args:
            provider: "openai" or "gemini"
            model: Model name (defaults based on provider)
            batch_size: Number of texts to embed in one API call
            api_key: API key (or use environment variable)
        """
        self.provider = EmbeddingProvider(provider)
        self.batch_size = batch_size
        
        # Set default models
        if model is None:
            model = {
                EmbeddingProvider.OPENAI: "text-embedding-3-small",
                EmbeddingProvider.GEMINI: "models/embedding-001",
                EmbeddingProvider.LOCAL: "all-mpnet-base-v2",  # Good balance of speed/quality
            }[self.provider]
        
        self.model = model
        
        # Initialize embedder
        if self.provider == EmbeddingProvider.OPENAI:
            self.embedder = OpenAIEmbedder(model=model, api_key=api_key)
            self.dimensions = 1536 if "small" in model else 3072
        elif self.provider == EmbeddingProvider.GEMINI:
            self.embedder = GeminiEmbedder(model=model, api_key=api_key)
            self.dimensions = 768
        elif self.provider == EmbeddingProvider.LOCAL:
            self.embedder = LocalEmbedder(model=model)
            self.dimensions = self.embedder.dimensions
        
        logger.info(f"Initialized {provider} embedding pipeline with model {model}")
    
    async def embed_chunks(
        self,
        chunks: List[ChunkResult],
        show_progress: bool = True,
    ) -> List[ChunkResult]:
        """
        Generate embeddings for all chunks (async).
        
        Args:
            chunks: List of ChunkResult objects from chunker
            show_progress: Whether to show progress bar
            
        Returns:
            Same chunks with embedding field populated
        """
        if not chunks:
            return chunks
        
        # Extract texts for embedding
        texts = [chunk.full_text for chunk in chunks]
        
        # Process in batches
        all_embeddings = []
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        iterator = tqdm(batches, desc="Embedding chunks") if show_progress else batches
        
        for batch in iterator:
            try:
                batch_embeddings = await self.embedder.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                # Add None embeddings for failed batch
                all_embeddings.extend([None] * len(batch))
        
        # Assign embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk.embedding = embedding
        
        # Count successful embeddings
        successful = sum(1 for c in chunks if c.embedding is not None)
        logger.info(f"Generated {successful}/{len(chunks)} embeddings")
        
        return chunks
    
    def embed_chunks_sync(
        self,
        chunks: List[ChunkResult],
        show_progress: bool = True,
    ) -> List[ChunkResult]:
        """
        Generate embeddings for all chunks (synchronous wrapper).
        
        Args:
            chunks: List of ChunkResult objects
            show_progress: Whether to show progress bar
            
        Returns:
            Chunks with embeddings populated
        """
        return asyncio.run(self.embed_chunks(chunks, show_progress))
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: The search query text
            
        Returns:
            Query embedding vector
        """
        return self.embedder.embed_text_sync(query)
    
    async def embed_query_async(self, query: str) -> List[float]:
        """Generate embedding for a search query (async)."""
        embeddings = await self.embedder.embed_texts([query])
        return embeddings[0]


# Convenience function
def create_embedding_pipeline(
    provider: str = "openai",
    model: Optional[str] = None,
) -> EmbeddingPipeline:
    """
    Create an embedding pipeline with default settings.
    
    Args:
        provider: "openai" or "gemini"
        model: Model name (optional)
        
    Returns:
        Configured EmbeddingPipeline
    """
    return EmbeddingPipeline(provider=provider, model=model)

