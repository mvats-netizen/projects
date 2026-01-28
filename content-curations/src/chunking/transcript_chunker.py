"""
Transcript Chunker

Implements sliding window chunking as per roadmap.pdf:
- 750-token window
- 150-token overlap
- Contextual pre-pending for embeddings
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import tiktoken  # For accurate token counting


@dataclass
class Chunk:
    """Represents a chunk of transcript content."""
    chunk_id: str
    text: str
    start_token: int
    end_token: int
    token_count: int
    
    # Parent info
    item_id: str
    item_name: str
    course_name: str
    module_name: str
    
    # For embedding
    contextual_text: Optional[str] = None
    
    # Metadata (from LLM extraction)
    bloom_level: Optional[str] = None
    difficulty: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'start_token': self.start_token,
            'end_token': self.end_token,
            'token_count': self.token_count,
            'item_id': self.item_id,
            'item_name': self.item_name,
            'course_name': self.course_name,
            'module_name': self.module_name,
            'contextual_text': self.contextual_text,
            'bloom_level': self.bloom_level,
            'difficulty': self.difficulty,
        }


class TranscriptChunker:
    """
    Chunks transcripts using sliding window approach.
    
    From roadmap.pdf:
    - Use 750-token window with 150-token overlap
    - Pre-pend context for embeddings:
      [Course: X] [Module: Y] [Level: Z] {Text}
    """
    
    def __init__(
        self,
        window_size: int = 750,
        overlap: int = 150,
        min_chunk_size: int = 100,
        encoding_name: str = "cl100k_base",  # GPT-4/Claude tokenizer
    ):
        """
        Initialize chunker.
        
        Args:
            window_size: Number of tokens per chunk (default: 750)
            overlap: Number of overlapping tokens (default: 150)
            min_chunk_size: Minimum tokens for a valid chunk (default: 100)
            encoding_name: Tiktoken encoding name
        """
        self.window_size = window_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.stride = window_size - overlap  # 750 - 150 = 600
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to simple word-based tokenization
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        # Fallback: ~4 chars per token
        return len(text) // 4
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to tokens."""
        if self.encoding:
            return self.encoding.encode(text)
        # Fallback: word-based
        return text.split()
    
    def detokenize(self, tokens) -> str:
        """Convert tokens back to text."""
        if self.encoding:
            return self.encoding.decode(tokens)
        # Fallback
        return " ".join(tokens)
    
    def chunk_text(
        self,
        text: str,
        item_id: str = "",
        item_name: str = "",
        course_name: str = "",
        module_name: str = "",
        bloom_level: str = None,
        difficulty: str = None,
    ) -> List[Chunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full transcript text
            item_id: Parent item ID
            item_name: Parent item name
            course_name: Parent course name
            module_name: Parent module name
            bloom_level: For contextual pre-pending
            difficulty: For contextual pre-pending
        
        Returns:
            List of Chunk objects
        """
        # Clean text
        text = self._clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        total_tokens = len(tokens)
        
        # If text is shorter than window, return as single chunk
        if total_tokens <= self.window_size:
            chunk = Chunk(
                chunk_id=f"{item_id}_c0",
                text=text,
                start_token=0,
                end_token=total_tokens,
                token_count=total_tokens,
                item_id=item_id,
                item_name=item_name,
                course_name=course_name,
                module_name=module_name,
                bloom_level=bloom_level,
                difficulty=difficulty,
            )
            chunk.contextual_text = self._build_contextual_text(chunk)
            return [chunk]
        
        # Sliding window chunking
        chunks = []
        chunk_idx = 0
        start = 0
        
        while start < total_tokens:
            end = min(start + self.window_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.detokenize(chunk_tokens)
            
            # Skip if too small (except for last chunk)
            if len(chunk_tokens) < self.min_chunk_size and start > 0:
                break
            
            chunk = Chunk(
                chunk_id=f"{item_id}_c{chunk_idx}",
                text=chunk_text,
                start_token=start,
                end_token=end,
                token_count=len(chunk_tokens),
                item_id=item_id,
                item_name=item_name,
                course_name=course_name,
                module_name=module_name,
                bloom_level=bloom_level,
                difficulty=difficulty,
            )
            chunk.contextual_text = self._build_contextual_text(chunk)
            chunks.append(chunk)
            
            # Move window
            start += self.stride
            chunk_idx += 1
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean transcript text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common transcript artifacts
        text = re.sub(r'\[.*?\]', '', text)  # [MUSIC], [APPLAUSE], etc.
        return text.strip()
    
    def _build_contextual_text(self, chunk: Chunk) -> str:
        """
        Build contextual pre-pending for embedding.
        
        From roadmap.pdf:
        "Pre-pend the chunk with its hierarchy:
         [Course: Machine Learning] [Module: Neural Networks] [Level: Apply] {Text}"
        """
        parts = []
        
        if chunk.course_name:
            parts.append(f"[Course: {chunk.course_name}]")
        if chunk.module_name:
            parts.append(f"[Module: {chunk.module_name}]")
        if chunk.bloom_level:
            parts.append(f"[Level: {chunk.bloom_level}]")
        if chunk.difficulty:
            parts.append(f"[Difficulty: {chunk.difficulty}]")
        
        prefix = " ".join(parts)
        return f"{prefix} {chunk.text}" if prefix else chunk.text
    
    def chunk_content_item(
        self,
        item: Dict[str, Any],
        derived_metadata: Dict[str, Any] = None,
    ) -> List[Chunk]:
        """
        Chunk a content item (from enriched data).
        
        Args:
            item: Content item dict with content_text, item_id, etc.
            derived_metadata: LLM-extracted metadata (bloom_level, etc.)
        
        Returns:
            List of Chunk objects
        """
        derived = derived_metadata or item.get('derived_metadata', {})
        op = item.get('operational_metadata', {})
        
        return self.chunk_text(
            text=item.get('content_text', ''),
            item_id=item.get('item_id', ''),
            item_name=item.get('item_name', ''),
            course_name=op.get('course_name', item.get('course_name', '')),
            module_name=op.get('module_name', item.get('module_name', '')),
            bloom_level=derived.get('bloom_level'),
            difficulty=op.get('difficulty_level'),
        )
    
    def chunk_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Chunk]:
        """
        Chunk a batch of content items.
        
        Returns:
            Flat list of all chunks
        """
        all_chunks = []
        for item in items:
            chunks = self.chunk_content_item(item)
            all_chunks.extend(chunks)
        return all_chunks
    
    def get_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get chunking statistics."""
        if not chunks:
            return {"total_chunks": 0}
        
        token_counts = [c.token_count for c in chunks]
        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(chunks),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "unique_items": len(set(c.item_id for c in chunks)),
        }
