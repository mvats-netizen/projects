"""
Sentence-Level Chunking with Sliding Window

This module implements Option 3: Maximum granularity chunking where each
sentence is indexed with surrounding context for better semantic matching.

Architecture:
    Transcript → Split into sentences → Create sliding windows → Ready for embedding
    
    Window: [S(n-2)] [S(n-1)] [CENTER] [S(n+1)] [S(n+2)]
                                ↑
                    Center sentence is the "anchor"
                    Surrounding sentences provide context
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Represents a single chunk with its metadata."""
    
    chunk_id: str
    center_sentence: str  # The primary sentence being indexed
    full_text: str  # Center + surrounding context
    
    # Position information
    sentence_index: int  # Index of center sentence in original transcript
    start_char: int  # Character offset in original transcript
    end_char: int  # Character offset end
    
    # Timing information (for video content)
    start_time: Optional[float] = None  # Seconds from start
    end_time: Optional[float] = None
    
    # Source information
    item_id: Optional[str] = None
    course_id: Optional[str] = None
    item_type: Optional[str] = None  # video, reading, lab
    
    # Context window details
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    
    # Embedding (populated later)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "center_sentence": self.center_sentence,
            "full_text": self.full_text,
            "sentence_index": self.sentence_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "item_id": self.item_id,
            "course_id": self.course_id,
            "item_type": self.item_type,
            "context_before": self.context_before,
            "context_after": self.context_after,
        }


@dataclass
class TimestampedSentence:
    """A sentence with optional timestamp information."""
    text: str
    index: int
    start_char: int
    end_char: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class SentenceChunker:
    """
    Sentence-level chunker with sliding window context.
    
    For each sentence in the transcript, creates a chunk that includes:
    - The center sentence (what we're indexing)
    - N sentences before for context
    - N sentences after for context
    
    This preserves co-occurrence: "pivot tables" and "multiple sheets"
    appearing together will be captured in the same chunk embedding.
    
    Example:
        >>> chunker = SentenceChunker(context_before=2, context_after=2)
        >>> chunks = chunker.chunk_transcript(transcript, item_id="video_123")
        >>> for chunk in chunks:
        ...     print(f"Center: {chunk.center_sentence[:50]}...")
        ...     print(f"Full context: {chunk.full_text[:100]}...")
    """
    
    def __init__(
        self,
        context_before: int = 2,
        context_after: int = 2,
        min_sentence_length: int = 10,
        max_chunk_tokens: int = 512,
    ):
        """
        Initialize the chunker.
        
        Args:
            context_before: Number of sentences to include before center
            context_after: Number of sentences to include after center
            min_sentence_length: Minimum characters for a valid sentence
            max_chunk_tokens: Maximum tokens per chunk (approximate)
        """
        self.context_before = context_before
        self.context_after = context_after
        self.min_sentence_length = min_sentence_length
        self.max_chunk_tokens = max_chunk_tokens
        
        # Sentence splitting pattern
        # Handles: periods, question marks, exclamation marks
        # Avoids splitting on: Mr., Mrs., Dr., abbreviations, decimals
        self._sentence_pattern = re.compile(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
        )
        
        # Approximate tokens per character (for English)
        self._chars_per_token = 4
    
    def _split_sentences(self, text: str) -> List[TimestampedSentence]:
        """
        Split text into sentences with position tracking.
        
        Args:
            text: The transcript text to split
            
        Returns:
            List of TimestampedSentence objects
        """
        # Clean the text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Split into sentences
        sentences = self._sentence_pattern.split(text)
        
        # Track positions and filter short sentences
        result = []
        current_pos = 0
        
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            
            if len(sentence) < self.min_sentence_length:
                # Skip very short sentences but track position
                current_pos = text.find(sentence, current_pos) + len(sentence)
                continue
            
            start_char = text.find(sentence, current_pos)
            end_char = start_char + len(sentence)
            
            result.append(TimestampedSentence(
                text=sentence,
                index=len(result),
                start_char=start_char,
                end_char=end_char,
            ))
            
            current_pos = end_char
        
        return result
    
    def _split_sentences_with_timestamps(
        self,
        timestamped_segments: List[dict]
    ) -> List[TimestampedSentence]:
        """
        Split timestamped transcript segments into sentences.
        
        Args:
            timestamped_segments: List of dicts with 'text', 'start', 'end' keys
                Example: [{"text": "Hello world.", "start": 0.0, "end": 2.5}, ...]
        
        Returns:
            List of TimestampedSentence objects with timing info
        """
        result = []
        global_char_offset = 0
        
        for segment in timestamped_segments:
            text = segment.get("text", "").strip()
            start_time = segment.get("start", segment.get("start_time"))
            end_time = segment.get("end", segment.get("end_time"))
            
            # Split segment into sentences
            sentences = self._sentence_pattern.split(text)
            
            # Distribute timing across sentences (linear interpolation)
            if start_time is not None and end_time is not None:
                duration = end_time - start_time
                total_chars = sum(len(s) for s in sentences)
                
                current_time = start_time
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < self.min_sentence_length:
                        continue
                    
                    # Estimate sentence duration based on character ratio
                    sentence_duration = (len(sentence) / max(total_chars, 1)) * duration
                    
                    result.append(TimestampedSentence(
                        text=sentence,
                        index=len(result),
                        start_char=global_char_offset,
                        end_char=global_char_offset + len(sentence),
                        start_time=current_time,
                        end_time=current_time + sentence_duration,
                    ))
                    
                    current_time += sentence_duration
                    global_char_offset += len(sentence) + 1
            else:
                # No timing info, just track position
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < self.min_sentence_length:
                        continue
                    
                    result.append(TimestampedSentence(
                        text=sentence,
                        index=len(result),
                        start_char=global_char_offset,
                        end_char=global_char_offset + len(sentence),
                    ))
                    global_char_offset += len(sentence) + 1
        
        return result
    
    def _create_window(
        self,
        sentences: List[TimestampedSentence],
        center_idx: int,
    ) -> tuple[List[str], str, List[str]]:
        """
        Create a context window around a center sentence.
        
        Args:
            sentences: All sentences in the transcript
            center_idx: Index of the center sentence
            
        Returns:
            Tuple of (context_before, center, context_after)
        """
        # Get context before
        start_idx = max(0, center_idx - self.context_before)
        context_before = [s.text for s in sentences[start_idx:center_idx]]
        
        # Get center
        center = sentences[center_idx].text
        
        # Get context after
        end_idx = min(len(sentences), center_idx + self.context_after + 1)
        context_after = [s.text for s in sentences[center_idx + 1:end_idx]]
        
        return context_before, center, context_after
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text."""
        return len(text) // self._chars_per_token
    
    def chunk_transcript(
        self,
        transcript: Union[str, List[dict]],
        item_id: str,
        course_id: Optional[str] = None,
        item_type: Optional[str] = None,
    ) -> List[ChunkResult]:
        """
        Chunk a transcript into sentence-level windows.
        
        Args:
            transcript: Either a plain text string or a list of timestamped segments
                        [{"text": "...", "start": 0.0, "end": 2.5}, ...]
            item_id: Unique identifier for this item (video/reading)
            course_id: Parent course identifier
            item_type: Type of content (video, reading, lab)
            
        Returns:
            List of ChunkResult objects ready for embedding
        """
        # Parse sentences based on input type
        if isinstance(transcript, str):
            sentences = self._split_sentences(transcript)
        elif isinstance(transcript, list):
            sentences = self._split_sentences_with_timestamps(transcript)
        else:
            raise ValueError(f"Unsupported transcript type: {type(transcript)}")
        
        if not sentences:
            logger.warning(f"No sentences extracted from item {item_id}")
            return []
        
        logger.info(f"Extracted {len(sentences)} sentences from item {item_id}")
        
        # Create chunks with sliding window
        chunks = []
        
        for i, sentence in enumerate(sentences):
            # Create context window
            context_before, center, context_after = self._create_window(sentences, i)
            
            # Build full text (context + center + context)
            full_parts = context_before + [center] + context_after
            full_text = " ".join(full_parts)
            
            # Check token limit
            if self._estimate_tokens(full_text) > self.max_chunk_tokens:
                # Reduce context if too long
                while self._estimate_tokens(full_text) > self.max_chunk_tokens and (context_before or context_after):
                    if len(context_before) >= len(context_after) and context_before:
                        context_before = context_before[1:]
                    elif context_after:
                        context_after = context_after[:-1]
                    full_parts = context_before + [center] + context_after
                    full_text = " ".join(full_parts)
            
            # Calculate timing for the window
            start_time = None
            end_time = None
            
            if sentences[max(0, i - len(context_before))].start_time is not None:
                start_idx = max(0, i - len(context_before))
                end_idx = min(len(sentences) - 1, i + len(context_after))
                start_time = sentences[start_idx].start_time
                end_time = sentences[end_idx].end_time
            
            # Create chunk result
            chunk = ChunkResult(
                chunk_id=f"{item_id}_chunk_{i:04d}",
                center_sentence=center,
                full_text=full_text,
                sentence_index=i,
                start_char=sentence.start_char,
                end_char=sentence.end_char,
                start_time=start_time,
                end_time=end_time,
                item_id=item_id,
                course_id=course_id,
                item_type=item_type,
                context_before=context_before,
                context_after=context_after,
            )
            
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks for item {item_id}")
        return chunks
    
    def chunk_batch(
        self,
        items: List[dict],
    ) -> List[ChunkResult]:
        """
        Chunk multiple items in batch.
        
        Args:
            items: List of dicts with keys:
                - transcript: str or List[dict]
                - item_id: str
                - course_id: str (optional)
                - item_type: str (optional)
                
        Returns:
            List of all ChunkResult objects
        """
        all_chunks = []
        
        for item in items:
            chunks = self.chunk_transcript(
                transcript=item["transcript"],
                item_id=item["item_id"],
                course_id=item.get("course_id"),
                item_type=item.get("item_type"),
            )
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(items)} items into {len(all_chunks)} total chunks")
        return all_chunks


# Convenience function
def chunk_transcript(
    transcript: Union[str, List[dict]],
    item_id: str,
    course_id: Optional[str] = None,
    context_size: int = 2,
) -> List[ChunkResult]:
    """
    Convenience function to chunk a transcript with default settings.
    
    Args:
        transcript: Text or timestamped segments
        item_id: Item identifier
        course_id: Course identifier
        context_size: Number of sentences before/after center
        
    Returns:
        List of ChunkResult objects
    """
    chunker = SentenceChunker(
        context_before=context_size,
        context_after=context_size,
    )
    return chunker.chunk_transcript(transcript, item_id, course_id)

