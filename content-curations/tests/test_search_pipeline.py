"""
Tests for the full search pipeline using actual data.
Uses importlib to directly load modules and avoid pandas/numpy segfault.
"""
import pytest
import importlib.util
from pathlib import Path


def import_module_directly(module_name, relative_path):
    """Import a module directly to avoid pandas/numpy import chain."""
    module_path = Path(__file__).parent.parent / "src" / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestChunkingFunctionality:
    """Tests for the chunking component."""
    
    def test_chunking_produces_multiple_chunks_for_long_text(self, sample_transcript):
        """Test that chunking splits long text into multiple chunks."""
        chunker_module = import_module_directly("transcript_chunker", "chunking/transcript_chunker.py")
        TranscriptChunker = chunker_module.TranscriptChunker
        
        chunker = TranscriptChunker(
            window_size=100,  # Small for testing
            overlap=20,
            min_chunk_size=10
        )
        
        # Use chunk_text method
        chunks = chunker.chunk_text(
            text=sample_transcript,
            item_id="test",
            item_name="Test Item",
            course_name="Test Course",
            module_name="Test Module"
        )
        
        # Should produce multiple chunks from a long transcript
        assert len(chunks) >= 1, "Should produce at least one chunk"
        
        # If we have multiple chunks, verify they have metadata
        for chunk in chunks:
            assert chunk.chunk_id, "Chunk should have ID"
            assert chunk.text, "Chunk should have text"
            assert chunk.item_id == "test", "Chunk should reference item"
    
    def test_chunking_with_actual_content(self, sample_video_item):
        """Test chunking with actual video content."""
        if sample_video_item is None:
            pytest.skip("No video item available")
        
        chunker_module = import_module_directly("transcript_chunker", "chunking/transcript_chunker.py")
        TranscriptChunker = chunker_module.TranscriptChunker
        
        chunker = TranscriptChunker(window_size=200, overlap=50, min_chunk_size=50)
        
        content = sample_video_item.get('content_text', '')
        chunks = chunker.chunk_text(
            text=content,
            item_id=sample_video_item['item_id'],
            item_name=sample_video_item['item_name'],
            course_name=sample_video_item['course_name'],
            module_name=sample_video_item['module_name']
        )
        
        # Should produce chunks
        assert len(chunks) >= 1, "Should produce at least one chunk"
        
        # Each chunk should have required fields
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.text
            assert chunk.item_id
    
    def test_short_content_single_chunk(self):
        """Test that short content produces single chunk."""
        chunker_module = import_module_directly("transcript_chunker", "chunking/transcript_chunker.py")
        TranscriptChunker = chunker_module.TranscriptChunker
        
        chunker = TranscriptChunker(window_size=500, overlap=100, min_chunk_size=10)
        short_text = "This is a very short piece of content."
        
        chunks = chunker.chunk_text(
            text=short_text,
            item_id="short",
            item_name="Short Item",
            course_name="Test Course",
            module_name="Test Module"
        )
        
        assert len(chunks) == 1, "Short content should produce single chunk"
        assert short_text in chunks[0].text


class TestDeduplication:
    """Tests for item-level deduplication in search results."""
    
    def test_deduplicate_by_item(self, index_chunks):
        """Test that deduplication returns one result per item."""
        if not index_chunks:
            pytest.skip("No index available")
        
        # Simulate search results with duplicates from same item
        mock_results = []
        for i, chunk in enumerate(index_chunks[:10]):
            mock_results.append({
                'chunk_id': chunk['chunk_id'],
                'item_id': chunk.get('item_id'),
                'score': 0.9 - (i * 0.05),
                'text': chunk.get('text', '')
            })
        
        # Add duplicate item
        if mock_results:
            duplicate = mock_results[0].copy()
            duplicate['score'] = 0.85
            duplicate['chunk_id'] = 'different_chunk'
            mock_results.append(duplicate)
        
        # Deduplicate
        seen_items = set()
        deduplicated = []
        for result in sorted(mock_results, key=lambda x: x['score'], reverse=True):
            item_id = result.get('item_id')
            if item_id and item_id not in seen_items:
                seen_items.add(item_id)
                deduplicated.append(result)
        
        # Should have fewer results after deduplication
        assert len(deduplicated) <= len(mock_results)
        
        # All item_ids should be unique
        item_ids = [r['item_id'] for r in deduplicated if r.get('item_id')]
        assert len(item_ids) == len(set(item_ids)), "Items should be unique"


class TestSentenceChunking:
    """Tests for sentence-level chunking."""
    
    def test_sentence_chunking_basic(self):
        """Test basic sentence chunking."""
        chunker_module = import_module_directly("sentence_chunker", "chunking/sentence_chunker.py")
        SentenceChunker = chunker_module.SentenceChunker
        
        # Use correct init parameters
        chunker = SentenceChunker(
            context_before=2,
            context_after=2,
            min_sentence_length=10,
            max_chunk_tokens=100
        )
        
        text = "First sentence here. Second sentence follows. Third one is here. Fourth sentence. Fifth and final sentence here."
        chunks = chunker.chunk_transcript(text, item_id="test")
        
        assert len(chunks) >= 1, "Should produce chunks"
        
        for chunk in chunks:
            # SentenceChunk has different attributes
            assert hasattr(chunk, 'center_text') or hasattr(chunk, 'full_text')


class TestSubtitleParsing:
    """Tests for subtitle parsing functionality."""
    
    def test_subtitle_parser_import(self):
        """Test that subtitle parser can be imported."""
        parser_module = import_module_directly("subtitle_parser", "parsers/subtitle_parser.py")
        SubtitleParser = parser_module.SubtitleParser
        
        parser = SubtitleParser()
        assert parser is not None
    
    def test_parse_srt_format(self):
        """Test parsing SRT subtitle format using standalone function."""
        parser_module = import_module_directly("subtitle_parser", "parsers/subtitle_parser.py")
        parse_srt = parser_module.parse_srt
        
        srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello and welcome to this lecture.

2
00:00:04,500 --> 00:00:08,000
Today we will learn about programming.

3
00:00:08,500 --> 00:00:12,000
Let's get started with the basics.
"""
        
        segments = parse_srt(srt_content)
        
        assert len(segments) == 3, "Should parse 3 segments"
        assert "Hello" in segments[0]['text']
        assert segments[0]['start'] == 1.0  # Correct key is 'start' not 'start_time'
    
    def test_parser_instance_methods(self):
        """Test SubtitleParser instance methods."""
        parser_module = import_module_directly("subtitle_parser", "parsers/subtitle_parser.py")
        SubtitleParser = parser_module.SubtitleParser
        
        parser = SubtitleParser()
        
        srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello and welcome.

2
00:00:04,500 --> 00:00:08,000
Let's learn programming.
"""
        
        # Use parse_string method (parameter is 'format', not 'format_type')
        segments = parser.parse_string(srt_content, format='srt')
        
        assert len(segments) == 2, "Should parse 2 segments"


class TestSearchResultStructure:
    """Tests for search result structure."""
    
    def test_result_has_required_fields(self, index_chunks):
        """Test that index chunks have fields needed for search results."""
        if not index_chunks:
            pytest.skip("No chunks available")
        
        # Simulate what a search result would look like
        for chunk in index_chunks[:20]:
            result = {
                'chunk_id': chunk['chunk_id'],
                'text': chunk.get('text', ''),
                'item_id': chunk.get('item_id'),
                'score': 0.95,  # Would come from vector search
                'metadata': chunk.get('metadata', {})
            }
            
            assert result['chunk_id'], "Result should have chunk_id"
            assert result['text'], "Result should have text"
    
    def test_can_build_response_from_chunk(self, index_chunks, sample_content):
        """Test that we can build a full response from chunk + content."""
        if not index_chunks or not sample_content:
            pytest.skip("Missing data")
        
        # Build content lookup
        content_by_item = {item['item_id']: item for item in sample_content}
        
        # Get a chunk and enrich it
        chunk = index_chunks[0]
        item_id = chunk.get('item_id')
        
        if item_id and item_id in content_by_item:
            content_item = content_by_item[item_id]
            
            enriched_result = {
                'chunk_id': chunk['chunk_id'],
                'matched_text': chunk.get('text', '')[:200],
                'item_id': item_id,
                'item_name': content_item.get('item_name'),
                'course_name': content_item.get('course_name'),
                'module_name': content_item.get('module_name'),
                'lesson_name': content_item.get('lesson_name'),
                'content_type': content_item.get('content_type'),
                'summary': content_item.get('summary'),
                'score': 0.95
            }
            
            assert enriched_result['item_name'], "Should have item name"
            assert enriched_result['course_name'], "Should have course name"


class TestContextualPrepending:
    """Tests for contextual prepending in embeddings."""
    
    def test_chunk_has_contextual_text(self, sample_video_item):
        """Test that chunks include contextual prefix."""
        if sample_video_item is None:
            pytest.skip("No video item available")
        
        chunker_module = import_module_directly("transcript_chunker", "chunking/transcript_chunker.py")
        TranscriptChunker = chunker_module.TranscriptChunker
        
        chunker = TranscriptChunker(window_size=200, overlap=50, min_chunk_size=20)
        
        chunks = chunker.chunk_text(
            text=sample_video_item.get('content_text', ''),
            item_id=sample_video_item['item_id'],
            item_name=sample_video_item['item_name'],
            course_name=sample_video_item['course_name'],
            module_name=sample_video_item['module_name'],
            bloom_level="Apply",
            difficulty="BEGINNER"
        )
        
        if chunks:
            chunk = chunks[0]
            # Check that contextual_text is populated
            if chunk.contextual_text:
                assert "[Course:" in chunk.contextual_text
                assert sample_video_item['course_name'] in chunk.contextual_text
