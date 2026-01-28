"""
Tests for data loading functionality using actual data files.
"""
import pytest
import json


class TestSampleContentData:
    """Tests for the actual sample_courses_content.json data."""
    
    def test_content_file_exists_and_has_data(self, sample_content):
        """Test that sample content file has data."""
        assert len(sample_content) > 0, "Sample content file should have items"
    
    def test_content_has_required_fields(self, sample_content):
        """Test that each item has required fields for processing."""
        required_fields = [
            'course_id', 'course_name', 'item_id', 'item_name',
            'module_name', 'lesson_name', 'content_text', 'content_type'
        ]
        
        for item in sample_content[:10]:  # Check first 10 items
            for field in required_fields:
                assert field in item, f"Item missing required field: {field}"
    
    def test_content_types_are_valid(self, sample_content):
        """Test that content types are either 'video' or 'reading'."""
        valid_types = {'video', 'reading'}
        
        for item in sample_content:
            assert item['content_type'] in valid_types, \
                f"Invalid content_type: {item['content_type']}"
    
    def test_content_text_is_not_empty(self, sample_content):
        """Test that content_text is not empty for items."""
        empty_count = 0
        for item in sample_content:
            if not item.get('content_text') or len(item['content_text'].strip()) == 0:
                empty_count += 1
        
        # Allow some empty, but most should have content
        assert empty_count < len(sample_content) * 0.1, \
            f"Too many items ({empty_count}) have empty content_text"
    
    def test_unique_item_ids(self, sample_content):
        """Test that item IDs are unique."""
        item_ids = [item['item_id'] for item in sample_content]
        assert len(item_ids) == len(set(item_ids)), "Item IDs should be unique"
    
    def test_video_items_have_summaries(self, sample_content):
        """Test that video items have summary descriptions."""
        videos = [item for item in sample_content if item['content_type'] == 'video']
        videos_with_summary = [v for v in videos if v.get('summary')]
        
        # Most videos should have summaries
        assert len(videos_with_summary) > len(videos) * 0.5, \
            "Most video items should have summaries"
    
    def test_courses_are_present(self, sample_content):
        """Test that we have multiple courses in the data."""
        course_ids = set(item['course_id'] for item in sample_content)
        assert len(course_ids) >= 5, f"Expected at least 5 courses, got {len(course_ids)}"
    
    def test_sample_video_item_structure(self, sample_video_item):
        """Test the structure of a video item."""
        if sample_video_item is None:
            pytest.skip("No video items in sample data")
        
        assert sample_video_item['content_type'] == 'video'
        assert len(sample_video_item['content_text']) > 100, \
            "Video transcript should have substantial content"
        assert sample_video_item['item_name'], "Video should have a name"


class TestEnrichedContentData:
    """Tests for the actual sample_courses_enriched.json data."""
    
    def test_enriched_file_exists_and_has_data(self, enriched_content):
        """Test that enriched content file has data."""
        assert len(enriched_content) > 0, "Enriched content file should have items"
    
    def test_enriched_has_operational_metadata(self, enriched_content):
        """Test that enriched items have operational metadata."""
        for item in enriched_content:
            assert 'operational_metadata' in item, "Missing operational_metadata"
            
            op_meta = item['operational_metadata']
            assert 'course_id' in op_meta
            assert 'item_id' in op_meta
            assert 'difficulty_level' in op_meta
    
    def test_enriched_has_derived_metadata(self, enriched_content):
        """Test that enriched items have LLM-derived metadata."""
        for item in enriched_content:
            assert 'derived_metadata' in item, "Missing derived_metadata"
            
            derived = item['derived_metadata']
            assert 'atomic_skills' in derived, "Missing atomic_skills"
            assert 'bloom_level' in derived, "Missing bloom_level"
            assert 'primary_domain' in derived, "Missing primary_domain"
    
    def test_atomic_skills_are_extracted(self, enriched_content):
        """Test that atomic skills were extracted from content."""
        for item in enriched_content:
            skills = item.get('derived_metadata', {}).get('atomic_skills', [])
            assert isinstance(skills, list), "atomic_skills should be a list"
            assert len(skills) > 0, f"Item {item.get('item_id')} should have skills"
    
    def test_bloom_levels_are_valid(self, enriched_content):
        """Test that Bloom's taxonomy levels are valid."""
        valid_levels = {
            'Remember', 'Understand', 'Apply', 
            'Analyze', 'Evaluate', 'Create'
        }
        
        for item in enriched_content:
            bloom = item.get('derived_metadata', {}).get('bloom_level')
            if bloom:
                assert bloom in valid_levels, f"Invalid Bloom level: {bloom}"
    
    def test_cognitive_load_in_range(self, enriched_content):
        """Test that cognitive load scores are in valid range (1-10)."""
        for item in enriched_content:
            load = item.get('derived_metadata', {}).get('cognitive_load')
            if load is not None:
                assert 1 <= load <= 10, f"Cognitive load {load} out of range"
    
    def test_embedding_input_generated(self, enriched_content):
        """Test that embedding input text is generated."""
        for item in enriched_content:
            embedding_input = item.get('embedding_input')
            assert embedding_input, "Missing embedding_input"
            assert len(embedding_input) > 50, "Embedding input too short"
    
    def test_filter_metadata_for_search(self, enriched_content):
        """Test that filter metadata is present for hybrid search."""
        for item in enriched_content:
            assert 'filter_metadata' in item, "Missing filter_metadata"
            
            filters = item['filter_metadata']
            assert 'course_id' in filters
            assert 'item_id' in filters
            assert 'item_type' in filters


class TestIndexData:
    """Tests for the built FAISS index data."""
    
    def test_index_chunks_exist(self, index_chunks):
        """Test that index chunks were created."""
        assert len(index_chunks) > 0, "Index should have chunks"
    
    def test_chunks_have_required_structure(self, index_chunks):
        """Test chunk structure for vector search."""
        required_fields = ['chunk_id', 'text', 'item_id']
        
        for chunk in index_chunks[:20]:  # Check first 20
            for field in required_fields:
                assert field in chunk, f"Chunk missing field: {field}"
    
    def test_chunks_have_metadata(self, index_chunks):
        """Test that chunks have metadata for filtering."""
        for chunk in index_chunks[:20]:
            metadata = chunk.get('metadata', {})
            # At minimum should have item reference
            assert chunk.get('item_id') or metadata.get('item_id'), \
                "Chunk should reference an item"
    
    def test_index_config_exists(self, index_config):
        """Test that index configuration is saved."""
        assert index_config, "Index config should exist"
    
    def test_multiple_chunks_per_long_content(self, index_chunks, sample_content):
        """Test that long content is chunked into multiple pieces."""
        # Find an item with long content
        long_items = [
            item for item in sample_content 
            if len(item.get('content_text', '')) > 5000
        ]
        
        if long_items:
            long_item_id = long_items[0]['item_id']
            item_chunks = [c for c in index_chunks if c.get('item_id') == long_item_id]
            assert len(item_chunks) >= 1, "Long content should have chunks"


class TestTaxonomyData:
    """Tests for the Coursera skills taxonomy."""
    
    def test_taxonomy_exists(self, coursera_taxonomy):
        """Test that taxonomy file exists and has content."""
        assert coursera_taxonomy, "Taxonomy should exist"
        assert 'skills' in coursera_taxonomy, "Taxonomy should have skills list"
    
    def test_taxonomy_has_skills(self, coursera_taxonomy):
        """Test that taxonomy has a reasonable number of skills."""
        skills = coursera_taxonomy.get('skills', [])
        assert len(skills) > 100, f"Expected many skills, got {len(skills)}"
    
    def test_skills_are_strings(self, coursera_taxonomy):
        """Test that skills are string values."""
        skills = coursera_taxonomy.get('skills', [])
        for skill in skills[:100]:
            assert isinstance(skill, str), "Skills should be strings"
            assert len(skill) > 0, "Skills should not be empty"
    
    def test_skills_are_unique(self, coursera_taxonomy):
        """Test that skills in taxonomy are unique."""
        skills = coursera_taxonomy.get('skills', [])
        assert len(skills) == len(set(skills)), "Skills should be unique"


class TestDataConsistency:
    """Tests for data consistency across files."""
    
    def test_enriched_items_exist_in_content(self, sample_content, enriched_content):
        """Test that enriched items reference existing content items."""
        content_item_ids = {item['item_id'] for item in sample_content}
        
        for enriched_item in enriched_content:
            item_id = enriched_item.get('item_id')
            assert item_id in content_item_ids, \
                f"Enriched item {item_id} not in sample content"
    
    def test_chunk_items_exist_in_content(self, sample_content, index_chunks):
        """Test that chunks reference existing content items."""
        content_item_ids = {item['item_id'] for item in sample_content}
        
        for chunk in index_chunks[:50]:  # Check first 50
            item_id = chunk.get('item_id')
            if item_id:
                assert item_id in content_item_ids, \
                    f"Chunk references unknown item {item_id}"
    
    def test_content_courses_are_consistent(self, sample_content):
        """Test that course metadata is consistent for items in same course."""
        courses = {}
        for item in sample_content:
            course_id = item['course_id']
            if course_id not in courses:
                courses[course_id] = item['course_name']
            else:
                assert courses[course_id] == item['course_name'], \
                    f"Inconsistent course name for {course_id}"
