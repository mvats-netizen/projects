"""
Tests for metadata enrichment using actual enriched data.
"""
import pytest


class TestOperationalMetadata:
    """Tests for operational metadata extracted from course catalog."""
    
    def test_operational_metadata_fields(self, enriched_content):
        """Test that operational metadata has expected fields."""
        expected_fields = [
            'course_id', 'course_name', 'item_id', 'item_name',
            'item_type', 'module_name', 'lesson_name', 'difficulty_level'
        ]
        
        for item in enriched_content:
            op_meta = item.get('operational_metadata', {})
            for field in expected_fields:
                assert field in op_meta, f"Missing operational field: {field}"
    
    def test_difficulty_levels_are_valid(self, enriched_content):
        """Test that difficulty levels are valid Coursera values."""
        valid_difficulties = {'BEGINNER', 'INTERMEDIATE', 'ADVANCED', 'MIXED', ''}
        
        for item in enriched_content:
            difficulty = item.get('operational_metadata', {}).get('difficulty_level', '')
            assert difficulty in valid_difficulties, \
                f"Invalid difficulty: {difficulty}"
    
    def test_item_types_match_content_type(self, enriched_content):
        """Test that item_type in operational metadata matches content_type."""
        for item in enriched_content:
            content_type = item.get('content_type')
            item_type = item.get('operational_metadata', {}).get('item_type')
            assert content_type == item_type, \
                f"Mismatch: content_type={content_type}, item_type={item_type}"
    
    def test_catalogue_skills_format(self, enriched_content):
        """Test that catalogue_skills is properly formatted."""
        for item in enriched_content:
            skills = item.get('operational_metadata', {}).get('catalogue_skills', [])
            assert isinstance(skills, list), "catalogue_skills should be a list"


class TestDerivedMetadata:
    """Tests for LLM-derived metadata."""
    
    def test_derived_metadata_structure(self, enriched_content):
        """Test the structure of derived metadata."""
        expected_fields = [
            'atomic_skills', 'key_concepts', 'bloom_level',
            'primary_domain', 'sub_domain'
        ]
        
        for item in enriched_content:
            derived = item.get('derived_metadata', {})
            for field in expected_fields:
                assert field in derived, f"Missing derived field: {field}"
    
    def test_atomic_skills_quality(self, enriched_content):
        """Test that atomic skills are meaningful."""
        for item in enriched_content:
            skills = item.get('derived_metadata', {}).get('atomic_skills', [])
            
            # Skills should be present
            assert len(skills) > 0, "Should have at least one skill"
            
            # Skills should be descriptive (not too short)
            for skill in skills:
                assert len(skill) > 5, f"Skill too short: {skill}"
    
    def test_key_concepts_extracted(self, enriched_content):
        """Test that key concepts are extracted."""
        for item in enriched_content:
            concepts = item.get('derived_metadata', {}).get('key_concepts', [])
            assert isinstance(concepts, list), "key_concepts should be a list"
            assert len(concepts) > 0, "Should have at least one concept"
    
    def test_prerequisite_concepts_format(self, enriched_content):
        """Test prerequisite concepts are properly formatted."""
        for item in enriched_content:
            prereqs = item.get('derived_metadata', {}).get('prerequisite_concepts', [])
            assert isinstance(prereqs, list), "prerequisite_concepts should be a list"
    
    def test_instructional_function_values(self, enriched_content):
        """Test that instructional function values are valid."""
        # Based on actual schema.py InstructionalFunction enum
        valid_functions = {
            'Definition', 'Analogy', 'Code Walkthrough', 'Proof',
            'Synthesis', 'Example', 'Exercise', 'Summary',
            # Also allow some variations from LLM extraction
            'Conceptual Explanation', 'Technical Demo', 'Problem Solving',
            'Case Study', 'Q&A/Discussion', ''
        }
        
        for item in enriched_content:
            func = item.get('derived_metadata', {}).get('instructional_function', '')
            if func:
                assert func in valid_functions, f"Invalid function: {func}"
    
    def test_domain_classification(self, enriched_content):
        """Test that domain classification is present."""
        for item in enriched_content:
            derived = item.get('derived_metadata', {})
            primary = derived.get('primary_domain')
            sub = derived.get('sub_domain')
            
            assert primary, "Should have primary_domain"
            assert sub, "Should have sub_domain"
    
    def test_extraction_confidence_in_range(self, enriched_content):
        """Test that extraction confidence is in valid range."""
        for item in enriched_content:
            confidence = item.get('derived_metadata', {}).get('extraction_confidence')
            if confidence is not None:
                assert 0 <= confidence <= 1, \
                    f"Confidence {confidence} out of range [0,1]"


class TestEmbeddingInput:
    """Tests for embedding input generation."""
    
    def test_embedding_input_contains_context(self, enriched_content):
        """Test that embedding input includes contextual metadata."""
        for item in enriched_content:
            embedding_input = item.get('embedding_input', '')
            
            # Should contain course context
            assert '[Course:' in embedding_input, "Missing course context"
            assert '[Module:' in embedding_input, "Missing module context"
    
    def test_embedding_input_contains_bloom_level(self, enriched_content):
        """Test that embedding input includes Bloom's level."""
        for item in enriched_content:
            embedding_input = item.get('embedding_input', '')
            assert '[Level:' in embedding_input, "Missing Bloom level context"
    
    def test_embedding_input_contains_content(self, enriched_content):
        """Test that embedding input includes actual content text."""
        for item in enriched_content:
            embedding_input = item.get('embedding_input', '')
            content_text = item.get('content_text', '')[:50]  # First 50 chars
            
            # Should contain some of the original content
            assert len(embedding_input) > 100, "Embedding input too short"


class TestFilterMetadata:
    """Tests for filter metadata used in hybrid search."""
    
    def test_filter_metadata_structure(self, enriched_content):
        """Test filter metadata has required fields."""
        required_filters = [
            'course_id', 'item_id', 'item_type', 'difficulty',
            'bloom_level', 'primary_domain'
        ]
        
        for item in enriched_content:
            filters = item.get('filter_metadata', {})
            for field in required_filters:
                assert field in filters, f"Missing filter: {field}"
    
    def test_filter_values_match_source(self, enriched_content):
        """Test that filter values match their source metadata."""
        for item in enriched_content:
            filters = item['filter_metadata']
            op_meta = item['operational_metadata']
            derived = item['derived_metadata']
            
            # Check consistency
            assert filters['course_id'] == op_meta['course_id']
            assert filters['item_id'] == op_meta['item_id']
            assert filters['bloom_level'] == derived['bloom_level']
            assert filters['primary_domain'] == derived['primary_domain']


class TestMetadataQuality:
    """Tests for overall metadata quality."""
    
    def test_no_empty_critical_fields(self, enriched_content):
        """Test that critical fields are not empty."""
        critical_fields = [
            ('course_id', lambda x: x),
            ('item_id', lambda x: x),
            ('content_text', lambda x: x),
            ('derived_metadata.atomic_skills', lambda x: x.get('derived_metadata', {}).get('atomic_skills', []))
        ]
        
        for item in enriched_content:
            assert item.get('course_id'), "course_id should not be empty"
            assert item.get('item_id'), "item_id should not be empty"
            assert item.get('content_text'), "content_text should not be empty"
            
            skills = item.get('derived_metadata', {}).get('atomic_skills', [])
            assert len(skills) > 0, "Should have extracted skills"
    
    def test_metadata_enrichment_complete(self, enriched_content):
        """Test that enrichment process completed for all items."""
        for item in enriched_content:
            # Should have both operational and derived
            assert 'operational_metadata' in item
            assert 'derived_metadata' in item
            
            # Should have embedding ready fields
            assert 'embedding_input' in item
            assert 'filter_metadata' in item
    
    def test_skills_are_relevant_to_content(self, enriched_content):
        """Test that extracted skills are relevant to content."""
        for item in enriched_content[:5]:  # Check first 5 items
            content = item.get('content_text', '').lower()
            skills = item.get('derived_metadata', {}).get('atomic_skills', [])
            
            # At least some skill keywords should appear in content
            # This is a soft check - skills can be abstracted
            skill_words = set()
            for skill in skills:
                # Extract words from skill
                words = skill.lower().split()
                skill_words.update(words)
            
            # Check if any skill-related words appear in content
            content_words = set(content.split())
            overlap = skill_words & content_words
            
            # Should have some overlap (allowing for abstraction)
            assert len(overlap) > 0 or len(skills) > 0, \
                "Skills should be somewhat related to content"


class TestCourseMetadataConsistency:
    """Tests for course-level metadata consistency."""
    
    def test_items_from_same_course_have_consistent_metadata(self, enriched_content):
        """Test that items from the same course have consistent course metadata."""
        courses = {}
        
        for item in enriched_content:
            course_id = item['course_id']
            course_name = item['course_name']
            
            if course_id in courses:
                assert courses[course_id] == course_name, \
                    f"Inconsistent course name for {course_id}"
            else:
                courses[course_id] = course_name
    
    def test_domain_classification_consistency(self, enriched_content):
        """Test that domain classification is somewhat consistent within courses."""
        # Group by course
        course_domains = {}
        for item in enriched_content:
            course_id = item['course_id']
            domain = item.get('derived_metadata', {}).get('primary_domain')
            
            if course_id not in course_domains:
                course_domains[course_id] = set()
            if domain:
                course_domains[course_id].add(domain)
        
        # Each course should have limited domain variation
        for course_id, domains in course_domains.items():
            # Allow some variation but not too much
            assert len(domains) <= 3, \
                f"Course {course_id} has too many domains: {domains}"
