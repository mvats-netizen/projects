"""
Tests for metadata schema validation using actual data.
Schema tests import directly from schema.py to avoid pandas/numpy segfault.
"""
import pytest
import importlib.util
from pathlib import Path


def import_schema_directly():
    """Import schema classes directly to avoid pandas/numpy segfault."""
    schema_path = Path(__file__).parent.parent / "src" / "metadata" / "schema.py"
    spec = importlib.util.spec_from_file_location("schema", schema_path)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    return schema


class TestBloomLevelEnum:
    """Tests for Bloom's Taxonomy level enumeration."""
    
    def test_all_bloom_levels_exist(self):
        """Test that all Bloom's levels are defined."""
        schema = import_schema_directly()
        BloomLevel = schema.BloomLevel
        
        expected_levels = ['REMEMBER', 'UNDERSTAND', 'APPLY', 'ANALYZE', 'EVALUATE', 'CREATE']
        for level in expected_levels:
            assert hasattr(BloomLevel, level), f"Missing Bloom level: {level}"
    
    def test_bloom_level_values(self):
        """Test Bloom level string values."""
        schema = import_schema_directly()
        BloomLevel = schema.BloomLevel
        
        assert BloomLevel.APPLY.value == "Apply"
        assert BloomLevel.UNDERSTAND.value == "Understand"
        assert BloomLevel.REMEMBER.value == "Remember"
    
    def test_bloom_level_from_value(self):
        """Test getting Bloom level from value string."""
        schema = import_schema_directly()
        BloomLevel = schema.BloomLevel
        
        # Can access by value
        assert BloomLevel("Apply") == BloomLevel.APPLY
        assert BloomLevel("Understand") == BloomLevel.UNDERSTAND


class TestInstructionalFunctionEnum:
    """Tests for instructional function enumeration."""
    
    def test_instructional_functions_exist(self):
        """Test that instructional functions are defined."""
        schema = import_schema_directly()
        InstructionalFunction = schema.InstructionalFunction
        
        expected_functions = [
            'DEFINITION', 'ANALOGY', 'CODE_WALKTHROUGH', 'PROOF',
            'SYNTHESIS', 'EXAMPLE', 'EXERCISE', 'SUMMARY'
        ]
        for func in expected_functions:
            assert hasattr(InstructionalFunction, func), f"Missing function: {func}"
    
    def test_instructional_function_values(self):
        """Test instructional function string values."""
        schema = import_schema_directly()
        InstructionalFunction = schema.InstructionalFunction
        
        assert InstructionalFunction.CODE_WALKTHROUGH.value == "Code Walkthrough"
        assert InstructionalFunction.DEFINITION.value == "Definition"


class TestContentTypeEnum:
    """Tests for content type enumeration."""
    
    def test_content_types_exist(self):
        """Test that content types are defined."""
        schema = import_schema_directly()
        ContentType = schema.ContentType
        
        assert hasattr(ContentType, 'VIDEO')
        assert hasattr(ContentType, 'READING')
    
    def test_content_type_values(self):
        """Test content type string values."""
        schema = import_schema_directly()
        ContentType = schema.ContentType
        
        assert ContentType.VIDEO.value == "video"
        assert ContentType.READING.value == "reading"


class TestOperationalMetadataDataclass:
    """Tests for OperationalMetadata dataclass."""
    
    def test_create_operational_metadata(self):
        """Test creating OperationalMetadata instance."""
        schema = import_schema_directly()
        OperationalMetadata = schema.OperationalMetadata
        ContentType = schema.ContentType
        
        metadata = OperationalMetadata(
            course_id="test_course_123",
            course_name="Test Course",
            course_slug="test-course",
            course_url="https://coursera.org/learn/test-course",
            item_id="item_456",
            item_name="Test Item",
            item_type=ContentType.VIDEO,
            module_name="Test Module",
            lesson_name="Test Lesson",
            difficulty_level="BEGINNER"
        )
        
        assert metadata.course_id == "test_course_123"
        assert metadata.difficulty_level == "BEGINNER"
    
    def test_operational_metadata_to_dict(self):
        """Test converting OperationalMetadata to dictionary."""
        schema = import_schema_directly()
        OperationalMetadata = schema.OperationalMetadata
        ContentType = schema.ContentType
        
        metadata = OperationalMetadata(
            course_id="test_123",
            course_name="Test Course",
            course_slug="test-course",
            course_url="https://coursera.org/learn/test-course",
            item_id="item_456",
            item_name="Test Item",
            item_type=ContentType.VIDEO,
            module_name="Module 1",
            lesson_name="Lesson 1",
            difficulty_level="INTERMEDIATE"
        )
        
        result = metadata.to_dict()
        
        assert isinstance(result, dict)
        assert result['course_id'] == "test_123"
        assert result['difficulty_level'] == "INTERMEDIATE"
    
    def test_operational_metadata_from_actual_data(self, enriched_content):
        """Test that actual data matches OperationalMetadata structure."""
        if not enriched_content:
            pytest.skip("No enriched content available")
        
        schema = import_schema_directly()
        OperationalMetadata = schema.OperationalMetadata
        
        item = enriched_content[0]
        op_data = item.get('operational_metadata', {})
        
        # Should be able to create from actual data (with defaults)
        metadata = OperationalMetadata(
            course_id=op_data.get('course_id', ''),
            course_name=op_data.get('course_name', ''),
            course_slug=op_data.get('course_slug', ''),
            course_url=op_data.get('course_url', ''),
            item_id=op_data.get('item_id', ''),
            item_name=op_data.get('item_name', ''),
            item_type=op_data.get('item_type', 'video'),
            module_name=op_data.get('module_name', ''),
            lesson_name=op_data.get('lesson_name', ''),
            difficulty_level=op_data.get('difficulty_level', 'BEGINNER')
        )
        
        assert metadata.course_id == op_data.get('course_id', '')


class TestDerivedMetadataDataclass:
    """Tests for DerivedMetadata dataclass."""
    
    def test_create_derived_metadata(self):
        """Test creating DerivedMetadata instance."""
        schema = import_schema_directly()
        DerivedMetadata = schema.DerivedMetadata
        BloomLevel = schema.BloomLevel
        
        metadata = DerivedMetadata(
            chunk_id="chunk_001",
            chunk_text="Sample text for testing",
            atomic_skills=["skill1", "skill2"],
            key_concepts=["concept1", "concept2"],
            bloom_level=BloomLevel.APPLY,
            primary_domain="Computer Science",
            sub_domain="Programming"
        )
        
        assert metadata.chunk_id == "chunk_001"
        assert len(metadata.atomic_skills) == 2
        assert metadata.bloom_level == BloomLevel.APPLY
    
    def test_derived_metadata_to_dict(self):
        """Test converting DerivedMetadata to dictionary."""
        schema = import_schema_directly()
        DerivedMetadata = schema.DerivedMetadata
        BloomLevel = schema.BloomLevel
        
        metadata = DerivedMetadata(
            chunk_id="chunk_002",
            chunk_text="Test content",
            atomic_skills=["Python programming"],
            key_concepts=["variables"],
            bloom_level=BloomLevel.UNDERSTAND,
            primary_domain="Technology",
            sub_domain="Software Development"
        )
        
        result = metadata.to_dict()
        
        assert isinstance(result, dict)
        assert 'atomic_skills' in result
        assert result['bloom_level'] == 'Understand'
    
    def test_derived_metadata_from_actual_data(self, enriched_content):
        """Test that actual data matches DerivedMetadata structure."""
        if not enriched_content:
            pytest.skip("No enriched content available")
        
        schema = import_schema_directly()
        DerivedMetadata = schema.DerivedMetadata
        BloomLevel = schema.BloomLevel
        
        item = enriched_content[0]
        derived_data = item.get('derived_metadata', {})
        
        # Get bloom level safely
        bloom_str = derived_data.get('bloom_level', 'Understand')
        try:
            bloom = BloomLevel(bloom_str)
        except ValueError:
            bloom = BloomLevel.UNDERSTAND
        
        metadata = DerivedMetadata(
            chunk_id=derived_data.get('chunk_id', item.get('item_id', '')),
            chunk_text=derived_data.get('chunk_text', '')[:500] if derived_data.get('chunk_text') else '',
            atomic_skills=derived_data.get('atomic_skills', []),
            key_concepts=derived_data.get('key_concepts', []),
            bloom_level=bloom,
            primary_domain=derived_data.get('primary_domain', ''),
            sub_domain=derived_data.get('sub_domain', '')
        )
        
        assert len(metadata.atomic_skills) > 0


class TestContentMetadataDataclass:
    """Tests for ContentMetadata combined dataclass."""
    
    def test_create_content_metadata(self):
        """Test creating ContentMetadata with both operational and derived."""
        schema = import_schema_directly()
        OperationalMetadata = schema.OperationalMetadata
        DerivedMetadata = schema.DerivedMetadata
        ContentMetadata = schema.ContentMetadata
        BloomLevel = schema.BloomLevel
        ContentType = schema.ContentType
        
        op_meta = OperationalMetadata(
            course_id="course_123",
            course_name="Test Course",
            course_slug="test-course",
            course_url="https://coursera.org/learn/test-course",
            item_id="item_456",
            item_name="Test Item",
            item_type=ContentType.VIDEO,
            module_name="Module 1",
            lesson_name="Lesson 1",
            difficulty_level="BEGINNER"
        )
        
        derived_meta = DerivedMetadata(
            chunk_id="chunk_001",
            chunk_text="Sample content",
            atomic_skills=["skill1"],
            key_concepts=["concept1"],
            bloom_level=BloomLevel.APPLY,
            primary_domain="Tech",
            sub_domain="Dev"
        )
        
        content_meta = ContentMetadata(
            id="content_123",
            operational=op_meta,
            derived=derived_meta
        )
        
        assert content_meta.operational.course_id == "course_123"
        assert content_meta.derived.bloom_level == BloomLevel.APPLY
    
    def test_content_metadata_get_embedding_input(self):
        """Test generating embedding input from ContentMetadata."""
        schema = import_schema_directly()
        OperationalMetadata = schema.OperationalMetadata
        DerivedMetadata = schema.DerivedMetadata
        ContentMetadata = schema.ContentMetadata
        BloomLevel = schema.BloomLevel
        ContentType = schema.ContentType
        
        op_meta = OperationalMetadata(
            course_id="course_123",
            course_name="Machine Learning Basics",
            course_slug="ml-basics",
            course_url="https://coursera.org/learn/ml-basics",
            item_id="item_456",
            item_name="Introduction to ML",
            item_type=ContentType.VIDEO,
            module_name="Getting Started",
            lesson_name="What is ML?",
            difficulty_level="BEGINNER"
        )
        
        derived_meta = DerivedMetadata(
            chunk_id="chunk_001",
            chunk_text="Machine learning is a subset of AI",
            atomic_skills=["Define ML"],
            key_concepts=["machine learning", "AI"],
            bloom_level=BloomLevel.UNDERSTAND,
            primary_domain="Computer Science",
            sub_domain="AI"
        )
        
        content_meta = ContentMetadata(
            id="content_123",
            operational=op_meta,
            derived=derived_meta
        )
        
        embedding_input = content_meta.get_embedding_input()
        
        assert "[Course: Machine Learning Basics]" in embedding_input
        assert "[Module: Getting Started]" in embedding_input
        assert "[Level: Understand]" in embedding_input
        assert "Machine learning is a subset of AI" in embedding_input


class TestActualDataValidation:
    """Tests to validate actual enriched data against schema."""
    
    def test_all_enriched_items_have_valid_bloom_levels(self, enriched_content):
        """Test that all enriched items have valid Bloom levels."""
        valid_levels = {'Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'}
        
        for item in enriched_content:
            bloom = item.get('derived_metadata', {}).get('bloom_level')
            if bloom:
                assert bloom in valid_levels, f"Invalid Bloom level: {bloom}"
    
    def test_all_enriched_items_have_valid_content_types(self, enriched_content):
        """Test that all enriched items have valid content types."""
        valid_types = {'video', 'reading'}
        
        for item in enriched_content:
            content_type = item.get('content_type')
            assert content_type in valid_types, f"Invalid type: {content_type}"
    
    def test_cognitive_load_values_are_valid(self, enriched_content):
        """Test that cognitive load values are within expected range."""
        for item in enriched_content:
            load = item.get('derived_metadata', {}).get('cognitive_load')
            if load is not None:
                assert isinstance(load, (int, float)), "Cognitive load should be numeric"
                assert 1 <= load <= 10, f"Cognitive load {load} out of range"
    
    def test_atomic_skills_are_lists(self, enriched_content):
        """Test that atomic skills are always lists."""
        for item in enriched_content:
            skills = item.get('derived_metadata', {}).get('atomic_skills')
            assert isinstance(skills, list), f"Skills should be list, got {type(skills)}"
    
    def test_filter_metadata_types(self, enriched_content):
        """Test that filter metadata has correct types."""
        for item in enriched_content:
            filters = item.get('filter_metadata', {})
            
            # String fields
            assert isinstance(filters.get('course_id', ''), str)
            assert isinstance(filters.get('item_id', ''), str)
            assert isinstance(filters.get('item_type', ''), str)
            
            # Cognitive load should be int or None
            cog_load = filters.get('cognitive_load')
            if cog_load is not None:
                assert isinstance(cog_load, (int, float))
