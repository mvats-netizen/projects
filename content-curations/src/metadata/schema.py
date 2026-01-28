"""
Metadata Schema for AI-Led Curations

Based on:
- data_asset_req.csv: Operational + Derived metadata requirements
- roadmap.pdf: Pedagogical metadata extraction approach
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


# =============================================================================
# ENUMS FOR DERIVED METADATA
# =============================================================================

class BloomLevel(str, Enum):
    """Bloom's Taxonomy cognitive levels."""
    REMEMBER = "Remember"      # Recall facts and basic concepts
    UNDERSTAND = "Understand"  # Explain ideas or concepts
    APPLY = "Apply"           # Use information in new situations
    ANALYZE = "Analyze"       # Draw connections among ideas
    EVALUATE = "Evaluate"     # Justify a decision or course of action
    CREATE = "Create"         # Produce new or original work


class InstructionalFunction(str, Enum):
    """Teaching method/style of the content chunk."""
    DEFINITION = "Definition"           # Explaining what something is
    ANALOGY = "Analogy"                 # Comparing to familiar concepts
    CODE_WALKTHROUGH = "Code Walkthrough"  # Step-by-step code explanation
    PROOF = "Proof"                     # Mathematical/logical demonstration
    SYNTHESIS = "Synthesis"             # Combining multiple concepts
    EXAMPLE = "Example"                 # Practical demonstration
    EXERCISE = "Exercise"               # Practice problem
    SUMMARY = "Summary"                 # Recap of key points


class ContentType(str, Enum):
    """Type of content item."""
    VIDEO = "video"
    READING = "reading"
    LAB = "lab"
    QUIZ = "quiz"
    ASSIGNMENT = "assignment"


# =============================================================================
# OPERATIONAL METADATA (from CourseCatalogue / Databricks)
# =============================================================================

@dataclass
class OperationalMetadata:
    """
    PART 1: Operational Metadata
    Sourced from CourseCatalogue.xlsx or Databricks
    """
    # Course identifiers
    course_id: str
    course_name: str
    course_slug: str
    course_url: str
    
    # Item identifiers
    item_id: str
    item_name: str
    item_type: ContentType
    
    # Hierarchy
    module_name: str
    lesson_name: str
    
    # Course-level metadata
    course_duration_minutes: float = 0.0
    lecture_count: int = 0
    reading_count: int = 0
    module_count: int = 0
    
    # Instructor & Partner
    instructor_name: Optional[str] = None
    instructor_title: Optional[str] = None
    partner_name: Optional[str] = None
    
    # Quality indicators
    difficulty_level: str = "BEGINNER"  # BEGINNER, INTERMEDIATE, ADVANCED
    star_rating: float = 0.0
    avg_pass_rate: Optional[float] = None
    
    # Temporal
    last_updated: Optional[datetime] = None
    
    # Existing skills (from catalogue)
    catalogue_skills: List[str] = field(default_factory=list)
    learning_objectives: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'course_id': self.course_id,
            'course_name': self.course_name,
            'course_slug': self.course_slug,
            'course_url': self.course_url,
            'item_id': self.item_id,
            'item_name': self.item_name,
            'item_type': self.item_type.value if isinstance(self.item_type, ContentType) else self.item_type,
            'module_name': self.module_name,
            'lesson_name': self.lesson_name,
            'course_duration_minutes': self.course_duration_minutes,
            'lecture_count': self.lecture_count,
            'reading_count': self.reading_count,
            'module_count': self.module_count,
            'instructor_name': self.instructor_name,
            'instructor_title': self.instructor_title,
            'partner_name': self.partner_name,
            'difficulty_level': self.difficulty_level,
            'star_rating': self.star_rating,
            'avg_pass_rate': self.avg_pass_rate,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'catalogue_skills': self.catalogue_skills,
            'learning_objectives': self.learning_objectives,
        }


# =============================================================================
# DERIVED METADATA (from LLM Transcript Analysis)
# =============================================================================

@dataclass
class DerivedMetadata:
    """
    PART 2: Derived Metadata
    Extracted via LLM analysis of transcript chunks
    
    Based on roadmap.pdf pedagogical metadata schema:
    - bloom_level, instr_function, cognitive_load
    - entities, prerequisites, timestamp
    """
    # Chunk identification
    chunk_id: str
    chunk_text: str
    
    # Timestamp (for video segments)
    start_time: Optional[str] = None  # e.g., "04:12"
    end_time: Optional[str] = None    # e.g., "06:45"
    
    # Skills & Concepts
    atomic_skills: List[str] = field(default_factory=list)  # 3-5 specific skills
    key_concepts: List[str] = field(default_factory=list)   # Entities mentioned
    prerequisite_concepts: List[str] = field(default_factory=list)  # Required prior knowledge
    
    # Classification
    primary_domain: Optional[str] = None    # e.g., "Computer Science"
    sub_domain: Optional[str] = None        # e.g., "Machine Learning"
    
    # Pedagogical indicators
    bloom_level: Optional[BloomLevel] = None
    instructional_function: Optional[InstructionalFunction] = None
    cognitive_load: int = 5  # 1-10 scale (technical density)
    
    # Confidence scores
    extraction_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'chunk_text': self.chunk_text,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'atomic_skills': self.atomic_skills,
            'key_concepts': self.key_concepts,
            'prerequisite_concepts': self.prerequisite_concepts,
            'primary_domain': self.primary_domain,
            'sub_domain': self.sub_domain,
            'bloom_level': self.bloom_level.value if self.bloom_level else None,
            'instructional_function': self.instructional_function.value if self.instructional_function else None,
            'cognitive_load': self.cognitive_load,
            'extraction_confidence': self.extraction_confidence,
        }


# =============================================================================
# COMBINED CONTENT METADATA
# =============================================================================

@dataclass
class ContentMetadata:
    """
    Complete metadata for a content chunk.
    Combines operational + derived metadata.
    """
    # Core identifiers
    id: str  # Unique chunk identifier
    
    # Operational (from catalogue/DB)
    operational: OperationalMetadata
    
    # Derived (from LLM analysis)
    derived: Optional[DerivedMetadata] = None
    
    # Embedding info
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    # For contextual embedding (from roadmap.pdf)
    # Pre-pend format: [Course: X] [Module: Y] [Level: Z] {Text}
    contextual_prefix: Optional[str] = None
    
    def get_embedding_input(self) -> str:
        """
        Generate input for embedding with contextual pre-pending.
        
        From roadmap.pdf:
        "Pre-pend the chunk with its hierarchy to ensure context-aware embeddings"
        """
        if self.contextual_prefix:
            return self.contextual_prefix
        
        # Build contextual prefix
        parts = []
        parts.append(f"[Course: {self.operational.course_name}]")
        parts.append(f"[Module: {self.operational.module_name}]")
        
        if self.derived and self.derived.bloom_level:
            parts.append(f"[Level: {self.derived.bloom_level.value}]")
        
        if self.operational.difficulty_level:
            parts.append(f"[Difficulty: {self.operational.difficulty_level}]")
        
        # Add the actual content
        content_text = self.derived.chunk_text if self.derived else ""
        
        self.contextual_prefix = " ".join(parts) + " " + content_text
        return self.contextual_prefix
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'operational': self.operational.to_dict(),
            'derived': self.derived.to_dict() if self.derived else None,
            'embedding_model': self.embedding_model,
            'contextual_prefix': self.contextual_prefix,
        }
    
    def get_filter_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for vector DB filtering.
        
        From roadmap.pdf:
        "Metadata Filtering - narrow search to specific bloom_level, domain, etc."
        """
        filters = {
            'course_id': self.operational.course_id,
            'item_id': self.operational.item_id,
            'item_type': self.operational.item_type.value if isinstance(self.operational.item_type, ContentType) else self.operational.item_type,
            'difficulty': self.operational.difficulty_level,
            'partner': self.operational.partner_name,
        }
        
        if self.derived:
            filters.update({
                'bloom_level': self.derived.bloom_level.value if self.derived.bloom_level else None,
                'cognitive_load': self.derived.cognitive_load,
                'primary_domain': self.derived.primary_domain,
                'sub_domain': self.derived.sub_domain,
            })
        
        return {k: v for k, v in filters.items() if v is not None}
