"""
Operational Metadata Loader

Loads PART 1 metadata from CourseCatalogue.xlsx
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
import logging

from .schema import OperationalMetadata, ContentType

logger = logging.getLogger(__name__)


class OperationalMetadataLoader:
    """
    Loads operational metadata from CourseCatalogue.xlsx.
    
    Maps columns to OperationalMetadata schema:
    - course_name, course_id, course_slug, course_url
    - total_duration_minutes → course_duration_minutes
    - lecture_count, reading_count
    - partner_names → partner_name
    - course_difficulty_level → difficulty_level
    - course_star_rating → star_rating
    - all_skills → catalogue_skills
    - learning_objectives
    """
    
    def __init__(self, catalogue_path: str = None):
        """
        Initialize loader.
        
        Args:
            catalogue_path: Path to CourseCatalogue.xlsx
        """
        if catalogue_path is None:
            # Default path
            project_root = Path(__file__).parent.parent.parent
            catalogue_path = project_root / "data" / "CourseCatalogue.xlsx"
        
        self.catalogue_path = Path(catalogue_path)
        self._catalogue_df: Optional[pd.DataFrame] = None
        self._course_lookup: Dict[str, Dict[str, Any]] = {}
    
    def load(self) -> pd.DataFrame:
        """Load the catalogue into memory."""
        if self._catalogue_df is None:
            logger.info(f"Loading catalogue from {self.catalogue_path}")
            self._catalogue_df = pd.read_excel(self.catalogue_path)
            
            # Build course lookup
            for _, row in self._catalogue_df.iterrows():
                course_id = row.get('course_id')
                if course_id:
                    self._course_lookup[course_id] = row.to_dict()
            
            logger.info(f"Loaded {len(self._course_lookup)} courses")
        
        return self._catalogue_df
    
    def get_course_metadata(self, course_id: str) -> Optional[Dict[str, Any]]:
        """Get operational metadata for a course."""
        if not self._course_lookup:
            self.load()
        
        return self._course_lookup.get(course_id)
    
    def enrich_content_item(
        self,
        content_item: Dict[str, Any]
    ) -> OperationalMetadata:
        """
        Enrich a content item (video/reading) with operational metadata.
        
        Args:
            content_item: Dict with course_id, item_id, item_name, etc.
                         (from sample_courses_content.json)
        
        Returns:
            OperationalMetadata object
        """
        course_id = content_item.get('course_id', '')
        course_meta = self.get_course_metadata(course_id) or {}
        
        # Parse skills from comma-separated string
        skills_str = course_meta.get('all_skills', '')
        catalogue_skills = []
        if isinstance(skills_str, str) and skills_str:
            catalogue_skills = [s.strip() for s in skills_str.split(',')]
        
        # Parse content type
        content_type_str = content_item.get('content_type', 'video')
        try:
            content_type = ContentType(content_type_str)
        except ValueError:
            content_type = ContentType.VIDEO
        
        # Parse last updated date
        last_updated = None
        launch_ts = course_meta.get('course_launch_ts')
        if isinstance(launch_ts, datetime):
            last_updated = launch_ts
        
        return OperationalMetadata(
            # Course identifiers
            course_id=course_id,
            course_name=course_meta.get('course_name', content_item.get('course_name', '')),
            course_slug=content_item.get('course_slug', course_meta.get('course_slug', '')),
            course_url=content_item.get('course_url', course_meta.get('course_url', '')),
            
            # Item identifiers
            item_id=content_item.get('item_id', ''),
            item_name=content_item.get('item_name', ''),
            item_type=content_type,
            
            # Hierarchy
            module_name=content_item.get('module_name', ''),
            lesson_name=content_item.get('lesson_name', ''),
            
            # Course-level metadata
            course_duration_minutes=float(course_meta.get('total_duration_minutes', 0) or 0),
            lecture_count=int(course_meta.get('lecture_count', 0) or 0),
            reading_count=int(course_meta.get('reading_count', 0) or 0),
            
            # Partner
            partner_name=course_meta.get('partner_names', ''),
            
            # Quality indicators
            difficulty_level=course_meta.get('course_difficulty_level', 'BEGINNER') or 'BEGINNER',
            star_rating=float(course_meta.get('course_star_rating', 0) or 0),
            
            # Temporal
            last_updated=last_updated,
            
            # Skills
            catalogue_skills=catalogue_skills,
            learning_objectives=course_meta.get('learning_objectives'),
        )
    
    def enrich_batch(
        self,
        content_items: List[Dict[str, Any]]
    ) -> List[OperationalMetadata]:
        """Enrich a batch of content items."""
        if not self._course_lookup:
            self.load()
        
        return [self.enrich_content_item(item) for item in content_items]
    
    def get_courses_by_domain(
        self,
        domain: str,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Get courses filtered by domain."""
        if not self._course_lookup:
            self.load()
        
        df = self._catalogue_df
        filtered = df[df['course_primary_domain'].str.contains(domain, case=False, na=False)]
        
        if limit:
            filtered = filtered.head(limit)
        
        return filtered.to_dict('records')
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the catalogue."""
        if not self._course_lookup:
            self.load()
        
        df = self._catalogue_df
        
        return {
            'total_courses': len(df),
            'domains': df['course_primary_domain'].value_counts().to_dict(),
            'difficulty_levels': df['course_difficulty_level'].value_counts().to_dict(),
            'total_lectures': int(df['lecture_count'].sum()),
            'total_readings': int(df['reading_count'].sum()),
            'avg_rating': float(df['course_star_rating'].mean()),
        }
