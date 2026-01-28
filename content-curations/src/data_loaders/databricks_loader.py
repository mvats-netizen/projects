"""
Databricks Data Loader

Connects to Databricks to fetch transcript and reading material data.

Tables used:
- prod.online_prep.foundation_data_gen_ai_course_video_subtitles (video transcripts)
- prod.online_prep.foundation_data_gen_ai_course_readings (reading materials)

Usage:
    >>> loader = DatabricksLoader()
    >>> 
    >>> # Fetch video transcripts
    >>> videos = loader.fetch_transcripts(limit=100, domain="Software Development")
    >>> 
    >>> # Fetch reading materials
    >>> readings = loader.fetch_reading_materials(limit=100, domain="Software Development")
    >>> 
    >>> # Get combined items for indexing
    >>> items = loader.get_items_for_indexing(limit=1000)
"""

import os
import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_databricks_config():
    """Get Databricks configuration from unified config."""
    try:
        from ..config import get_config
        return get_config()
    except ImportError:
        return None


class DatabricksLoader:
    """
    Loads transcript and content data from Databricks.
    
    Data Sources:
    - Video transcripts: prod.online_prep.foundation_data_gen_ai_course_video_subtitles
    - Reading materials: prod.online_prep.foundation_data_gen_ai_course_readings
    - Course metadata: prod.gold_base.courses, prod.gold.course_branches_vw
    
    Example:
        >>> loader = DatabricksLoader()
        >>> 
        >>> # Test connection
        >>> if loader.test_connection():
        ...     # Fetch Software Development courses in English
        ...     videos = loader.fetch_transcripts(
        ...         domain="Software Development",
        ...         language="en",
        ...         limit=100
        ...     )
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        token: Optional[str] = None,
        http_path: Optional[str] = None,
    ):
        """
        Initialize Databricks connection.
        
        Credentials are loaded from (in order of priority):
        1. Explicit constructor arguments
        2. config/secrets.env file
        3. Environment variables
        """
        cfg = get_databricks_config()
        
        self.host = host or (cfg.DATABRICKS_HOST if cfg else None) or os.getenv("DATABRICKS_HOST")
        self.token = token or (cfg.DATABRICKS_TOKEN if cfg else None) or os.getenv("DATABRICKS_TOKEN")
        self.http_path = http_path or (cfg.DATABRICKS_HTTP_PATH if cfg else None) or os.getenv("DATABRICKS_HTTP_PATH")
        
        self._connection = None
        
        if not self.host or not self.token:
            logger.warning(
                "Databricks credentials not provided. "
                "Set DATABRICKS_HOST and DATABRICKS_TOKEN in config/secrets.env"
            )
    
    def _get_connection(self):
        """Get or create Databricks SQL connection."""
        if self._connection is None:
            try:
                from databricks import sql
                
                self._connection = sql.connect(
                    server_hostname=self.host,
                    http_path=self.http_path,
                    access_token=self.token,
                )
                logger.info(f"Connected to Databricks: {self.host}")
            except ImportError:
                raise ImportError(
                    "databricks-sql-connector not installed. "
                    "Install with: pip install databricks-sql-connector"
                )
            except Exception as e:
                logger.error(f"Failed to connect to Databricks: {e}")
                raise
        
        return self._connection
    
    def test_connection(self) -> bool:
        """Test the Databricks connection."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            logger.info("✅ Databricks connection successful!")
            return result[0] == 1
        except Exception as e:
            logger.error(f"❌ Databricks connection failed: {e}")
            return False
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results as list of dicts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            results = [dict(zip(columns, row)) for row in rows]
            logger.info(f"Query returned {len(results)} rows")
            return results
        finally:
            cursor.close()
    
    def fetch_transcripts(
        self,
        domain: Optional[str] = None,
        language: str = "en",
        limit: Optional[int] = None,
        course_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch video transcripts from Databricks.
        
        Source: prod.online_prep.foundation_data_gen_ai_course_video_subtitles
        
        Args:
            domain: Filter by domain (e.g., "Software Development")
            language: Filter by language code (default: "en")
            limit: Maximum number of videos to fetch
            course_ids: Filter by specific course IDs
            
        Returns:
            List of video items with:
            - course_id, course_name
            - item_id, item_name
            - module_name, lesson_name
            - content_text (full transcript)
            - summary (atom_summary)
            - content_type ("video")
        """
        query = """
        SELECT 
            v.course_id,
            c.course_name,
            c.course_slug,
            d.course_url,
            v.course_branch_id,
            cb.course_branch_status,
            m.course_branch_module_order,
            m.course_branch_module_name as module_name,
            v.course_module_id,
            CASE 
                WHEN v.course_lesson_name = 'Lesson' OR v.course_lesson_name IS NULL 
                THEN v.atom_title 
                ELSE v.course_lesson_name 
            END as lesson_name,
            v.course_item_id as item_id,
            v.course_item_name as item_name,
            v.atom_summary as summary,
            v.subtitle_text as content_text,
            v.atom_title,
            'video' as content_type
        FROM prod.online_prep.foundation_data_gen_ai_course_video_subtitles v
        JOIN prod.gold_base.courses c
            ON v.course_id = c.course_id
        JOIN prod.gold.course_branches_vw cb
            ON v.course_id = cb.course_id
            AND v.course_branch_id = cb.course_branch_id
        JOIN prod.gold_base.course_branch_modules m
            ON v.course_branch_id = m.course_branch_id
            AND v.course_module_id = m.course_module_id
         JOIN coursera_warehouse.edw_bi.enterprise_catalog_course_metadata d
            ON v.course_id = d.course_id
        WHERE cb.course_branch_status = 'Live'
            AND v.subtitle_text IS NOT NULL
            AND LENGTH(v.subtitle_text) > 100
        """
        
        # Add filters
        if language:
            query += f" AND v.subtitle_language_cd = '{language}'"
        
        if domain:
            query += f" AND c.course_primary_domain = '{domain}'"
        
        if course_ids:
            ids_str = ", ".join(f"'{cid}'" for cid in course_ids)
            query += f" AND v.course_id IN ({ids_str})"
        
        query += " ORDER BY c.course_name, m.course_branch_module_order"
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = self.execute_query(query)
        
        # Clean up results
        for r in results:
            r['content_text'] = (r.get('content_text') or '').strip()
            r['summary'] = (r.get('summary') or '').strip()
        
        return results
    
    def fetch_reading_materials(
        self,
        domain: Optional[str] = None,
        language: str = "en",
        limit: Optional[int] = None,
        course_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch reading materials from Databricks.
        
        Source: prod.online_prep.foundation_data_gen_ai_course_readings
        
        Args:
            domain: Filter by domain (e.g., "Software Development")
            language: Filter by language code (default: "en")
            limit: Maximum number of readings to fetch
            course_ids: Filter by specific course IDs
            
        Returns:
            List of reading items with:
            - course_id, course_name
            - item_id, item_name
            - module_name, lesson_name
            - content_text (parsed from reading_content)
            - summary (atom_summary)
            - content_type ("reading")
        """
        query = """
        SELECT 
            r.course_id,
            c.course_name,
            c.course_slug,
            d.course_url,
            r.course_branch_id,
            cb.course_branch_status,
            m.course_branch_module_order,
            m.course_branch_module_name as module_name,
            r.course_module_id,
            CASE 
                WHEN r.course_lesson_name = 'Lesson' OR r.course_lesson_name IS NULL 
                THEN r.course_item_name 
                ELSE r.course_lesson_name 
            END as lesson_name,
            r.course_item_id as item_id,
            r.course_item_name as item_name,
            r.atom_summary as summary,
            r.reading_content as raw_content,
            'reading' as content_type
        FROM prod.online_prep.foundation_data_gen_ai_course_readings r
        JOIN prod.gold_base.courses c
            ON r.course_id = c.course_id
        JOIN prod.gold.course_branches_vw cb
            ON r.course_id = cb.course_id
            AND r.course_branch_id = cb.course_branch_id
        JOIN prod.gold_base.course_branch_modules m
            ON r.course_branch_id = m.course_branch_id
            AND r.course_module_id = m.course_module_id
        JOIN coursera_warehouse.edw_bi.enterprise_catalog_course_metadata d
            ON r.course_id = d.course_id
        WHERE cb.course_branch_status = 'Live'
            AND r.reading_content IS NOT NULL
            AND LENGTH(r.reading_content) > 100
        """
        
        if language:
            query += f" AND c.course_language_cd = '{language}'"
        
        if domain:
            query += f" AND c.course_primary_domain = '{domain}'"
        
        if course_ids:
            ids_str = ", ".join(f"'{cid}'" for cid in course_ids)
            query += f" AND r.course_id IN ({ids_str})"
        
        query += " ORDER BY c.course_name, m.course_branch_module_order"
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = self.execute_query(query)
        
        # Parse reading content (JSON/HTML) to extract text
        for r in results:
            raw_content = r.get('raw_content', '')
            r['content_text'] = self._parse_reading_content(raw_content)
            r['summary'] = (r.get('summary') or '').strip()
            # Remove raw_content to save memory
            del r['raw_content']
        
        return results
    
    def _parse_reading_content(self, content: str) -> str:
        """Parse JSON/HTML reading content to extract plain text."""
        if not content:
            return ''
        
        text = content
        
        # Try parsing as JSON
        try:
            if content.strip().startswith('{'):
                data = json.loads(content)
                text = data.get('value', data.get('text', content))
        except json.JSONDecodeError:
            pass
        
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text[:10000]  # Limit length
    
    def fetch_course_slugs(
        self,
        domain: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Fetch course slugs for filtering.
        
        Args:
            domain: Filter by domain
            limit: Maximum courses to return
            
        Returns:
            List of {course_id, course_slug, course_name}
        """
        query = """
        SELECT DISTINCT
            course_id,
            course_slug,
            course_name,
            course_primary_domain
        FROM prod.gold_base.courses
        WHERE course_slug IS NOT NULL
        """
        
        if domain:
            query += f" AND course_primary_domain = '{domain}'"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query)
    
    def get_items_for_indexing(
        self,
        domain: Optional[str] = "Computer Science",
        language: str = "en",
        limit: Optional[int] = None,
        include_readings: bool = True,
        course_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get combined video + reading items ready for indexing pipeline.
        
        Args:
            domain: Filter by domain (default: Software Development)
            language: Filter by language (default: English)
            limit: Maximum total items to fetch
            include_readings: Whether to include reading materials
            
        Returns:
            List of items with standardized fields for indexing:
            - course_id, course_name, course_slug
            - item_id, item_name
            - module_name, lesson_name
            - content_text, summary
            - content_type (video/reading)
            - atom_title
        """
        items = []
        
        # Calculate limits
        if limit and include_readings:
            video_limit = limit // 2
            reading_limit = limit - video_limit
        else:
            video_limit = limit
            reading_limit = limit if include_readings else 0
        
        # Fetch video transcripts
        logger.info(f"Fetching video transcripts (domain={domain}, language={language})...")
        try:
            videos = self.fetch_transcripts(
                domain=domain,
                language=language,
                limit=video_limit,
                course_ids=course_ids,
            )
            
            for v in videos:
                if v.get('content_text') and len(v['content_text']) > 50:
                    items.append({
                        'course_id': v.get('course_id'),
                        'course_name': v.get('course_name'),
                        'course_slug': v.get('course_slug'),
                        'course_url': v.get('course_url'),
                        'item_id': v.get('item_id'),
                        'item_name': v.get('item_name'),
                        'module_name': v.get('module_name'),
                        'lesson_name': v.get('lesson_name'),
                        'summary': v.get('summary'),
                        'content_text': v.get('content_text'),
                        'content_type': 'video',
                        'atom_title': v.get('atom_title'),
                    })
            
            logger.info(f"✅ Loaded {len(items)} video transcripts")
        except Exception as e:
            logger.error(f"❌ Error fetching transcripts: {e}")
        
        # Fetch reading materials
        if include_readings and reading_limit:
            logger.info(f"Fetching reading materials...")
            try:
                readings = self.fetch_reading_materials(
                    domain=domain,
                    language=language,
                    limit=reading_limit,
                    course_ids=course_ids,
                )
                
                reading_count = 0
                for r in readings:
                    if r.get('content_text') and len(r['content_text']) > 50:
                        items.append({
                            'course_id': r.get('course_id'),
                            'course_name': r.get('course_name'),
                            'course_slug': r.get('course_slug'),
                            'course_url': r.get('course_url'),
                            'item_id': r.get('item_id'),
                            'item_name': r.get('item_name'),
                            'module_name': r.get('module_name'),
                            'lesson_name': r.get('lesson_name'),
                            'summary': r.get('summary'),
                            'content_text': r.get('content_text'),
                            'content_type': 'reading',
                            'atom_title': r.get('item_name'),
                        })
                        reading_count += 1
                
                logger.info(f"✅ Loaded {reading_count} reading materials")
            except Exception as e:
                logger.error(f"❌ Error fetching reading materials: {e}")
        
        logger.info(f"📊 Total items for indexing: {len(items)}")
        return items
    
    def run_custom_query(self, query: str) -> List[Dict[str, Any]]:
        """Run a custom SQL query."""
        return self.execute_query(query)
    
    def close(self):
        """Close the Databricks connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Databricks connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
