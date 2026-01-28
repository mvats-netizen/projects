"""
Incremental Metadata Extraction Script

Extracts derived metadata from content using Gemini LLM.
Saves ALL extracted metadata persistently to avoid reprocessing.

Features:
- Persistent storage: All extractions saved to metadata store
- Incremental: Only processes new/unprocessed items
- Resume-capable: Can stop and restart without losing progress
- Course-aware: Tracks which courses have been fully processed

Usage:
    # Process all new items
    python scripts/extract_metadata.py
    
    # Process specific number of new items
    python scripts/extract_metadata.py --max-items 100
    
    # Process specific course only
    python scripts/extract_metadata.py --course-id "abc123"
    
    # Force reprocess a course (overwrite existing)
    python scripts/extract_metadata.py --course-id "abc123" --force
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Set
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.metadata.llm_extractor import LLMMetadataExtractor
from src.metadata.operational_loader import OperationalMetadataLoader
from src.metadata.schema import ContentMetadata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# METADATA STORE - Persistent storage for all extracted metadata
# =============================================================================

class MetadataStore:
    """
    Persistent storage for extracted metadata.
    
    Structure:
    - data/metadata_store/
        ├── index.json           # Master index of all processed items
        ├── courses/
        │   ├── {course_id}.json # Per-course metadata
        │   └── ...
        └── enriched_items.json  # Combined output for indexing
    """
    
    def __init__(self, store_dir: str = "data/metadata_store"):
        self.store_dir = Path(__file__).parent.parent / store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self.courses_dir = self.store_dir / "courses"
        self.courses_dir.mkdir(exist_ok=True)
        
        self.index_file = self.store_dir / "index.json"
        self.output_file = self.store_dir / "enriched_items.json"
        
        # Load index
        self._load_index()
    
    def _load_index(self):
        """Load the master index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                "created_at": datetime.now().isoformat(),
                "last_updated": None,
                "total_items": 0,
                "total_courses": 0,
                "processed_items": {},  # item_id -> {course_id, processed_at}
                "processed_courses": {},  # course_id -> {item_count, processed_at}
            }
    
    def _save_index(self):
        """Save the master index."""
        self.index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def is_item_processed(self, item_id: str) -> bool:
        """Check if an item has already been processed."""
        return item_id in self.index["processed_items"]
    
    def get_processed_item_ids(self) -> Set[str]:
        """Get all processed item IDs."""
        return set(self.index["processed_items"].keys())
    
    def get_processed_course_ids(self) -> Set[str]:
        """Get all processed course IDs."""
        return set(self.index["processed_courses"].keys())
    
    def save_item(self, item_id: str, course_id: str, enriched_data: Dict):
        """Save an enriched item to the store."""
        # Update index
        self.index["processed_items"][item_id] = {
            "course_id": course_id,
            "processed_at": datetime.now().isoformat(),
        }
        
        # Load or create course file
        course_file = self.courses_dir / f"{course_id}.json"
        if course_file.exists():
            with open(course_file, 'r') as f:
                course_data = json.load(f)
        else:
            course_data = {
                "course_id": course_id,
                "course_name": enriched_data.get("course_name", ""),
                "items": {},
                "created_at": datetime.now().isoformat(),
            }
        
        # Add/update item
        course_data["items"][item_id] = enriched_data
        course_data["last_updated"] = datetime.now().isoformat()
        
        # Save course file
        with open(course_file, 'w') as f:
            json.dump(course_data, f, indent=2)
        
        # Update course stats in index
        self.index["processed_courses"][course_id] = {
            "course_name": enriched_data.get("course_name", ""),
            "item_count": len(course_data["items"]),
            "last_updated": datetime.now().isoformat(),
        }
        
        self.index["total_items"] = len(self.index["processed_items"])
        self.index["total_courses"] = len(self.index["processed_courses"])
    
    def save_batch(self, items: List[Dict]):
        """Save a batch of items efficiently."""
        # Group by course
        by_course = {}
        for item in items:
            course_id = item.get("course_id", "unknown")
            if course_id not in by_course:
                by_course[course_id] = []
            by_course[course_id].append(item)
        
        # Save each course
        for course_id, course_items in by_course.items():
            course_file = self.courses_dir / f"{course_id}.json"
            
            if course_file.exists():
                with open(course_file, 'r') as f:
                    course_data = json.load(f)
            else:
                course_data = {
                    "course_id": course_id,
                    "course_name": course_items[0].get("course_name", "") if course_items else "",
                    "items": {},
                    "created_at": datetime.now().isoformat(),
                }
            
            for item in course_items:
                item_id = item.get("item_id")
                course_data["items"][item_id] = item
                
                self.index["processed_items"][item_id] = {
                    "course_id": course_id,
                    "processed_at": datetime.now().isoformat(),
                }
            
            course_data["last_updated"] = datetime.now().isoformat()
            
            with open(course_file, 'w') as f:
                json.dump(course_data, f, indent=2)
            
            self.index["processed_courses"][course_id] = {
                "course_name": course_data.get("course_name", ""),
                "item_count": len(course_data["items"]),
                "last_updated": datetime.now().isoformat(),
            }
        
        self.index["total_items"] = len(self.index["processed_items"])
        self.index["total_courses"] = len(self.index["processed_courses"])
        self._save_index()
    
    def export_all_items(self, output_file: Optional[str] = None) -> List[Dict]:
        """Export all enriched items to a single file for indexing."""
        all_items = []
        
        for course_file in self.courses_dir.glob("*.json"):
            with open(course_file, 'r') as f:
                course_data = json.load(f)
            
            for item_id, item_data in course_data.get("items", {}).items():
                all_items.append(item_data)
        
        # Sort by course name, then item name
        all_items.sort(key=lambda x: (x.get("course_name", ""), x.get("item_name", "")))
        
        # Save to output file
        output_path = Path(output_file) if output_file else self.output_file
        with open(output_path, 'w') as f:
            json.dump(all_items, f, indent=2)
        
        logger.info(f"Exported {len(all_items)} items to {output_path}")
        return all_items
    
    def get_stats(self) -> Dict:
        """Get statistics about the metadata store."""
        return {
            "total_items": self.index.get("total_items", 0),
            "total_courses": self.index.get("total_courses", 0),
            "last_updated": self.index.get("last_updated"),
            "courses": list(self.index.get("processed_courses", {}).keys()),
        }
    
    def delete_course(self, course_id: str):
        """Delete all metadata for a course (for reprocessing)."""
        course_file = self.courses_dir / f"{course_id}.json"
        
        if course_file.exists():
            # Load to get item IDs
            with open(course_file, 'r') as f:
                course_data = json.load(f)
            
            # Remove items from index
            for item_id in course_data.get("items", {}).keys():
                if item_id in self.index["processed_items"]:
                    del self.index["processed_items"][item_id]
            
            # Remove course from index
            if course_id in self.index["processed_courses"]:
                del self.index["processed_courses"][course_id]
            
            # Delete file
            course_file.unlink()
            
            self.index["total_items"] = len(self.index["processed_items"])
            self.index["total_courses"] = len(self.index["processed_courses"])
            self._save_index()
            
            logger.info(f"Deleted metadata for course {course_id}")


# =============================================================================
# EXTRACTION PIPELINE
# =============================================================================

def extract_metadata(
    input_file: str = "data/sample_courses_content.json",
    max_items: Optional[int] = None,
    course_id: Optional[str] = None,
    force: bool = False,
    rate_limit_delay: float = 0.5,
    save_every: int = 10,
):
    """
    Extract metadata for content items incrementally.
    
    Args:
        input_file: Path to content JSON file
        max_items: Limit number of NEW items to process (None = all)
        course_id: Only process specific course (None = all)
        force: Force reprocess even if already done
        rate_limit_delay: Seconds between API calls
        save_every: Save progress every N items
    """
    # Load config
    config = get_config()
    if not config.is_gemini_configured():
        logger.error("❌ Gemini API key not configured! Set GOOGLE_API_KEY in config/secrets.env")
        return None
    
    # Initialize metadata store
    store = MetadataStore()
    stats = store.get_stats()
    logger.info(f"📊 Metadata Store: {stats['total_items']} items from {stats['total_courses']} courses")
    
    # Delete course if force reprocessing
    if force and course_id:
        logger.info(f"🗑️ Force flag set - deleting existing metadata for course {course_id}")
        store.delete_course(course_id)
    
    # Load input data
    input_path = Path(__file__).parent.parent / input_file
    if not input_path.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        return None
    
    with open(input_path, 'r') as f:
        all_items = json.load(f)
    
    logger.info(f"📂 Loaded {len(all_items)} items from {input_file}")
    
    # Filter by course if specified
    if course_id:
        all_items = [item for item in all_items if item.get("course_id") == course_id]
        logger.info(f"  Filtered to {len(all_items)} items for course {course_id}")
    
    # Get already processed items
    processed_ids = store.get_processed_item_ids()
    
    # Filter to unprocessed items only
    items_to_process = [
        item for item in all_items 
        if item.get("item_id") not in processed_ids
    ]
    
    logger.info(f"  Already processed: {len(processed_ids)} items")
    logger.info(f"  New items to process: {len(items_to_process)}")
    
    if not items_to_process:
        logger.info("✅ All items already processed!")
        return store.export_all_items()
    
    # Apply max_items limit
    if max_items and len(items_to_process) > max_items:
        items_to_process = items_to_process[:max_items]
        logger.info(f"  Limited to {max_items} items")
    
    # Initialize LLM extractor and operational loader
    llm_extractor = LLMMetadataExtractor(
        provider="gemini",
        api_key=config.GOOGLE_API_KEY,
        model="gemini-2.0-flash"
    )
    op_loader = OperationalMetadataLoader()
    
    # Process items
    results = []
    start_time = time.time()
    errors = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 STARTING EXTRACTION ({len(items_to_process)} items)")
    logger.info(f"{'='*60}\n")
    
    for i, item in enumerate(items_to_process):
        item_id = item.get('item_id', f"item_{i}")
        item_name = item.get('item_name', 'Unknown')[:40]
        course_name = item.get('course_name', 'Unknown')[:30]
        
        logger.info(f"[{i+1}/{len(items_to_process)}] {item_name}... ({course_name})")
        
        try:
            # Get operational metadata
            op_meta = op_loader.enrich_content_item(item)
            
            # Get content text
            content_text = item.get('content_text', '')
            if len(content_text) < 50:
                logger.warning(f"  ⚠️ Skipping - content too short ({len(content_text)} chars)")
                continue
            
            # Extract derived metadata using LLM
            derived = llm_extractor.extract_all(
                transcript=content_text[:4000],  # First 4000 chars
                course_name=op_meta.course_name,
                module_name=op_meta.module_name,
                lesson_name=item.get('lesson_name', ''),
                content_type=item.get('content_type', 'video'),
                chunk_id=item_id,
            )
            
            # Combine metadata
            combined = ContentMetadata(
                id=item_id,
                operational=op_meta,
                derived=derived,
            )
            
            # Build enriched result
            enriched_item = {
                **item,
                "operational_metadata": op_meta.to_dict(),
                "derived_metadata": derived.to_dict(),
                "embedding_input": combined.get_embedding_input()[:500],
                "filter_metadata": combined.get_filter_metadata(),
                "extracted_at": datetime.now().isoformat(),
            }
            
            results.append(enriched_item)
            
            # Save to store periodically
            if len(results) >= save_every:
                store.save_batch(results)
                logger.info(f"  💾 Saved {len(results)} items to store")
                results = []
            
            # Rate limiting
            time.sleep(rate_limit_delay)
            
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            errors.append({"item_id": item_id, "error": str(e)})
            time.sleep(2)  # Wait longer on error
            continue
    
    # Save remaining items
    if results:
        store.save_batch(results)
        logger.info(f"💾 Saved final {len(results)} items")
    
    # Export all items for indexing
    all_enriched = store.export_all_items("data/sample_courses_enriched.json")
    
    # Summary
    elapsed = time.time() - start_time
    final_stats = store.get_stats()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  New items processed: {len(items_to_process) - len(errors)}")
    logger.info(f"  Errors: {len(errors)}")
    logger.info(f"  Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"  Total in store: {final_stats['total_items']} items from {final_stats['total_courses']} courses")
    logger.info(f"  Output: data/sample_courses_enriched.json")
    logger.info(f"{'='*60}\n")
    
    if errors:
        logger.warning(f"⚠️ Failed items: {[e['item_id'] for e in errors]}")
    
    return all_enriched


def show_stats():
    """Show metadata store statistics."""
    store = MetadataStore()
    stats = store.get_stats()
    
    print(f"\n{'='*60}")
    print(f"📊 METADATA STORE STATISTICS")
    print(f"{'='*60}")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Total courses: {stats['total_courses']}")
    print(f"  Last updated: {stats['last_updated']}")
    print(f"\n  Courses processed:")
    
    for course_id in stats['courses'][:20]:  # Show first 20
        course_info = store.index["processed_courses"].get(course_id, {})
        print(f"    - {course_id[:20]}... ({course_info.get('item_count', 0)} items)")
    
    if len(stats['courses']) > 20:
        print(f"    ... and {len(stats['courses']) - 20} more")
    
    print(f"{'='*60}\n")


def export_enriched():
    """Export all enriched items to a single file."""
    store = MetadataStore()
    items = store.export_all_items("data/sample_courses_enriched.json")
    print(f"✅ Exported {len(items)} items to data/sample_courses_enriched.json")


def migrate_existing():
    """Migrate existing enriched data to the new metadata store."""
    enriched_file = Path(__file__).parent.parent / "data/sample_courses_enriched.json"
    
    if not enriched_file.exists():
        print("❌ No existing enriched data to migrate")
        return
    
    with open(enriched_file, 'r') as f:
        items = json.load(f)
    
    if not items:
        print("❌ Enriched file is empty")
        return
    
    print(f"📂 Found {len(items)} existing enriched items")
    
    store = MetadataStore()
    existing_ids = store.get_processed_item_ids()
    
    # Filter to items not already in store
    new_items = [item for item in items if item.get("item_id") not in existing_ids]
    
    if not new_items:
        print("✅ All items already in metadata store")
        return
    
    print(f"  Migrating {len(new_items)} new items...")
    
    store.save_batch(new_items)
    
    stats = store.get_stats()
    print(f"✅ Migration complete!")
    print(f"  Total in store: {stats['total_items']} items from {stats['total_courses']} courses")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract metadata from content items (incremental)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all new items
  python scripts/extract_metadata.py
  
  # Process up to 100 new items
  python scripts/extract_metadata.py --max-items 100
  
  # Process specific course
  python scripts/extract_metadata.py --course-id "-k4wp39DEfCzbgr_1h6UfQ"
  
  # Force reprocess a course
  python scripts/extract_metadata.py --course-id "-k4wp39DEfCzbgr_1h6UfQ" --force
  
  # Show statistics
  python scripts/extract_metadata.py --stats
  
  # Export all enriched items
  python scripts/extract_metadata.py --export
  
  # Migrate existing enriched data to new store format
  python scripts/extract_metadata.py --migrate
        """
    )
    
    parser.add_argument("--max-items", type=int, default=None, 
                        help="Max NEW items to process (default: all)")
    parser.add_argument("--course-id", type=str, default=None,
                        help="Only process specific course")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocess (delete existing metadata first)")
    parser.add_argument("--delay", type=float, default=0.5, 
                        help="Delay between API calls (default: 0.5s)")
    parser.add_argument("--stats", action="store_true",
                        help="Show metadata store statistics")
    parser.add_argument("--export", action="store_true",
                        help="Export all enriched items to JSON")
    parser.add_argument("--migrate", action="store_true",
                        help="Migrate existing enriched data to metadata store")
    
    args = parser.parse_args()
    
    if args.stats:
        show_stats()
    elif args.export:
        export_enriched()
    elif args.migrate:
        migrate_existing()
    else:
        extract_metadata(
            max_items=args.max_items,
            course_id=args.course_id,
            force=args.force,
            rate_limit_delay=args.delay,
        )
