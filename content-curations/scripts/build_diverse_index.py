#!/usr/bin/env python3
"""
Build Diverse Test Index

Creates a high-quality test index from top-rated courses across multiple domains.
Domains selected: Data Science, Computer Science, Business, Health, Personal Development.

Usage:
    python scripts/build_diverse_index.py --courses-per-domain 10
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loaders.databricks_loader import DatabricksLoader
from scripts.build_domain_index import DomainIndexBuilder
from scripts.build_curated_index import get_top_courses

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DOMAINS = [
    "Data Science",
    "Computer Science",
    "Business",
    "Health",
    "Personal Development"
]

def build_diverse_index(
    courses_per_domain: int = 10,
    output_dir: str = "data/test_indexes/diverse_50"
):
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("  BUILD DIVERSE TEST INDEX")
    logger.info("=" * 70)
    logger.info(f"  Domains: {', '.join(DOMAINS)}")
    logger.info(f"  Courses per domain: {courses_per_domain}")
    logger.info(f"  Target total courses: {len(DOMAINS) * courses_per_domain}")
    logger.info("=" * 70 + "\n")
    
    # Initialize builder to get paths
    builder = DomainIndexBuilder(
        domain="Diverse",
        output_dir=output_dir,
    )
    
    all_items = []
    
    # Check if content already exists to skip fetch
    if builder.content_file.exists():
        logger.info(f"✅ Found existing content file: {builder.content_file}")
        with open(builder.content_file, 'r') as f:
            all_items = json.load(f)
        logger.info(f"   Loaded {len(all_items)} items. Skipping Databricks fetch.")
    else:
        all_course_ids = []
        
        # Step 1: Get top course IDs for each domain
        for domain in DOMAINS:
            try:
                domain_course_ids = get_top_courses(
                    domain=domain,
                    top_n=courses_per_domain,
                    min_rating=4.5,
                    min_enrollments=500,  # Slightly lower to ensure we get 10 even for smaller domains
                )
                all_course_ids.extend(domain_course_ids)
                logger.info(f"✅ Selected {len(domain_course_ids)} courses from {domain}")
            except Exception as e:
                logger.error(f"❌ Error getting courses for {domain}: {e}")
                continue
                
        if not all_course_ids:
            logger.error("❌ No courses found for any domain!")
            return
            
        logger.info(f"\nTotal unique courses selected: {len(set(all_course_ids))}")
        
        # Step 2: Fetch content for these specific courses
        logger.info(f"\nFetching content for {len(all_course_ids)} courses...")
        
        loader = DatabricksLoader()
        if not loader.test_connection():
            raise ConnectionError("Failed to connect to Databricks!")
        
        # Fetch in batches of 50
        batch_size = 50
        unique_course_ids = list(set(all_course_ids))
        for i in range(0, len(unique_course_ids), batch_size):
            batch = unique_course_ids[i:i+batch_size]
            logger.info(f"  Fetching batch {i//batch_size + 1}/{(len(unique_course_ids) + batch_size - 1)//batch_size}...")
            
            try:
                batch_items = loader.get_items_for_indexing(
                    domain=None,
                    course_ids=batch,
                    include_readings=True,
                )
                all_items.extend(batch_items)
            except Exception as e:
                logger.warning(f"  Error fetching batch: {e}")
                continue
        
        loader.close()
        logger.info(f"✅ Fetched {len(all_items)} total items")
        
        # Save the fetched items
        with open(builder.content_file, 'w') as f:
            json.dump(all_items, f, indent=2)
        logger.info(f"💾 Saved content → {builder.content_file}")

    # Extract metadata (Stage 2)
    logger.info("\nExtracting metadata for items...")
    enriched = builder.stage2_extract_metadata(
        items=all_items,
        max_items=None,
    )
    
    # Build index (Stage 3)
    logger.info("\nBuilding FAISS index...")
    builder.stage3_build_index(enriched)
    
    # Final summary
    elapsed = datetime.now() - start_time
    unique_courses = set(item.get('course_id') for item in all_items if item.get('course_id'))
    metadata = {
        "domain": "Diverse",
        "index_type": "diverse_test",
        "domains_included": DOMAINS,
        "courses_per_domain": courses_per_domain,
        "total_courses": len(unique_courses),
        "total_items": len(all_items),
        "enriched_items": len(enriched),
        "build_time_minutes": elapsed.total_seconds() / 60,
        "created_at": start_time.isoformat(),
    }
    
    with open(builder.metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    logger.info("\n" + "=" * 70)
    logger.info("  ✅ DIVERSE INDEX BUILD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Courses: {len(unique_courses)}")
    logger.info(f"  Items: {len(all_items)}")
    logger.info(f"  Build time: {elapsed.total_seconds() / 60:.1f} minutes")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 70 + "\n")
    
    print(f"To run with this index: streamlit run app.py -- --index-dir {output_dir}/index")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build diverse test index")
    parser.add_argument("--courses-per-domain", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="data/test_indexes/diverse_50")
    
    args = parser.parse_args()
    
    build_diverse_index(
        courses_per_domain=args.courses_per_domain,
        output_dir=args.output_dir
    )
