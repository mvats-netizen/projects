#!/usr/bin/env python3
"""
Build Curated Test Index

Creates a high-quality test index from top-rated, popular courses.
Perfect for testing without waiting days for full domain builds.

Selection Criteria:
- Top N courses by enrollment/rating
- Minimum quality threshold (4.5+ stars)
- Recent content (updated in last 2 years)
- Representative of domain

Usage:
    # Build index from top 100 Data Science courses
    python scripts/build_curated_index.py \
      --domain "Data Science" \
      --top-courses 100
    
    # Custom criteria
    python scripts/build_curated_index.py \
      --domain "Computer Science" \
      --top-courses 50 \
      --min-rating 4.7 \
      --min-enrollments 50000
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loaders.databricks_loader import DatabricksLoader
from scripts.build_domain_index import DomainIndexBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_top_courses(
    domain: str,
    top_n: int = 100,
    min_rating: float = 4.5,
    min_enrollments: int = 1000,
    max_age_years: int = 3,
) -> list:
    """
    Query Databricks for top courses based on quality criteria.
    
    Args:
        domain: Course domain (e.g., "Data Science")
        top_n: Number of top courses to select
        min_rating: Minimum star rating
        min_enrollments: Minimum number of ratings/enrollments
        max_age_years: Maximum age of course (years)
        
    Returns:
        List of course IDs meeting criteria
    """
    logger.info(f"Querying top {top_n} courses in {domain}...")
    logger.info(f"  Criteria: rating >= {min_rating}, enrollments >= {min_enrollments}")
    
    loader = DatabricksLoader()
    
    if not loader.test_connection():
        raise ConnectionError("Failed to connect to Databricks!")
    
    # Calculate date threshold
    cutoff_date = (datetime.now() - timedelta(days=max_age_years * 365)).strftime("%Y-%m-%d")
    
    query = f"""
    SELECT 
        course_id,
        course_name,
        course_slug,
        course_star_rating as rating,
        course_star_ratings_count as enrollments,
        course_primary_domain as domain,
        course_update_ts as last_updated
    FROM prod.gold_base.courses
    WHERE course_primary_domain = '{domain}'
        AND course_star_rating >= {min_rating}
        AND course_star_ratings_count >= {min_enrollments}
        AND course_update_ts >= '{cutoff_date}'
        AND course_slug IS NOT NULL
    ORDER BY 
        course_star_ratings_count DESC,
        course_star_rating DESC
    LIMIT {top_n * 2}
    """
    
    results = loader.execute_query(query)
    loader.close()
    
    # Take top N by a composite score (rating * log(enrollments))
    import math
    for r in results:
        rating = float(r.get('rating', 0))
        enrollments = int(r.get('enrollments', 1))
        r['score'] = rating * math.log10(enrollments + 1)
    
    results.sort(key=lambda x: x['score'], reverse=True)
    top_courses = results[:top_n]
    
    logger.info(f"✅ Selected {len(top_courses)} courses:")
    for i, course in enumerate(top_courses[:10], 1):
        logger.info(
            f"  {i}. {course['course_name'][:50]:<50} "
            f"({course['rating']:.1f}★, {course['enrollments']:,} ratings)"
        )
    if len(top_courses) > 10:
        logger.info(f"  ... and {len(top_courses) - 10} more")
    
    return [c['course_id'] for c in top_courses]


def build_curated_index(
    domain: str,
    top_n: int = 100,
    min_rating: float = 4.5,
    min_enrollments: int = 1000,
    max_age_years: int = 3,
    output_dir: str = None,
):
    """
    Build a curated test index from top courses.
    
    This is much faster than building a full domain index:
    - 100 courses → ~5K items
    - Metadata extraction: ~6-8 hours
    - Total: < 1 day vs 7+ days for full domain
    """
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("  BUILD CURATED TEST INDEX")
    logger.info("=" * 70)
    logger.info(f"  Domain: {domain}")
    logger.info(f"  Top courses: {top_n}")
    logger.info(f"  Quality threshold: {min_rating}+ stars, {min_enrollments}+ ratings")
    logger.info("=" * 70 + "\n")
    
    # Step 1: Get top course IDs
    course_ids = get_top_courses(
        domain=domain,
        top_n=top_n,
        min_rating=min_rating,
        min_enrollments=min_enrollments,
        max_age_years=max_age_years,
    )
    
    if not course_ids:
        logger.error("❌ No courses found matching criteria!")
        return
    
    # Step 2: Fetch content for these specific courses
    logger.info(f"\nFetching content for {len(course_ids)} courses...")
    
    loader = DatabricksLoader()
    if not loader.test_connection():
        raise ConnectionError("Failed to connect to Databricks!")
    
    items = []
    
    # Fetch in batches
    batch_size = 50
    for i in range(0, len(course_ids), batch_size):
        batch = course_ids[i:i+batch_size]
        logger.info(f"  Fetching batch {i//batch_size + 1}/{(len(course_ids) + batch_size - 1)//batch_size}...")
        
        try:
            batch_items = loader.get_items_for_indexing(
                domain=domain,
                course_ids=batch,
                include_readings=True,
            )
            items.extend(batch_items)
        except Exception as e:
            logger.warning(f"  Error fetching batch: {e}")
            continue
    
    loader.close()
    
    logger.info(f"✅ Fetched {len(items)} items from {len(course_ids)} courses")
    
    video_count = sum(1 for i in items if i.get('content_type') == 'video')
    reading_count = sum(1 for i in items if i.get('content_type') == 'reading')
    logger.info(f"   Videos: {video_count}, Readings: {reading_count}")
    
    # Step 3: Use DomainIndexBuilder for the rest
    if not output_dir:
        output_dir = f"data/test_indexes/{domain.lower().replace(' ', '_')}_top{top_n}"
    
    builder = DomainIndexBuilder(
        domain=domain,
        output_dir=output_dir,
    )
    
    # Save the fetched items
    with open(builder.content_file, 'w') as f:
        json.dump(items, f, indent=2)
    
    logger.info(f"💾 Saved content → {builder.content_file}")
    
    # Extract metadata
    enriched = builder.stage2_extract_metadata(
        items=items,
        max_items=None,  # Process all
    )
    
    # Build index
    builder.stage3_build_index(enriched)
    
    # Save metadata
    elapsed = datetime.now() - start_time
    metadata = {
        "domain": domain,
        "index_type": "curated_test",
        "top_courses": top_n,
        "min_rating": min_rating,
        "min_enrollments": min_enrollments,
        "total_courses": len(course_ids),
        "total_items": len(items),
        "enriched_items": len(enriched),
        "videos": video_count,
        "readings": reading_count,
        "build_time_minutes": elapsed.total_seconds() / 60,
        "created_at": start_time.isoformat(),
    }
    
    with open(builder.metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("  ✅ CURATED INDEX BUILD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Domain: {domain}")
    logger.info(f"  Courses: {len(course_ids)} (top {top_n})")
    logger.info(f"  Items: {len(items)} ({video_count} videos, {reading_count} readings)")
    logger.info(f"  Enriched: {len(enriched)}")
    logger.info(f"  Build time: {elapsed.total_seconds() / 3600:.1f} hours")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 70 + "\n")
    
    logger.info("🚀 Ready for testing!")
    logger.info(f"   Load index: {builder.index_dir}")
    logger.info(f"   Streamlit: streamlit run app.py -- --index-dir {builder.index_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build curated test index from top courses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Top 100 Data Science courses (recommended for testing)
  python scripts/build_curated_index.py --domain "Data Science" --top-courses 100
  
  # Top 50 high-quality Computer Science courses
  python scripts/build_curated_index.py \
    --domain "Computer Science" \
    --top-courses 50 \
    --min-rating 4.7 \
    --min-enrollments 50000
  
  # Quick test with top 25 courses
  python scripts/build_curated_index.py --domain "Business" --top-courses 25
        """
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain to build index for (e.g., 'Data Science')"
    )
    parser.add_argument(
        "--top-courses",
        type=int,
        default=100,
        help="Number of top courses to include (default: 100)"
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=4.5,
        help="Minimum course rating (default: 4.5)"
    )
    parser.add_argument(
        "--min-enrollments",
        type=int,
        default=1000,
        help="Minimum number of ratings/enrollments (default: 1000)"
    )
    parser.add_argument(
        "--max-age-years",
        type=int,
        default=3,
        help="Maximum course age in years (default: 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: data/test_indexes/{domain}_top{N})"
    )
    
    args = parser.parse_args()
    
    build_curated_index(
        domain=args.domain,
        top_n=args.top_courses,
        min_rating=args.min_rating,
        min_enrollments=args.min_enrollments,
        max_age_years=args.max_age_years,
        output_dir=args.output_dir,
    )
