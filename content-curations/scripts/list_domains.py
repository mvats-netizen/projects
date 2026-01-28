#!/usr/bin/env python3
"""
List Available Domains

Extracts all available domains from prod.gold_base.courses and saves them
for reference when building domain indexes.

Usage:
    python scripts/list_domains.py
    
    # Save to custom location
    python scripts/list_domains.py --output data/available_domains.json
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loaders.databricks_loader import DatabricksLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_domains(output_file: str = "data/available_domains.json"):
    """
    Query all available domains from Databricks and save to file.
    
    Args:
        output_file: Path to save the domain list
    """
    logger.info("=" * 70)
    logger.info("EXTRACTING AVAILABLE DOMAINS FROM DATABRICKS")
    logger.info("=" * 70)
    
    # Connect to Databricks
    loader = DatabricksLoader()
    
    if not loader.test_connection():
        raise ConnectionError("Failed to connect to Databricks!")
    
    # Query domains with statistics
    logger.info("\nQuerying domains from prod.gold_base.courses...")
    
    query = """
    SELECT 
        course_primary_domain as domain,
        COUNT(DISTINCT course_id) as course_count,
        COUNT(DISTINCT CASE WHEN course_launch_ts IS NOT NULL THEN course_id END) as launched_courses,
        AVG(course_star_rating) as avg_rating,
        SUM(CASE WHEN course_language_cd = 'en' THEN 1 ELSE 0 END) as english_courses
    FROM prod.gold_base.courses
    WHERE course_primary_domain IS NOT NULL
        AND course_slug IS NOT NULL
    GROUP BY course_primary_domain
    ORDER BY course_count DESC
    """
    
    results = loader.execute_query(query)
    loader.close()
    
    # Format results
    domains = []
    for r in results:
        domain_info = {
            "domain": r.get('domain'),
            "course_count": int(r.get('course_count', 0)),
            "launched_courses": int(r.get('launched_courses', 0)),
            "avg_rating": float(r.get('avg_rating', 0)) if r.get('avg_rating') else 0,
            "english_courses": int(r.get('english_courses', 0)),
        }
        domains.append(domain_info)
    
    # Save to file
    output_path = PROJECT_ROOT / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "extracted_at": datetime.now().isoformat(),
        "total_domains": len(domains),
        "domains": domains,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Display summary
    logger.info(f"\n✅ Found {len(domains)} domains")
    logger.info(f"   Saved to: {output_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TOP DOMAINS BY COURSE COUNT")
    logger.info("=" * 70)
    
    for i, domain in enumerate(domains[:20], 1):
        logger.info(
            f"{i:2d}. {domain['domain']:<35} "
            f"Courses: {domain['course_count']:>5} "
            f"(EN: {domain['english_courses']:>5}) "
            f"Rating: {domain['avg_rating']:.2f}"
        )
    
    if len(domains) > 20:
        logger.info(f"\n... and {len(domains) - 20} more domains")
    
    logger.info("\n" + "=" * 70)
    logger.info("DOMAIN RECOMMENDATIONS")
    logger.info("=" * 70)
    
    # Recommend top domains for testing
    recommended = [d for d in domains if d['english_courses'] >= 100][:10]
    
    logger.info("\nRecommended domains for testing (100+ English courses):\n")
    for i, domain in enumerate(recommended, 1):
        logger.info(
            f"  {i}. {domain['domain']:<35} "
            f"({domain['english_courses']} EN courses, "
            f"{domain['avg_rating']:.2f}★)"
        )
    
    logger.info("\n" + "=" * 70)
    logger.info("BUILD DOMAIN INDEX COMMANDS")
    logger.info("=" * 70)
    logger.info("\nTo build an index for a domain, run:\n")
    
    for domain in recommended[:5]:
        logger.info(f'  python scripts/build_domain_index.py --domain "{domain["domain"]}"')
    
    logger.info("\n" + "=" * 70)
    
    return domains


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List all available domains from Databricks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/available_domains.json",
        help="Output file path (default: data/available_domains.json)"
    )
    
    args = parser.parse_args()
    
    list_domains(output_file=args.output)
