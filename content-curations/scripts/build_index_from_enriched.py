#!/usr/bin/env python3
"""
Build FAISS Index from Enriched Data

Takes enriched items (with metadata) and builds a searchable FAISS index.
Useful for rebuilding indexes or building from pre-enriched data.

Usage:
    python scripts/build_index_from_enriched.py \
      --enriched data/domain_indexes/data_science_test/enriched.json \
      --output data/domain_indexes/data_science_test/index
"""

import json
import logging
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_domain_index import DomainIndexBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_index_from_enriched(enriched_file: str, output_dir: str):
    """Build FAISS index from pre-enriched data."""
    
    enriched_path = Path(enriched_file)
    if not enriched_path.exists():
        raise FileNotFoundError(f"Enriched file not found: {enriched_file}")
    
    # Load enriched items
    logger.info(f"Loading enriched items from {enriched_file}...")
    with open(enriched_path, 'r') as f:
        enriched_items = json.load(f)
    
    logger.info(f"✅ Loaded {len(enriched_items)} enriched items")
    
    # Create a dummy builder just to use its stage3 method
    builder = DomainIndexBuilder(
        domain="Test",
        output_dir=output_dir,
    )
    
    # Build index
    logger.info("\nBuilding FAISS index...")
    index, chunks, embeddings, model = builder.stage3_build_index(enriched_items)
    
    logger.info(f"\n✅ Index built successfully!")
    logger.info(f"   Location: {builder.index_dir}")
    logger.info(f"   Chunks: {len(chunks)}")
    logger.info(f"   Dimensions: {embeddings.shape[1]}")
    
    return builder.index_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from enriched data")
    parser.add_argument("--enriched", required=True, help="Path to enriched.json file")
    parser.add_argument("--output", required=True, help="Output directory for index")
    
    args = parser.parse_args()
    
    build_index_from_enriched(args.enriched, args.output)
