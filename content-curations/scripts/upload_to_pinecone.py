#!/usr/bin/env python3
"""
Upload Existing Embeddings to Pinecone

Reads the existing FAISS index embeddings and chunk metadata,
then uploads them to Pinecone for V2 exploration.

Usage:
    python scripts/upload_to_pinecone.py
    python scripts/upload_to_pinecone.py --index-dir data/test_indexes/diverse_50/index
    python scripts/upload_to_pinecone.py --delete-first  # Clear index before upload
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "config" / "secrets.env")

from src.vector_store.pinecone_store import PineconeStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_existing_index(index_dir: str) -> tuple:
    """
    Load embeddings and chunks from existing FAISS index.
    
    Returns:
        (embeddings, chunks, config)
    """
    index_path = Path(index_dir)
    
    # Load embeddings
    embeddings_file = index_path / "embeddings.npy"
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_file}")
    
    embeddings = np.load(embeddings_file)
    logger.info(f"Loaded embeddings: {embeddings.shape}")
    
    # Load chunk metadata
    chunks_file = index_path / "chunks.json"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks not found: {chunks_file}")
    
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    logger.info(f"Loaded chunks: {len(chunks)}")
    
    # Load config
    config_file = index_path / "config.json"
    config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config: {config}")
    
    return embeddings, chunks, config


def _safe_float(value, default: float = 0.0) -> float:
    """Convert value to float, handling NaN and None."""
    import math
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_str(value, default: str = "") -> str:
    """Convert value to string, handling NaN and None."""
    import math
    if value is None:
        return default
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return default
    # Check for string "NaN", "nan", "None", "null"
    str_val = str(value)
    if str_val.lower() in ("nan", "none", "null", ""):
        return default
    return str_val


def _sanitize_metadata(metadata: dict) -> dict:
    """Remove or replace NaN/None/invalid values from metadata dict."""
    import math
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            continue  # Skip None values
        elif isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                continue  # Skip NaN/inf
            sanitized[key] = value
        elif isinstance(value, str):
            if value.lower() in ("nan", "none", "null"):
                continue  # Skip NaN strings
            if value:  # Only include non-empty strings
                sanitized[key] = value
        elif isinstance(value, list):
            # Filter list items
            sanitized[key] = [v for v in value if v and str(v).lower() not in ("nan", "none", "null")]
        else:
            sanitized[key] = value
    return sanitized


def prepare_metadata(chunk: dict) -> dict:
    """
    Prepare metadata for Pinecone storage.
    
    Pinecone metadata supports: str, int, float, bool, list of str
    We extract the most useful fields for filtering.
    """
    derived = chunk.get("derived_metadata", {})
    operational = chunk.get("operational_metadata", {})
    
    metadata = {
        # Identifiers
        "chunk_id": chunk.get("chunk_id", ""),
        "item_id": chunk.get("item_id", ""),
        "course_id": chunk.get("course_id", ""),
        
        # Course info
        "course_name": chunk.get("course_name", "")[:200],  # Truncate for Pinecone limits
        "course_slug": chunk.get("course_slug", ""),
        "item_name": chunk.get("item_name", "")[:200],
        "module_name": chunk.get("module_name", "")[:200],
        "content_type": chunk.get("content_type", "video"),
        
        # Derived metadata (for filtering)
        "primary_domain": derived.get("primary_domain", ""),
        "sub_domain": derived.get("sub_domain", ""),
        "bloom_level": derived.get("bloom_level", ""),
        "cognitive_load": derived.get("cognitive_load", ""),
        
        # Operational metadata
        "difficulty_level": operational.get("difficulty_level", ""),
        "star_rating": _safe_float(operational.get("star_rating", 0)),
        
        # Text preview (for display, truncated)
        "text_preview": (chunk.get("chunk_text", "") or chunk.get("text", ""))[:500],
    }
    
    # Add skills as list (Pinecone supports list of strings)
    atomic_skills = derived.get("atomic_skills", [])
    if atomic_skills:
        metadata["atomic_skills"] = atomic_skills[:10]  # Limit to 10 skills
    
    key_concepts = derived.get("key_concepts", [])
    if key_concepts:
        metadata["key_concepts"] = key_concepts[:10]
    
    # Sanitize all metadata to remove NaN/None values
    return _sanitize_metadata(metadata)


def upload_to_pinecone(
    index_dir: str,
    pinecone_index_name: str = "content-curations-v2",
    delete_first: bool = False,
    batch_size: int = 100,
):
    """
    Upload existing embeddings to Pinecone.
    """
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("  UPLOAD TO PINECONE")
    logger.info("=" * 70)
    logger.info(f"  Source: {index_dir}")
    logger.info(f"  Target: {pinecone_index_name}")
    logger.info("=" * 70 + "\n")
    
    # Load existing data
    logger.info("Loading existing index data...")
    embeddings, chunks, config = load_existing_index(index_dir)
    
    # Verify dimensions match
    dimension = embeddings.shape[1]
    logger.info(f"Embedding dimension: {dimension}")
    
    # Initialize Pinecone
    logger.info("\nInitializing Pinecone...")
    store = PineconeStore(
        index_name=pinecone_index_name,
        dimension=dimension,
    )
    
    # Optionally clear existing data
    if delete_first:
        logger.info("Clearing existing vectors...")
        store.delete_all()
    
    # Prepare data for upload
    logger.info("\nPreparing data for upload...")
    ids = []
    metadata_list = []
    
    for i, chunk in enumerate(chunks):
        # Use chunk_id as vector ID
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        ids.append(chunk_id)
        
        # Prepare metadata
        metadata = prepare_metadata(chunk)
        metadata_list.append(metadata)
    
    logger.info(f"Prepared {len(ids)} vectors with metadata")
    
    # Upload to Pinecone
    logger.info("\nUploading to Pinecone...")
    total_upserted = store.upsert_vectors(
        ids=ids,
        embeddings=embeddings,
        metadata_list=metadata_list,
        batch_size=batch_size,
    )
    
    # Verify upload
    logger.info("\nVerifying upload...")
    stats = store.get_stats()
    
    elapsed = datetime.now() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("  ✅ UPLOAD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Vectors uploaded: {total_upserted}")
    logger.info(f"  Total in index: {stats['total_vectors']}")
    logger.info(f"  Dimension: {stats['dimension']}")
    logger.info(f"  Time elapsed: {elapsed.total_seconds():.1f} seconds")
    logger.info("=" * 70 + "\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Upload existing embeddings to Pinecone"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/test_indexes/diverse_50/index",
        help="Directory containing FAISS index (default: data/test_indexes/diverse_50/index)"
    )
    parser.add_argument(
        "--pinecone-index",
        type=str,
        default="content-curations-v2",
        help="Pinecone index name (default: content-curations-v2)"
    )
    parser.add_argument(
        "--delete-first",
        action="store_true",
        help="Delete existing vectors before upload"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for upsert (default: 100)"
    )
    
    args = parser.parse_args()
    
    upload_to_pinecone(
        index_dir=args.index_dir,
        pinecone_index_name=args.pinecone_index,
        delete_first=args.delete_first,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
