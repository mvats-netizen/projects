#!/usr/bin/env python3
"""
End-to-End Pipeline Runner

Runs the complete content curation pipeline from scratch:
1. Fetch data from Databricks (videos + readings)
2. Extract metadata using LLM (Gemini)
3. Build vector index (FAISS)
4. Run test search queries

Usage:
    # Full pipeline with 50 items
    python scripts/run_pipeline.py --items 50
    
    # Skip data fetch (use existing)
    python scripts/run_pipeline.py --skip-fetch
    
    # Only fetch data
    python scripts/run_pipeline.py --only-fetch --items 100
    
    # Only build index
    python scripts/run_pipeline.py --only-index
"""

import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output paths
DATA_DIR = PROJECT_ROOT / "data"
CONTENT_FILE = DATA_DIR / "pipeline_content.json"
ENRICHED_FILE = DATA_DIR / "pipeline_enriched.json"
INDEX_DIR = DATA_DIR / "pipeline_index"


# =============================================================================
# STAGE 1: DATA INGESTION FROM DATABRICKS
# =============================================================================

def fetch_from_databricks(
    domain: str = "Computer Science",
    language: str = "en",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Fetch video transcripts and reading materials from Databricks.
    
    Source tables:
    - prod.online_prep.foundation_data_gen_ai_course_video_subtitles
    - prod.online_prep.foundation_data_gen_ai_course_readings
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA INGESTION FROM DATABRICKS")
    logger.info("=" * 60)
    
    from src.data_loaders.databricks_loader import DatabricksLoader
    
    loader = DatabricksLoader()
    
    # Test connection
    logger.info("Testing Databricks connection...")
    if not loader.test_connection():
        raise ConnectionError("Failed to connect to Databricks!")
    
    # Fetch items
    logger.info(f"Fetching items (domain={domain}, language={language}, limit={limit})...")
    items = loader.get_items_for_indexing(
        domain=domain,
        language=language,
        limit=limit,
        include_readings=True,
    )
    
    loader.close()
    
    # Save to file
    with open(CONTENT_FILE, 'w') as f:
        json.dump(items, f, indent=2)
    
    logger.info(f"✅ Fetched {len(items)} items → {CONTENT_FILE.name}")
    
    # Stats
    video_count = sum(1 for i in items if i.get('content_type') == 'video')
    reading_count = sum(1 for i in items if i.get('content_type') == 'reading')
    courses = set(i.get('course_id') for i in items)
    
    logger.info(f"   Videos: {video_count}, Readings: {reading_count}")
    logger.info(f"   Courses: {len(courses)}")
    
    return items


# =============================================================================
# STAGE 2: METADATA EXTRACTION
# =============================================================================

def extract_metadata(
    items: List[Dict[str, Any]],
    max_items: Optional[int] = None,
    rate_limit: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Extract operational + derived metadata for each item.
    
    - Operational: from CourseCatalogue.xlsx
    - Derived: from Gemini LLM
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: METADATA EXTRACTION")
    logger.info("=" * 60)
    
    from src.config import get_config
    from src.metadata.llm_extractor import LLMMetadataExtractor
    from src.metadata.operational_loader import OperationalMetadataLoader
    from src.metadata.schema import ContentMetadata
    
    config = get_config()
    if not config.is_gemini_configured():
        raise ValueError("Gemini API key not configured! Set GOOGLE_API_KEY.")
    
    # Initialize extractors
    llm_extractor = LLMMetadataExtractor(
        provider="gemini",
        api_key=config.GOOGLE_API_KEY,
        model="gemini-2.0-flash"
    )
    op_loader = OperationalMetadataLoader()
    
    # Process items
    if max_items:
        items = items[:max_items]
    
    enriched_items = []
    errors = []
    start_time = time.time()
    
    for i, item in enumerate(items):
        item_id = item.get('item_id', f"item_{i}")
        item_name = item.get('item_name', 'Unknown')[:40]
        
        logger.info(f"[{i+1}/{len(items)}] {item_name}...")
        
        try:
            # Get operational metadata
            op_meta = op_loader.enrich_content_item(item)
            
            # Get content text
            content_text = item.get('content_text', '')
            if len(content_text) < 50:
                logger.warning(f"  ⚠️ Skipping - content too short")
                continue
            
            # Extract derived metadata
            derived = llm_extractor.extract_all(
                transcript=content_text[:4000],
                course_name=op_meta.course_name,
                module_name=op_meta.module_name,
                lesson_name=item.get('lesson_name', ''),
                content_type=item.get('content_type', 'video'),
                chunk_id=item_id,
            )
            
            # Combine
            combined = ContentMetadata(
                id=item_id,
                operational=op_meta,
                derived=derived,
            )
            
            enriched_item = {
                **item,
                "operational_metadata": op_meta.to_dict(),
                "derived_metadata": derived.to_dict(),
                "embedding_input": combined.get_embedding_input()[:500],
                "filter_metadata": combined.get_filter_metadata(),
                "extracted_at": datetime.now().isoformat(),
            }
            enriched_items.append(enriched_item)
            
            # Rate limit
            time.sleep(rate_limit)
            
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            errors.append({"item_id": item_id, "error": str(e)})
            time.sleep(2)
    
    # Save
    with open(ENRICHED_FILE, 'w') as f:
        json.dump(enriched_items, f, indent=2)
    
    elapsed = time.time() - start_time
    logger.info(f"\n✅ Enriched {len(enriched_items)} items → {ENRICHED_FILE.name}")
    logger.info(f"   Time: {elapsed/60:.1f} minutes")
    logger.info(f"   Errors: {len(errors)}")
    
    return enriched_items


# =============================================================================
# STAGE 3: INDEX BUILDING
# =============================================================================

def build_index(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build FAISS vector index from enriched items.
    
    - Chunking: 750-token sliding window
    - Embeddings: sentence-transformers (all-mpnet-base-v2)
    - Index: FAISS HNSW
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: INDEX BUILDING")
    logger.info("=" * 60)
    
    import numpy as np
    
    # Import chunker
    from src.chunking.transcript_chunker import TranscriptChunker
    
    # Initialize chunker (750-token window, 150-token overlap per roadmap)
    chunker = TranscriptChunker(
        window_size=750,
        overlap=150,
    )
    
    # Chunk all items
    logger.info("Chunking content...")
    all_chunks = []
    
    for item in items:
        content_text = item.get('content_text', '')
        if not content_text or len(content_text) < 50:
            continue
        
        # Use the chunker's content_item method
        chunks = chunker.chunk_text(
            text=content_text,
            item_id=item.get('item_id', ''),
            item_name=item.get('item_name', ''),
            course_name=item.get('course_name', ''),
            module_name=item.get('module_name', ''),
        )
        
        # Add metadata to each chunk
        for chunk in chunks:
            all_chunks.append({
                "chunk_id": chunk.chunk_id,
                "item_id": item.get('item_id'),
                "course_id": item.get('course_id'),
                "course_name": item.get('course_name'),
                "course_slug": item.get('course_slug'),
                "item_name": item.get('item_name'),
                "module_name": item.get('module_name'),
                "lesson_name": item.get('lesson_name'),
                "content_type": item.get('content_type'),
                "summary": item.get('summary'),
                "chunk_text": chunk.contextual_text or chunk.text,  # Use contextual text for embedding
                "chunk_index": chunk.start_token,
                "derived_metadata": item.get('derived_metadata', {}),
            })
    
    logger.info(f"  Created {len(all_chunks)} chunks from {len(items)} items")
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    from sentence_transformers import SentenceTransformer
    
    model_name = "all-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    
    texts = [c['chunk_text'] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    logger.info(f"  Generated {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    import faiss
    
    dimension = embeddings.shape[1]
    
    # Use HNSW index for better search quality
    index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 100
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    logger.info(f"  Index size: {index.ntotal} vectors")
    
    # Save index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    
    with open(INDEX_DIR / "chunks.json", 'w') as f:
        json.dump(all_chunks, f, indent=2)
    
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    
    config = {
        "model_name": model_name,
        "dimension": dimension,
        "num_chunks": len(all_chunks),
        "index_type": "HNSW",
        "created_at": datetime.now().isoformat(),
    }
    with open(INDEX_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\n✅ Index saved → {INDEX_DIR.name}/")
    
    return {
        "index": index,
        "chunks": all_chunks,
        "embeddings": embeddings,
        "model": model,
    }


# =============================================================================
# STAGE 4: TEST SEARCH
# =============================================================================

def test_search(
    index_data: Dict[str, Any],
    queries: List[str] = None,
    top_k: int = 5,
):
    """
    Test the search functionality with sample queries.
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: TEST SEARCH")
    logger.info("=" * 60)
    
    import faiss
    import numpy as np
    
    index = index_data["index"]
    chunks = index_data["chunks"]
    model = index_data["model"]
    
    if queries is None:
        queries = [
            "what is a variable in programming",
            "how to use functions in python",
            "object oriented programming concepts",
            "debugging techniques",
            "version control with git",
        ]
    
    for query in queries:
        logger.info(f"\n🔍 Query: \"{query}\"")
        
        # Embed query
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = index.search(query_embedding, top_k)
        
        # Display results
        seen_items = set()
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(chunks):
                continue
            
            chunk = chunks[idx]
            item_id = chunk.get('item_id')
            
            # Skip duplicates from same item
            if item_id in seen_items:
                continue
            seen_items.add(item_id)
            
            score = float(dist)
            item_name = chunk.get('item_name', 'Unknown')[:50]
            course_name = chunk.get('course_name', 'Unknown')[:30]
            content_type = chunk.get('content_type', 'video')
            
            logger.info(f"   {rank+1}. [{content_type}] {item_name}")
            logger.info(f"      Course: {course_name}")
            logger.info(f"      Score: {score:.3f}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(
    items_limit: int = 50,
    extract_limit: Optional[int] = None,
    skip_fetch: bool = False,
    only_fetch: bool = False,
    only_index: bool = False,
    domain: str = "Software Development",
    language: str = "en",
):
    """
    Run the complete end-to-end pipeline.
    """
    start_time = time.time()
    
    logger.info("\n" + "=" * 70)
    logger.info("  AI-LED CURATIONS - END-TO-END PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Items limit: {items_limit}")
    logger.info(f"  Domain: {domain}")
    logger.info(f"  Language: {language}")
    logger.info("=" * 70 + "\n")
    
    # Stage 1: Data Fetch
    if only_index:
        # Load existing enriched data
        if ENRICHED_FILE.exists():
            with open(ENRICHED_FILE, 'r') as f:
                enriched_items = json.load(f)
            logger.info(f"Loaded {len(enriched_items)} enriched items from {ENRICHED_FILE.name}")
        else:
            raise FileNotFoundError(f"No enriched data found: {ENRICHED_FILE}")
    elif skip_fetch:
        # Load existing content
        if CONTENT_FILE.exists():
            with open(CONTENT_FILE, 'r') as f:
                items = json.load(f)
            logger.info(f"Loaded {len(items)} items from {CONTENT_FILE.name}")
        else:
            raise FileNotFoundError(f"No content data found: {CONTENT_FILE}")
        
        # Stage 2: Metadata Extraction
        enriched_items = extract_metadata(items, max_items=extract_limit)
    else:
        # Fresh fetch from Databricks
        items = fetch_from_databricks(
            domain=domain,
            language=language,
            limit=items_limit,
        )
        
        if only_fetch:
            logger.info("\n✅ Data fetch complete (--only-fetch)")
            return
        
        # Stage 2: Metadata Extraction
        enriched_items = extract_metadata(items, max_items=extract_limit)
    
    # Stage 3: Index Building
    index_data = build_index(enriched_items)
    
    # Stage 4: Test Search
    test_search(index_data)
    
    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time: {elapsed/60:.1f} minutes")
    logger.info(f"  Content file: {CONTENT_FILE}")
    logger.info(f"  Enriched file: {ENRICHED_FILE}")
    logger.info(f"  Index dir: {INDEX_DIR}")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the AI-Led Curations pipeline end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--items", type=int, default=50,
                        help="Number of items to fetch from Databricks (default: 50)")
    parser.add_argument("--extract-limit", type=int, default=None,
                        help="Limit metadata extraction (default: all)")
    parser.add_argument("--domain", type=str, default="Computer Science",
                        help="Filter by domain (default: Computer Science)")
    parser.add_argument("--language", type=str, default="en",
                        help="Filter by language (default: en)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip Databricks fetch, use existing content file")
    parser.add_argument("--only-fetch", action="store_true",
                        help="Only fetch data, skip extraction and indexing")
    parser.add_argument("--only-index", action="store_true",
                        help="Only build index from existing enriched data")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        items_limit=args.items,
        extract_limit=args.extract_limit,
        skip_fetch=args.skip_fetch,
        only_fetch=args.only_fetch,
        only_index=args.only_index,
        domain=args.domain,
        language=args.language,
    )
