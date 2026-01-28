#!/usr/bin/env python3
"""
Build Domain Index - Production Pipeline

Fetches ALL content for a domain, extracts metadata, and builds a complete
searchable index. The index can then be used directly by the Streamlit app.

This pipeline is:
- Resumable: Uses persistent metadata store to avoid re-extraction
- Efficient: Batch processing with rate limiting
- Complete: Processes all available content for a domain

Usage:
    # Build index for Computer Science domain
    python scripts/build_domain_index.py --domain "Computer Science"
    
    # Build for Data Science with custom limits
    python scripts/build_domain_index.py --domain "Data Science" --fetch-limit 500
    
    # Force rebuild (delete existing metadata)
    python scripts/build_domain_index.py --domain "Computer Science" --force
"""

import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import pipeline components
from src.config import get_config
from src.data_loaders.databricks_loader import DatabricksLoader
from src.metadata.llm_extractor import LLMMetadataExtractor
from src.metadata.operational_loader import OperationalMetadataLoader
from src.metadata.schema import ContentMetadata
from src.chunking.transcript_chunker import TranscriptChunker


class DomainIndexBuilder:
    """
    Builds a complete searchable index for a domain.
    
    Stages:
    1. Fetch all items from Databricks for the domain
    2. Extract metadata (using persistent store to avoid reprocessing)
    3. Chunk content with contextual pre-pending
    4. Generate embeddings
    5. Build FAISS index
    6. Save everything for production use
    """
    
    def __init__(
        self,
        domain: str,
        output_dir: str = None,
        language: str = "en",
    ):
        self.domain = domain
        self.language = language
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "data" / "domain_indexes" / self._sanitize_name(domain)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths
        self.content_file = self.output_dir / "content.json"
        self.enriched_file = self.output_dir / "enriched.json"
        self.chunks_file = self.output_dir / "chunks.json"
        self.index_dir = self.output_dir / "index"
        self.metadata_file = self.output_dir / "metadata.json"
        
        logger.info(f"Domain Index Builder for: {domain}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _sanitize_name(self, name: str) -> str:
        """Convert domain name to safe directory name."""
        return name.lower().replace(" ", "_").replace("/", "_")
    
    def stage1_fetch_data(
        self,
        fetch_limit: Optional[int] = None,
        skip_if_exists: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Fetch all items from Databricks.
        
        Args:
            fetch_limit: Limit items to fetch (None = all available)
            skip_if_exists: Skip if content file already exists
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"STAGE 1: FETCH DATA FROM DATABRICKS")
        logger.info("=" * 70)
        
        # Check if already fetched
        if skip_if_exists and self.content_file.exists():
            logger.info(f"✅ Content file exists: {self.content_file.name}")
            with open(self.content_file, 'r') as f:
                items = json.load(f)
            logger.info(f"   Loaded {len(items)} items")
            return items
        
        # Fetch from Databricks
        logger.info(f"Fetching items (domain={self.domain}, language={self.language})...")
        
        loader = DatabricksLoader()
        
        if not loader.test_connection():
            raise ConnectionError("Failed to connect to Databricks!")
        
        items = loader.get_items_for_indexing(
            domain=self.domain,
            language=self.language,
            limit=fetch_limit,
            include_readings=True,
        )
        
        loader.close()
        
        # Save
        with open(self.content_file, 'w') as f:
            json.dump(items, f, indent=2)
        
        # Stats
        video_count = sum(1 for i in items if i.get('content_type') == 'video')
        reading_count = sum(1 for i in items if i.get('content_type') == 'reading')
        courses = set(i.get('course_id') for i in items)
        
        logger.info(f"✅ Fetched {len(items)} items")
        logger.info(f"   Videos: {video_count}, Readings: {reading_count}")
        logger.info(f"   Courses: {len(courses)}")
        logger.info(f"   Saved → {self.content_file.name}")
        
        return items
    
    def stage2_extract_metadata(
        self,
        items: List[Dict[str, Any]],
        max_items: Optional[int] = None,
        rate_limit: float = 0.1,
        batch_size: int = 50,
        max_workers: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Extract metadata using persistent store with PARALLEL processing.
        
        Uses the metadata store to:
        - Track what's already been processed
        - Avoid duplicate LLM API calls
        - Resume from where it left off
        - Process items in parallel for 10x speedup
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"STAGE 2: EXTRACT METADATA (PARALLEL - {max_workers} workers)")
        logger.info("=" * 70)
        
        # Import metadata store
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from extract_metadata import MetadataStore
        
        config = get_config()
        if not config.is_gemini_configured():
            raise ValueError("Gemini API key not configured!")
        
        # Initialize components
        store = MetadataStore()
        op_loader = OperationalMetadataLoader()
        
        # Filter to unprocessed items
        processed_ids = store.get_processed_item_ids()
        items_to_process = [
            item for item in items
            if item.get('item_id') not in processed_ids
        ]
        
        logger.info(f"Total items: {len(items)}")
        logger.info(f"Already processed: {len(processed_ids)}")
        logger.info(f"New items to process: {len(items_to_process)}")
        
        if max_items and len(items_to_process) > max_items:
            items_to_process = items_to_process[:max_items]
            logger.info(f"Limited to: {max_items} items")
        
        # Thread-safe counters and results
        results_lock = threading.Lock()
        batch_results = []
        errors = []
        completed_count = [0]  # Use list for mutable in closure
        
        def process_single_item(item_with_index):
            """Process a single item - runs in parallel."""
            idx, item = item_with_index
            item_id = item.get('item_id', f"item_{idx}")
            
            try:
                # Each thread needs its own LLM client
                llm_extractor = LLMMetadataExtractor(
                    provider="gemini",
                    api_key=config.GOOGLE_API_KEY,
                    model="gemini-2.0-flash"
                )
                
                # Operational metadata
                op_meta = op_loader.enrich_content_item(item)
                
                # Get content
                content_text = item.get('content_text', '')
                if len(content_text) < 50:
                    return None  # Skip short content
                
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
                
                return enriched_item
                
            except Exception as e:
                return {"error": str(e), "item_id": item_id}
        
        # Process items in parallel
        if items_to_process:
            logger.info(f"\n🚀 Extracting metadata for {len(items_to_process)} items with {max_workers} parallel workers...")
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_single_item, (i, item)): i
                    for i, item in enumerate(items_to_process)
                }
                
                # Process results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    
                    with results_lock:
                        completed_count[0] += 1
                        
                        if result is None:
                            pass  # Skipped item
                        elif isinstance(result, dict) and "error" in result:
                            errors.append(result)
                        else:
                            batch_results.append(result)
                        
                        # Log progress every 50 items
                        if completed_count[0] % 50 == 0:
                            elapsed = time.time() - start_time
                            rate = completed_count[0] / elapsed if elapsed > 0 else 0
                            remaining = len(items_to_process) - completed_count[0]
                            eta = remaining / rate if rate > 0 else 0
                            logger.info(f"  ⏳ Progress: {completed_count[0]}/{len(items_to_process)} "
                                       f"({rate:.1f} items/sec, ETA: {eta/60:.1f} min)")
                        
                        # Save batch periodically
                        if len(batch_results) >= batch_size:
                            store.save_batch(batch_results)
                            logger.info(f"  💾 Saved batch of {len(batch_results)} items")
                            batch_results = []
            
            # Save remaining
            if batch_results:
                store.save_batch(batch_results)
                logger.info(f"💾 Saved final batch of {len(batch_results)} items")
            
            elapsed = time.time() - start_time
            logger.info(f"\n✅ Metadata extraction complete")
            logger.info(f"   Processed: {completed_count[0] - len(errors)} items")
            logger.info(f"   Errors: {len(errors)}")
            logger.info(f"   Time: {elapsed/60:.1f} minutes")
            logger.info(f"   Speed: {completed_count[0]/elapsed:.1f} items/sec")
        
        # Export all enriched items for this domain
        logger.info(f"\nExporting all enriched items...")
        all_enriched = []
        
        for item in items:
            item_id = item.get('item_id')
            if item_id in store.get_processed_item_ids():
                # Get from store
                course_id = item.get('course_id')
                course_file = store.courses_dir / f"{course_id}.json"
                if course_file.exists():
                    with open(course_file, 'r') as f:
                        course_data = json.load(f)
                    if item_id in course_data.get('items', {}):
                        all_enriched.append(course_data['items'][item_id])
        
        # Save
        with open(self.enriched_file, 'w') as f:
            json.dump(all_enriched, f, indent=2)
        
        logger.info(f"✅ Saved {len(all_enriched)} enriched items → {self.enriched_file.name}")
        
        return all_enriched
    
    def stage3_build_index(
        self,
        enriched_items: List[Dict[str, Any]],
        model_name: str = "all-mpnet-base-v2",
    ):
        """
        Stage 3: Chunk content, generate embeddings, build FAISS index.
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"STAGE 3: BUILD VECTOR INDEX")
        logger.info("=" * 70)
        
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        
        # Chunk content
        logger.info("Chunking content...")
        chunker = TranscriptChunker(window_size=750, overlap=150)
        
        all_chunks = []
        for item in enriched_items:
            content_text = item.get('content_text', '')
            if not content_text or len(content_text) < 50:
                continue
            
            chunks = chunker.chunk_text(
                text=content_text,
                item_id=item.get('item_id', ''),
                item_name=item.get('item_name', ''),
                course_name=item.get('course_name', ''),
                module_name=item.get('module_name', ''),
            )
            
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
                    "chunk_text": chunk.contextual_text or chunk.text,
                    "chunk_index": chunk.start_token,
                    "derived_metadata": item.get('derived_metadata', {}),
                    "operational_metadata": item.get('operational_metadata', {}),
                })
        
        logger.info(f"  Created {len(all_chunks)} chunks from {len(enriched_items)} items")
        
        # Generate embeddings using Gemini API
        # Using gemini-embedding-001 (3072 dims) - text-embedding-004 was deprecated Jan 14, 2026
        EMBEDDING_MODEL = "models/gemini-embedding-001"
        logger.info(f"Generating embeddings using {EMBEDDING_MODEL}...")
        
        import google.generativeai as genai
        import time
        
        config = get_config()
        genai.configure(api_key=config.GOOGLE_API_KEY)
        
        texts = [c['chunk_text'][:2000] for c in all_chunks]  # Truncate to fit token limit
        
        # Process in batches of 100 using batch_embed_contents
        batch_size = 100
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)...")
            
            try:
                # Use batch embedding for efficiency
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch,
                    task_type="retrieval_document",
                )
                # Result contains list of embeddings when content is a list
                if isinstance(result['embedding'][0], list):
                    all_embeddings.extend(result['embedding'])
                else:
                    all_embeddings.append(result['embedding'])
                    
            except Exception as e:
                logger.warning(f"  Batch failed, trying individual: {e}")
                # Fallback to individual calls
                for text in batch:
                    result = genai.embed_content(
                        model=EMBEDDING_MODEL,
                        content=text,
                        task_type="retrieval_document",
                    )
                    all_embeddings.append(result['embedding'])
                    time.sleep(0.1)
            
            # Brief pause between batches to respect rate limits
            time.sleep(0.3)
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"  Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
        
        # Build FAISS index
        logger.info("Building FAISS HNSW index...")
        dimension = embeddings.shape[1]
        
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 100
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        logger.info(f"  Index size: {index.ntotal} vectors")
        
        # Save index
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, str(self.index_dir / "faiss.index"))
        
        with open(self.index_dir / "chunks.json", 'w') as f:
            json.dump(all_chunks, f, indent=2)
        
        np.save(self.index_dir / "embeddings.npy", embeddings)
        
        index_config = {
            "domain": self.domain,
            "language": self.language,
            "model_name": EMBEDDING_MODEL,  # The actual model used for embeddings
            "embedding_provider": "gemini",
            "dimension": dimension,
            "num_chunks": len(all_chunks),
            "num_items": len(enriched_items),
            "index_type": "HNSW",
            "created_at": datetime.now().isoformat(),
        }
        with open(self.index_dir / "config.json", 'w') as f:
            json.dump(index_config, f, indent=2)
        
        logger.info(f"✅ Index saved → {self.index_dir}/")
        
        return index, all_chunks, embeddings, EMBEDDING_MODEL
    
    def build_complete_index(
        self,
        fetch_limit: Optional[int] = None,
        extract_limit: Optional[int] = None,
        skip_fetch: bool = False,
    ):
        """Run the complete pipeline."""
        start_time = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"  BUILD DOMAIN INDEX: {self.domain}")
        logger.info("=" * 70)
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Language: {self.language}")
        logger.info("=" * 70 + "\n")
        
        # Stage 1: Fetch
        items = self.stage1_fetch_data(
            fetch_limit=fetch_limit,
            skip_if_exists=skip_fetch,
        )
        
        # Stage 2: Extract metadata
        enriched = self.stage2_extract_metadata(
            items=items,
            max_items=extract_limit,
        )
        
        # Stage 3: Build index
        index, chunks, embeddings, model = self.stage3_build_index(enriched)
        
        # Save metadata
        metadata = {
            "domain": self.domain,
            "language": self.language,
            "total_items": len(items),
            "enriched_items": len(enriched),
            "total_chunks": len(chunks),
            "courses": len(set(i.get('course_id') for i in items)),
            "videos": sum(1 for i in items if i.get('content_type') == 'video'),
            "readings": sum(1 for i in items if i.get('content_type') == 'reading'),
            "index_dir": str(self.index_dir),
            "created_at": datetime.now().isoformat(),
            "build_time_minutes": (time.time() - start_time) / 60,
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info(f"  ✅ DOMAIN INDEX BUILD COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Domain: {self.domain}")
        logger.info(f"  Total items: {len(items)}")
        logger.info(f"  Enriched items: {len(enriched)}")
        logger.info(f"  Chunks: {len(chunks)}")
        logger.info(f"  Courses: {metadata['courses']}")
        logger.info(f"  Build time: {elapsed/60:.1f} minutes")
        logger.info(f"  Output: {self.output_dir}")
        logger.info("=" * 70 + "\n")
        
        logger.info("📁 Files created:")
        logger.info(f"  - {self.content_file.name} ({len(items)} items)")
        logger.info(f"  - {self.enriched_file.name} ({len(enriched)} items)")
        logger.info(f"  - {self.index_dir}/ (FAISS index)")
        logger.info(f"  - {self.metadata_file.name} (metadata)")
        
        logger.info("\n🚀 Ready for production!")
        logger.info(f"   Use this index in Streamlit app: {self.index_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build complete searchable index for a domain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index for Computer Science (all available items)
  python scripts/build_domain_index.py --domain "Computer Science"
  
  # Build with limits for testing
  python scripts/build_domain_index.py --domain "Data Science" --fetch-limit 100 --extract-limit 50
  
  # Skip fetch if data already exists
  python scripts/build_domain_index.py --domain "Computer Science" --skip-fetch
  
  # Available domains: Computer Science, Data Science, Business, etc.
        """
    )
    
    parser.add_argument("--domain", type=str, required=True,
                        help="Domain to build index for (e.g., 'Computer Science')")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code (default: en)")
    parser.add_argument("--fetch-limit", type=int, default=None,
                        help="Limit items to fetch from Databricks (default: all)")
    parser.add_argument("--extract-limit", type=int, default=None,
                        help="Limit metadata extraction (default: all)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory (default: data/domain_indexes/{domain})")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip fetch if content file exists")
    
    args = parser.parse_args()
    
    builder = DomainIndexBuilder(
        domain=args.domain,
        language=args.language,
        output_dir=args.output_dir,
    )
    
    builder.build_complete_index(
        fetch_limit=args.fetch_limit,
        extract_limit=args.extract_limit,
        skip_fetch=args.skip_fetch,
    )
