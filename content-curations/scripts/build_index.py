"""
Build Vector Index Pipeline

Complete pipeline to:
1. Load enriched content
2. Chunk transcripts (750-token window, 150 overlap)
3. Generate embeddings with contextual pre-pending
4. Build FAISS index with metadata
"""

import json
import pickle
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.transcript_chunker import TranscriptChunker, Chunk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_enriched_data(path: str = "data/sample_courses_enriched.json") -> List[Dict]:
    """Load enriched content items."""
    full_path = Path(__file__).parent.parent / path
    with open(full_path, 'r') as f:
        return json.load(f)


def chunk_all_items(items: List[Dict]) -> List[Chunk]:
    """Chunk all items using 750-token window."""
    chunker = TranscriptChunker(window_size=750, overlap=150)
    
    all_chunks = []
    for item in items:
        chunks = chunker.chunk_content_item(item)
        all_chunks.extend(chunks)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(items)} items")
    return all_chunks


def generate_embeddings(chunks: List[Chunk], model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    """
    Generate embeddings for all chunks using contextual text.
    
    Uses local sentence-transformers (FREE, no API needed).
    """
    from sentence_transformers import SentenceTransformer
    
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Use contextual text for embedding (with pre-pended metadata)
    texts = [chunk.contextual_text for chunk in chunks]
    
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> Any:
    """Build FAISS index from embeddings."""
    import faiss
    
    dimension = embeddings.shape[1]
    
    # Use HNSW index for fast approximate search (as recommended in roadmap)
    # M=32 connections, efConstruction=200 for build quality
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128  # Trade-off between speed and accuracy
    
    # Add embeddings
    index.add(embeddings.astype('float32'))
    
    logger.info(f"Built FAISS HNSW index with {index.ntotal} vectors")
    return index


def save_index(
    index,
    chunks: List[Chunk],
    embeddings: np.ndarray,
    output_dir: str = "data/index"
):
    """Save index and metadata."""
    import faiss
    
    output_path = Path(__file__).parent.parent / output_dir
    output_path.mkdir(exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, str(output_path / "faiss.index"))
    logger.info(f"Saved FAISS index to {output_path / 'faiss.index'}")
    
    # Save chunk metadata
    chunk_data = [chunk.to_dict() for chunk in chunks]
    with open(output_path / "chunks.json", 'w') as f:
        json.dump(chunk_data, f, indent=2)
    logger.info(f"Saved {len(chunk_data)} chunk metadata to chunks.json")
    
    # Save embeddings (for backup/debugging)
    np.save(output_path / "embeddings.npy", embeddings)
    logger.info(f"Saved embeddings to embeddings.npy")
    
    # Save index config
    config = {
        "model": "all-mpnet-base-v2",
        "dimensions": embeddings.shape[1],
        "num_chunks": len(chunks),
        "window_size": 750,
        "overlap": 150,
    }
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)


def main(
    input_file: str = "data/sample_courses_enriched.json",
    model_name: str = "all-mpnet-base-v2",
    output_dir: str = "data/index"
):
    """Run the full indexing pipeline."""
    logger.info("=" * 60)
    logger.info("BUILDING VECTOR INDEX")
    logger.info("=" * 60)
    
    # Step 1: Load data
    logger.info("\n📂 Step 1: Loading enriched data...")
    items = load_enriched_data(input_file)
    logger.info(f"  Loaded {len(items)} items")
    
    # Step 2: Chunk
    logger.info("\n✂️ Step 2: Chunking transcripts (750 tokens, 150 overlap)...")
    chunks = chunk_all_items(items)
    
    # Step 3: Embed
    logger.info(f"\n🧠 Step 3: Generating embeddings ({model_name})...")
    embeddings = generate_embeddings(chunks, model_name)
    
    # Step 4: Build index
    logger.info("\n📊 Step 4: Building FAISS HNSW index...")
    index = build_faiss_index(embeddings)
    
    # Step 5: Save
    logger.info(f"\n💾 Step 5: Saving to {output_dir}...")
    save_index(index, chunks, embeddings, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ INDEX BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total chunks: {len(chunks)}")
    logger.info(f"  Embedding dims: {embeddings.shape[1]}")
    logger.info(f"  Index location: {output_dir}/")
    
    return index, chunks, embeddings


if __name__ == "__main__":
    main()
