"""
Search Engine with Metadata Filtering

Implements two-stage retrieval from roadmap.pdf:
1. Stage 1 (Coarse): Vector similarity search
2. Stage 2 (Fine): Metadata filtering (bloom_level, difficulty, etc.)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Vector search engine with metadata filtering.
    
    Features:
    - FAISS vector similarity search
    - Metadata filtering (bloom_level, difficulty, domain, etc.)
    - Contextual result ranking
    """
    
    def __init__(self, index_dir: str = "data/test_indexes/diverse_50/index"):
        """
        Initialize search engine.
        
        Args:
            index_dir: Directory containing faiss.index, chunks.json, embeddings.npy
        """
        self.index_dir = Path(index_dir)
        
        # Load components
        self._load_index()
        self._load_chunks()
        self._load_embedder()
    
    def _load_index(self):
        """Load FAISS index."""
        import faiss
        
        index_path = self.index_dir / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
    
    def _load_chunks(self):
        """Load chunk metadata."""
        chunks_path = self.index_dir / "chunks.json"
        with open(chunks_path, 'r') as f:
            self.chunks = json.load(f)
        logger.info(f"Loaded {len(self.chunks)} chunk metadata")
    
    def _load_embedder(self):
        """Load embedding model for query encoding."""
        config_path = self.index_dir / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_name = config.get("model_name", config.get("model", "all-mpnet-base-v2"))
        embedding_provider = config.get("embedding_provider", "local")
        
        if embedding_provider == "gemini" or "gemini" in model_name:
            # Use Gemini API for query embeddings
            import os
            import google.generativeai as genai
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set for Gemini embeddings")
            
            genai.configure(api_key=api_key)
            self._gemini_model = model_name
            self._embedding_provider = "gemini"
            self.embedder = None  # Will use _embed_query_gemini instead
            logger.info(f"Using Gemini embedder: {model_name}")
        else:
            # Use local SentenceTransformer
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(model_name)
            self._embedding_provider = "local"
            logger.info(f"Loaded local embedder: {model_name}")
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query text based on configured provider."""
        if self._embedding_provider == "gemini":
            import google.generativeai as genai
            result = genai.embed_content(
                model=self._gemini_model,
                content=query,
                task_type="retrieval_query",  # Use retrieval_query for queries!
            )
            return np.array([result['embedding']], dtype=np.float32)
        else:
            return self.embedder.encode([query]).astype(np.float32)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        target_domain: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Search for relevant content with domain filtering and confidence scoring.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional metadata filters, e.g.:
                     {"bloom_level": "Apply", "difficulty": "BEGINNER"}
            target_domain: Target domain to filter/boost (e.g., "Math and Logic")
            min_confidence: Minimum confidence threshold (0-1)
        
        Returns:
            Dict with results, confidence metrics, and domain match info
        """
        # Stage 1: Vector similarity search
        query_embedding = self._embed_query(query)
        
        # Normalize query embedding for cosine similarity (index is already normalized)
        import faiss
        faiss.normalize_L2(query_embedding)
        
        # Search more than needed for filtering and confidence analysis
        search_k = min(top_k * 5, len(self.chunks))
        
        distances, indices = self.index.search(
            query_embedding,
            search_k
        )
        
        # Collect all candidate results with domain info
        all_candidates = []
        domain_matched = []
        domain_unmatched = []
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            
            # Stage 2: Apply metadata filters
            if filters and not self._matches_filters(chunk, filters):
                continue
            
            # Get text from either 'text' or 'chunk_text' field
            text = chunk.get("text") or chunk.get("chunk_text", "")
            url = chunk.get("item_url") or chunk.get("url") or chunk.get("course_url", "")
            
            # Get domain from derived metadata
            derived_meta = chunk.get("derived_metadata", {})
            chunk_domain = derived_meta.get("primary_domain", "")
            chunk_subdomain = derived_meta.get("sub_domain", "")
            key_concepts = derived_meta.get("key_concepts", [])
            
            # Calculate base score
            base_score = float(1 / (1 + dist))
            
            # Domain matching boost/filter
            domain_match = False
            if target_domain:
                target_lower = target_domain.lower()
                domain_lower = chunk_domain.lower() if chunk_domain else ""
                subdomain_lower = chunk_subdomain.lower() if chunk_subdomain else ""
                
                # Check if domain matches
                if target_lower in domain_lower or target_lower in subdomain_lower:
                    domain_match = True
                # Also check key concepts for domain relevance
                elif any(target_lower in c.lower() for c in key_concepts):
                    domain_match = True
            
            result = {
                "rank": 0,  # Will be set after sorting
                "score": base_score,
                "distance": float(dist),
                "chunk_id": chunk["chunk_id"],
                "item_id": chunk.get("item_id") or chunk["chunk_id"].rsplit('_c', 1)[0],
                "item_name": chunk["item_name"],
                "url": url,
                "course_name": chunk["course_name"],
                "course_slug": chunk.get("course_slug"),
                "module_name": chunk["module_name"],
                "lesson_name": chunk.get("lesson_name", chunk["module_name"]),
                "content_type": chunk.get("content_type", "video"),
                "bloom_level": chunk.get("bloom_level"),
                "difficulty": chunk.get("difficulty"),
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                "text": text,
                "summary": chunk.get("summary", ""),
                "derived_metadata": derived_meta,
                "operational_metadata": chunk.get("operational_metadata", {}),
                "domain_match": domain_match,
                "primary_domain": chunk_domain,
            }
            
            all_candidates.append(result)
            if domain_match:
                domain_matched.append(result)
            else:
                domain_unmatched.append(result)
        
        # Determine final results based on domain matching
        if target_domain and domain_matched:
            # Prefer domain-matched results
            final_results = sorted(domain_matched, key=lambda x: x["score"], reverse=True)[:top_k]
            domain_coverage = len(domain_matched) / len(all_candidates) if all_candidates else 0
        else:
            # Use all results if no domain filter or no matches
            final_results = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:top_k]
            domain_coverage = 1.0 if not target_domain else 0.0
        
        # Assign ranks
        for i, r in enumerate(final_results):
            r["rank"] = i + 1
        
        # Calculate confidence metrics
        if final_results:
            avg_score = sum(r["score"] for r in final_results) / len(final_results)
            top_score = final_results[0]["score"] if final_results else 0
            # Confidence based on: top score, score consistency, domain coverage
            confidence = (top_score * 0.5) + (avg_score * 0.3) + (domain_coverage * 0.2)
        else:
            confidence = 0.0
            avg_score = 0.0
            top_score = 0.0
        
        # Determine confidence level
        if confidence >= 0.6:
            confidence_level = "high"
        elif confidence >= 0.45:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return {
            "results": final_results,
            "total_candidates": len(all_candidates),
            "domain_matched_count": len(domain_matched),
            "confidence": confidence,
            "confidence_level": confidence_level,
            "top_score": top_score,
            "avg_score": avg_score,
            "target_domain": target_domain,
            "domain_coverage": domain_coverage,
        }
    
    def _matches_filters(self, chunk: Dict, filters: Dict) -> bool:
        """Check if chunk matches all filters."""
        for key, value in filters.items():
            if isinstance(value, list):
                # Match any in list
                if chunk.get(key) not in value:
                    return False
            else:
                # Exact match
                if chunk.get(key) != value:
                    return False
        return True
    
    def search_with_context(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search with additional context and summary.
        
        Returns structured response for chatbot UI.
        """
        results = self.search(query, top_k, filters)
        
        # Group by course
        courses = {}
        for r in results:
            course = r["course_name"]
            if course not in courses:
                courses[course] = []
            courses[course].append(r)
        
        return {
            "query": query,
            "filters": filters,
            "total_results": len(results),
            "results": results,
            "by_course": courses,
            "top_result": results[0] if results else None,
        }


def test_search():
    """Test the search engine."""
    # Find index directory
    project_root = Path(__file__).parent.parent.parent
    index_dir = project_root / "data" / "index"
    
    if not index_dir.exists():
        print("❌ Index not found. Run build_index.py first.")
        return
    
    engine = SearchEngine(str(index_dir))
    
    # Test queries
    test_queries = [
        ("What is a do-while loop?", None),
        ("Docker containers", None),
        ("Java programming basics", {"difficulty": "BEGINNER"}),
        ("How to use Kubernetes", {"bloom_level": "Apply"}),
    ]
    
    print("=" * 70)
    print("SEARCH ENGINE TEST")
    print("=" * 70)
    
    for query, filters in test_queries:
        print(f"\n🔍 Query: '{query}'")
        if filters:
            print(f"   Filters: {filters}")
        
        results = engine.search(query, top_k=3, filters=filters)
        
        print(f"   Found {len(results)} results:")
        for r in results:
            print(f"   {r['rank']}. [{r['score']:.3f}] {r['item_name']}")
            print(f"      Course: {r['course_name']}")
            print(f"      Bloom: {r['bloom_level']}, Difficulty: {r['difficulty']}")


if __name__ == "__main__":
    test_search()
