"""
Pinecone Search Engine

Search engine using Pinecone as vector store.
Maintains same interface as SearchEngine (FAISS-based) for easy swapping.

This is the V2 exploration - keeps the same API as the original search_engine.py
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PineconeSearchEngine:
    """
    Vector search engine using Pinecone.
    
    Same interface as SearchEngine (FAISS-based) for compatibility.
    
    Features:
    - Pinecone vector similarity search
    - Metadata filtering (bloom_level, difficulty, domain, etc.)
    - Contextual result ranking
    """
    
    def __init__(
        self,
        index_name: str = "content-curations-v2",
        chunks_file: Optional[str] = None,
    ):
        """
        Initialize Pinecone search engine.
        
        Args:
            index_name: Pinecone index name
            chunks_file: Optional path to chunks.json for full text lookup
        """
        self.index_name = index_name
        
        # Load Pinecone store
        self._load_pinecone()
        
        # Load chunks for full text if provided
        self.chunks = {}
        if chunks_file:
            self._load_chunks(chunks_file)
        
        # Load embedder
        self._load_embedder()
    
    def _load_pinecone(self):
        """Initialize Pinecone connection."""
        from ..vector_store.pinecone_store import PineconeStore
        
        self.store = PineconeStore(index_name=self.index_name)
        stats = self.store.get_stats()
        logger.info(f"Connected to Pinecone index: {self.index_name}")
        logger.info(f"Total vectors: {stats['total_vectors']}")
    
    def _load_chunks(self, chunks_file: str):
        """Load chunk metadata for full text lookup."""
        chunks_path = Path(chunks_file)
        if chunks_path.exists():
            with open(chunks_path, 'r') as f:
                chunks_list = json.load(f)
            # Index by chunk_id for fast lookup
            self.chunks = {c.get("chunk_id", f"chunk_{i}"): c for i, c in enumerate(chunks_list)}
            logger.info(f"Loaded {len(self.chunks)} chunks for text lookup")
    
    def _load_embedder(self):
        """Load Gemini embedding model for query encoding."""
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set for embeddings")
        
        genai.configure(api_key=api_key)
        self._gemini_model = "models/gemini-embedding-001"
        logger.info(f"Using Gemini embedder: {self._gemini_model}")
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query text using Gemini."""
        import google.generativeai as genai
        
        result = genai.embed_content(
            model=self._gemini_model,
            content=query,
            task_type="retrieval_query",
        )
        return np.array(result['embedding'], dtype=np.float32)
    
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
        
        Same interface as SearchEngine.search() for compatibility.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional metadata filters, e.g.:
                     {"bloom_level": "Apply", "difficulty_level": "BEGINNER"}
            target_domain: Target domain to filter/boost (e.g., "Computer Science")
            min_confidence: Minimum confidence threshold (0-1)
        
        Returns:
            Dict with results, confidence metrics, and domain match info
        """
        # Embed query
        query_embedding = self._embed_query(query)
        
        # Build Pinecone filter
        pinecone_filter = None
        if filters or target_domain:
            pinecone_filter = {}
            
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        pinecone_filter[key] = {"$in": value}
                    else:
                        pinecone_filter[key] = value
            
            if target_domain:
                # Add domain filter
                pinecone_filter["primary_domain"] = target_domain
        
        # Search more than needed for analysis
        search_k = min(top_k * 3, 100)
        
        # Query Pinecone (single query - skip the domain coverage query for performance)
        raw_results = self.store.query(
            query_embedding=query_embedding,
            top_k=search_k,
            filter_dict=pinecone_filter,
            include_metadata=True,
        )
        
        # Use results count for stats (skip expensive second query)
        domain_matched_count = len(raw_results)
        total_candidates = len(raw_results)
        
        # Format results
        formatted_results = []
        for i, r in enumerate(raw_results[:top_k]):
            meta = r.get("metadata", {})
            chunk_id = r.get("id", "")
            
            # Get full text from chunks if available
            full_text = ""
            if chunk_id in self.chunks:
                chunk_data = self.chunks[chunk_id]
                full_text = chunk_data.get("chunk_text", "") or chunk_data.get("text", "")
            else:
                full_text = meta.get("text_preview", "")
            
            # Calculate score (Pinecone returns similarity score 0-1 for cosine)
            score = r.get("score", 0)
            
            # Check domain match
            domain_match = False
            if target_domain:
                chunk_domain = meta.get("primary_domain", "").lower()
                if target_domain.lower() in chunk_domain:
                    domain_match = True
            
            result = {
                "rank": i + 1,
                "score": score,
                "distance": 1 - score,  # Convert similarity to distance for compatibility
                "chunk_id": chunk_id,
                "item_id": meta.get("item_id", chunk_id.rsplit('_c', 1)[0]),
                "item_name": meta.get("item_name", ""),
                "url": "",  # URL not stored in metadata
                "course_name": meta.get("course_name", ""),
                "course_slug": meta.get("course_slug", ""),
                "module_name": meta.get("module_name", ""),
                "lesson_name": meta.get("module_name", ""),  # Fallback
                "content_type": meta.get("content_type", "video"),
                "bloom_level": meta.get("bloom_level"),
                "difficulty": meta.get("difficulty_level"),
                "text_preview": meta.get("text_preview", "")[:200] + "..." if len(meta.get("text_preview", "")) > 200 else meta.get("text_preview", ""),
                "text": full_text,
                "summary": "",
                "derived_metadata": {
                    "primary_domain": meta.get("primary_domain", ""),
                    "sub_domain": meta.get("sub_domain", ""),
                    "bloom_level": meta.get("bloom_level", ""),
                    "cognitive_load": meta.get("cognitive_load", ""),
                    "atomic_skills": meta.get("atomic_skills", []),
                    "key_concepts": meta.get("key_concepts", []),
                },
                "operational_metadata": {
                    "difficulty_level": meta.get("difficulty_level", ""),
                    "star_rating": meta.get("star_rating", 0),
                },
                "domain_match": domain_match,
                "primary_domain": meta.get("primary_domain", ""),
            }
            
            formatted_results.append(result)
        
        # Calculate confidence metrics
        if formatted_results:
            avg_score = sum(r["score"] for r in formatted_results) / len(formatted_results)
            top_score = formatted_results[0]["score"] if formatted_results else 0
            domain_coverage = domain_matched_count / total_candidates if total_candidates > 0 else 0
            confidence = (top_score * 0.5) + (avg_score * 0.3) + (domain_coverage * 0.2)
        else:
            confidence = 0.0
            avg_score = 0.0
            top_score = 0.0
            domain_coverage = 0.0
        
        # Determine confidence level
        if confidence >= 0.6:
            confidence_level = "high"
        elif confidence >= 0.45:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return {
            "results": formatted_results,
            "total_candidates": total_candidates,
            "domain_matched_count": domain_matched_count,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "top_score": top_score,
            "avg_score": avg_score,
            "target_domain": target_domain,
            "domain_coverage": domain_coverage,
            "search_engine": "pinecone",  # Identify which engine was used
        }
    
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
        result = self.search(query, top_k, filters)
        results = result.get("results", [])
        
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
            "search_engine": "pinecone",
        }


def test_pinecone_search():
    """Test the Pinecone search engine."""
    from dotenv import load_dotenv
    
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env")
    
    try:
        engine = PineconeSearchEngine(
            index_name="content-curations-v2",
            chunks_file=str(project_root / "data/test_indexes/diverse_50/index/chunks.json"),
        )
        
        # Test query
        print("=" * 70)
        print("PINECONE SEARCH ENGINE TEST")
        print("=" * 70)
        
        query = "What is a KeyError in Python?"
        print(f"\n🔍 Query: '{query}'")
        
        results = engine.search(query, top_k=5)
        
        print(f"   Found {len(results['results'])} results")
        print(f"   Confidence: {results['confidence']:.3f} ({results['confidence_level']})")
        print(f"   Search engine: {results.get('search_engine', 'unknown')}")
        
        for r in results['results']:
            print(f"\n   {r['rank']}. [{r['score']:.3f}] {r['item_name']}")
            print(f"      Course: {r['course_name']}")
            print(f"      Domain: {r['primary_domain']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pinecone search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_pinecone_search()
