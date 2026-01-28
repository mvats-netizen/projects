"""
Pinecone Vector Store

Wrapper for Pinecone client to store and query embeddings.
Used for V2 exploration while keeping FAISS-based V1 unchanged.

Usage:
    >>> store = PineconeStore(index_name="content-curations-v2")
    >>> store.upsert_vectors(ids, embeddings, metadata_list)
    >>> results = store.query(query_embedding, top_k=10)
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PineconeStore:
    """
    Pinecone vector store for content curation embeddings.
    
    Attributes:
        index_name: Name of the Pinecone index
        dimension: Vector dimension (3072 for Gemini embeddings)
        metric: Distance metric (cosine)
    """
    
    def __init__(
        self,
        index_name: str = "content-curations-v2",
        dimension: int = 3072,
        metric: str = "cosine",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Pinecone connection.
        
        Args:
            index_name: Name of the index to use/create
            dimension: Vector dimension (default: 3072 for Gemini)
            metric: Distance metric (default: cosine)
            api_key: Pinecone API key (or use PINECONE_API_KEY env var)
        """
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        # Get API key
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Pinecone API key not found. "
                "Set PINECONE_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize Pinecone
        self._init_pinecone()
    
    def _init_pinecone(self):
        """Initialize Pinecone client and index."""
        from pinecone import Pinecone, ServerlessSpec
        
        # Create Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info(f"Index '{self.index_name}' created successfully")
        else:
            logger.info(f"Using existing Pinecone index: {self.index_name}")
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
        
        # Get index stats
        stats = self.index.describe_index_stats()
        logger.info(f"Index stats: {stats.total_vector_count} vectors")
    
    def upsert_vectors(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        batch_size: int = 100,
        namespace: str = "",
    ) -> int:
        """
        Upsert vectors with metadata to Pinecone.
        
        Args:
            ids: List of unique vector IDs
            embeddings: Numpy array of embeddings (n_vectors x dimension)
            metadata_list: List of metadata dicts for each vector
            batch_size: Number of vectors per upsert batch
            namespace: Optional namespace for organization
            
        Returns:
            Number of vectors upserted
        """
        if len(ids) != len(embeddings) or len(ids) != len(metadata_list):
            raise ValueError("ids, embeddings, and metadata_list must have same length")
        
        total_upserted = 0
        
        # Process in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size]
            
            # Prepare vectors for upsert
            vectors = []
            for vid, emb, meta in zip(batch_ids, batch_embeddings, batch_metadata):
                # Filter out None values and ensure metadata values are valid types
                clean_meta = {}
                for k, v in meta.items():
                    if v is not None:
                        # Pinecone metadata supports: str, int, float, bool, list of str
                        if isinstance(v, (str, int, float, bool)):
                            clean_meta[k] = v
                        elif isinstance(v, list):
                            # Convert list items to strings
                            clean_meta[k] = [str(item) for item in v]
                        else:
                            clean_meta[k] = str(v)
                
                vectors.append({
                    "id": vid,
                    "values": emb.tolist() if isinstance(emb, np.ndarray) else emb,
                    "metadata": clean_meta
                })
            
            # Upsert batch
            self.index.upsert(vectors=vectors, namespace=namespace)
            total_upserted += len(vectors)
            
            if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(ids):
                logger.info(f"Upserted {total_upserted}/{len(ids)} vectors")
        
        return total_upserted
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone for similar vectors.
        
        Args:
            query_embedding: Query vector (dimension,)
            top_k: Number of results to return
            filter_dict: Metadata filter (e.g., {"primary_domain": "Computer Science"})
            namespace: Optional namespace to query
            include_metadata: Whether to return metadata with results
            
        Returns:
            List of results with id, score, and metadata
        """
        # Convert numpy array to list
        query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Flatten if needed (handle (1, dim) shape)
        if isinstance(query_vector[0], list):
            query_vector = query_vector[0]
        
        # Query Pinecone
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter_dict,
            namespace=namespace,
            include_metadata=include_metadata,
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata if include_metadata else {},
            })
        
        return formatted_results
    
    def delete_all(self, namespace: str = ""):
        """Delete all vectors in the index (or namespace)."""
        self.index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Deleted all vectors from index '{self.index_name}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
        }


def test_pinecone_store():
    """Test Pinecone store connection."""
    try:
        store = PineconeStore(index_name="content-curations-v2-test")
        stats = store.get_stats()
        print(f"✅ Pinecone connection successful!")
        print(f"   Index: {store.index_name}")
        print(f"   Total vectors: {stats['total_vectors']}")
        print(f"   Dimension: {stats['dimension']}")
        return True
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        return False


if __name__ == "__main__":
    test_pinecone_store()
