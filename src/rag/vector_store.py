"""
Vector store for RAG using ChromaDB.

ChromaDB stores document embeddings and enables semantic similarity search.
We use it because it's lightweight, runs locally, and requires no external services.

Why ChromaDB?
- No external dependencies (runs in-process)
- Persistent storage (survives restarts)
- Built-in embedding support (but we use Gemini embeddings)
- Good for development and small-to-medium scale
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Optional
from pathlib import Path
from src.core.config import settings
from src.core.llm_client import embed_text, embed_query
from src.core.observability import get_langfuse, get_logger
from src.rag.chunker import Chunk

logger = get_logger(__name__)

class VectorStore:
    """
    Wrapper around ChromaDB for document storage and retrieval.
    
    How it works:
    1. Documents are chunked and embedded using Gemini
    2. Embeddings are stored in ChromaDB with metadata
    3. Queries are embedded and compared to stored embeddings
    4. Most similar chunks are returned
    """
    
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Where to store the database (default from settings)
            
        Why persist?
        - Don't re-embed documents on every restart
        - Embeddings are expensive (API calls)
        - Persistence makes the system production-ready
        """
        persist_dir = persist_directory or settings.chroma_persist_dir
        
        # Ensure directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False  # Disable telemetry for privacy
            )
        )
        
        # Get or create the collection
        # We don't use ChromaDB's built-in embedding - we use Gemini
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Product Success Tracking Agent knowledge base"}
        )
        
        self.collection_name = collection_name
    
    def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects to add
            
        Returns:
            Number of chunks added
            
        How it works:
        1. Generate embeddings for each chunk using Gemini
        2. Store embeddings with metadata in ChromaDB
        3. Metadata enables filtering during retrieval
        """
        if not chunks:
            return 0
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            # Create unique ID for this chunk
            chunk_id = f"{chunk.document_id}_{chunk.chunk_index}"
            
            # Generate embedding using Gemini
            embedding = embed_text(chunk.content)
            
            # Prepare metadata for filtering
            metadata = {
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "section_title": chunk.section_title or "",
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char
            }
            
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk.content)
            metadatas.append(metadata)
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def search(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Optional[dict] = None
    ) -> list[dict]:
        """
        Search for similar chunks.
        
        Args:
            query: The search query
            top_k: Number of results to return (default from settings)
            filter_metadata: Optional metadata filter (e.g., {"document_id": "prd"})
            
        Returns:
            List of dicts with 'content', 'metadata', 'distance'
            
        How it works:
        1. Embed the query using Gemini (task_type="retrieval_query")
        2. Find nearest neighbors in vector space
        3. Return chunks sorted by similarity
        """
        k = top_k or settings.retrieval_top_k
        
        # Embed the query
        query_embedding = embed_query(query)
        
        # Build query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        # Add filter if provided
        if filter_metadata:
            query_params["where"] = filter_metadata
        
        # Execute search
        results = self.collection.query(**query_params)
        
        # Format results
        formatted = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
        
        return formatted
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks from a document.
        
        Args:
            document_id: The document to delete
            
        Returns:
            Number of chunks deleted
            
        Why delete?
        - Document was updated, need to re-index
        - Document was removed from knowledge base
        """
        # Find all chunks with this document_id
        results = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"]
        )
        
        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])
        
        return 0
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict with count, documents, etc.
        """
        count = self.collection.count()
        
        # Get unique documents
        all_metadata = self.collection.get(include=["metadatas"])
        documents = set()
        if all_metadata and all_metadata['metadatas']:
            for meta in all_metadata['metadatas']:
                documents.add(meta.get('document_id', 'unknown'))
        
        return {
            "total_chunks": count,
            "unique_documents": len(documents),
            "document_ids": list(documents),
            "collection_name": self.collection_name
        }
    
    def clear(self):
        """
        Clear all data from the collection.
        
        Warning: This is destructive! Use with caution.
        """
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Product Success Tracking Agent knowledge base"}
        )


# Singleton instance for convenience
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get the singleton vector store instance.
    
    Why singleton?
    - Only need one connection to ChromaDB
    - Reuse across all RAG operations
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
