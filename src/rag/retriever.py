"""
Hybrid retriever combining keyword (BM25) and semantic search.

Why hybrid?
- Keywords alone miss synonyms ("AI search" vs "intelligent search")
- Vectors alone can return semantically similar but irrelevant content
- Hybrid gets the best of both: precision from keywords + recall from semantics

How it works:
1. BM25 (keyword) search filters to candidate chunks
2. Semantic search ranks by meaning
3. Reciprocal Rank Fusion combines both scores
"""

from dataclasses import dataclass
from typing import Optional
from rank_bm25 import BM25Okapi
from src.rag.vector_store import VectorStore, get_vector_store
from src.rag.chunker import Chunk


@dataclass
class RetrievalResult:
    """
    A single retrieval result with combined scoring.
    
    Why this structure?
    - content: The actual text to use in RAG
    - metadata: Source info for citations
    - score: Combined relevance score for ranking
    - source: Which method found this (for debugging)
    """
    content: str
    metadata: dict
    score: float
    source: str  # "semantic", "keyword", or "hybrid"


class HybridRetriever:
    """
    Combines BM25 keyword search with semantic vector search.
    
    Strategy:
    1. Semantic search: Find top-K similar chunks by embedding
    2. BM25 search: Find top-K matching chunks by keywords
    3. Reciprocal Rank Fusion: Combine rankings into final score
    
    Why RRF (Reciprocal Rank Fusion)?
    - Simple but effective fusion method
    - Doesn't require score normalization
    - Works well when combining different ranking methods
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        rrf_k: int = 60
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store: The vector store for semantic search
            semantic_weight: Weight for semantic results (0-1)
            keyword_weight: Weight for keyword results (0-1)
            rrf_k: RRF constant (higher = more weight to lower ranks)
            
        Why these default weights?
        - Semantic search is generally more powerful for understanding intent
        - Keywords are important for exact matches (feature names, acronyms)
        - 60/40 split balances both approaches
        """
        self.vector_store = vector_store or get_vector_store()
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        
        # BM25 index - built lazily when needed
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_docs: list[dict] = []
    
    def _build_bm25_index(self):
        """
        Build BM25 index from all documents in vector store.
        
        Why rebuild?
        - BM25 needs all documents in memory
        - We rebuild when documents change
        - For large collections, consider incremental updates
        """
        # Get all documents from vector store
        all_docs = self.vector_store.collection.get(
            include=["documents", "metadatas"]
        )
        
        if not all_docs or not all_docs['documents']:
            self._bm25_index = None
            self._bm25_docs = []
            return
        
        # Tokenize documents for BM25
        # Simple whitespace tokenization - could use better tokenizer
        tokenized = []
        self._bm25_docs = []
        
        for i, doc in enumerate(all_docs['documents']):
            # Lowercase and split by whitespace
            tokens = doc.lower().split()
            tokenized.append(tokens)
            
            self._bm25_docs.append({
                "content": doc,
                "metadata": all_docs['metadatas'][i] if all_docs['metadatas'] else {},
                "id": all_docs['ids'][i] if all_docs['ids'] else str(i)
            })
        
        # Build BM25 index
        self._bm25_index = BM25Okapi(tokenized)
    
    def _keyword_search(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        """
        Search using BM25 keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (doc_dict, score) tuples
        """
        # Rebuild index if needed
        if self._bm25_index is None:
            self._build_bm25_index()
        
        if self._bm25_index is None or not self._bm25_docs:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self._bm25_index.get_scores(query_tokens)
        
        # Get top-K indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Return docs with scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                results.append((self._bm25_docs[idx], scores[idx]))
        
        return results
    
    def _semantic_search(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        """
        Search using semantic vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (doc_dict, score) tuples
        """
        results = self.vector_store.search(query, top_k=top_k)
        
        # Convert distance to similarity score
        # ChromaDB returns L2 distance, lower is better
        # Convert to similarity: 1 / (1 + distance)
        formatted = []
        for r in results:
            similarity = 1 / (1 + r['distance'])
            formatted.append((
                {"content": r['content'], "metadata": r['metadata']},
                similarity
            ))
        
        return formatted
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[tuple[dict, float]],
        keyword_results: list[tuple[dict, float]]
    ) -> list[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF formula: score = sum(1 / (k + rank)) for each ranking
        
        Why RRF?
        - Doesn't require score normalization
        - Robust to different score scales
        - Simple and effective
        """
        # Build score map: content -> (rrf_score, metadata, sources)
        score_map = {}
        
        # Process semantic results
        for rank, (doc, _) in enumerate(semantic_results):
            content = doc['content']
            rrf_score = self.semantic_weight / (self.rrf_k + rank + 1)
            
            if content not in score_map:
                score_map[content] = {
                    'score': 0,
                    'metadata': doc['metadata'],
                    'sources': set()
                }
            
            score_map[content]['score'] += rrf_score
            score_map[content]['sources'].add('semantic')
        
        # Process keyword results
        for rank, (doc, _) in enumerate(keyword_results):
            content = doc['content']
            rrf_score = self.keyword_weight / (self.rrf_k + rank + 1)
            
            if content not in score_map:
                score_map[content] = {
                    'score': 0,
                    'metadata': doc['metadata'],
                    'sources': set()
                }
            
            score_map[content]['score'] += rrf_score
            score_map[content]['sources'].add('keyword')
        
        # Convert to results and sort
        results = []
        for content, data in score_map.items():
            sources = data['sources']
            if len(sources) > 1:
                source = 'hybrid'
            else:
                source = list(sources)[0]
            
            results.append(RetrievalResult(
                content=content,
                metadata=data['metadata'],
                score=data['score'],
                source=source
            ))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search (True) or semantic only (False)
            
        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        if use_hybrid:
            # Get results from both methods
            # Fetch more than top_k from each to allow for deduplication
            semantic_results = self._semantic_search(query, top_k * 2)
            keyword_results = self._keyword_search(query, top_k * 2)
            
            # Combine using RRF
            combined = self._reciprocal_rank_fusion(semantic_results, keyword_results)
            
            return combined[:top_k]
        else:
            # Semantic only
            semantic_results = self._semantic_search(query, top_k)
            
            return [
                RetrievalResult(
                    content=doc['content'],
                    metadata=doc['metadata'],
                    score=score,
                    source='semantic'
                )
                for doc, score in semantic_results
            ]
    
    def refresh_index(self):
        """
        Force rebuild of BM25 index.
        
        Call this after adding/removing documents.
        """
        self._bm25_index = None
        self._build_bm25_index()


# Singleton instance
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Get the singleton retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
