"""
Document indexer for the RAG pipeline.

Handles loading documents from the knowledge base and indexing them
into the vector store.

Why a separate indexer?
- Separates document loading from storage
- Handles different file formats (markdown, txt, etc.)
- Provides batch indexing with progress tracking
"""

from pathlib import Path
from typing import Optional
from src.rag.chunker import SemanticChunker, Chunk
from src.rag.vector_store import VectorStore, get_vector_store
from src.rag.retriever import get_retriever
from src.core.observability import get_logger

logger = get_logger(__name__)


class DocumentIndexer:
    """
    Indexes documents from the knowledge base into the vector store.
    
    Workflow:
    1. Scan knowledge_base directory for documents
    2. Chunk each document using SemanticChunker
    3. Add chunks to VectorStore
    4. Refresh BM25 index for hybrid search
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        knowledge_base_path: str = "./knowledge_base"
    ):
        """
        Initialize the indexer.
        
        Args:
            vector_store: Vector store instance (default: singleton)
            knowledge_base_path: Path to knowledge base directory
        """
        self.vector_store = vector_store or get_vector_store()
        self.knowledge_base_path = Path(knowledge_base_path)
        self.chunker = SemanticChunker()
        
        # Supported file extensions
        self.supported_extensions = {'.md', '.txt', '.markdown'}
    
    def index_file(self, file_path: Path) -> int:
        """
        Index a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Number of chunks indexed
        """
        if file_path.suffix.lower() not in self.supported_extensions:
            logger.warning("unsupported_file_type", path=str(file_path))
            return 0
        
        # Read file content
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error("file_read_error", path=str(file_path), error=str(e))
            return 0
        
        # Generate document ID from path
        # Use relative path from knowledge_base as ID
        try:
            relative_path = file_path.relative_to(self.knowledge_base_path)
            document_id = str(relative_path).replace('/', '_').replace('\\', '_')
        except ValueError:
            document_id = file_path.stem
        
        # Delete existing chunks for this document (in case of re-index)
        deleted = self.vector_store.delete_document(document_id)
        if deleted > 0:
            logger.info("deleted_existing_chunks", document_id=document_id, count=deleted)
        
        # Chunk the document
        chunks = self.chunker.chunk_document(
            content=content,
            document_id=document_id,
            document_name=file_path.name
        )
        
        if not chunks:
            logger.warning("no_chunks_generated", path=str(file_path))
            return 0
        
        # Add to vector store
        added = self.vector_store.add_chunks(chunks)
        
        logger.info("indexed_file", 
                   path=str(file_path), 
                   document_id=document_id,
                   chunks=added)
        
        return added
    
    def index_directory(self, directory: Optional[Path] = None) -> dict:
        """
        Index all documents in a directory.
        
        Args:
            directory: Directory to index (default: knowledge_base_path)
            
        Returns:
            Dict with indexing statistics
        """
        dir_path = directory or self.knowledge_base_path
        
        if not dir_path.exists():
            logger.error("directory_not_found", path=str(dir_path))
            return {"error": "Directory not found", "files": 0, "chunks": 0}
        
        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        # Find all supported files recursively
        for ext in self.supported_extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                # Skip hidden files and directories
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                
                try:
                    chunks = self.index_file(file_path)
                    if chunks > 0:
                        stats["files_processed"] += 1
                        stats["total_chunks"] += chunks
                    else:
                        stats["files_skipped"] += 1
                except Exception as e:
                    stats["errors"].append({
                        "file": str(file_path),
                        "error": str(e)
                    })
                    logger.error("indexing_error", path=str(file_path), error=str(e))
        
        # Refresh BM25 index after adding documents
        retriever = get_retriever()
        retriever.refresh_index()
        
        logger.info("indexing_complete", **stats)
        
        return stats
    
    def index_all(self) -> dict:
        """
        Index the entire knowledge base.
        
        Convenience method that indexes both company_docs and industry_standards.
        """
        return self.index_directory(self.knowledge_base_path)


def index_knowledge_base(knowledge_base_path: str = "./knowledge_base") -> dict:
    """
    Convenience function to index the knowledge base.
    
    Args:
        knowledge_base_path: Path to knowledge base directory
        
    Returns:
        Indexing statistics
    """
    indexer = DocumentIndexer(knowledge_base_path=knowledge_base_path)
    return indexer.index_all()
