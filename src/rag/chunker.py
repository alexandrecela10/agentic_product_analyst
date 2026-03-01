"""
Semantic chunking for RAG pipeline.

Chunks documents by structure (headers, paragraphs) rather than fixed size.
This preserves meaning and context within each chunk.

Why semantic chunking?
- Fixed-size chunks can split sentences mid-thought
- Semantic chunks respect document structure
- Better retrieval precision because chunks are coherent units
"""

import re
from dataclasses import dataclass
from typing import Optional
from src.core.config import settings


@dataclass
class Chunk:
    """
    A single chunk of text with metadata.
    
    Why metadata?
    - Track which document the chunk came from
    - Know the section/header for context
    - Enable filtering during retrieval
    """
    content: str                    # The actual text content
    document_id: str                # Source document identifier
    document_name: str              # Human-readable document name
    section_title: Optional[str]    # Header/section this chunk belongs to
    chunk_index: int                # Position in the document (0-indexed)
    start_char: int                 # Character offset in original document
    end_char: int                   # End character offset


class SemanticChunker:
    """
    Chunks markdown documents by structure.
    
    Strategy:
    1. Split by headers (##, ###, etc.)
    2. Within sections, split by paragraphs if too long
    3. Merge small chunks to avoid tiny fragments
    
    Why this approach?
    - Headers indicate topic boundaries
    - Paragraphs are natural thought units
    - Merging prevents retrieval of useless tiny chunks
    """
    
    def __init__(
        self,
        max_chunk_size: int = None,
        min_chunk_size: int = 100,
        overlap: int = None
    ):
        """
        Initialize the chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk (default from settings)
            min_chunk_size: Minimum characters per chunk (merge smaller ones)
            overlap: Characters to overlap between chunks (default from settings)
        """
        self.max_chunk_size = max_chunk_size or settings.chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap or settings.chunk_overlap
    
    def chunk_document(
        self,
        content: str,
        document_id: str,
        document_name: str
    ) -> list[Chunk]:
        """
        Chunk a markdown document into semantic units.
        
        Args:
            content: The full document text
            document_id: Unique identifier for the document
            document_name: Human-readable name
            
        Returns:
            List of Chunk objects
            
        How it works:
        1. Find all headers and their positions
        2. Split content into sections based on headers
        3. Further split large sections by paragraphs
        4. Merge small adjacent chunks
        """
        chunks = []
        
        # Step 1: Find all markdown headers (## Header, ### Header, etc.)
        # Pattern matches lines starting with 1-6 # characters followed by space
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        # Find all headers with their positions
        headers = []
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            headers.append({
                'level': len(match.group(1)),  # Number of # characters
                'title': match.group(2).strip(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Step 2: Split content into sections
        sections = []
        
        if not headers:
            # No headers found - treat entire document as one section
            sections.append({
                'title': None,
                'content': content,
                'start': 0,
                'end': len(content)
            })
        else:
            # Add content before first header (if any)
            if headers[0]['start'] > 0:
                sections.append({
                    'title': None,
                    'content': content[:headers[0]['start']].strip(),
                    'start': 0,
                    'end': headers[0]['start']
                })
            
            # Add each header section
            for i, header in enumerate(headers):
                # Section ends at next header or end of document
                if i + 1 < len(headers):
                    section_end = headers[i + 1]['start']
                else:
                    section_end = len(content)
                
                section_content = content[header['end']:section_end].strip()
                
                sections.append({
                    'title': header['title'],
                    'content': section_content,
                    'start': header['start'],
                    'end': section_end
                })
        
        # Step 3: Process each section into chunks
        chunk_index = 0
        for section in sections:
            section_chunks = self._chunk_section(
                section['content'],
                section['title'],
                section['start']
            )
            
            for chunk_content, start_offset in section_chunks:
                chunks.append(Chunk(
                    content=chunk_content,
                    document_id=document_id,
                    document_name=document_name,
                    section_title=section['title'],
                    chunk_index=chunk_index,
                    start_char=section['start'] + start_offset,
                    end_char=section['start'] + start_offset + len(chunk_content)
                ))
                chunk_index += 1
        
        # Step 4: Merge small chunks
        chunks = self._merge_small_chunks(chunks)
        
        return chunks
    
    def _chunk_section(
        self,
        content: str,
        section_title: Optional[str],
        base_offset: int
    ) -> list[tuple[str, int]]:
        """
        Split a section into chunks if it's too large.
        
        Args:
            content: Section text
            section_title: Header of this section (for context)
            base_offset: Character offset of section start
            
        Returns:
            List of (chunk_content, offset) tuples
            
        Strategy:
        - If section fits in max_chunk_size, return as-is
        - Otherwise, split by paragraphs (double newlines)
        - If paragraphs are still too large, split by sentences
        """
        if not content.strip():
            return []
        
        # Include section title in chunk for context
        if section_title:
            prefix = f"## {section_title}\n\n"
        else:
            prefix = ""
        
        full_content = prefix + content
        
        # If small enough, return as single chunk
        if len(full_content) <= self.max_chunk_size:
            return [(full_content, 0)]
        
        # Split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\n+', content)
        
        chunks = []
        current_chunk = prefix
        current_offset = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds max size
            if len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                # Save current chunk if it has content
                if current_chunk.strip() and current_chunk != prefix:
                    chunks.append((current_chunk.strip(), current_offset))
                
                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk:
                    # Take last N characters as overlap
                    overlap_text = current_chunk[-self.overlap:]
                    current_chunk = prefix + overlap_text + "\n\n" + para
                else:
                    current_chunk = prefix + para
                
                current_offset = len(content) - len(para) - len(prefix)
            else:
                # Add paragraph to current chunk
                if current_chunk and current_chunk != prefix:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = prefix + para
        
        # Don't forget the last chunk
        if current_chunk.strip() and current_chunk != prefix:
            chunks.append((current_chunk.strip(), current_offset))
        
        return chunks
    
    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Merge chunks that are too small.
        
        Why merge?
        - Tiny chunks (< min_chunk_size) are often not useful for retrieval
        - Merging adjacent small chunks improves context
        - Reduces total number of embeddings to compute
        """
        if not chunks:
            return chunks
        
        merged = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            # If current chunk is small and same document, merge
            if (len(current.content) < self.min_chunk_size and 
                current.document_id == next_chunk.document_id):
                
                # Merge chunks
                current = Chunk(
                    content=current.content + "\n\n" + next_chunk.content,
                    document_id=current.document_id,
                    document_name=current.document_name,
                    section_title=current.section_title or next_chunk.section_title,
                    chunk_index=current.chunk_index,
                    start_char=current.start_char,
                    end_char=next_chunk.end_char
                )
            else:
                merged.append(current)
                current = next_chunk
        
        merged.append(current)
        
        return merged


def chunk_markdown_file(
    file_path: str,
    document_id: Optional[str] = None
) -> list[Chunk]:
    """
    Convenience function to chunk a markdown file.
    
    Args:
        file_path: Path to the markdown file
        document_id: Optional ID (defaults to filename)
        
    Returns:
        List of chunks from the file
    """
    from pathlib import Path
    
    path = Path(file_path)
    content = path.read_text(encoding='utf-8')
    
    doc_id = document_id or path.stem
    doc_name = path.name
    
    chunker = SemanticChunker()
    return chunker.chunk_document(content, doc_id, doc_name)
