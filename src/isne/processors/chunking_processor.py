"""
Chunking processor for the ISNE pipeline.

This module provides a processor for chunking documents into smaller segments
for more granular processing in the ISNE pipeline.
"""

from typing import Dict, List, Optional, Union, Any, Set, Tuple, Callable, cast
import re
import uuid
import logging
from datetime import datetime

from src.isne.types.models import IngestDocument, IngestDataset, DocumentRelation, RelationType
from src.isne.processors.base_processor import BaseProcessor, ProcessorConfig, ProcessorResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkingProcessor(BaseProcessor):
    """
    Processor for chunking documents into smaller segments.
    
    This processor splits documents into smaller chunks for more granular
    processing and analysis in the ISNE pipeline.
    """
    
    def __init__(
        self,
        processor_config: Optional[ProcessorConfig] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitting_strategy: str = "paragraph",
        preserve_metadata: bool = True,
        create_relationships: bool = True
    ) -> None:
        """
        Initialize the chunking processor.
        
        Args:
            processor_config: Configuration for the processor
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            splitting_strategy: Strategy for splitting content ('paragraph', 'sentence', 'token', 'fixed')
            preserve_metadata: Whether to preserve document metadata in chunks
            create_relationships: Whether to create relationships between chunks and parent documents
        """
        super().__init__(processor_config)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitting_strategy = splitting_strategy.lower()
        self.preserve_metadata = preserve_metadata
        self.create_relationships = create_relationships
    
    def process(
        self, 
        documents: List[IngestDocument],
        relations: Optional[List[DocumentRelation]] = None,
        dataset: Optional[IngestDataset] = None
    ) -> ProcessorResult:
        """
        Process documents by chunking them into smaller segments.
        
        Args:
            documents: List of documents to chunk
            relations: Optional list of relationships between documents
            dataset: Optional dataset containing documents and relationships
            
        Returns:
            ProcessorResult containing chunked documents and relationships
        """
        logger.info(f"Chunking {len(documents)} documents using {self.splitting_strategy} strategy")
        
        chunked_documents: List[IngestDocument] = []
        chunk_relations: List[DocumentRelation] = []
        errors: List[Dict[str, Any]] = []
        
        # Process each document
        for doc in documents:
            try:
                # Skip empty documents
                if not doc.content or not doc.content.strip():
                    chunked_documents.append(doc)
                    continue
                
                # Create chunks based on strategy
                chunks = self._create_chunks(doc)
                
                # If no chunks were created or chunking failed, keep original document
                if not chunks:
                    chunked_documents.append(doc)
                    continue
                
                # Store chunk information in original document
                updated_metadata = doc.metadata.copy()
                updated_metadata["chunk_count"] = len(chunks)
                updated_metadata["chunking_strategy"] = self.splitting_strategy
                updated_metadata["chunk_size"] = self.chunk_size
                updated_metadata["chunk_overlap"] = self.chunk_overlap
                updated_metadata["chunk_ids"] = [chunk.id for chunk in chunks]
                
                updated_doc = IngestDocument(
                    id=doc.id,
                    content=doc.content,
                    source=doc.source,
                    document_type=doc.document_type,
                    title=doc.title,
                    author=doc.author,
                    created_at=doc.created_at,
                    updated_at=datetime.now(),
                    metadata=updated_metadata,
                    embedding=doc.embedding,
                    embedding_model=doc.embedding_model,
                    chunks=[{
                        "id": chunk.id,
                        "content": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                        "metadata": chunk.metadata
                    } for chunk in chunks],
                    tags=doc.tags
                )
                
                # Add updated original document and chunks
                chunked_documents.append(updated_doc)
                chunked_documents.extend(chunks)
                
                # Create relationships between chunks and parent document
                if self.create_relationships:
                    for chunk in chunks:
                        relation = DocumentRelation(
                            source_id=doc.id,
                            target_id=chunk.id,
                            relation_type=RelationType.CONTAINS,
                            weight=1.0,
                            bidirectional=False,
                            metadata={
                                "relationship_source": "document_chunking",
                                "chunk_index": chunk.metadata.get("chunk_index", 0)
                            }
                        )
                        chunk_relations.append(relation)
                
            except Exception as e:
                logger.error(f"Error chunking document {doc.id}: {e}")
                errors.append({
                    "document_id": doc.id,
                    "error": str(e),
                    "type": type(e).__name__
                })
                chunked_documents.append(doc)
        
        # Combine existing and new relationships
        combined_relations = list(relations or []) + chunk_relations
        
        # Create updated dataset if provided
        updated_dataset = None
        if dataset:
            updated_dataset = IngestDataset(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                metadata={
                    **dataset.metadata,
                    "chunking_applied": True,
                    "chunking_strategy": self.splitting_strategy,
                    "original_document_count": len(documents),
                    "chunked_document_count": len(chunked_documents),
                    "chunk_relation_count": len(chunk_relations)
                },
                created_at=dataset.created_at,
                updated_at=datetime.now()
            )
            
            # Add chunked documents
            for doc in chunked_documents:
                updated_dataset.add_document(doc)
            
            # Add relationships
            for rel in combined_relations:
                updated_dataset.add_relation(rel)
        
        logger.info(f"Chunking completed. Original documents: {len(documents)}, "
                   f"After chunking: {len(chunked_documents)}")
        
        return ProcessorResult(
            documents=chunked_documents,
            relations=combined_relations,
            dataset=updated_dataset,
            errors=errors,
            metadata={
                "processor": "ChunkingProcessor",
                "strategy": self.splitting_strategy,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "original_document_count": len(documents),
                "chunked_document_count": len(chunked_documents),
                "chunk_relation_count": len(chunk_relations),
                "error_count": len(errors)
            }
        )
    
    def _create_chunks(self, document: IngestDocument) -> List[IngestDocument]:
        """
        Create chunks from a document based on the configured strategy.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunk documents
        """
        content = document.content
        
        # Select splitting function based on strategy
        if self.splitting_strategy == "paragraph":
            chunks_text = self._split_by_paragraph(content)
        elif self.splitting_strategy == "sentence":
            chunks_text = self._split_by_sentence(content)
        elif self.splitting_strategy == "token":
            chunks_text = self._split_by_token(content)
        elif self.splitting_strategy == "fixed":
            chunks_text = self._split_fixed_size(content)
        else:
            # Default to paragraph splitting
            chunks_text = self._split_by_paragraph(content)
        
        # Create chunk documents
        chunk_docs: List[IngestDocument] = []
        
        for i, chunk_text in enumerate(chunks_text):
            # Skip empty chunks
            if not chunk_text.strip():
                continue
            
            # Create chunk metadata
            chunk_metadata = {
                "chunk_index": i,
                "chunk_count": len(chunks_text),
                "parent_document_id": document.id,
                "parent_document_title": document.title,
                "chunking_strategy": self.splitting_strategy
            }
            
            # Add parent metadata if requested
            if self.preserve_metadata:
                for key, value in document.metadata.items():
                    if key not in chunk_metadata:
                        chunk_metadata[f"parent_{key}"] = value
            
            # Create chunk title
            chunk_title = None
            if document.title:
                chunk_title = f"{document.title} (Chunk {i+1}/{len(chunks_text)})"
            
            # Create chunk document
            chunk_doc = IngestDocument(
                id=f"{document.id}_chunk_{i}_{str(uuid.uuid4())[:8]}",
                content=chunk_text,
                source=f"{document.source}#chunk{i+1}",
                document_type=f"{document.document_type}_chunk",
                title=chunk_title,
                author=document.author,
                created_at=document.created_at,
                updated_at=datetime.now(),
                metadata=chunk_metadata,
                embedding=None,  # Chunks need to be embedded separately
                embedding_model=None,
                tags=document.tags + ["chunk"]
            )
            
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def _split_by_paragraph(self, content: str) -> List[str]:
        """
        Split content by paragraphs with overlapping.
        
        Args:
            content: Text content to split
            
        Returns:
            List of paragraph chunks
        """
        # Identify paragraph boundaries
        paragraphs = re.split(r'\n\s*\n', content)
        
        # Filter empty paragraphs
        paragraphs = [p for p in paragraphs if p.strip()]
        
        # If no paragraphs or single paragraph, use fixed size splitting
        if not paragraphs or (len(paragraphs) == 1 and len(paragraphs[0]) > self.chunk_size):
            return self._split_fixed_size(content)
        
        # Create chunks from paragraphs
        chunks: List[str] = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, start a new chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap from previous chunk
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    # Get last part of previous chunk for overlap
                    words = current_chunk.split()
                    overlap_word_count = len(' '.join(words[-30:]))  # Approximate overlap by word count
                    
                    if overlap_word_count <= self.chunk_overlap:
                        current_chunk = current_chunk[-self.chunk_overlap:]
                    else:
                        # If overlap is too large, just take the last N characters
                        current_chunk = current_chunk[-self.chunk_overlap:]
                else:
                    current_chunk = ""
            
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_sentence(self, content: str) -> List[str]:
        """
        Split content by sentences with overlapping.
        
        Args:
            content: Text content to split
            
        Returns:
            List of sentence-based chunks
        """
        # Identify sentence boundaries (simplified pattern)
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, content)
        
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Create chunks from sentences
        chunks: List[str] = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds chunk size, start a new chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Get overlap for next chunk
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    current_chunk = current_chunk[-self.chunk_overlap:]
                else:
                    current_chunk = ""
            
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_token(self, content: str) -> List[str]:
        """
        Split content by tokens (words) with overlapping.
        
        Args:
            content: Text content to split
            
        Returns:
            List of token-based chunks
        """
        # Split by words/tokens
        tokens = content.split()
        
        # Estimate characters per token
        avg_token_len = sum(len(t) for t in tokens) / max(1, len(tokens))
        tokens_per_chunk = int(self.chunk_size / (avg_token_len + 1))  # +1 for spaces
        overlap_tokens = int(self.chunk_overlap / (avg_token_len + 1))
        
        # Create chunks
        chunks: List[str] = []
        
        for i in range(0, len(tokens), tokens_per_chunk - overlap_tokens):
            chunk_tokens = tokens[i:i + tokens_per_chunk]
            if chunk_tokens:
                chunks.append(" ".join(chunk_tokens))
        
        return chunks
    
    def _split_fixed_size(self, content: str) -> List[str]:
        """
        Split content into fixed size chunks with overlapping.
        
        Args:
            content: Text content to split
            
        Returns:
            List of fixed-size chunks
        """
        chunks: List[str] = []
        
        # Create chunks of fixed size with overlap
        for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
            chunk = content[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        
        return chunks
