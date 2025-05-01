"""
Semantic chunking processor using Chonky for the ISNE pipeline.

This module provides a processor that uses the Chonky neural chunking approach
to segment documents into semantically meaningful chunks for more effective
processing in the ISNE pipeline.
"""

from typing import Dict, List, Optional, Union, Any, Set, Tuple, Callable, cast
import re
import uuid
import logging
from datetime import datetime

from chonky import ParagraphSplitter
from src.isne.types.models import IngestDocument, IngestDataset, DocumentRelation, RelationType
from src.isne.processors.base_processor import BaseProcessor, ProcessorConfig, ProcessorResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChonkyProcessor(BaseProcessor):
    """
    Processor for semantic chunking using Chonky.
    
    This processor uses a fine-tuned transformer model to intelligently split
    documents into semantically meaningful chunks, providing better context
    preservation than traditional character or token-based chunking methods.
    """
    
    def __init__(
        self,
        processor_config: Optional[ProcessorConfig] = None,
        model_id: str = "mirth/chonky_distilbert_uncased_1",
        device: str = "cpu",
        preserve_metadata: bool = True,
        create_relationships: bool = True,
        text_only: bool = True
    ) -> None:
        """
        Initialize the Chonky semantic chunking processor.
        
        Args:
            processor_config: Configuration for the processor
            model_id: ID of the Chonky model to use
            device: Device to run the model on ('cpu' or 'cuda')
            preserve_metadata: Whether to preserve document metadata in chunks
            create_relationships: Whether to create relationships between chunks and parent documents
            text_only: Whether to only apply Chonky to text documents (not code)
        """
        super().__init__(processor_config)
        
        self.model_id = model_id
        self.device = device
        self.preserve_metadata = preserve_metadata
        self.create_relationships = create_relationships
        self.text_only = text_only
        
        logger.info(f"Initializing Chonky semantic chunker with model {model_id} on {device}")
        try:
            # Initialize the Chonky paragraph splitter
            self.splitter = ParagraphSplitter(model_id=model_id, device=device)
            logger.info("Successfully initialized Chonky splitter")
        except Exception as e:
            logger.error(f"Error initializing Chonky splitter: {e}")
            self.splitter = None
    
    def process(
        self, 
        documents: List[IngestDocument],
        relations: Optional[List[DocumentRelation]] = None,
        dataset: Optional[IngestDataset] = None
    ) -> ProcessorResult:
        """
        Process documents using Chonky semantic chunking.
        
        Args:
            documents: List of documents to chunk
            relations: Optional list of relationships between documents
            dataset: Optional dataset containing documents and relationships
            
        Returns:
            ProcessorResult containing semantically chunked documents and relationships
        """
        logger.info(f"Semantically chunking {len(documents)} documents using Chonky")
        
        if not self.splitter:
            logger.error("Chonky splitter not initialized, skipping chunking")
            return ProcessorResult(
                documents=documents,
                relations=relations or [],
                dataset=dataset,
                errors=[{"error": "Chonky splitter not initialized"}],
                metadata={"status": "error", "processor": "ChonkyProcessor"}
            )
        
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
                
                # Skip code documents if text_only is True
                if self.text_only and doc.document_type in ["code", "python", "javascript", "java", "cpp", "c", "go", "rust"]:
                    chunked_documents.append(doc)
                    continue
                
                # Create chunks using Chonky
                chunks = self._create_semantic_chunks(doc)
                
                # If no chunks were created or chunking failed, keep original document
                if not chunks:
                    chunked_documents.append(doc)
                    continue
                
                # Store chunk information in original document
                updated_metadata = doc.metadata.copy() if doc.metadata else {}
                updated_metadata["chunk_count"] = len(chunks)
                updated_metadata["chunking_strategy"] = "chonky_semantic"
                updated_metadata["chunking_model"] = self.model_id
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
                        # Create parent-child relationship
                        relation = DocumentRelation(
                            source_id=doc.id,
                            target_id=chunk.id,
                            relation_type=RelationType.CONTAINS,
                            weight=1.0,
                            bidirectional=False,
                            metadata={
                                "relationship_source": "semantic_chunking",
                                "chunk_index": chunk.metadata.get("chunk_index", 0)
                            }
                        )
                        chunk_relations.append(relation)
                        
                        # Create sequential relationships between chunks
                        if chunks.index(chunk) < len(chunks) - 1:
                            next_chunk = chunks[chunks.index(chunk) + 1]
                            sequence_relation = DocumentRelation(
                                source_id=chunk.id,
                                target_id=next_chunk.id,
                                relation_type=RelationType.RELATED_TO,
                                weight=1.0,
                                bidirectional=False,
                                metadata={
                                    "relationship_source": "semantic_chunking",
                                    "sequential": True
                                }
                            )
                            chunk_relations.append(sequence_relation)
                
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
                    **(dataset.metadata or {}),
                    "semantic_chunking_applied": True,
                    "chunking_model": self.model_id,
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
        
        logger.info(f"Semantic chunking completed. Original documents: {len(documents)}, "
                  f"After chunking: {len(chunked_documents)}")
        
        return ProcessorResult(
            documents=chunked_documents,
            relations=combined_relations,
            dataset=updated_dataset,
            errors=errors,
            metadata={
                "processor": "ChonkyProcessor",
                "model": self.model_id,
                "device": self.device,
                "original_document_count": len(documents),
                "chunked_document_count": len(chunked_documents),
                "chunk_relation_count": len(chunk_relations),
                "error_count": len(errors)
            }
        )
    
    def _create_semantic_chunks(self, document: IngestDocument) -> List[IngestDocument]:
        """
        Create semantically meaningful chunks from a document using Chonky.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunk documents with semantic boundaries
        """
        content = document.content
        
        # Use Chonky ParagraphSplitter to get semantic chunks
        try:
            # Use Chonky to split the document into semantic chunks
            chunks_text = list(self.splitter(content))
            
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
                    "chunking_strategy": "chonky_semantic",
                    "chunking_model": self.model_id
                }
                
                # Add parent metadata if requested
                if self.preserve_metadata and document.metadata:
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
                    document_type=document.document_type,
                    title=chunk_title,
                    author=document.author,
                    created_at=document.created_at,
                    updated_at=datetime.now(),
                    metadata=chunk_metadata,
                    embedding=None,  # Chunks need to be embedded separately
                    embedding_model=None,
                    tags=(document.tags or []) + ["chunk", "semantic_chunk"]
                )
                
                chunk_docs.append(chunk_doc)
            
            return chunk_docs
            
        except Exception as e:
            logger.error(f"Error creating semantic chunks: {e}")
            return []
