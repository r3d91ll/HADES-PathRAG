"""Repository ingestor orchestrator.

This module contains the RepositoryIngestor class, which orchestrates the process
of ingesting code repositories into a knowledge graph with embeddings.
"""

from __future__ import annotations

import os
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
import uuid
import asyncio
from multiprocessing.pool import ThreadPool
from tqdm.asyncio import tqdm as tqdm_async

from src.storage.arango.connection import ArangoConnection
from src.storage.arango.repository import ArangoRepository
from src.docproc.manager import DocumentProcessorManager
from src.embedding.base import EmbeddingAdapter, get_adapter
from src.embedding.batch import batch_embed
from src.chunking.code_chunkers import chunk_code
from src.chunking.text_chunkers.chonky_chunker import chunk_text, ParagraphSplitter
from src.isne.types.models import DocumentRelation, IngestDocument, RelationType
from src.types.common import EmbeddingVector
from src.pipelines.ingest.orchestrator.config import IngestionConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics for an ingestion run."""
    
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    
    # File statistics
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    
    # Entity statistics
    entities_created: int = 0
    relationships_created: int = 0
    embeddings_created: int = 0
    
    # Database statistics
    nodes_created: int = 0
    edges_created: int = 0
    vector_entries_created: int = 0
    
    def duration(self) -> float:
        """Calculate the duration of the ingestion run in seconds."""
        end_time = self.ended_at if self.ended_at > 0 else time.time()
        return end_time - self.started_at
    
    def mark_complete(self) -> None:
        """Mark the ingestion run as complete."""
        self.ended_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the statistics to a dictionary."""
        return {
            "duration_seconds": self.duration(),
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "skipped_files": self.skipped_files,
            "failed_files": self.failed_files,
            "entities_created": self.entities_created,
            "relationships_created": self.relationships_created,
            "embeddings_created": self.embeddings_created,
            "nodes_created": self.nodes_created,
            "edges_created": self.edges_created,
            "vector_entries_created": self.vector_entries_created,
        }


class RepositoryIngestor:
    """Main orchestrator for the ingestion pipeline.
    
    This class coordinates the process of ingesting a code repository
    into a knowledge graph with embeddings.
    """
    
    def __init__(
        self,
        *,
        connection: Optional[ArangoConnection] = None,
        doc_processor: Optional[DocumentProcessorManager] = None,
        embedding_adapter: Optional[EmbeddingAdapter] = None,
        config: Optional[IngestionConfig] = None,
        initialize_db: Optional[bool] = None,
        batch_size: Optional[int] = None,
        max_concurrency: Optional[int] = None,
    ):
        """Initialize the repository ingestor.
        
        Args:
            connection: ArangoDB connection to use. If None, a new connection will be created.
            doc_processor: Document processor to use. If None, a new processor will be created.
            embedding_adapter: Embedding adapter to use. If None, the default adapter will be used based on config.
            config: Ingestion configuration. If None, the default configuration will be used.
            initialize_db: Whether to initialize the database (overrides config if provided).
            batch_size: Batch size for processing documents (overrides config if provided).
            max_concurrency: Maximum number of concurrent operations (overrides config if provided).
        """
        # Load configuration
        self.config = config or DEFAULT_CONFIG
        
        # Override config with explicit parameters if provided
        if initialize_db is not None:
            self.config.initialize_db = initialize_db
        if batch_size is not None:
            self.config.batch_size = batch_size
        if max_concurrency is not None:
            self.config.max_concurrency = max_concurrency
            
        # Initialize components
        self.connection = connection or ArangoConnection()
        self.repository = ArangoRepository(self.connection)
        self.doc_processor = doc_processor or DocumentProcessorManager()
        
        # Initialize embedding adapter based on config if not explicitly provided
        if embedding_adapter is None:
            adapter_name = self.config.embedding.adapter
            self.embedding_adapter = get_adapter(adapter_name)
            # Set device for embedding adapter if possible
            if hasattr(self.embedding_adapter, "set_device"):
                self.embedding_adapter.set_device(self.config.embedding.device)
        else:
            self.embedding_adapter = embedding_adapter
            
        # Cache for paragraph splitter
        self._paragraph_splitter = None
        
        # Statistics
        self.stats = IngestionStats()
        
        # If initialize_db is True, create necessary database structures
        if self.config.initialize_db:
            self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize the database with necessary collections and indices."""
        logger.info("Initializing database structures...")
        self.repository.initialize()
        logger.info("Database initialization complete.")
    
    async def discover_files(
        self, 
        repo_path: Union[str, Path],
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Path]:
        """Discover files in the repository.
        
        Args:
            repo_path: Path to the repository.
            include_patterns: Glob patterns for files to include.
            exclude_patterns: Glob patterns for files to exclude.
            
        Returns:
            List of file paths to process.
        """
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        logger.info(f"Discovering files in {repo_path}...")
        
        # Default patterns if none provided
        include_patterns = include_patterns or ["**/*.py", "**/*.md", "**/*.txt", "**/*.json"]
        exclude_patterns = exclude_patterns or ["**/__pycache__/**", "**/.*/**", "**/venv/**", "**/node_modules/**"]
        
        # Discover files matching patterns
        files: List[Path] = []
        for pattern in include_patterns:
            matching_files = list(repo_path.glob(pattern))
            logger.debug(f"Found {len(matching_files)} files matching pattern: {pattern}")
            files.extend(matching_files)
        
        # Filter out excluded files
        for exclude_pattern in exclude_patterns:
            excluded = list(repo_path.glob(exclude_pattern))
            files = [f for f in files if f not in excluded]
        
        # Sort for reproducibility
        files.sort()
        
        logger.info(f"Discovered {len(files)} files to process.")
        self.stats.total_files = len(files)
        return files
    
    async def process_files(
        self,
        files: List[Path],
        repo_path: Union[str, Path],
    ) -> Tuple[List[Dict[str, Any]], List[DocumentRelation]]:
        """Process a batch of files.
        
        Args:
            files: List of files to process.
            repo_path: Path to the repository.
            
        Returns:
            Tuple of (entities, relationships).
        """
        repo_path = Path(repo_path)
        entities: List[Dict[str, Any]] = []
        relationships: List[DocumentRelation] = []
        
        # Process files in batches
        batches = [files[i:i + self.batch_size] for i in range(0, len(files), self.batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} files)...")
            
            # Process each file in the batch
            batch_tasks = []
            for file_path in batch:
                relative_path = file_path.relative_to(repo_path)
                task = asyncio.create_task(self._process_single_file(file_path, relative_path))
                batch_tasks.append(task)
            
            # Wait for all tasks to complete
            batch_results = await tqdm_async.gather(
                *batch_tasks, 
                desc=f"Batch {batch_idx + 1}/{len(batches)}"
            )
            
            # Collect results
            for file_entities, file_relationships in batch_results:
                entities.extend(file_entities)
                relationships.extend(file_relationships)
        
        return entities, relationships
    
    async def _process_single_file(
        self, 
        file_path: Path,
        relative_path: Path,
    ) -> Tuple[List[Dict[str, Any]], List[DocumentRelation]]:
        """Process a single file.
        
        Args:
            file_path: Path to the file.
            relative_path: Path relative to the repository.
            
        Returns:
            Tuple of (entities, relationships).
        """
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            # Process the document using the document processor
            doc_type = self._determine_doc_type(file_path)
            document = self.doc_processor.process_document(
                content=content,
                path=str(relative_path),
                doc_type=doc_type,
            )
            
            # Extract entities and relationships
            entities, relationships = self._extract_entities_and_relationships(document)
            
            self.stats.processed_files += 1
            return entities, relationships
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self.stats.failed_files += 1
            return [], []
    
    def _determine_doc_type(self, file_path: Path) -> str:
        """Determine the document type based on file extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".py":
            return "python"
        elif suffix == ".md":
            return "markdown"
        elif suffix == ".txt":
            return "text"
        elif suffix == ".json":
            return "json"
        else:
            return "text"  # Default to text
    
    def _chunk_text_cpu(
        self,
        document: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Process text document using CPU-based semantic chunking with parallel processing.
        
        This method implements optimized CPU-based chunking using the ParagraphSplitter
        with multithreading for better performance.
        
        Args:
            document: Document dictionary with content, ID, and metadata.
            
        Returns:
            List of chunk dictionaries.
        """
        # Get document content and metadata
        content = document.get("content", "")
        doc_id = document.get("id", f"doc:{uuid.uuid4().hex}")
        doc_path = document.get("path", "unknown")
        doc_type = document.get("type", "text")
        
        # Get config settings
        config = self.config.chunking
        model_id = config.model_id
        max_tokens = config.max_tokens
        num_workers = config.num_cpu_workers
        
        # Initialize the paragraph splitter if not already cached
        if self._paragraph_splitter is None:
            logger.info(f"Creating CPU ParagraphSplitter with model {model_id}")
            self._paragraph_splitter = ParagraphSplitter(
                model_id=model_id,
                device="cpu",
                use_model_engine=False
            )
        
        # For large documents, process in segments with parallel workers
        segments = []
        char_length = len(content)
        
        # If document is large, split into segments for more efficient processing
        if char_length > 10000:
            logger.info(f"Processing large document ({char_length} chars) in segments")
            segment_size = 10000  # ~10k characters per segment
            segment_count = (char_length // segment_size) + (1 if char_length % segment_size > 0 else 0)
            
            # Create segments with slight overlap to avoid cutting in the middle of paragraphs
            for i in range(segment_count):
                start = max(0, i * segment_size - 200 if i > 0 else 0) 
                end = min(char_length, (i + 1) * segment_size + 200 if i < segment_count - 1 else char_length)
                segment_text = content[start:end]
                segments.append((i+1, segment_count, segment_text))
                
            # Process segments in parallel using ThreadPool
            if num_workers > 1 and len(segments) > 1:
                with ThreadPool(min(num_workers, len(segments))) as pool:
                    segment_results = pool.map(self._process_text_segment, segments)
            else:
                segment_results = [self._process_text_segment(segment) for segment in segments]
                
            # Combine paragraphs from all segments
            all_paragraphs = []
            for segment_idx, paragraphs in enumerate(segment_results):
                logger.info(f"Segment {segment_idx+1} produced {len(paragraphs)} paragraphs")
                all_paragraphs.extend(paragraphs)
                
            # Create chunks from paragraphs
            chunks = []
            for i, para_text in enumerate(all_paragraphs):
                chunk_id = f"{doc_id}:chunk:{i}"
                chunks.append({
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "type": "chunk",
                    "symbol_type": "paragraph",
                    "name": f"paragraph_{i}",
                    "path": doc_path,
                    "content": para_text,
                    "parent": doc_id,
                    "metadata": {},
                })
            
            logger.info(f"Split document into {len(chunks)} semantic paragraphs across {len(segments)} segments")
            return chunks
            
        else:
            # For smaller documents, process directly
            paragraphs = self._paragraph_splitter.split_text(content)
            
            # Create chunks from paragraphs
            chunks = []
            for i, para_text in enumerate(paragraphs):
                chunk_id = f"{doc_id}:chunk:{i}"
                chunks.append({
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "type": "chunk",
                    "symbol_type": "paragraph",
                    "name": f"paragraph_{i}",
                    "path": doc_path,
                    "content": para_text,
                    "parent": doc_id,
                    "metadata": {},
                })
                
            logger.info(f"Split document into {len(chunks)} semantic paragraphs")
            return chunks
    
    def _process_text_segment(self, segment_data: Tuple[int, int, str]) -> List[str]:
        """Process a single text segment with the paragraph splitter.
        
        Args:
            segment_data: Tuple of (segment_index, total_segments, segment_text)
            
        Returns:
            List of paragraph texts from this segment.
        """
        segment_idx, total_segments, segment_text = segment_data
        logger.info(f"Processing segment {segment_idx}/{total_segments} ({len(segment_text)} chars)")
        
        # Split the segment into paragraphs
        paragraphs = self._paragraph_splitter.split_text(segment_text)
        return paragraphs
    
    def _extract_entities_and_relationships(
        self, 
        document: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[DocumentRelation]]:
        """Extract entities and relationships from a processed document.
        
        Args:
            document: Processed document dictionary.
            
        Returns:
            Tuple of (entities, relationships).
        """
        entities: List[Dict[str, Any]] = []
        relationships: List[DocumentRelation] = []
        
        # Extract document type and content
        doc_type = document.get("type", "text")
        doc_content = document.get("content", "")
        
        if doc_type == "python":
            # Use AST-based chunking for Python files
            chunks = chunk_code(document)
        else:
            # Use semantic chunking for text files, respecting config settings
            chunking_config = self.config.chunking
            
            if chunking_config.use_semantic_chunking:
                if chunking_config.use_gpu:
                    # Use GPU-based semantic chunking if configured
                    logger.info(f"Using GPU semantic chunking with device {chunking_config.device}")
                    chunks = chunk_text(
                        document, 
                        max_tokens=chunking_config.max_tokens,
                        model_id=chunking_config.model_id,
                        device=chunking_config.device
                    )
                else:
                    # Use CPU-based semantic chunking with parallel processing
                    logger.info(f"Using CPU semantic chunking with {chunking_config.num_cpu_workers} workers")
                    chunks = self._chunk_text_cpu(document)
            else:
                # Use basic chunking if semantic chunking is disabled
                logger.info("Using basic chunking (semantic chunking disabled)")
                chunks = chunk_text(document, semantic_chunking=False)
        
        # Create document-level entity
        doc_id = document.get("id") or f"doc:{uuid.uuid4().hex}"
        doc_entity = {
            "id": doc_id,
            "type": "document",
            "doc_type": doc_type,
            "path": document.get("path", ""),
            "content": document.get("content", ""),
            "metadata": document.get("metadata", {}),
        }
        entities.append(doc_entity)
        
        # Create entities and relationships for chunks
        for chunk in chunks:
            chunk_id = chunk.get("id") or f"chunk:{uuid.uuid4().hex}"
            chunk_entity = {
                "id": chunk_id,
                "type": "chunk",
                "doc_type": doc_type,
                "symbol_type": chunk.get("symbol_type", "chunk"),
                "name": chunk.get("name", ""),
                "path": chunk.get("path", ""),
                "content": chunk.get("content", ""),
                "line_start": chunk.get("line_start", 0),
                "line_end": chunk.get("line_end", 0),
                "metadata": {},
            }
            entities.append(chunk_entity)
            
            # Create relationship between document and chunk
            chunk_rel = DocumentRelation(
                source_id=doc_id,
                target_id=chunk_id,
                relation_type=RelationType.CONTAINS,
                weight=1.0,
                metadata={},
            )
            relationships.append(chunk_rel)
            
            # Create relationships between chunks if parent info is available
            parent_id = chunk.get("parent")
            if parent_id and parent_id != "file" and parent_id != doc_id:
                parent_rel = DocumentRelation(
                    source_id=parent_id,
                    target_id=chunk_id,
                    relation_type=RelationType.CONTAINS,
                    weight=1.0,
                    metadata={},
                )
                relationships.append(parent_rel)
        
        self.stats.entities_created += len(entities)
        self.stats.relationships_created += len(relationships)
        
        return entities, relationships
    
    async def generate_embeddings(
        self,
        entities: List[Dict[str, Any]],
    ) -> Dict[str, EmbeddingVector]:
        """Generate embeddings for entities.
        
        Args:
            entities: List of entities to generate embeddings for.
            
        Returns:
            Dictionary mapping entity IDs to embeddings.
        """
        logger.info(f"Generating embeddings for {len(entities)} entities...")
        
        # Filter out entities without content
        entities_with_content = [e for e in entities if e.get("content")]
        
        # Prepare texts and IDs
        texts = [e["content"] for e in entities_with_content]
        entity_ids = [e["id"] for e in entities_with_content]
        
        # Configure embedding generation based on config
        batch_size = self.config.batch_size
        max_concurrency = self.config.max_concurrency
        
        logger.info(f"Using {'GPU' if self.config.embedding.use_gpu else 'CPU'} for embeddings")
        logger.info(f"Embedding device: {self.config.embedding.device}")
        
        # Generate embeddings in batches
        embeddings = await batch_embed(
            texts,
            self.embedding_adapter,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            show_progress=True,
        )
        
        # Create dictionary mapping entity IDs to embeddings
        embeddings_dict = dict(zip(entity_ids, embeddings))
        
        self.stats.embeddings_created += len(embeddings)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        
        return embeddings_dict
    
    async def store_entities(
        self,
        entities: List[Dict[str, Any]],
        embeddings: Dict[str, EmbeddingVector],
    ) -> None:
        """Store entities in the repository.
        
        Args:
            entities: List of entities to store.
            embeddings: Dictionary mapping entity IDs to embeddings.
        """
        logger.info(f"Storing {len(entities)} entities...")
        
        # Process entities in batches
        batches = [entities[i:i + self.batch_size] for i in range(0, len(entities), self.batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Storing entity batch {batch_idx + 1}/{len(batches)} ({len(batch)} entities)...")
            
            # Store each entity
            store_tasks = []
            for entity in batch:
                # Get embedding if available
                embedding = embeddings.get(entity["id"])
                
                task = asyncio.create_task(
                    self._store_single_entity(entity, embedding)
                )
                store_tasks.append(task)
            
            # Wait for all storage tasks to complete
            await tqdm_async.gather(
                *store_tasks,
                desc=f"Storing batch {batch_idx + 1}/{len(batches)}"
            )
        
        logger.info(f"Stored {len(entities)} entities.")
    
    async def _store_single_entity(
        self,
        entity: Dict[str, Any],
        embedding: Optional[EmbeddingVector] = None,
    ) -> None:
        """Store a single entity in the repository.
        
        Args:
            entity: Entity to store.
            embedding: Embedding for the entity (if available).
        """
        try:
            # Store the entity
            await self.repository.store_node(entity)
            self.stats.nodes_created += 1
            
            # Store the embedding if available
            if embedding is not None:
                await self.repository.store_embedding(entity["id"], embedding)
                self.stats.vector_entries_created += 1
        
        except Exception as e:
            logger.error(f"Error storing entity {entity['id']}: {e}")
    
    async def store_relationships(
        self,
        relationships: List[DocumentRelation],
    ) -> None:
        """Store relationships in the repository.
        
        Args:
            relationships: List of relationships to store.
        """
        logger.info(f"Storing {len(relationships)} relationships...")
        
        # Process relationships in batches
        batches = [
            relationships[i:i + self.batch_size] 
            for i in range(0, len(relationships), self.batch_size)
        ]
        
        for batch_idx, batch in enumerate(batches):
            logger.info(
                f"Storing relationship batch {batch_idx + 1}/{len(batches)} "
                f"({len(batch)} relationships)..."
            )
            
            # Store each relationship
            store_tasks = []
            for relation in batch:
                task = asyncio.create_task(self._store_single_relationship(relation))
                store_tasks.append(task)
            
            # Wait for all storage tasks to complete
            await tqdm_async.gather(
                *store_tasks,
                desc=f"Storing batch {batch_idx + 1}/{len(batches)}"
            )
        
        logger.info(f"Stored {len(relationships)} relationships.")
    
    async def _store_single_relationship(self, relation: DocumentRelation) -> None:
        """Store a single relationship in the repository.
        
        Args:
            relation: Relationship to store.
        """
        try:
            # Store the relationship
            await self.repository.store_edge(
                source_id=relation.source_id,
                target_id=relation.target_id,
                edge_type=relation.relation_type.value,
                properties={
                    "weight": relation.weight,
                    "metadata": relation.metadata,
                },
            )
            self.stats.edges_created += 1
        
        except Exception as e:
            logger.error(
                f"Error storing relationship from {relation.source_id} "
                f"to {relation.target_id}: {e}"
            )
    
    async def ingest(
        self,
        repo_path: Union[str, Path],
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> IngestionStats:
        """Ingest a repository into the knowledge graph.
        
        This method orchestrates the entire ingestion process:
        1. Discover files in the repository
        2. Process files to extract entities and relationships
        3. Generate embeddings for entities
        4. Store entities and relationships in the repository
        
        Args:
            repo_path: Path to the repository.
            include_patterns: Glob patterns for files to include.
            exclude_patterns: Glob patterns for files to exclude.
            
        Returns:
            Statistics about the ingestion run.
        """
        logger.info(f"Starting ingestion of repository: {repo_path}")
        
        # Reset statistics
        self.stats = IngestionStats()
        
        try:
            # Step 1: Discover files
            files = await self.discover_files(repo_path, include_patterns, exclude_patterns)
            
            # Step 2: Process files
            entities, relationships = await self.process_files(files, repo_path)
            
            # Step 3: Generate embeddings
            embeddings = await self.generate_embeddings(entities)
            
            # Step 4: Store entities and relationships
            await self.store_entities(entities, embeddings)
            await self.store_relationships(relationships)
            
            # Mark ingestion as complete
            self.stats.mark_complete()
            logger.info(f"Ingestion completed in {self.stats.duration():.2f} seconds.")
            logger.info(f"Statistics: {self.stats.to_dict()}")
            
            return self.stats
        
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            self.stats.mark_complete()
            return self.stats
