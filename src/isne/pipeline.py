"""
Main pipeline implementation for the ISNE module.

This module provides the core pipeline functionality to orchestrate loading,
processing, and embedding documents using the ISNE approach.
"""

import uuid
import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Type, Callable, cast
from datetime import datetime
import time

from src.isne.types.models import (
    IngestDocument, 
    IngestDataset, 
    DocumentRelation, 
    EmbeddingConfig, 
    ISNEConfig
)
from src.isne.loaders.base_loader import BaseLoader, LoaderResult, LoaderConfig
from src.isne.processors.base_processor import BaseProcessor, ProcessorResult, ProcessorConfig
from src.isne.processors.embedding_processor import EmbeddingProcessor
from src.isne.processors.graph_processor import GraphProcessor
from src.isne.processors.chunking_processor import ChunkingProcessor
from src.isne.processors.chonking_processor import ChonkyProcessor
from src.isne.models.isne_model import ISNEModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineConfig:
    """
    Configuration for the ISNE pipeline.
    
    This class defines the configuration parameters for the entire ISNE pipeline,
    including options for loading, processing, and model components.
    """
    
    def __init__(
        self,
        # Pipeline options
        pipeline_name: str = "isne_pipeline",
        output_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        
        # Loader options
        loader_type: Optional[Type[BaseLoader]] = None,
        loader_config: Optional[LoaderConfig] = None,
        
        # Processor options
        embedding_config: Optional[EmbeddingConfig] = None,
        chunking_config: Optional[ProcessorConfig] = None,
        chonky_config: Optional[ProcessorConfig] = None,
        graph_config: Optional[ProcessorConfig] = None,
        
        # Model options
        isne_config: Optional[ISNEConfig] = None,
        
        # Processing options
        enable_chunking: bool = True,
        chonky_model_id: str = "mirth/chonky_distilbert_uncased_1",  # Default Chonky model for non-code content
        enable_embedding: bool = True,
        enable_graph_processing: bool = True,
        enable_isne_model: bool = True,
        
        # Performance options
        use_gpu: bool = True,
        parallel_processing: bool = False,
        batch_size: int = 32,
        
        # Additional options
        save_intermediate_results: bool = False,
        verbose: bool = True
    ) -> None:
        """
        Initialize the pipeline configuration.
        
        Args:
            pipeline_name: Name of the pipeline
            output_dir: Directory to save pipeline outputs
            cache_dir: Directory to save cache files
            loader_type: Type of document loader to use
            loader_config: Configuration for the document loader
            embedding_config: Configuration for embedding processor
            chunking_config: Configuration for chunking processor
            graph_config: Configuration for graph processor
            isne_config: Configuration for ISNE model
            enable_chunking: Whether to enable document chunking
            use_chonky: Whether to use Chonky for semantic chunking of non-code content
            chonky_model_id: Chonky model ID to use
            enable_embedding: Whether to enable document embedding
            enable_graph_processing: Whether to enable graph processing
            enable_isne_model: Whether to enable ISNE model
            use_gpu: Whether to use GPU for processing
            parallel_processing: Whether to use parallel processing
            batch_size: Batch size for processing
            save_intermediate_results: Whether to save intermediate results
            verbose: Whether to enable verbose logging
        """
        self.pipeline_name = pipeline_name
        self.output_dir = output_dir or os.path.join(os.getcwd(), "isne_output")
        self.cache_dir = cache_dir or os.path.join(self.output_dir, "cache")
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Loader configuration
        self.loader_type = loader_type
        self.loader_config = loader_config or LoaderConfig()
        
        # Processor configurations
        self.embedding_config = embedding_config or EmbeddingConfig(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_dimension=768,
            batch_size=batch_size,
            use_gpu=use_gpu
        )
        
        self.chunking_config = chunking_config or ProcessorConfig(
            batch_size=batch_size,
            use_gpu=use_gpu,
            cache_dir=os.path.join(self.cache_dir, "chunking")
        )
        
        self.chonky_config = chonky_config or ProcessorConfig(
            batch_size=batch_size,
            use_gpu=use_gpu,
            cache_dir=os.path.join(self.cache_dir, "chonky")
        )
        
        self.graph_config = graph_config or ProcessorConfig(
            batch_size=batch_size,
            use_gpu=use_gpu,
            cache_dir=os.path.join(self.cache_dir, "graph")
        )
        
        # Model configuration
        self.isne_config = isne_config or ISNEConfig(
            input_dim=self.embedding_config.model_dimension,
            hidden_dim=128,
            output_dim=128,
            num_layers=2,
            use_gpu=use_gpu,
            batch_size=batch_size
        )
        
        # Pipeline options
        self.enable_chunking = enable_chunking
        self.enable_embedding = enable_embedding
        self.enable_graph_processing = enable_graph_processing
        self.enable_isne_model = enable_isne_model
        
        # Performance options
        self.use_gpu = use_gpu
        self.parallel_processing = parallel_processing
        self.batch_size = batch_size
        
        # Additional options
        self.save_intermediate_results = save_intermediate_results
        self.verbose = verbose
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "pipeline_name": self.pipeline_name,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "loader_type": self.loader_type.__name__ if self.loader_type else None,
            "loader_config": vars(self.loader_config),
            "embedding_config": self.embedding_config.to_dict(),
            "chunking_config": vars(self.chunking_config),
            "graph_config": vars(self.graph_config),
            "isne_config": self.isne_config.to_dict(),
            "enable_chunking": self.enable_chunking,
            "enable_embedding": self.enable_embedding,
            "enable_graph_processing": self.enable_graph_processing,
            "enable_isne_model": self.enable_isne_model,
            "use_gpu": self.use_gpu,
            "parallel_processing": self.parallel_processing,
            "batch_size": self.batch_size,
            "save_intermediate_results": self.save_intermediate_results,
            "verbose": self.verbose
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration to
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PipelineConfig':
        """
        Load configuration from file.
        
        Args:
            path: Path to load configuration from
            
        Returns:
            Loaded PipelineConfig
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Create base configuration
        config = cls(
            pipeline_name=config_dict.get("pipeline_name", "isne_pipeline"),
            output_dir=config_dict.get("output_dir"),
            cache_dir=config_dict.get("cache_dir"),
            enable_chunking=config_dict.get("enable_chunking", True),
            enable_embedding=config_dict.get("enable_embedding", True),
            enable_graph_processing=config_dict.get("enable_graph_processing", True),
            enable_isne_model=config_dict.get("enable_isne_model", True),
            use_gpu=config_dict.get("use_gpu", True),
            parallel_processing=config_dict.get("parallel_processing", False),
            batch_size=config_dict.get("batch_size", 32),
            save_intermediate_results=config_dict.get("save_intermediate_results", False),
            verbose=config_dict.get("verbose", True)
        )
        
        # Load embedding configuration
        if "embedding_config" in config_dict:
            config.embedding_config = EmbeddingConfig(**config_dict["embedding_config"])
        
        # Load ISNE configuration
        if "isne_config" in config_dict:
            config.isne_config = ISNEConfig(**config_dict["isne_config"])
        
        # Update loader type (requires runtime resolution)
        if "loader_type" in config_dict and config_dict["loader_type"]:
            # This would require dynamically loading the class
            logger.info(f"Loader type {config_dict['loader_type']} must be set manually")
        
        logger.info(f"Configuration loaded from {path}")
        return config


class ISNEPipeline:
    """
    Main pipeline for ISNE document processing.
    
    This class orchestrates the entire ISNE pipeline process, from loading
    documents to creating ISNE embeddings and managing the graph structure.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """
        Initialize the ISNE pipeline.
        
        Args:
            config: Pipeline configuration (or use default)
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.loader: Optional[BaseLoader] = None
        self.chunking_processor: Optional[ChunkingProcessor] = None
        self.chonky_processor: Optional[ChonkyProcessor] = None
        self.embedding_processor: Optional[EmbeddingProcessor] = None
        self.graph_processor: Optional[GraphProcessor] = None
        self.isne_model: Optional[ISNEModel] = None
        
        # Initialize state
        self.dataset: Optional[IngestDataset] = None
        self.documents: List[IngestDocument] = []
        self.relations: List[DocumentRelation] = []
        
        # Initialize processors if enabled
        self._init_processors()
        
        # Track pipeline runs
        self.run_id: Optional[str] = None
        self.run_stats: Dict[str, Any] = {}
    
    def _init_processors(self) -> None:
        """Initialize pipeline processors based on configuration."""
        # Initialize chunking processors
        if self.config.enable_chunking:
            # Traditional text chunking processor (used for all documents if use_chonky=False,
            # or only for code documents if use_chonky=True)
            self.chunking_processor = ChunkingProcessor(
                processor_config=self.config.chunking_config,
                chunk_size=1000,
                chunk_overlap=200,
                splitting_strategy="paragraph",
                preserve_metadata=True,
                create_relationships=True
            )
            
            # Always initialize Chonky for semantic chunking of non-code content
            try:
                self.chonky_processor = ChonkyProcessor(
                    processor_config=self.config.chonky_config,
                    model_id=self.config.chonky_model_id,
                    device="cuda" if self.config.use_gpu else "cpu",
                    preserve_metadata=True,
                    create_relationships=True,
                    text_only=True  # Only use Chonky for non-code content
                )
                logger.info(f"Initialized Chonky processor with model {self.config.chonky_model_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Chonky processor: {e}")
                logger.warning("Falling back to traditional chunking for all documents")
                self.chonky_processor = None
        
        # Initialize embedding processor
        if self.config.enable_embedding:
            self.embedding_processor = EmbeddingProcessor(
                embedding_config=self.config.embedding_config,
                processor_config=ProcessorConfig(
                    batch_size=self.config.batch_size,
                    use_gpu=self.config.use_gpu,
                    cache_dir=os.path.join(self.config.cache_dir, "embeddings")
                )
            )
        
        # Initialize graph processor
        if self.config.enable_graph_processing:
            self.graph_processor = GraphProcessor(
                processor_config=self.config.graph_config,
                min_edge_weight=0.1,
                include_self_loops=True,
                bidirectional_edges=True
            )
        
        # Initialize ISNE model
        if self.config.enable_isne_model:
            self.isne_model = ISNEModel(config=self.config.isne_config)
    
    def set_loader(self, loader: BaseLoader) -> None:
        """
        Set the document loader for the pipeline.
        
        Args:
            loader: Loader instance to use
        """
        self.loader = loader
    
    def load(self, source: Union[str, Path]) -> LoaderResult:
        """
        Load documents from a source.
        
        Args:
            source: Source to load documents from (e.g., directory or file)
            
        Returns:
            Loader result containing documents and relationships
        """
        # Ensure loader is initialized
        if self.loader is None:
            if self.config.loader_type is None:
                raise ValueError("No loader configured. Call set_loader() or specify loader_type in config")
            
            # Initialize default loader
            self.loader = self.config.loader_type(self.config.loader_config)
        
        # Load documents
        logger.info(f"Loading documents from {source}")
        result = self.loader.load(source)
        
        # Update pipeline state
        self.documents = result.documents
        self.relations = result.relations
        self.dataset = result.dataset
        
        # Save intermediate results if configured
        if self.config.save_intermediate_results:
            self._save_intermediate_result("load", result)
        
        return result
    
    def process(self) -> Dict[str, Any]:
        """
        Process loaded documents through the pipeline.
        
        Returns:
            Dictionary with process results and statistics
        """
        start_time = time.time()
        
        # Generate run ID
        self.run_id = str(uuid.uuid4())
        self.run_stats = {
            "run_id": self.run_id,
            "pipeline_name": self.config.pipeline_name,
            "start_time": datetime.now().isoformat(),
            "document_count": len(self.documents),
            "relation_count": len(self.relations),
            "steps": {}
        }
        
        logger.info(f"Starting ISNE pipeline run {self.run_id} with {len(self.documents)} documents")
        
        # 1. Chunking (optional)
        if self.config.enable_chunking:
            # Always separate documents into code and non-code, using Chonky for non-code
            if self.chonky_processor:
                # Split documents into code and non-code
                code_docs = [doc for doc in self.documents 
                            if doc.document_type in ["code", "python", "javascript", "java", "cpp", "c", "go", "rust"]]
                non_code_docs = [doc for doc in self.documents 
                               if doc.document_type not in ["code", "python", "javascript", "java", "cpp", "c", "go", "rust"]]
                
                logger.info(f"Processing {len(non_code_docs)} non-code documents with Chonky semantic chunking")
                logger.info(f"Processing {len(code_docs)} code documents with traditional chunking")
                
                # Process non-code docs with Chonky
                if non_code_docs:
                    chonky_result = self._run_step(
                        "semantic_chunking",
                        lambda: self.chonky_processor.process(
                            non_code_docs,
                            self.relations,
                            self.dataset
                        )
                    )
                    
                    if chonky_result:
                        processed_non_code = chonky_result.documents
                        non_code_relations = chonky_result.relations
                    else:
                        # If Chonky fails, fall back to traditional chunking
                        logger.warning("Falling back to traditional chunking for non-code documents")
                        fallback_result = self.chunking_processor.process(
                            non_code_docs,
                            self.relations,
                            self.dataset
                        )
                        processed_non_code = fallback_result.documents
                        non_code_relations = fallback_result.relations
                else:
                    processed_non_code = []
                    non_code_relations = []
                
                # Process code docs with traditional chunking
                if code_docs and self.chunking_processor:
                    code_result = self._run_step(
                        "code_chunking",
                        lambda: self.chunking_processor.process(
                            code_docs,
                            self.relations,
                            self.dataset
                        )
                    )
                    
                    if code_result:
                        processed_code = code_result.documents
                        code_relations = code_result.relations
                    else:
                        processed_code = code_docs
                        code_relations = []
                else:
                    processed_code = code_docs
                    code_relations = []
                
                # Combine results
                self.documents = processed_non_code + processed_code
                
                # Deduplicate relations (some might appear in both sets)
                unique_relations = {}
                for rel in self.relations + non_code_relations + code_relations:
                    key = f"{rel.source_id}_{rel.target_id}_{rel.relation_type}"
                    unique_relations[key] = rel
                
                self.relations = list(unique_relations.values())
                
                # Update dataset if it exists
                if self.dataset:
                    self.dataset.documents = {doc.id: doc for doc in self.documents}
                    self.dataset.relations = {i: rel for i, rel in enumerate(self.relations)}
                    self.dataset.metadata = self.dataset.metadata or {}
                    self.dataset.metadata.update({
                        "chunking_applied": True,
                        "semantic_chunking_applied": True,
                        "document_count": len(self.documents),
                        "relation_count": len(self.relations)
                    })
            
            # Traditional chunking for all documents if Chonky failed to initialize
            elif self.chunking_processor:
                chunk_result = self._run_step(
                    "chunking",
                    lambda: self.chunking_processor.process(
                        self.documents,
                        self.relations,
                        self.dataset
                    )
                )
                
                if chunk_result:
                    self.documents = chunk_result.documents
                    self.relations = chunk_result.relations
                    self.dataset = chunk_result.dataset
        
        # 2. Embedding (optional)
        if self.config.enable_embedding and self.embedding_processor:
            embed_result = self._run_step(
                "embedding",
                lambda: self.embedding_processor.process(
                    self.documents,
                    self.relations,
                    self.dataset
                )
            )
            
            if embed_result:
                self.documents = embed_result.documents
                self.relations = embed_result.relations
                self.dataset = embed_result.dataset
        
        # 3. Graph processing (optional)
        if self.config.enable_graph_processing and self.graph_processor:
            graph_result = self._run_step(
                "graph_processing",
                lambda: self.graph_processor.process(
                    self.documents,
                    self.relations,
                    self.dataset
                )
            )
            
            if graph_result:
                self.documents = graph_result.documents
                self.relations = graph_result.relations
                self.dataset = graph_result.dataset
                
                # Extract graph data for ISNE model
                graph_data = graph_result.metadata.get("graph_data")
        
        # 4. ISNE model (optional)
        isne_embeddings = None
        if self.config.enable_isne_model and self.isne_model:
            # TODO: Implement ISNE model training and embedding
            pass
        
        # Update run statistics
        elapsed_time = time.time() - start_time
        self.run_stats.update({
            "end_time": datetime.now().isoformat(),
            "elapsed_seconds": elapsed_time,
            "final_document_count": len(self.documents),
            "final_relation_count": len(self.relations)
        })
        
        # Save final results
        self._save_result()
        
        logger.info(f"ISNE pipeline completed in {elapsed_time:.2f}s")
        
        return self.run_stats
    
    def _run_step(
        self, 
        step_name: str, 
        step_fn: Callable[[], ProcessorResult]
    ) -> Optional[ProcessorResult]:
        """
        Run a pipeline step and track statistics.
        
        Args:
            step_name: Name of the step
            step_fn: Function to execute the step
            
        Returns:
            Step result or None if error
        """
        step_start = time.time()
        logger.info(f"Running {step_name} step")
        
        try:
            result = step_fn()
            
            # Track step statistics
            elapsed = time.time() - step_start
            self.run_stats["steps"][step_name] = {
                "success": True,
                "elapsed_seconds": elapsed,
                "document_count": len(result.documents),
                "relation_count": len(result.relations),
                "error_count": len(result.errors)
            }
            
            # Save intermediate results if configured
            if self.config.save_intermediate_results:
                self._save_intermediate_result(step_name, result)
            
            logger.info(f"Completed {step_name} in {elapsed:.2f}s with {len(result.documents)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Error in {step_name} step: {e}")
            
            # Track error
            self.run_stats["steps"][step_name] = {
                "success": False,
                "elapsed_seconds": time.time() - step_start,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            return None
    
    def _save_intermediate_result(self, step_name: str, result: Union[LoaderResult, ProcessorResult]) -> None:
        """
        Save intermediate step results.
        
        Args:
            step_name: Name of the step
            result: Step result to save
        """
        if not self.run_id:
            self.run_id = str(uuid.uuid4())
        
        # Create step output directory
        step_dir = os.path.join(self.config.output_dir, self.run_id, step_name)
        os.makedirs(step_dir, exist_ok=True)
        
        # Save documents
        docs_path = os.path.join(step_dir, "documents.json")
        with open(docs_path, 'w') as f:
            json.dump([doc.to_dict() for doc in result.documents], f, indent=2)
        
        # Save relationships
        rels_path = os.path.join(step_dir, "relations.json")
        with open(rels_path, 'w') as f:
            json.dump([rel.to_dict() for rel in result.relations], f, indent=2)
        
        # Save dataset if available
        if result.dataset:
            dataset_path = os.path.join(step_dir, "dataset.json")
            with open(dataset_path, 'w') as f:
                json.dump(result.dataset.to_dict(), f, indent=2)
        
        # Save metadata
        meta_path = os.path.join(step_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump({
                "step": step_name,
                "timestamp": datetime.now().isoformat(),
                "document_count": len(result.documents),
                "relation_count": len(result.relations),
                **result.metadata
            }, f, indent=2)
    
    def _save_result(self) -> None:
        """Save final pipeline results."""
        if not self.run_id:
            self.run_id = str(uuid.uuid4())
        
        # Create output directory
        output_dir = os.path.join(self.config.output_dir, self.run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save documents
        if self.documents:
            docs_path = os.path.join(output_dir, "documents.json")
            with open(docs_path, 'w') as f:
                json.dump([doc.to_dict() for doc in self.documents], f, indent=2)
        
        # Save relationships
        if self.relations:
            rels_path = os.path.join(output_dir, "relations.json")
            with open(rels_path, 'w') as f:
                json.dump([rel.to_dict() for rel in self.relations], f, indent=2)
        
        # Save dataset if available
        if self.dataset:
            dataset_path = os.path.join(output_dir, "dataset.json")
            with open(dataset_path, 'w') as f:
                json.dump(self.dataset.to_dict(), f, indent=2)
        
        # Save run statistics
        stats_path = os.path.join(output_dir, "run_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.run_stats, f, indent=2)
        
        # Save configuration
        config_path = os.path.join(output_dir, "config.json")
        self.config.save(config_path)
        
        logger.info(f"Pipeline results saved to {output_dir}")
    
    def save_model(self, path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """
        Save trained ISNE model.
        
        Args:
            path: Path to save model to (or use default location)
            
        Returns:
            Path to saved model or None if no model
        """
        if not self.isne_model:
            logger.warning("No ISNE model to save")
            return None
        
        # Use default path if not specified
        if path is None:
            if not self.run_id:
                self.run_id = str(uuid.uuid4())
            
            path = os.path.join(self.config.output_dir, self.run_id, "isne_model.pt")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model
        self.isne_model.save(path)
        logger.info(f"ISNE model saved to {path}")
        
        return str(path)
    
    @classmethod
    def load_model(
        cls, 
        path: Union[str, Path], 
        device: Optional[str] = None
    ) -> ISNEModel:
        """
        Load ISNE model from file.
        
        Args:
            path: Path to load model from
            device: Device to load model on ('cuda' or 'cpu')
            
        Returns:
            Loaded ISNE model
        """
        return ISNEModel.load(path, device)
