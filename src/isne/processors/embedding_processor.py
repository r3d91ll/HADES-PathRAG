"""
Embedding processor for the ISNE pipeline.

This module provides a processor for calculating embeddings for documents
in the ISNE pipeline, supporting various embedding models and configurations.
"""

from typing import Dict, List, Optional, Union, Any, Set, Tuple, Callable
import os
import json
import logging
import numpy as np
from pathlib import Path
import hashlib
from datetime import datetime
import time

from src.isne.adapters.vllm_adapter import VLLMAdapter

from src.isne.types.models import IngestDocument, IngestDataset, DocumentRelation, EmbeddingConfig
from src.isne.processors.base_processor import BaseProcessor, ProcessorConfig, ProcessorResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
EmbeddingModel = Any  # Type for embedding models
EmbeddingFunction = Callable[[List[str]], List[List[float]]]  # Function to compute embeddings


class EmbeddingProcessor(BaseProcessor):
    """
    Processor for calculating embeddings for documents.
    
    This processor computes embeddings for documents using specified models,
    with support for caching, batching, and various embedding models.
    """
    
    def __init__(
        self,
        embedding_config: EmbeddingConfig,
        processor_config: Optional[ProcessorConfig] = None,
        model: Optional[EmbeddingModel] = None,
        embedding_fn: Optional[EmbeddingFunction] = None
    ) -> None:
        """
        Initialize the embedding processor.
        
        Args:
            embedding_config: Configuration for the embedding model
            processor_config: Configuration for the processor
            model: Optional pre-initialized embedding model
            embedding_fn: Optional custom embedding function
        """
        super().__init__(processor_config)
        
        self.embedding_config = embedding_config
        self.model = model
        self.embedding_fn = embedding_fn
        self.initialized = False
        
        # Initialize cache
        self._cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Create cache directory if needed
        if self.config.use_cache and self.config.cache_dir:
            os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def _initialize_model(self) -> None:
        """
        Initialize the embedding model if not already done.
        """
        if self.initialized:
            return
        
        # Skip initialization if a custom embedding function is provided
        if self.embedding_fn is not None:
            self.initialized = True
            return
        
        # Initialize the embedding model based on configuration
        model_name = self.embedding_config.model_name.lower()
        device = self._get_device()
        
        try:
            # Load appropriate embedding model based on name
            if self.embedding_config.use_vllm:
                # Use vLLM for accelerated embeddings
                self._initialize_vllm(model_name, device)
            elif "sentence-transformers" in model_name or "sbert" in model_name:
                self._initialize_sentence_transformers(model_name, device)
            elif "openai" in model_name:
                self._initialize_openai(model_name)
            elif "huggingface" in model_name or "hf" in model_name:
                self._initialize_huggingface(model_name, device)
            elif "tensorflow" in model_name or "tf" in model_name:
                self._initialize_tensorflow(model_name)
            else:
                # Default to sentence-transformers
                self._initialize_sentence_transformers(model_name, device)
                
            self.initialized = True
            logger.info(f"Initialized embedding model: {model_name} on {device}")
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise RuntimeError(f"Failed to initialize embedding model: {e}")
    
    def _initialize_sentence_transformers(self, model_name: str, device: str) -> None:
        """
        Initialize a sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run the model on ('cuda' or 'cpu')
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Extract the actual model name from the full name
            if "/" in model_name:
                model_name = model_name.split("/")[-1]
            
            self.model = SentenceTransformer(model_name, device=device)
            
            # Create embedding function
            self.embedding_fn = lambda texts: self.model.encode(
                texts,
                batch_size=self.embedding_config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.embedding_config.normalize_embeddings
            ).tolist()
            
        except ImportError:
            logger.error("sentence-transformers not installed. Please install it with 'pip install sentence-transformers'")
            raise
    
    def _initialize_openai(self, model_name: str) -> None:
        """
        Initialize OpenAI embeddings.
        
        Args:
            model_name: Name of the OpenAI embedding model
        """
        try:
            import openai
            
            # Extract the actual model name if needed
            if "/" in model_name:
                model_name = model_name.split("/")[-1]
            
            # Create embedding function using OpenAI API
            def get_openai_embeddings(texts: List[str]) -> List[List[float]]:
                results = []
                for i in range(0, len(texts), 20):  # OpenAI batch size limit
                    batch = texts[i:i+20]
                    response = openai.Embedding.create(
                        model=model_name,
                        input=batch
                    )
                    batch_embeddings = [item["embedding"] for item in response["data"]]
                    results.extend(batch_embeddings)
                return results
            
            self.embedding_fn = get_openai_embeddings
            
        except ImportError:
            logger.error("openai not installed. Please install it with 'pip install openai'")
            raise
    
    def _initialize_huggingface(self, model_name: str, device: str) -> None:
        """
        Initialize a Hugging Face model for embeddings.
        
        Args:
            model_name: Name of the Hugging Face model
            device: Device to run the model on ('cuda' or 'cpu')
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # Extract the actual model name if needed
            if "huggingface/" in model_name or "hf/" in model_name:
                model_name = model_name.split("/", 1)[1]
            
            # Initialize tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            
            # Create embedding function
            def get_hf_embeddings(texts: List[str]) -> List[List[float]]:
                embeddings = []
                
                for i in range(0, len(texts), self.embedding_config.batch_size):
                    batch = texts[i:i+self.embedding_config.batch_size]
                    
                    # Tokenize
                    encoded_input = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.embedding_config.max_length,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Get embeddings
                    with torch.no_grad():
                        model_output = model(**encoded_input)
                        
                    # Use appropriate pooling strategy
                    if self.embedding_config.pooling_strategy == "cls":
                        batch_embeddings = model_output.last_hidden_state[:, 0].cpu().numpy()
                    elif self.embedding_config.pooling_strategy == "mean":
                        # Mean pooling
                        attention_mask = encoded_input["attention_mask"]
                        token_embeddings = model_output.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        batch_embeddings = batch_embeddings.cpu().numpy()
                    else:
                        # Default to CLS token
                        batch_embeddings = model_output.last_hidden_state[:, 0].cpu().numpy()
                    
                    # Normalize if requested
                    if self.embedding_config.normalize_embeddings:
                        batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    
                    embeddings.extend(batch_embeddings.tolist())
                
                return embeddings
            
            self.embedding_fn = get_hf_embeddings
            self.model = model
            
        except ImportError:
            logger.error("transformers not installed. Please install it with 'pip install transformers'")
            raise
    
    def _initialize_tensorflow(self, model_name: str) -> None:
        """
        Initialize a TensorFlow embedding model.
        
        Args:
            model_name: Name of the TensorFlow model
        """
        # TODO: Implement TensorFlow model initialization
        raise NotImplementedError("TensorFlow embedding model integration not implemented yet")
    
    def _initialize_vllm(self, model_name: str, device: str) -> None:
        """
        Initialize vLLM-accelerated embeddings.
        
        Args:
            model_name: Name of the embedding model
            device: Device to run the model on ('cuda' or 'cpu')
        """
        try:
            # Configure vLLM adapter
            server_url = self.embedding_config.vllm_server_url or "http://localhost:8000"
            
            # Create vLLM adapter
            vllm_adapter = VLLMAdapter(
                model_name=model_name,
                server_url=server_url,
                batch_size=self.embedding_config.batch_size,
                device=device,
                normalize_embeddings=self.embedding_config.normalize_embeddings,
                use_openai_api=True  # Use OpenAI-compatible API
            )
            
            # Store the adapter
            self.model = vllm_adapter
            
            # Create embedding function
            def get_tf_embeddings(texts: List[str]) -> List[List[float]]:
                embeddings = []
                
                for i in range(0, len(texts), self.embedding_config.batch_size):
                    batch = texts[i:i+self.embedding_config.batch_size]
                    batch_embeddings = self.model(batch).numpy()
                    
                    # Normalize if requested
                    if self.embedding_config.normalize_embeddings:
                        batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    
                    embeddings.extend(batch_embeddings.tolist())
                
                return embeddings
            
            self.embedding_fn = get_tf_embeddings
            
        except ImportError:
            logger.error("tensorflow and tensorflow-hub not installed. Please install them with 'pip install tensorflow tensorflow-hub'")
            raise
    
    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for a list of texts.
        
        Args:
            texts: List of text strings to compute embeddings for
            
        Returns:
            List of embedding vectors (as lists of floats)
        """
        # Initialize model if needed
        if not self.initialized:
            self._initialize_model()
        
        # Use the embedding function
        if self.embedding_fn is not None:
            return self.embedding_fn(texts)
        else:
            raise RuntimeError("No embedding function available")
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for a text string.
        
        Args:
            text: Text to generate a cache key for
            
        Returns:
            Cache key string
        """
        # Create a hash of the text and model name
        text_hash = hashlib.md5(text.encode()).hexdigest()
        model_id = self.embedding_config.model_name.replace('/', '_')
        return f"{model_id}_{text_hash}"
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """
        Retrieve a cached embedding for a text.
        
        Args:
            text: Text to retrieve embedding for
            
        Returns:
            Cached embedding vector or None if not found
        """
        if not self.config.use_cache:
            return None
        
        cache_key = self._get_cache_key(text)
        
        # Check in-memory cache
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        # Check disk cache if configured
        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir) / f"{cache_key}.json"
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        cached_data = json.load(f)
                        self._cache[cache_key] = cached_data["embedding"]
                        self._cache_hits += 1
                        return cached_data["embedding"]
                except Exception as e:
                    logger.warning(f"Error reading cache file: {e}")
        
        self._cache_misses += 1
        return None
    
    def _save_embedding_to_cache(self, text: str, embedding: List[float]) -> None:
        """
        Save an embedding to the cache.
        
        Args:
            text: Text the embedding is for
            embedding: Embedding vector to cache
        """
        if not self.config.use_cache:
            return
        
        cache_key = self._get_cache_key(text)
        
        # Save to in-memory cache
        self._cache[cache_key] = embedding
        
        # Save to disk cache if configured
        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir) / f"{cache_key}.json"
            try:
                with open(cache_path, 'w') as f:
                    json.dump({
                        "text_hash": cache_key.split('_')[1],
                        "model": self.embedding_config.model_name,
                        "embedding": embedding,
                        "created_at": datetime.now().isoformat()
                    }, f)
            except Exception as e:
                logger.warning(f"Error writing to cache file: {e}")
    
    def process(
        self, 
        documents: List[IngestDocument],
        relations: Optional[List[DocumentRelation]] = None,
        dataset: Optional[IngestDataset] = None
    ) -> ProcessorResult:
        """
        Process documents by computing embeddings.
        
        Args:
            documents: List of documents to compute embeddings for
            relations: Optional list of relationships between documents
            dataset: Optional dataset containing documents and relationships
            
        Returns:
            ProcessorResult containing documents with embeddings
        """
        start_time = time.time()
        logger.info(f"Computing embeddings for {len(documents)} documents")
        
        # Reset cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Process documents in batches
        processed_documents: List[IngestDocument] = []
        errors: List[Dict[str, Any]] = []
        
        # Track documents that need embedding computation
        docs_to_embed: List[Tuple[int, IngestDocument, str]] = []
        
        # Check cache for existing embeddings
        for i, doc in enumerate(documents):
            # Skip documents with existing embeddings of the correct model
            if doc.embedding is not None and doc.embedding_model == self.embedding_config.model_name:
                processed_documents.append(doc)
                continue
            
            # Try to get from cache
            cached_embedding = self._get_cached_embedding(doc.content)
            if cached_embedding is not None:
                # Create a copy of the document with the embedding
                updated_doc = IngestDocument(
                    id=doc.id,
                    content=doc.content,
                    source=doc.source,
                    document_type=doc.document_type,
                    title=doc.title,
                    author=doc.author,
                    created_at=doc.created_at,
                    updated_at=doc.updated_at,
                    metadata=doc.metadata,
                    embedding=cached_embedding,
                    embedding_model=self.embedding_config.model_name,
                    chunks=doc.chunks,
                    tags=doc.tags
                )
                processed_documents.append(updated_doc)
            else:
                # Queue for embedding computation
                docs_to_embed.append((i, doc, doc.content))
        
        # Compute embeddings for documents not in cache
        if docs_to_embed:
            try:
                # Extract text content for batch processing
                indices = [item[0] for item in docs_to_embed]
                docs = [item[1] for item in docs_to_embed]
                texts = [item[2] for item in docs_to_embed]
                
                # Compute embeddings in batches
                all_embeddings = self._compute_embeddings(texts)
                
                # Update documents with embeddings
                for i, (doc_idx, doc, text) in enumerate(docs_to_embed):
                    embedding = all_embeddings[i]
                    
                    # Cache the embedding
                    self._save_embedding_to_cache(text, embedding)
                    
                    # Create a copy of the document with the embedding
                    updated_doc = IngestDocument(
                        id=doc.id,
                        content=doc.content,
                        source=doc.source,
                        document_type=doc.document_type,
                        title=doc.title,
                        author=doc.author,
                        created_at=doc.created_at,
                        updated_at=doc.updated_at,
                        metadata=doc.metadata,
                        embedding=embedding,
                        embedding_model=self.embedding_config.model_name,
                        chunks=doc.chunks,
                        tags=doc.tags
                    )
                    processed_documents.append(updated_doc)
                    
            except Exception as e:
                logger.error(f"Error computing embeddings: {e}")
                errors.append({
                    "error": str(e),
                    "type": type(e).__name__
                })
                
                # Add original documents without embeddings
                for _, doc, _ in docs_to_embed:
                    processed_documents.append(doc)
        
        # Sort processed documents to maintain original order
        processed_documents.sort(key=lambda d: documents.index(d) if d.id in [doc.id for doc in documents] else -1)
        
        # Create updated dataset if provided
        updated_dataset = None
        if dataset:
            updated_dataset = IngestDataset(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                metadata=dataset.metadata,
                created_at=dataset.created_at,
                updated_at=datetime.now()
            )
            
            # Add processed documents
            for doc in processed_documents:
                updated_dataset.add_document(doc)
            
            # Add relationships
            if relations:
                for rel in relations:
                    updated_dataset.add_relation(rel)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Embedding completed in {elapsed_time:.2f}s. Cache hits: {self._cache_hits}, misses: {self._cache_misses}")
        
        return ProcessorResult(
            documents=processed_documents,
            relations=relations or [],
            dataset=updated_dataset,
            errors=errors,
            metadata={
                "processor": "EmbeddingProcessor",
                "model": self.embedding_config.model_name,
                "document_count": len(processed_documents),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "processing_time": elapsed_time
            }
        )
