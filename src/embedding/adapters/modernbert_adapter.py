"""ModernBERT adapter for embedding generation using Haystack.

This module provides an implementation of the EmbeddingAdapter protocol
that uses Haystack to manage and run the ModernBERT model for high-quality,
long-context embeddings. It supports both CPU and GPU configurations through
the embedding_config.yaml configuration file.

The adapter is designed to work with the default Haystack model engine infrastructure,
allowing resource management through configuration rather than code changes.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, cast, Protocol, TypeVar, Literal, overload
from types import TracebackType

import logging
import os
import torch
from typing import Any, Dict, List, Optional, Tuple, Union, cast

# Local imports
from src.embedding.base import EmbeddingAdapter, register_adapter, EmbeddingVector
from src.model_engine.engines.haystack import HaystackModelEngine
from transformers import AutoModel, AutoTokenizer
from src.config.embedding_config import get_adapter_config, load_config

# Type variables for better type inference
T = TypeVar('T')
ResponseDict = Dict[str, Any]
PoolingStrategy = Literal["cls", "mean", "max"]

logger = logging.getLogger(__name__)

class ModernBERTEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for generating embeddings using ModernBERT via Haystack."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: Optional[int] = None,
        pooling_strategy: Optional[PoolingStrategy] = None,
        batch_size: Optional[int] = None,
        normalize_embeddings: Optional[bool] = None,
        device: Optional[str] = None,
        adapter_name: str = "modernbert",
        **kwargs: Any
    ) -> None:
        """Initialize the ModernBERT embedding adapter using configuration.
        
        This constructor loads settings from the embedding_config.yaml file,
        while allowing parameter overrides through constructor arguments.
        
        Args:
            model_name: Override the model name from config
            max_length: Override the max sequence length from config
            pooling_strategy: Override the pooling strategy from config (cls, mean, max)
            batch_size: Override the batch size from config
            normalize_embeddings: Override normalization setting from config
            device: Override the device from config (cpu, cuda:0, etc.)
            adapter_name: Name of the adapter configuration to use
            **kwargs: Additional parameters for the embedding model
        """
        # Load configuration
        config = get_adapter_config(adapter_name=adapter_name)
        
        # Set parameters, allowing constructor arguments to override config
        self.model_name: str = model_name or config.get("model_name", "answerdotai/ModernBERT-base")
        self.max_length: int = max_length or config.get("max_length", 8192)
        self.pooling_strategy: PoolingStrategy = pooling_strategy or config.get("pooling_strategy", "cls")
        self.batch_size: int = batch_size or config.get("batch_size", 8)
        self.normalize_embeddings: bool = normalize_embeddings if normalize_embeddings is not None else config.get("normalize_embeddings", True)
        self.device: str = device or config.get("device", "cpu")
        
        # Engine configuration
        self.use_model_engine: bool = config.get("use_model_engine", True)
        self.engine_type: str = config.get("engine_type", "haystack")
        
        # Store the full configuration for reference
        self.config = config
        
        # Haystack engine instance
        self._engine: Optional[HaystackModelEngine] = None
        
        # Model loading state
        self._model_loaded: bool = False
        
        # Log initialization
        logger.info(
            f"Initialized ModernBERT adapter with model={self.model_name}, "
            f"device={self.device}, pooling={self.pooling_strategy}"
        )
        
    @property
    def engine(self) -> HaystackModelEngine:
        """Lazy-load the Haystack engine when first needed."""
        if self._engine is None:
            try:
                from src.model_engine.engines.haystack import HaystackModelEngine
                
                logger.info("Initializing Haystack engine for ModernBERT embeddings")
                self._engine = HaystackModelEngine()
                
                # Make sure the engine is running
                status: Dict[str, Any] = self._engine.get_status()
                if not status.get("running", False):
                    logger.info("Starting Haystack engine")
                    start_result: bool = self._engine.start()
                    if not start_result:
                        raise RuntimeError("Failed to start Haystack engine")
                
                logger.info("Haystack engine initialized successfully")
            except ImportError as e:
                logger.error(f"Failed to import HaystackModelEngine: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Haystack engine: {e}")
                raise
        
        if self._engine is None:  # This is for type checker only
            raise RuntimeError("Engine initialization failed but no exception was raised")
            
        return self._engine
    
    async def _ensure_model_loaded(self) -> None:
        """Ensure the ModernBERT model is loaded and available.
        
        This method loads the model directly using the HuggingFace transformers library
        for CPU inference to avoid issues with the Haystack caching system.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._model_loaded:
            return
        
        try:
            # For CPU-based inference, we'll load the model directly rather than using Haystack
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading ModernBERT model directly: {self.model_name} on {self.device}")
            
            # Load the tokenizer - this will automatically download if needed
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            logger.info(f"ModernBERT tokenizer loaded successfully")
            
            # Load the model directly on CPU or specified device
            # We don't convert to half precision on CPU as it's not supported
            model_device = torch.device(self.device)
            self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Only convert to half precision if on CUDA device
            if 'cuda' in self.device:
                self._model = self._model.half()
            
            self._model = self._model.to(model_device)
            self._model.eval()  # Set model to evaluation mode
            
            logger.info(f"ModernBERT model loaded successfully on {self.device}")
            self._model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load ModernBERT model: {e}")
            raise RuntimeError(f"Failed to load ModernBERT model: {e}") from e
    
    async def _get_embeddings_from_model(self, texts: List[str]) -> List[EmbeddingVector]:
        """Generate embeddings directly using the ModernBERT model.
        
        This method directly interfaces with the transformer model to generate embeddings,
        rather than using the Haystack embedding calculation function. This approach ensures
        we have full control over the embedding process.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If the embedding generation fails
        """
        if not self._model_loaded:
            await self._ensure_model_loaded()
        
        try:
            # Process the texts in batches to avoid OOM issues
            all_embeddings = []
            
            # Create mini-batches to process
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                logger.debug(f"Processing batch {i//self.batch_size + 1} with {len(batch_texts)} texts")
                
                with torch.no_grad():
                    # Tokenize the input texts
                    encoding = self._tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    
                    # Move tensors to the correct device
                    input_ids = encoding["input_ids"].to(self._model.device)
                    attention_mask = encoding["attention_mask"].to(self._model.device)
                    
                    # Forward pass
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    
                    # Apply pooling based on strategy
                    if self.pooling_strategy == "cls":
                        # CLS token embedding (first token)
                        batch_embeddings = outputs.last_hidden_state[:, 0, :]
                    elif self.pooling_strategy == "mean":
                        # Mean pooling - average token embeddings
                        token_embeddings = outputs.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    elif self.pooling_strategy == "max":
                        # Max pooling - take maximum values
                        token_embeddings = outputs.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        batch_embeddings = torch.max(token_embeddings * input_mask_expanded, dim=1)[0]
                    else:
                        raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")
                    
                    # Normalize if requested
                    if self.normalize_embeddings:
                        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                    
                    # Convert to list and add to results
                    batch_embeddings_np = batch_embeddings.cpu().numpy()
                    all_embeddings.extend(batch_embeddings_np.tolist())
            
            logger.info(f"Generated {len(all_embeddings)} embeddings with dimension {len(all_embeddings[0]) if all_embeddings else 0}")
            return cast(List[EmbeddingVector], all_embeddings)
            
        except Exception as e:
            logger.error(f"ModernBERT embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e
    
    async def embed(self, texts: List[str], **kwargs: Any) -> List[EmbeddingVector]:
        """Generate embeddings for a list of texts using ModernBERT.
        
        Args:
            texts: List of text strings to embed
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            List of embedding vectors, one for each input text
            
        Raises:
            RuntimeError: If the embedding operation fails
        """
        if not texts:
            return []
        
        # Log information about the chunking
        token_estimates: List[float] = [len(text.split()) * 1.3 for text in texts]  # Rough token estimate
        for i, (text, tokens) in enumerate(zip(texts, token_estimates)):
            if tokens > self.max_length:
                logger.warning(f"Text {i} may exceed the model's context length: {tokens:.0f} tokens (approx)")
        
        batch_size: int = kwargs.get("batch_size", self.batch_size)
        logger.info(f"Processing {len(texts)} texts with batch size {batch_size}")
        
        results: List[EmbeddingVector] = []
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch: List[str] = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_results: List[EmbeddingVector] = await self._get_embeddings_from_model(batch)
            results.extend(batch_results)
        
        return results
    
    async def embed_single(self, text: str, **kwargs: Any) -> EmbeddingVector:
        """Generate an embedding for a single text using ModernBERT.
        
        Args:
            text: Text string to embed
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Embedding vector for the input text
            
        Raises:
            RuntimeError: If the embedding operation fails
        """
        results: List[EmbeddingVector] = await self.embed([text], **kwargs)
        if not results:
            raise RuntimeError("Empty results from ModernBERT embedding model")
        return results[0]


# Register the adapter
register_adapter("modernbert", ModernBERTEmbeddingAdapter)
