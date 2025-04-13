"""
Semantic embedding models for different content modalities.

This module provides models for generating semantic embeddings for
different content modalities such as code and text, using ModernBERT models.
"""
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, cast

import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

from hades_pathrag.embeddings.base import BaseEmbedder
from hades_pathrag.ingestion.models import IngestDocument

logger = logging.getLogger(__name__)


class ContentModality(Enum):
    """Content modality types."""
    CODE = "code"
    TEXT = "text"
    # Future modalities
    # IMAGE = "image"
    # AUDIO = "audio"


class ModernBERTEmbedder(BaseEmbedder):
    """
    Semantic embedder using ModernBERT models.
    
    This class provides semantic embeddings for different content modalities
    using specialized ModernBERT models from Hugging Face.
    """
    
    # Model checkpoints for different modalities
    MODEL_CHECKPOINTS: Dict[ContentModality, str] = {
        ContentModality.CODE: "juanwisz/modernbert-python-code-retrieval",
        ContentModality.TEXT: "answerdotai/ModernBERT-base",
    }
    
    def __init__(
        self,
        embedding_dim: int = 768,
        max_length: int = 512,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the ModernBERT embedder.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            max_length: Maximum sequence length for tokenization
            device: Device to run models on (cpu, cuda, mps)
            cache_dir: Directory to cache models
        """
        # Initialize base class with embedding dim
        super().__init__()
        self._embedding_dim = embedding_dim
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # Initialize models and tokenizers for each modality
        self.models: Dict[ContentModality, Any] = {}
        self.tokenizers: Dict[ContentModality, Any] = {}
        
        logger.info(f"Initializing ModernBERT models on {self.device}")
        for modality, checkpoint in self.MODEL_CHECKPOINTS.items():
            self._load_model(modality, checkpoint)
    
    def _load_model(self, modality: ContentModality, checkpoint: str) -> None:
        """
        Load a model and tokenizer for a specific modality.
        
        Args:
            modality: Content modality
            checkpoint: Model checkpoint name
        """
        try:
            logger.info(f"Loading model for {modality.value}: {checkpoint}")
            self.models[modality] = AutoModel.from_pretrained(
                checkpoint, 
                cache_dir=self.cache_dir
            ).to(self.device)
            self.tokenizers[modality] = AutoTokenizer.from_pretrained(
                checkpoint,
                cache_dir=self.cache_dir
            )
            logger.info(f"Successfully loaded model for {modality.value}")
        except Exception as e:
            logger.error(f"Error loading model for {modality.value}: {e}")
            # Create placeholder instead of failing completely
            self.models[modality] = None
            self.tokenizers[modality] = None
    
    def detect_modality(self, document: IngestDocument) -> ContentModality:
        """
        Detect the content modality of a document.
        
        Args:
            document: Document to detect modality for
            
        Returns:
            Detected content modality
        """
        # Check metadata first if available
        content_type = document.metadata.get("content_type", "").lower() if document.metadata else ""
        file_path = document.metadata.get("path", "") if document.metadata else ""
        
        # Determine based on file extension or content type
        if content_type.startswith("text/x-python") or (file_path and file_path.endswith((".py", ".pyi"))):
            return ContentModality.CODE
        else:
            # Default to TEXT for all other content
            return ContentModality.TEXT
    
    def _embed_with_model(
        self, 
        text: str, 
        modality: ContentModality
    ) -> Optional[List[float]]:
        """
        Embed text using the appropriate model for the given modality.
        
        Args:
            text: Text to embed
            modality: Content modality
            
        Returns:
            Embedding vector or None if embedding fails
        """
        model = self.models.get(modality)
        tokenizer = self.tokenizers.get(modality)
        
        if model is None or tokenizer is None:
            logger.error(f"Model or tokenizer not available for {modality.value}")
            return None
        
        try:
            # Prepare inputs
            inputs = tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Use CLS token embedding as document representation
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return cast(List[float], embedding.tolist())
        except Exception as e:
            logger.error(f"Error generating embedding for {modality.value}: {e}")
            return None
    
    def encode(self, data: Union[List[Tuple[str, List[str], Optional[str]]], Tuple[List[str], List[List[str]]]]) -> np.ndarray:  # type: ignore
        """Encode data (implements abstract method from BaseEmbedder)."""
        # Handle the expected input format from BaseEmbedder
        # This is a simplified implementation to satisfy the type checker
        text_content = ""
        if isinstance(data, list):
            # First format: List[Tuple[str, List[str], Optional[str]]]
            if data and isinstance(data[0], tuple) and len(data[0]) >= 1:
                text_content = data[0][0]  # First element of first tuple
        elif isinstance(data, tuple) and len(data) >= 1:
            # Second format: Tuple[List[str], List[List[str]]]
            if data[0] and isinstance(data[0], list):
                text_content = data[0][0]  # First element of first list
        
        # Default to TEXT modality for BaseEmbedder's encode method
        embedding = self._embed_with_model(text_content, ContentModality.TEXT)
        if embedding is None:
            return np.zeros(self._embedding_dim)
        return np.array(embedding)
        
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text (implements abstract method from BaseEmbedder)."""
        # Create proper input format for encode
        dummy_input: List[Tuple[str, List[str], Optional[str]]] = [(text, [], None)]
        # Use cast to ensure type compatibility
        compatible_input = cast(Union[List[Tuple[str, List[str], Optional[str]]], Tuple[List[str], List[List[str]]]], dummy_input)
        return self.encode(compatible_input)
        
    def batch_encode(self, batch_data: Union[List[Tuple[str, List[str], Optional[str]]], Tuple[List[str], List[List[str]]]]) -> np.ndarray:
        """Batch encode data (implements abstract method from BaseEmbedder)."""
        # Create a matrix of embeddings
        results = []
        # Handle each possible format
        if isinstance(batch_data, list):
            for item in batch_data:
                if isinstance(item, tuple) and len(item) >= 1:
                    text = item[0]
                    embedding = self._embed_with_model(text, ContentModality.TEXT)
                    if embedding is not None:
                        results.append(np.array(embedding))
                    else:
                        results.append(np.zeros(self._embedding_dim))
        elif isinstance(batch_data, tuple) and len(batch_data) >= 1:
            for text in batch_data[0]:
                embedding = self._embed_with_model(text, ContentModality.TEXT)
                if embedding is not None:
                    results.append(np.array(embedding))
                else:
                    results.append(np.zeros(self._embedding_dim))
        
        if not results:
            # Return empty array with correct shape if no results
            return np.zeros((0, self._embedding_dim))
        
        return np.vstack(results)
        
    def fit(self, data: Any) -> None:
        """Fit model to data (implements abstract method from BaseEmbedder)."""
        # Pre-trained models don't need fitting
        pass
    
    def embed(self, document: Union[IngestDocument, str]) -> Optional[List[float]]:
        """
        Embed a document using the appropriate model for its modality.
        
        Args:
            document: Document or text to embed
            
        Returns:
            Embedding vector or None if embedding fails
        """
        # Handle document object
        if isinstance(document, IngestDocument):
            text = document.content
            modality = self.detect_modality(document)
        # Handle raw text
        else:
            text = document
            modality = ContentModality.TEXT  # Default for raw text
        
        # Get embedding for the detected modality
        return self._embed_with_model(text, modality)
    
    def embed_batch(
        self, 
        documents: List[Union[IngestDocument, str]]
    ) -> List[Optional[List[float]]]:
        """
        Embed a batch of documents.
        
        Args:
            documents: List of documents or texts to embed
            
        Returns:
            List of embedding vectors
        """
        return [self.embed(doc) for doc in documents]
        
    def save(self, path: str) -> None:
        """
        Save the embedder configuration.
        
        Args:
            path: Path to save configuration to
        """
        # We don't save the models as they can be reloaded from HuggingFace
        config = {
            "embedding_dim": self._embedding_dim,
            "max_length": self.max_length,
            "model_checkpoints": {m.value: c for m, c in self.MODEL_CHECKPOINTS.items()}
        }
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w") as f:
            import json
            json.dump(config, f)
    
    @classmethod
    def load(cls, path: str) -> "ModernBERTEmbedder":
        """
        Load an embedder from a saved configuration.
        
        Args:
            path: Path to load configuration from
            
        Returns:
            Loaded embedder instance
        """
        with open(path, "r") as f:
            import json
            config = json.load(f)
        
        return cls(
            embedding_dim=config.get("embedding_dim", 768),
            max_length=config.get("max_length", 512)
        )
