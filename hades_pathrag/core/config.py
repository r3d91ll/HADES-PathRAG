"""
Configuration classes for PathRAG.

This module contains the configuration classes for PathRAG,
allowing for flexible parameterization of the system.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Literal

@dataclass
class PathRAGConfig:
    """Configuration for PathRAG system.
    
    This configuration class contains parameters for the PathRAG system,
    including embedding model configuration, path pruning parameters,
    and storage settings.
    """
    
    # Embedding parameters
    embedding_dim: int = 128
    embedding_model: str = "isne"  # Options: "isne", "node2vec", "sentence-bert"
    
    # Flow-based path pruning parameters
    decay_rate: float = 0.8
    pruning_threshold: float = 0.01
    max_path_length: int = 4
    max_paths_per_node_pair: int = 3
    use_diverse_paths: bool = True
    diversity_weight: float = 0.3
    
    # Retrieval parameters
    top_k_nodes: int = 10
    top_k_paths: int = 5
    
    # Training parameters (for embeddings)
    training_batch_size: int = 32
    training_epochs: int = 10
    learning_rate: float = 0.01
    negative_samples: int = 5
    
    # Storage parameters
    # Legacy parameters (kept for backward compatibility)
    arango_host: str = "localhost"
    arango_port: int = 8529
    arango_username: str = "root"
    arango_password: str = "root"
    arango_database: str = "pathrag"
    
    # Standardized database parameters
    db_host: str = "localhost"
    db_port: int = 8529
    db_username: str = "root"
    db_password: str = "root"
    db_name: str = "pathrag"
    db_use_ssl: bool = False
    
    # Collection names
    vector_collection_name: str = "embeddings"
    document_collection_name: str = "documents"
    node_collection_name: str = "nodes"
    edge_collection_name: str = "edges"
    graph_name: str = "knowledge_graph"
    
    # Text processing
    chunk_size: int = 1024
    chunk_overlap: int = 128
    
    # Enable/disable features
    use_cache: bool = True
    
    # Retraining strategy
    retraining_strategy: Literal["manual", "threshold", "schedule"] = "manual"
    retraining_threshold: float = 0.2  # Retrain after 20% new data
    
    # Operational mode
    mode: Literal["inductive", "retrained"] = "inductive"
    
    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate parameters
        if self.decay_rate <= 0 or self.decay_rate >= 1:
            raise ValueError("decay_rate must be between 0 and 1")
        
        if self.pruning_threshold <= 0:
            raise ValueError("pruning_threshold must be positive")
        
        if self.max_path_length < 2:
            raise ValueError("max_path_length must be at least 2")
            
        # Sync legacy and new database parameters for backward compatibility
        # Only overwrite if the legacy parameters are different from defaults
        if self.arango_host != "localhost" and self.db_host == "localhost":
            self.db_host = self.arango_host
            
        if self.arango_port != 8529 and self.db_port == 8529:
            self.db_port = self.arango_port
            
        if self.arango_username != "root" and self.db_username == "root":
            self.db_username = self.arango_username
            
        if self.arango_password != "root" and self.db_password == "root":
            self.db_password = self.arango_password
            
        if self.arango_database != "pathrag" and self.db_name == "pathrag":
            self.db_name = self.arango_database
