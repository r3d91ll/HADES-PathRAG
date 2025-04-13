"""
Configuration settings for the MCP server.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union


class SecurityConfig(BaseModel):
    """Security configuration for the MCP server."""
    secret_key: str = Field(
        default="please-change-this-secret-key-in-production",
        description="Secret key for JWT token encoding/decoding"
    )
    token_expire_minutes: int = Field(
        default=30,
        description="Minutes until authentication token expires"
    )
    algorithm: str = Field(
        default="HS256", 
        description="Algorithm used for JWT encoding/decoding"
    )
    api_key_header_name: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication"
    )
    allowed_api_keys: List[str] = Field(
        default=["dev-testing-key"],
        description="List of allowed API keys (should be properly secured in production)"
    )


class ServerConfig(BaseModel):
    """Server configuration for the MCP server."""
    host: str = Field(
        default="127.0.0.1",
        description="Host IP to bind the server to"
    )
    port: int = Field(
        default=8000,
        description="Port to bind the server to"
    )
    debug: bool = Field(
        default=False,
        description="Run server in debug mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    allowed_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    timeout_seconds: int = Field(
        default=120,
        description="Request timeout in seconds for tool calls"
    )


class DatabaseConfig(BaseModel):
    """Database configuration for connecting to ArangoDB."""
    url: str = Field(
        default="http://localhost:8529",
        description="ArangoDB connection URL"
    )
    username: str = Field(
        default="root",
        description="ArangoDB username"
    )
    password: str = Field(
        default="",
        description="ArangoDB password"
    )
    database_name: str = Field(
        default="pathrag",
        description="ArangoDB database name"
    )
    connection_pool_size: int = Field(
        default=5,
        description="Number of connections to keep in the pool"
    )


class PathRAGConfig(BaseModel):
    """Configuration for PathRAG integration."""
    max_paths: int = Field(
        default=5,
        description="Maximum number of paths to return in a response"
    )
    max_nodes: int = Field(
        default=20,
        description="Maximum number of nodes to explore"
    )
    min_similarity_threshold: float = Field(
        default=0.6,
        description="Minimum similarity threshold for embedding queries"
    )
    embedding_model: str = Field(
        default="isne",
        description="Embedding model to use (isne, enhanced_isne)"
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )


class MCPServerConfig(BaseModel):
    """Main configuration for the MCP server."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    pathrag: PathRAGConfig = Field(default_factory=PathRAGConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MCPServerConfig":
        """Create a config object from a dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "MCPServerConfig":
        """Load configuration from environment variables."""
        # This would be implemented to read from env vars
        # For now, return default config
        return cls()
    
    @classmethod
    def from_file(cls, filepath: str) -> "MCPServerConfig":
        """Load configuration from a file."""
        # This would be implemented to read from a config file
        # For now, return default config
        return cls()


# Single instance of configuration used across the application
_config: Optional[MCPServerConfig] = None


def get_config() -> MCPServerConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = MCPServerConfig()
    return _config


def load_config(filepath: Optional[str] = None) -> MCPServerConfig:
    """Load configuration from environment or file."""
    global _config
    
    if filepath:
        _config = MCPServerConfig.from_file(filepath)
    else:
        _config = MCPServerConfig.from_env()
    
    return _config
