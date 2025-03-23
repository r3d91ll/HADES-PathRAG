"""
Domain-specific model configuration for HADES-PathRAG.

This module contains utilities for selecting appropriate models
based on content domains and handling fallbacks between providers.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Domain identifiers
DOMAIN_CODE = "code"
DOMAIN_GENERAL = "general"

# Domain-specific model configuration
DOMAIN_MODELS = {
    DOMAIN_CODE: {
        "ollama": {
            "model": os.getenv("OLLAMA_CODE_MODEL", "qwen:7b"),  # Default code model
            "embed_model": os.getenv("OLLAMA_CODE_EMBED_MODEL", "nomic-embed-text"),
            "parameters": {
                "temperature": 0.2,  # Lower temperature for more precise code responses
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 4096,  # Larger context for code generation
            }
        },
        "openai": {
            "model": os.getenv("OPENAI_CODE_MODEL", "gpt-4"),
            "embed_model": os.getenv("OPENAI_CODE_EMBED_MODEL", "text-embedding-3-small"),
            "parameters": {
                "temperature": 0.3,
                "max_tokens": 4096,
            }
        }
    },
    DOMAIN_GENERAL: {
        "ollama": {
            "model": os.getenv("OLLAMA_GENERAL_MODEL", "llama3"),  # Default general model
            "embed_model": os.getenv("OLLAMA_GENERAL_EMBED_MODEL", "nomic-embed-text"),
            "parameters": {
                "temperature": 0.7,  # Higher temperature for more creative responses
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 2048,
            }
        },
        "openai": {
            "model": os.getenv("OPENAI_GENERAL_MODEL", "gpt-3.5-turbo"),
            "embed_model": os.getenv("OPENAI_GENERAL_EMBED_MODEL", "text-embedding-3-small"),
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 2048,
            }
        }
    }
}

# Code indicators for domain detection
CODE_INDICATORS = [
    ".py", ".js", ".cpp", ".java", ".go", ".rust", ".c", ".h", ".rb", 
    "function", "class", "def ", "import ", "from ", "#include", 
    "```python", "```javascript", "```cpp", "```java", "```go", "```rust",
    "package", "namespace", "module", "struct", "interface"
]

def detect_domain(content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Detect the domain of the content.
    
    Args:
        content: The text content to analyze
        metadata: Optional metadata that might contain domain information
        
    Returns:
        Domain identifier (e.g., "code" or "general")
    """
    # Check metadata first if available
    if metadata and metadata.get("domain"):
        return metadata["domain"]
        
    # Check for code indicators
    if any(indicator in content for indicator in CODE_INDICATORS):
        return DOMAIN_CODE
    
    # Default to general domain
    return DOMAIN_GENERAL

def get_model_config_for_domain(
    domain: str, 
    provider: str = "ollama"
) -> Dict[str, Any]:
    """
    Get the model configuration for a specific domain and provider.
    
    Args:
        domain: Domain identifier (e.g., "code" or "general")
        provider: LLM provider (e.g., "ollama" or "openai")
        
    Returns:
        Model configuration dictionary
    """
    # If domain not found, fall back to general
    if domain not in DOMAIN_MODELS:
        logger.warning(f"Domain '{domain}' not found, falling back to general domain")
        domain = DOMAIN_GENERAL
    
    # Get domain-specific config for provider
    domain_config = DOMAIN_MODELS[domain]
    
    # If provider not found in domain config, fall back to first available provider
    if provider not in domain_config:
        fallback_provider = next(iter(domain_config.keys()))
        logger.warning(
            f"Provider '{provider}' not found in domain '{domain}', "
            f"falling back to '{fallback_provider}'"
        )
        provider = fallback_provider
    
    return domain_config[provider]
