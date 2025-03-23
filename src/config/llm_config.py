"""
LLM Configuration for HADES-PathRAG.

This module contains configuration settings for Language Models,
with Ollama as the default model engine. Includes domain-specific
model selection and fallback mechanisms.
"""

import os
import logging
import requests
from typing import Dict, Any, Optional, Tuple, Callable
from functools import wraps
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

# Default Ollama settings
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
DEFAULT_OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# LLM provider options
LLM_PROVIDER_OLLAMA = "ollama"
LLM_PROVIDER_OPENAI = "openai"
LLM_PROVIDER_AZURE = "azure"
LLM_PROVIDER_BEDROCK = "bedrock"
LLM_PROVIDER_HF = "huggingface"

# Default LLM provider is Ollama (can be overridden by env var)
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", LLM_PROVIDER_OLLAMA)

# LLM configuration
LLM_CONFIG: Dict[str, Any] = {
    "provider": DEFAULT_LLM_PROVIDER,
    
    # Ollama configuration
    "ollama": {
        "host": DEFAULT_OLLAMA_HOST,
        "model": DEFAULT_OLLAMA_MODEL,
        "embed_model": DEFAULT_OLLAMA_EMBED_MODEL,
        "timeout": DEFAULT_OLLAMA_TIMEOUT,
        "parameters": {
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
            "top_p": float(os.getenv("OLLAMA_TOP_P", "0.9")),
            "top_k": int(os.getenv("OLLAMA_TOP_K", "40")),
            "max_tokens": int(os.getenv("OLLAMA_MAX_TOKENS", "2048")),
        }
    },
    
    # OpenAI configuration (if needed as fallback)
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4"),
        "embed_model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        "parameters": {
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "2048")),
        }
    }
}

def get_llm_config() -> Dict[str, Any]:
    """Get the current LLM configuration."""
    return LLM_CONFIG

def get_ollama_config() -> Dict[str, Any]:
    """Get Ollama-specific configuration."""
    return LLM_CONFIG["ollama"]

def is_ollama_available(host: Optional[str] = None) -> bool:
    """
    Check if Ollama service is available at the specified host.
    
    Args:
        host: Ollama host URL (defaults to the one in config)
        
    Returns:
        True if Ollama is available, False otherwise
    """
    host = host or LLM_CONFIG["ollama"]["host"]
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        logger.warning(f"Ollama service not available at {host}")
        return False

def with_fallback_provider(fallback_provider: str = LLM_PROVIDER_OPENAI):
    """
    Decorator for functions that need fallback to another provider if Ollama fails.
    
    Args:
        fallback_provider: Provider to use as fallback (default: OpenAI)
        
    Returns:
        Decorated function with fallback capability
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_provider = kwargs.get('provider') or LLM_CONFIG["provider"]
            
            # If not using Ollama, just call the function directly
            if current_provider != LLM_PROVIDER_OLLAMA:
                return func(*args, **kwargs)
                
            # Check if Ollama is available
            if is_ollama_available():
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error with Ollama: {e}. Trying fallback to {fallback_provider}")
            else:
                logger.warning(f"Ollama not available. Using fallback to {fallback_provider}")
            
            # If we got here, Ollama failed - use fallback provider
            kwargs['provider'] = fallback_provider
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_model_func_for_provider(provider: Optional[str] = None) -> str:
    """Get the appropriate model function name based on the provider."""
    provider = provider or LLM_CONFIG["provider"]
    
    # If Ollama is requested but not available, fall back to OpenAI
    if provider == LLM_PROVIDER_OLLAMA and not is_ollama_available():
        logger.warning("Ollama requested but not available. Falling back to OpenAI")
        provider = LLM_PROVIDER_OPENAI
    
    if provider == LLM_PROVIDER_OLLAMA:
        return "ollama_model_complete"
    elif provider == LLM_PROVIDER_OPENAI:
        return "openai_complete"
    elif provider == LLM_PROVIDER_AZURE:
        return "azure_openai_complete"
    elif provider == LLM_PROVIDER_BEDROCK:
        return "bedrock_complete"
    elif provider == LLM_PROVIDER_HF:
        return "hf_model_complete"
    else:
        # Default to Ollama if provider is unknown
        return "ollama_model_complete"

def get_embedding_func_for_provider(provider: Optional[str] = None) -> str:
    """Get the appropriate embedding function name based on the provider."""
    provider = provider or LLM_CONFIG["provider"]
    
    # If Ollama is requested but not available, fall back to OpenAI
    if provider == LLM_PROVIDER_OLLAMA and not is_ollama_available():
        logger.warning("Ollama requested but not available. Falling back to OpenAI for embeddings")
        provider = LLM_PROVIDER_OPENAI
    
    if provider == LLM_PROVIDER_OLLAMA:
        return "ollama_embed"
    elif provider == LLM_PROVIDER_OPENAI:
        return "openai_embedding"
    elif provider == LLM_PROVIDER_AZURE:
        return "azure_openai_embedding"
    elif provider == LLM_PROVIDER_BEDROCK:
        return "bedrock_embedding"
    else:
        # Default to Ollama if provider is unknown
        return "ollama_embed"

# Import domain-specific model functionality
from .domain_models import (
    detect_domain, 
    get_model_config_for_domain, 
    DOMAIN_CODE,
    DOMAIN_GENERAL
)

def get_domain_specific_config(
    content: Optional[str] = None, 
    domain: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get domain-specific model configuration based on content or explicit domain.
    
    Args:
        content: Text content to analyze for domain detection
        domain: Explicit domain to use (overrides content-based detection)
        metadata: Optional metadata that might contain domain information
        
    Returns:
        Provider-specific model configuration for the detected/specified domain
    """
    provider = LLM_CONFIG["provider"]
    
    # Determine domain
    if domain:
        detected_domain = domain
    elif content:
        detected_domain = detect_domain(content, metadata)
    else:
        # Default to general if no content or domain provided
        detected_domain = DOMAIN_GENERAL
    
    # Get domain-specific config
    domain_config = get_model_config_for_domain(detected_domain, provider)
    
    # Merge with base provider config
    provider_config = LLM_CONFIG.get(provider, {}).copy()
    
    # Override with domain-specific settings
    for key, value in domain_config.items():
        if isinstance(value, dict) and key in provider_config and isinstance(provider_config[key], dict):
            # For nested dicts like 'parameters', update rather than replace
            provider_config[key].update(value)
        else:
            provider_config[key] = value
    
    return provider_config
