"""Haystack-based model engine for HADES-PathRAG.

This engine uses Haystack and a Unix domain socket for managing models in a separate
process. It provides efficient GPU memory management through the ModelClient interface.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from src.model_engine.base import ModelEngine
from src.model_engine.engines.haystack.runtime import ModelClient


class HaystackModelEngine(ModelEngine):
    """Haystack-based model engine implementation.
    
    This engine uses the Haystack framework for managing models, with a custom
    runtime service that communicates over a Unix domain socket to ensure
    efficient GPU memory usage.
    
    The engine can manage multiple models simultaneously, with an LRU cache that
    will automatically unload the least recently used models if memory limits are reached.
    Different model types (embedding, classification, etc.) can be loaded and used
    concurrently.
    
    Attributes:
        client: The ModelClient instance for communicating with the runtime service
        running: Whether the model manager service is currently running
    """
    
    def __init__(self, socket_path: Optional[str] = None) -> None:
        """Initialize the Haystack model engine.
        
        Args:
            socket_path: Optional custom path to the Unix domain socket
        """
        self.socket_path = socket_path
        self.client = None
        self.running = False
        
    def start(self) -> bool:
        """Start the model engine service.
        
        This will start the model manager server if it's not already running.
        
        Returns:
            True if the service was started successfully, False otherwise
        """
        if self.running and self.client is not None:
            print("[HaystackModelEngine] Service is already running")
            return True
            
        try:
            self.client = ModelClient(socket_path=self.socket_path)
            # Test connection with a ping
            response = self.client.ping()
            self.running = response == "pong"
            return self.running
        except Exception as e:
            print(f"[HaystackModelEngine] Failed to start service: {e}")
            self.running = False
            return False
            
    def stop(self) -> bool:
        """Stop the model engine service.
        
        This will shut down the model manager server and release all resources.
        
        Returns:
            True if the service was stopped successfully, False otherwise
        """
        if not self.running or self.client is None:
            print("[HaystackModelEngine] Service is not running")
            return True
            
        try:
            # Request server shutdown
            self.client.shutdown()
            self.running = False
            self.client = None
            return True
        except Exception as e:
            print(f"[HaystackModelEngine] Failed to stop service: {e}")
            return False
            
    def restart(self) -> bool:
        """Restart the model engine service.
        
        This will stop and then start the model manager server.
        
        Returns:
            True if the service was restarted successfully, False otherwise
        """
        self.stop()
        return self.start()
        
    def status(self) -> Dict[str, Any]:
        """Get the status of the model engine service.
        
        Returns:
            Dict containing status information including whether the service is running
            and information about loaded models
        """
        status_info = {"running": self.running}
        
        if self.running and self.client is not None:
            try:
                # Check server health
                ping_result = self.client.ping() == "pong"
                status_info["healthy"] = ping_result
                
                # Get info about loaded models
                try:
                    loaded_models = self.get_loaded_models()
                    status_info["loaded_models"] = loaded_models
                    status_info["model_count"] = len(loaded_models)
                except Exception as e:
                    status_info["model_info_error"] = str(e)
            except Exception as e:
                status_info["healthy"] = False
                status_info["error"] = str(e)
                
        return status_info
        
    def __enter__(self):
        """Context manager entry - automatically start the service."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically stop the service."""
        self.stop()
    
    def load_model(self, model_id: str, 
                   device: Optional[Union[str, List[str]]] = None) -> str:
        """Load a model into memory using the runtime service.
        
        Args:
            model_id: The ID of the model to load (HF model ID or local path)
            device: The device to load the model onto (e.g., "cuda:0")
                    If None, uses the default device.
                    
        Returns:
            Status string ("loaded" or "already_loaded")
            
        Raises:
            RuntimeError: If the service is not running or another error occurs
        """
        if not self.running or self.client is None:
            self.start()
            if not self.running:
                raise RuntimeError("Model engine service is not running")
        
        return self.client.load(model_id, device=device if isinstance(device, str) else None)
    
    def unload_model(self, model_id: str) -> str:
        """Unload a model from memory.
        
        Args:
            model_id: The ID of the model to unload
            
        Returns:
            Status string ("unloaded")
            
        Raises:
            RuntimeError: If the service is not running or another error occurs
        """
        if not self.running or self.client is None:
            raise RuntimeError("Model engine service is not running")
            
        return self.client.unload(model_id)
    
    def infer(self, model_id: str, inputs: Union[str, List[str]],
             task: str, **kwargs: Any) -> Any:
        """Run inference with a model.
        
        Args:
            model_id: The ID of the model to use
            inputs: Input text or list of texts
            task: The type of task to perform (e.g., "generate", "embed", "chunk")
            **kwargs: Task-specific parameters
            
        Returns:
            Model outputs in an appropriate format for the task
            
        Raises:
            ValueError: If the model doesn't support the task
            RuntimeError: If inference fails or the service is not running
        """
        if not self.running or self.client is None:
            self.start()
            if not self.running:
                raise RuntimeError("Model engine service is not running")
                
        # TODO: Implement when we add infer support to the runtime service
        raise NotImplementedError("Inference is not yet implemented for the Haystack engine")
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently loaded models.
        
        Returns:
            Dict mapping model_id to metadata about the model
            
        Raises:
            RuntimeError: If the service is not running or another error occurs
        """
        if not self.running or self.client is None:
            raise RuntimeError("Model engine service is not running")
        
        # Get basic information about cached models from the runtime service
        # The info result contains model_ids as keys and timestamps as values
        cache_info = self.client.info()
        
        # Transform into expected format
        result: Dict[str, Dict[str, Any]] = {}
        for model_id, timestamp in cache_info.items():
            result[model_id] = {
                "load_time": timestamp,
                "status": "loaded",
                "engine": "haystack"
            }
        
        return result
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the engine.
        
        Returns:
            Dict containing health information
        """
        if not self.running or self.client is None:
            return {"status": "not_running"}
            
        # Ping the server to check its health
        try:
            return {"status": "ok" if self.client.ping() == "pong" else "error"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
