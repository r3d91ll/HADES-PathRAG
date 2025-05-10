"""Model Manager server – owns GPU-resident models and exposes a minimal
JSON-RPC interface over a Unix-domain socket.
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import threading
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration (could later read from a yaml or env-vars)
# ---------------------------------------------------------------------------
MAX_MODELS: int = int(os.getenv("HADES_MAX_MODELS", "3"))
DEFAULT_DEVICE: str = os.getenv("HADES_DEFAULT_DEVICE", "cuda:0")
SOCKET_PATH: str = os.getenv("HADES_MODEL_MGR_SOCKET", "/tmp/hades_model_mgr.sock")


# ---------------------------------------------------------------------------
# In-memory LRU cache
# ---------------------------------------------------------------------------
class _LRUCache:
    def __init__(self, max_size: int) -> None:
        self._max: int = max_size
        self._lock: threading.Lock = threading.Lock()
        self._data: Dict[str, Tuple[Any, Any, float]] = {}

    def get(self, key: str) -> tuple[Any, Any] | None:
        with self._lock:
            if key not in self._data:
                return None
            model, tok, _ = self._data[key]
            self._data[key] = (model, tok, time.time())
            return model, tok

    def put(self, key: str, model: Any, tok: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data[key] = (model, tok, time.time())
                return
            if len(self._data) >= self._max:
                # evict oldest
                oldest = min(self._data, key=lambda k: self._data[k][2])
                self.evict(oldest)
            self._data[key] = (model, tok, time.time())

    def evict(self, key: str) -> None:
        model, _, _ = self._data.pop(key, (None, None, None))
        if model is not None:
            model.cpu()
            del model
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def info(self) -> MappingProxyType[str, float]:
        with self._lock:
            return MappingProxyType({k: v[2] for k, v in self._data.items()})


_CACHE = _LRUCache(MAX_MODELS)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_model(model_id: str, device: str | None = None) -> str:
    device = device or DEFAULT_DEVICE
    cached = _CACHE.get(model_id)
    if cached:
        return "already_loaded"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).half()
    _CACHE.put(model_id, model, tok)
    return "loaded"


def _unload_model(model_id: str) -> str:
    _CACHE.evict(model_id)
    return "unloaded"


def _get_model_info() -> dict[str, Any]:
    """Get information about all loaded models.
    
    Returns:
        Dictionary mapping model IDs to their last access timestamps
    """
    # Convert the cache info to a proper dictionary
    cache_items = _CACHE.info()
    print(f"[ModelManager] Cache info: {cache_items}")
    
    # Make sure we're returning data in the expected format
    result = {}
    for model_id, data in cache_items.items():
        # If data is a timestamp (float), use it directly
        if isinstance(data, (int, float)):
            result[model_id] = data
        # Otherwise convert to a string for JSON compatibility
        else:
            result[model_id] = str(data)
    
    return result


def _debug_cache() -> dict[str, Any]:
    """Get detailed debug information about the cache state.
    
    This is useful for troubleshooting model loading issues.
    
    Returns:
        Dictionary with detailed debug information
    """
    # Get the current state of the cache
    debug_info = {
        "cache_size": len(_CACHE._data),
        "max_cache_size": _CACHE._max,
        "keys": list(_CACHE._data.keys()),
        "debug": str(_CACHE),
    }
    print(f"[ModelManager] Debug info: {debug_info}")
    return debug_info


def _shutdown_server() -> str:
    """Shutdown the server gracefully.
    
    Returns:
        Status message
    """
    # Run this in a separate thread to avoid blocking the response
    def delayed_shutdown():
        # Give time for the response to be sent back
        time.sleep(0.5)
        # Force exit which will trigger the atexit handler
        os._exit(0)
    
    print("[ModelManager] Shutdown requested by client - server will exit")
    threading.Thread(target=delayed_shutdown, daemon=True).start()
    return "shutdown_initiated"


def _handle_request(req: dict[str, Any]) -> dict[str, Any]:
    """Handle an incoming request from a client.
    
    Args:
        req: Request dict with action and parameters
        
    Returns:
        Response dict with results
    """
    action = req.get("action")
    if action == "ping":
        return {"result": "pong"}
    if action == "load":
        model_id = req["model_id"]
        device = req.get("device")
        return {"result": _load_model(model_id, device)}
    if action == "unload":
        model_id = req["model_id"]
        return {"result": _unload_model(model_id)}
    if action == "info":
        result = _get_model_info()
        return {"result": result}
    if action == "debug":
        result = _debug_cache()
        return {"result": result}
    if action == "shutdown":
        result = _shutdown_server()
        return {"result": result}
    
    # If we get here, it's an unknown action
    return {"error": f"unknown action {action}"}


def _handle_conn(conn: socket.socket) -> None:
    """Handle a client connection, processing requests and sending responses.
    
    Args:
        conn: Socket connection to client
    """
    with conn:
        data = conn.recv(65536)
        try:
            # Decode and log the request
            req_str = data.decode()
            print(f"[ModelManager] Received request: {req_str}")
            req = json.loads(req_str)
            
            # Process the request
            resp = _handle_request(req)
            print(f"[ModelManager] Sending response: {resp}")
            
            # Encode and send the response
            resp_str = json.dumps(resp)
            print(f"[ModelManager] Encoded response: {resp_str}")
            conn.sendall(resp_str.encode())
        except json.JSONDecodeError as e:
            print(f"[ModelManager] JSON decode error: {e}")
            resp = {"error": f"Invalid JSON: {str(e)}"}
            conn.sendall(json.dumps(resp).encode())
        except Exception as e:  # noqa: BLE001
            print(f"[ModelManager] Error handling request: {e}")
            resp = {"error": str(e)}
            conn.sendall(json.dumps(resp).encode())

# ---------------------------------------------------------------------------
# Server mainloop
# ---------------------------------------------------------------------------

def run_server(socket_path: str = SOCKET_PATH) -> None:
    # Clean up stale socket
    sock_path = Path(socket_path)
    if sock_path.exists():
        sock_path.unlink()

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server_sock.bind(socket_path)
    except Exception as e:
        print(f"[ModelManager] Failed to bind socket {socket_path}: {e}", flush=True)
        raise
    os.chmod(socket_path, 0o660)
    server_sock.listen()
    print(f"[ModelManager] Listening on {socket_path}", flush=True)

    def _shutdown(*_: object) -> None:
        server_sock.close()
        if sock_path.exists():
            sock_path.unlink()
        print("[ModelManager] Shutdown complete", flush=True)
        exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _shutdown)
    atexit.register(_shutdown)

    while True:
        conn, _ = server_sock.accept()
        threading.Thread(target=_handle_conn, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    import sys
    try:
        sock = sys.argv[1] if len(sys.argv) > 1 else SOCKET_PATH
        run_server(sock)
    except Exception as e:
        print(f"[ModelManager] Server crashed: {e}", flush=True)
        sys.exit(1)
