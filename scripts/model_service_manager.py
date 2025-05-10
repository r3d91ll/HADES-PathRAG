#!/usr/bin/env python3
"""
Model Service Manager script to manage the Haystack model engine using Linux service conventions.

This script provides a command-line interface to start, stop, restart, and check the status
of the model manager service, as well as load and unload models.
"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model_engine.engines.haystack import HaystackModelEngine
from src.model_engine.engines.haystack.runtime.server import SOCKET_PATH as DEFAULT_SOCKET_PATH

def start_service(socket_path: Optional[str] = None) -> bool:
    """Start the model manager service.
    
    Args:
        socket_path: Optional custom path to the Unix domain socket
        
    Returns:
        True if service was started successfully, False otherwise
    """
    print(f"Starting model manager service...")
    engine = HaystackModelEngine(socket_path=socket_path)
    result = engine.start()
    
    if result:
        print(f"Model manager service started successfully")
        # Print the socket path for reference
        socket_path = socket_path or DEFAULT_SOCKET_PATH
        print(f"Socket path: {socket_path}")
    else:
        print(f"Failed to start model manager service")
        
    return result

def stop_service(socket_path: Optional[str] = None) -> bool:
    """Stop the model manager service.
    
    Args:
        socket_path: Optional custom path to the Unix domain socket
        
    Returns:
        True if service was stopped successfully, False otherwise
    """
    print(f"Stopping model manager service...")
    engine = HaystackModelEngine(socket_path=socket_path)
    result = engine.stop()
    
    if result:
        print(f"Model manager service stopped successfully")
    else:
        print(f"Failed to stop model manager service")
        
    return result

def restart_service(socket_path: Optional[str] = None) -> bool:
    """Restart the model manager service.
    
    Args:
        socket_path: Optional custom path to the Unix domain socket
        
    Returns:
        True if service was restarted successfully, False otherwise
    """
    print(f"Restarting model manager service...")
    engine = HaystackModelEngine(socket_path=socket_path)
    result = engine.restart()
    
    if result:
        print(f"Model manager service restarted successfully")
    else:
        print(f"Failed to restart model manager service")
        
    return result

def status_service(socket_path: Optional[str] = None, json_output: bool = False) -> Dict[str, Any]:
    """Check the status of the model manager service.
    
    Args:
        socket_path: Optional custom path to the Unix domain socket
        json_output: Whether to format output as JSON
        
    Returns:
        Status information as a dictionary
    """
    engine = HaystackModelEngine(socket_path=socket_path)
    status_info = engine.status()
    
    if json_output:
        print(json.dumps(status_info, indent=2))
    else:
        print(f"Model Manager Service Status:")
        print(f"  Running: {status_info.get('running', False)}")
        if status_info.get('running', False):
            print(f"  Healthy: {status_info.get('healthy', False)}")
            if 'model_count' in status_info:
                print(f"  Loaded Models: {status_info.get('model_count', 0)}")
                if status_info.get('model_count', 0) > 0:
                    print("\nLoaded Models:")
                    for model_id, model_info in status_info.get('loaded_models', {}).items():
                        print(f"  - {model_id}:")
                        for key, value in model_info.items():
                            print(f"      {key}: {value}")
            if 'error' in status_info:
                print(f"  Error: {status_info['error']}")
        
    return status_info

def load_model(model_id: str, device: Optional[str] = None, socket_path: Optional[str] = None) -> str:
    """Load a model into memory.
    
    Args:
        model_id: The ID of the model to load
        device: Optional device to load the model onto
        socket_path: Optional custom path to the Unix domain socket
        
    Returns:
        Status message
    """
    print(f"Loading model: {model_id}")
    engine = HaystackModelEngine(socket_path=socket_path)
    
    try:
        if not engine.running:
            print("Starting model manager service...")
            engine.start()
        
        result = engine.load_model(model_id, device=device)
        print(f"Model loading result: {result}")
        
        # Show loaded models after loading
        status_info = engine.status()
        if status_info.get('model_count', 0) > 0:
            print("\nCurrently Loaded Models:")
            for model, info in status_info.get('loaded_models', {}).items():
                print(f"  - {model}")
                
        return result
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"error: {str(e)}"

def unload_model(model_id: str, socket_path: Optional[str] = None) -> str:
    """Unload a model from memory.
    
    Args:
        model_id: The ID of the model to unload
        socket_path: Optional custom path to the Unix domain socket
        
    Returns:
        Status message
    """
    print(f"Unloading model: {model_id}")
    engine = HaystackModelEngine(socket_path=socket_path)
    
    try:
        result = engine.unload_model(model_id)
        print(f"Model unloading result: {result}")
        
        # Show loaded models after unloading
        status_info = engine.status()
        if status_info.get('model_count', 0) > 0:
            print("\nCurrently Loaded Models:")
            for model, info in status_info.get('loaded_models', {}).items():
                print(f"  - {model}")
        else:
            print("No models currently loaded")
                
        return result
    except Exception as e:
        print(f"Error unloading model: {e}")
        return f"error: {str(e)}"

def list_models(socket_path: Optional[str] = None, json_output: bool = False) -> Dict[str, Any]:
    """List all loaded models.
    
    Args:
        socket_path: Optional custom path to the Unix domain socket
        json_output: Whether to format output as JSON
        
    Returns:
        Information about loaded models
    """
    engine = HaystackModelEngine(socket_path=socket_path)
    
    try:
        loaded_models = engine.get_loaded_models()
        
        if json_output:
            print(json.dumps(loaded_models, indent=2))
        else:
            if loaded_models:
                print(f"Loaded Models ({len(loaded_models)}):")
                for model_id, model_info in loaded_models.items():
                    print(f"  - {model_id}:")
                    for key, value in model_info.items():
                        print(f"      {key}: {value}")
            else:
                print("No models currently loaded")
                
        return loaded_models
    except Exception as e:
        print(f"Error listing models: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Manage the Haystack model engine service")
    
    # Socket path option for all commands
    parser.add_argument("--socket", help="Path to the Unix domain socket", default=None)
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the model manager service")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the model manager service")
    
    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart the model manager service")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check the status of the model manager service")
    status_parser.add_argument("--json", action="store_true", help="Output status in JSON format")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load a model into memory")
    load_parser.add_argument("model_id", help="ID of the model to load")
    load_parser.add_argument("--device", help="Device to load the model onto", default=None)
    
    # Unload command
    unload_parser = subparsers.add_parser("unload", help="Unload a model from memory")
    unload_parser.add_argument("model_id", help="ID of the model to unload")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all loaded models")
    list_parser.add_argument("--json", action="store_true", help="Output model list in JSON format")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "start":
        success = start_service(args.socket)
        sys.exit(0 if success else 1)
    elif args.command == "stop":
        success = stop_service(args.socket)
        sys.exit(0 if success else 1)
    elif args.command == "restart":
        success = restart_service(args.socket)
        sys.exit(0 if success else 1)
    elif args.command == "status":
        status = status_service(args.socket, args.json)
        sys.exit(0 if status.get("running", False) else 1)
    elif args.command == "load":
        result = load_model(args.model_id, args.device, args.socket)
        sys.exit(0 if not result.startswith("error") else 1)
    elif args.command == "unload":
        result = unload_model(args.model_id, args.socket)
        sys.exit(0 if not result.startswith("error") else 1)
    elif args.command == "list":
        result = list_models(args.socket, args.json)
        sys.exit(0 if result is not None else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
