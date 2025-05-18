#!/usr/bin/env python3
"""
Test script to verify that the Haystack engine can load the Chonky model.
"""
import sys
import os
import socket
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import all the modules we need
from src.model_engine.engines.haystack import HaystackModelEngine
from src.model_engine.engines.haystack.runtime.server import run_server, SOCKET_PATH

def check_server_running(socket_path):
    """Check if the model manager server is running."""
    if not os.path.exists(socket_path):
        print(f"Socket file not found at {socket_path}")
        return False
    
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        sock.close()
        print(f"Successfully connected to server at {socket_path}")
        return True
    except Exception as e:
        print(f"Failed to connect to server at {socket_path}: {e}")
        return False

def start_server_process():
    """Start the model manager server in a separate process."""
    print(f"Starting model manager server at {SOCKET_PATH}...")
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    
    import subprocess
    import threading
    
    def run_server_process():
        # Start the server as a separate process
        cmd = [sys.executable, "-m", "src.model_engine.engines.haystack.runtime.server", SOCKET_PATH]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Started server process with PID {proc.pid}")
        
        # We don't want to block our main thread, so we'll handle output in a separate thread
        def handle_output():
            while True:
                stdout_line = proc.stdout.readline().decode('utf-8').strip()
                stderr_line = proc.stderr.readline().decode('utf-8').strip()
                if stdout_line:
                    print(f"[Server stdout] {stdout_line}")
                if stderr_line:
                    print(f"[Server stderr] {stderr_line}")
                if proc.poll() is not None:
                    break
        
        # Start thread to monitor output
        threading.Thread(target=handle_output, daemon=True).start()
        
        # Wait for the socket to appear
        for _ in range(30):  # Wait up to 3 seconds
            if os.path.exists(SOCKET_PATH):
                print(f"Server socket created at {SOCKET_PATH}")
                return True
            time.sleep(0.1)
        
        print("Timed out waiting for server socket to appear")
        return False
    
    # Run the server in a separate thread so our main process can continue
    server_thread = threading.Thread(target=run_server_process)
    server_thread.daemon = True  # Allow the program to exit even if this thread is running
    server_thread.start()
    
    # Wait a moment for the server to start
    time.sleep(2)

def main():
    """Test loading the Chonky model with our Haystack engine."""
    print(f"Using socket path: {SOCKET_PATH}")
    
    # Check if the server is already running
    if not check_server_running(SOCKET_PATH):
        print("Server not running. Starting it automatically...")
        start_server_process()
    
    print("Initializing HaystackModelEngine...")
    engine = HaystackModelEngine(socket_path=SOCKET_PATH)
    
    # Make sure the server is running now
    if not check_server_running(SOCKET_PATH):
        print("Failed to start server. Cannot continue.")
        return False
    
    model_id = "mirth/chonky_modernbert_large_1"
    print(f"Attempting to load model: {model_id}")
    
    success = False
    model_loaded = False
    
    try:
        # Try to load the model
        try:
            result = engine.load_model(model_id)
            print(f"Model loading result: {result}")
            model_loaded = True
            
            # Try to get info about loaded models
            try:
                print("Fetching model info directly from client...")
                try:
                    client_info = engine.client.info()
                    print(f"Raw client info: {client_info}")
                except Exception as e:
                    print(f"Error getting client info: {e}")
                
                print("Getting loaded models via engine...")
                try:
                    info = engine.get_loaded_models()
                    print(f"Loaded models info: {info}")
                except Exception as e:
                    print(f"Error getting loaded models: {e}")
                
                success = True
            except Exception as e:
                import traceback
                print(f"Error getting model info: {e}")
                print("Traceback:")
                traceback.print_exc()
        except Exception as e:
            import traceback
            print(f"Error loading model: {e}")
            print("Traceback:")
            traceback.print_exc()
    finally:
        # Always try to unload the model, even if there were errors
        if model_loaded:
            try:
                print(f"Ensuring model {model_id} is unloaded...")
                unload_result = engine.unload_model(model_id)
                print(f"Model unloading result: {unload_result}")
            except Exception as e:
                print(f"Error unloading model: {e}")
        else:
            print("Model was not successfully loaded, nothing to unload")
            
        # Return our overall success/failure status
        return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
