#!/usr/bin/env python3
"""
Cleanup script to properly shut down the model manager server and release GPU resources.
"""
import os
import sys
import signal
import socket
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import server configuration
from src.model_engine.engines.haystack.runtime.server import SOCKET_PATH

def kill_server_process():
    """Find and kill any running model manager server processes."""
    # First, check if the socket file exists
    if os.path.exists(SOCKET_PATH):
        print(f"Found socket file at {SOCKET_PATH}")
        
        try:
            # Try to send a signal to the server by connecting and then closing
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(SOCKET_PATH)
            sock.close()
            print("Successfully connected to server socket")
        except Exception as e:
            print(f"Error connecting to server socket: {e}")
        
        # Try to remove the socket file to prevent new connections
        try:
            os.unlink(SOCKET_PATH)
            print(f"Removed socket file at {SOCKET_PATH}")
        except Exception as e:
            print(f"Error removing socket file: {e}")
    else:
        print(f"No socket file found at {SOCKET_PATH}")
    
    # Try to find and kill any Python processes running the server module
    import subprocess
    try:
        # Find processes that match the pattern
        process_pattern = "python.*src.model_engine.engines.haystack.runtime.server"
        command = f"pgrep -f '{process_pattern}'"
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            # Found some processes, kill them
            pids = result.stdout.strip().split('\n')
            print(f"Found {len(pids)} server processes: {pids}")
            
            for pid in pids:
                pid = pid.strip()
                if pid:
                    try:
                        # Try a graceful termination first
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"Sent SIGTERM to process {pid}")
                        time.sleep(1)  # Wait a bit for graceful shutdown
                        
                        # Check if it's still running
                        try:
                            os.kill(int(pid), 0)  # Signal 0 is used to check if process exists
                            # If we get here, process still exists, so force kill
                            os.kill(int(pid), signal.SIGKILL)
                            print(f"Sent SIGKILL to process {pid}")
                        except OSError:
                            # Process already terminated
                            print(f"Process {pid} already terminated")
                    except Exception as e:
                        print(f"Error killing process {pid}: {e}")
        else:
            print("No server processes found")
    except Exception as e:
        print(f"Error finding/killing server processes: {e}")

def main():
    """Clean up server processes and release resources."""
    print("Starting model server cleanup...")
    kill_server_process()
    print("Cleanup completed. Check for remaining GPU memory usage with nvtop")
    
    # Note: After this you may need to manually free GPU memory if it's still allocated
    print("\nTo manually free GPU memory if still allocated, you can try:")
    print("1. Running 'nvidia-smi -r' to reset the GPU driver (requires root)")
    print("2. Restarting your Python environment")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
