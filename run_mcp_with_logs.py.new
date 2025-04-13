#!/usr/bin/env python
"""
Wrapper script to run MCP server with explicit file logging.

This script launches the MCP server with explicit log file redirection,
ensuring logs are captured regardless of how stdio is handled.
"""
import os
import sys
import time
import datetime
import subprocess
import threading
from pathlib import Path
from typing import TextIO, List, Tuple, Optional, Any, NoReturn, Dict


def process_output_stream(stream: TextIO, log_file: TextIO) -> None:
    """Process output streams and write to both console and log file.
    
    Args:
        stream: The output stream to process (stdout or stderr)
        log_file: The log file to write output to
    """
    for line in stream:
        # Write to both console and log file
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()


def setup_log_directory() -> Path:
    """Set up the logs directory.
    
    Returns:
        Path to the logs directory
    """
    logs_dir: Path = Path("/home/todd/ML-Lab/HADES-PathRAG/hades_pathrag/logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def create_log_file(logs_dir: Path) -> Tuple[Path, str]:
    """Create a timestamped log file.
    
    Args:
        logs_dir: Directory to create the log file in
        
    Returns:
        Tuple of (log_file_path, timestamp)
    """
    timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file: Path = logs_dir / f"mcp_server_{timestamp}.log"
    return log_file, timestamp


def create_latest_symlink(log_file: Path, logs_dir: Path) -> None:
    """Create a symlink to the latest log file.
    
    Args:
        log_file: Path to the log file
        logs_dir: Directory containing log files
    """
    latest_log_link: Path = logs_dir / "mcp_server_latest.log"
    try:
        if os.path.exists(latest_log_link):
            os.unlink(latest_log_link)
        os.symlink(log_file, latest_log_link)
    except (OSError, IOError) as e:
        print(f"Warning: Could not create symlink to latest log: {e}")


def build_command_line(args: List[str]) -> List[str]:
    """Build the command line for the MCP server.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of command line arguments
    """
    # Get the start_mcp_server.py path
    mcp_script: Path = Path(__file__).parent / "start_mcp_server.py"
    
    # Always add --debug and --log-level DEBUG if not already present
    cmd_args: List[str] = args.copy()
    
    if "--debug" not in cmd_args:
        cmd_args.append("--debug")
    
    if "--log-level" not in cmd_args:
        cmd_args.extend(["--log-level", "DEBUG"])
    
    # Build the full command line
    return [sys.executable, str(mcp_script)] + cmd_args


def run_server_process(cmd: List[str], log_file_path: Path, timestamp: str) -> int:
    """Run the MCP server process with logging.
    
    Args:
        cmd: Command line to execute
        log_file_path: Path to the log file
        timestamp: Timestamp for log entries
        
    Returns:
        Exit code from the process
    """
    with open(log_file_path, "w") as log_fh:
        # Write initial marker
        log_fh.write(f"=== MCP Server Log Started at {timestamp} ===\n\n")
        log_fh.flush()
        
        print(f"Starting MCP server with logs to: {log_file_path}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Start the server process with tee-like behavior
            process: subprocess.Popen = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            # Process stdout and stderr in separate threads
            stdout_thread: threading.Thread = threading.Thread(
                target=process_output_stream, 
                args=(process.stdout, log_fh)
            )
            stderr_thread: threading.Thread = threading.Thread(
                target=process_output_stream, 
                args=(process.stderr, log_fh)
            )
            
            # Start threads
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            process.wait()
            
            # Wait for threads to finish processing outputs
            stdout_thread.join()
            stderr_thread.join()
            
            # Write completion marker
            log_fh.write(f"\n=== MCP Server Exited with code {process.returncode} ===\n")
            
            return process.returncode
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nKeyboard interrupt received, terminating MCP server...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            log_fh.write("\n=== MCP Server Terminated by User ===\n")
            return 130  # Standard exit code for SIGINT
        
        except Exception as e:
            # Handle other exceptions
            print(f"Error running MCP server: {e}")
            log_fh.write(f"\n=== Error running MCP server: {e} ===\n")
            return 1


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code
    """
    # Set up logs directory and file
    logs_dir: Path = setup_log_directory()
    log_file_path, timestamp = create_log_file(logs_dir)
    create_latest_symlink(log_file_path, logs_dir)
    
    # Build command line
    cmd: List[str] = build_command_line(sys.argv[1:])
    
    # Run the server
    return run_server_process(cmd, log_file_path, timestamp)


if __name__ == "__main__":
    sys.exit(main())
