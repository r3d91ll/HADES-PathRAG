#!/usr/bin/env python
"""
Type checking script for HADES-PathRAG.

This script runs mypy on specified modules or the entire project,
providing a clear report on type safety compliance.
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("type_check")

def run_mypy(modules: Optional[List[str]] = None, strict: bool = True) -> bool:
    """
    Run mypy on specified modules or the entire project.
    
    Args:
        modules: List of module paths to check, or None for the entire project
        strict: Whether to use strict mode
        
    Returns:
        True if mypy passes with no errors, False otherwise
    """
    project_root = Path(__file__).parent.parent
    
    # Build command
    cmd = ["mypy"]
    
    if strict:
        cmd.append("--strict")
    
    # Add modules if specified, otherwise check the whole package
    if modules:
        cmd.extend(modules)
    else:
        cmd.append("hades_pathrag")
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Run mypy
    try:
        result = subprocess.run(
            cmd, 
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on mypy errors
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            
        if result.returncode == 0:
            logger.info("✅ Type check passed!")
            return True
        else:
            logger.error(f"❌ Type check failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to run mypy: {e}")
        return False

def generate_report(modules: Optional[List[str]] = None) -> None:
    """
    Generate a type checking report for the specified modules.
    
    Args:
        modules: List of module paths to check, or None for the entire project
    """
    project_root = Path(__file__).parent.parent
    report_path = project_root / "type_check_report.txt"
    
    cmd = ["mypy", "--txt-report", str(report_path)]
    
    if modules:
        cmd.extend(modules)
    else:
        cmd.append("hades_pathrag")
    
    logger.info(f"Generating type report: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Type report generated at {report_path}")
        else:
            logger.error(f"❌ Failed to generate type report: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Failed to generate type report: {e}")

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run type checking on HADES-PathRAG modules")
    parser.add_argument(
        "--modules", 
        nargs="+", 
        help="Specific modules to check (e.g., hades_pathrag.embeddings)"
    )
    parser.add_argument(
        "--report", 
        action="store_true", 
        help="Generate a detailed type checking report"
    )
    parser.add_argument(
        "--non-strict", 
        action="store_true", 
        help="Run without --strict flag (use settings from mypy.ini only)"
    )
    
    args = parser.parse_args()
    
    # Run mypy
    success = run_mypy(args.modules, not args.non_strict)
    
    # Generate report if requested
    if args.report:
        generate_report(args.modules)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
