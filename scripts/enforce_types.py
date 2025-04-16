#!/usr/bin/env python
"""
Type enforcement script for HADES-PathRAG.

This script provides utilities to enforce type safety across the project,
with options to focus on specific modules and gradually improve type coverage.
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("enforce_types")

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
MYPY_CONFIG = PROJECT_ROOT / "mypy.ini"
DEFAULT_MODULES = [
    "hades_pathrag/embeddings",
    "hades_pathrag/ingestion",
    "hades_pathrag/storage",
    "hades_pathrag/mcp_server",
    "hades_pathrag/core",
]

def run_mypy(
    targets: List[str], 
    strict: bool = True, 
    verbose: bool = False,
    ignore_errors: Optional[List[str]] = None
) -> Tuple[bool, Dict[str, int]]:
    """
    Run mypy on specified targets.
    
    Args:
        targets: List of files or directories to check
        strict: Whether to use strict mode
        verbose: Whether to show verbose output
        ignore_errors: List of error codes to ignore
        
    Returns:
        Tuple of (success, error_counts_by_module)
    """
    cmd = ["mypy"]
    
    if strict:
        cmd.append("--strict")
        
    if verbose:
        cmd.append("--verbose")
        
    # Add configuration file if it exists
    if MYPY_CONFIG.exists():
        cmd.extend(["--config-file", str(MYPY_CONFIG)])
    
    # Add error codes to ignore if specified
    if ignore_errors:
        for error in ignore_errors:
            cmd.extend(["--disable-error-code", error])
    
    # Add targets
    cmd.extend(targets)
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Run mypy
    try:
        result = subprocess.run(
            cmd, 
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False
        )
        
        # Parse output
        errors_by_module = {}
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if line and ":" in line and ".py:" in line:
                    module_path = line.split(".py:")[0] + ".py"
                    module_name = module_path.replace("/", ".")
                    errors_by_module[module_name] = errors_by_module.get(module_name, 0) + 1
            
            # Print output
            print(result.stdout)
            
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            
        if result.returncode == 0:
            logger.info("✅ Type check passed!")
            return True, errors_by_module
        else:
            logger.error(f"❌ Type check failed with exit code {result.returncode}")
            return False, errors_by_module
            
    except Exception as e:
        logger.error(f"Failed to run mypy: {e}")
        return False, {}

def generate_type_coverage_report(
    modules: List[str],
    output_file: Optional[str] = None
) -> None:
    """
    Generate a report on type coverage for specified modules.
    
    Args:
        modules: List of modules to check
        output_file: Optional file to write the report to
    """
    logger.info("Generating type coverage report...")
    
    cmd = ["mypy", "--html-report"]
    
    if output_file:
        report_dir = PROJECT_ROOT / output_file
    else:
        report_dir = PROJECT_ROOT / "type_coverage_report"
        
    cmd.append(str(report_dir))
    cmd.extend(modules)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Type coverage report generated at {report_dir}")
            if os.path.isdir(report_dir):
                index_path = report_dir / "index.html"
                if index_path.exists():
                    logger.info(f"View the report at: file://{index_path.absolute()}")
        else:
            logger.error(f"❌ Failed to generate type report: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Failed to generate type report: {e}")

def suggest_fixes_for_common_errors(errors_by_module: Dict[str, int]) -> None:
    """
    Suggest fixes for common type errors based on error patterns.
    
    Args:
        errors_by_module: Dictionary mapping module names to error counts
    """
    if not errors_by_module:
        return
        
    print("\nSuggested fixes for common type errors:")
    print("---------------------------------------")
    
    # NetworkX related fixes
    networkx_modules = [m for m in errors_by_module if "graph" in m.lower()]
    if networkx_modules:
        print("\nFor NetworkX related errors:")
        print("  1. Add '# type: ignore[type-arg]' to Graph/DiGraph usages")
        print("  2. Consider using explicit typing.Any for node/edge data")
        print("  3. For comprehensive fix, create type stubs in scripts/type_stubs/")
    
    # numpy array errors
    numpy_modules = [m for m in errors_by_module if "embed" in m.lower()]
    if numpy_modules:
        print("\nFor NumPy array related errors:")
        print("  1. Use 'from numpy.typing import NDArray'")
        print("  2. Specify array types with 'NDArray[np.float32]'")
        print("  3. Ensure dtype is specified in array creation: np.zeros(shape, dtype=np.float32)")
    
    print("\nGeneral recommendations:")
    print("  1. Start with core interfaces and work outward")
    print("  2. Use Optional[T] for values that might be None")
    print("  3. Consider creating a custom type alias file (hades_pathrag/typings.py)")
    print("  4. Gradually remove 'ignore_errors' from mypy.ini as you fix modules")
    print("---------------------------------------")

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run type checking and generate reports for HADES-PathRAG"
    )
    parser.add_argument(
        "--modules", 
        nargs="+",
        default=DEFAULT_MODULES, 
        help="Modules to check (default: core modules)"
    )
    parser.add_argument(
        "--report", 
        action="store_true",
        help="Generate HTML type coverage report" 
    )
    parser.add_argument(
        "--report-dir",
        help="Directory to store HTML report"
    )
    parser.add_argument(
        "--non-strict", 
        action="store_true",
        help="Run without --strict mode"
    )
    parser.add_argument(
        "--ignore", 
        nargs="+",
        help="Error codes to ignore (e.g., type-arg no-untyped-call)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    # Run mypy
    success, errors = run_mypy(
        args.modules,
        not args.non_strict,
        args.verbose,
        args.ignore
    )
    
    # Generate report if requested
    if args.report:
        generate_type_coverage_report(args.modules, args.report_dir)
    
    # Suggest fixes
    suggest_fixes_for_common_errors(errors)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
