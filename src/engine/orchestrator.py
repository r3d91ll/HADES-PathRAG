"""
Pipeline orchestrator.

This module implements the orchestrator for the GPU-accelerated batch pipeline.
It manages the flow of batches through the pipeline stages and coordinates
GPU resources.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Type

import torch
import numpy as np

from .batch_types import PipelineBatch
from .stages.base import StageBase
from .stages.docproc_stage import DocProcStage
from .stages.chunk_stage import ChunkStage
from src.config.engine_config import get_engine_config, EngineConfig


logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Pipeline orchestrator.
    
    This class manages the flow of batches through the pipeline stages and
    coordinates GPU resources.
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        queue_depth: Optional[int] = None
    ) -> None:
        """
        Initialize the orchestrator.
        
        Args:
            config_path: Path to the engine configuration file
            batch_size: Override the batch size from the configuration
            queue_depth: Override the queue depth from the configuration
        """
        # Load configuration
        self.config = get_engine_config(config_path)
        
        # Override configuration if specified
        if batch_size is not None:
            self.config.pipeline.batch_size = batch_size
        if queue_depth is not None:
            self.config.pipeline.queue_depth = queue_depth
        
        # Initialize pipeline stages
        self.stages: Dict[str, StageBase] = {}
        self.stage_devices: Dict[str, torch.device] = {}
        self.queues: Dict[str, asyncio.Queue] = {}
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Set up GPU devices
        self.setup_devices()
        
        # Initialize metrics
        self.metrics: Dict[str, Any] = {
            "start_time": 0,
            "end_time": 0,
            "total_batches": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "errors": 0,
            "stage_metrics": {},
        }
    
    def setup_devices(self) -> None:
        """
        Set up GPU devices based on the configuration.
        """
        # Map stages to devices
        for gpu, stage_names in [
            ("gpu0", self.config.pipeline.layout.gpu0),
            ("gpu1", self.config.pipeline.layout.gpu1),
        ]:
            device_idx = int(gpu[-1])  # Extract device index from gpu name
            device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
            
            for stage_name in stage_names:
                self.stage_devices[stage_name] = device
        
        logger.info(f"Stage to device mapping: {self.stage_devices}")
    
    def create_stages(self) -> None:
        """
        Create pipeline stages based on the configuration.
        """
        # Create stages
        for stage_name, device in self.stage_devices.items():
            if stage_name == "DocProc":
                self.stages[stage_name] = DocProcStage(
                    device=device,
                    timeout_seconds=self.config.pipeline.stages.DocProc.timeout_seconds
                )
            elif stage_name == "Chunk":
                self.stages[stage_name] = ChunkStage(
                    device=device,
                    max_tokens=self.config.pipeline.stages.Chunk.max_tokens,
                    use_overlap=self.config.pipeline.stages.Chunk.use_overlap
                )
            else:
                # For now, we only implement DocProc and Chunk stages
                logger.warning(f"Stage {stage_name} not implemented yet")
        
        logger.info(f"Created stages: {list(self.stages.keys())}")
    
    def create_queues(self) -> None:
        """
        Create queues between pipeline stages.
        """
        # Create input queue
        self.queues["input"] = asyncio.Queue(maxsize=self.config.pipeline.queue_depth)
        
        # Create queues between stages
        stage_names = list(self.stages.keys())
        for i in range(len(stage_names) - 1):
            from_stage = stage_names[i]
            to_stage = stage_names[i + 1]
            queue_name = f"{from_stage}_to_{to_stage}"
            self.queues[queue_name] = asyncio.Queue(maxsize=self.config.pipeline.queue_depth)
        
        # Create output queue
        self.queues["output"] = asyncio.Queue(maxsize=self.config.pipeline.queue_depth)
        
        logger.info(f"Created queues: {list(self.queues.keys())}")
    
    async def stage_worker(
        self, stage_name: str, input_queue: asyncio.Queue, output_queue: asyncio.Queue
    ) -> None:
        """
        Worker function for a pipeline stage.
        
        Args:
            stage_name: Name of the stage
            input_queue: Queue to get batches from
            output_queue: Queue to put processed batches into
        """
        stage = self.stages[stage_name]
        
        # Set up the stage
        stage.setup()
        
        try:
            while self.running:
                # Get a batch from the input queue
                batch = await input_queue.get()
                
                # Check for poison pill
                if batch is None:
                    logger.info(f"Stage {stage_name} received poison pill, shutting down")
                    await output_queue.put(None)  # Forward the poison pill
                    break
                
                # Process the batch
                try:
                    result = await stage(batch)
                    
                    # Put the result in the output queue
                    await output_queue.put(result)
                    
                    # Mark the task as done
                    input_queue.task_done()
                    
                    logger.debug(f"Stage {stage_name} processed batch {batch.batch_id}")
                except Exception as e:
                    logger.error(f"Error in stage {stage_name}: {str(e)}", exc_info=True)
                    
                    # Add the error to the batch
                    batch.add_error(stage_name, e)
                    
                    # Put the batch with error in the output queue
                    await output_queue.put(batch)
                    
                    # Mark the task as done
                    input_queue.task_done()
        finally:
            # Tear down the stage
            stage.teardown()
            
            # Update metrics
            self.metrics["stage_metrics"][stage_name] = stage.get_metrics()
    
    async def output_worker(self, output_queue: asyncio.Queue) -> None:
        """
        Worker function for the output queue.
        
        Args:
            output_queue: Queue to get processed batches from
        """
        while self.running:
            # Get a batch from the output queue
            batch = await output_queue.get()
            
            # Check for poison pill
            if batch is None:
                logger.info("Output worker received poison pill, shutting down")
                break
            
            # Process the batch
            try:
                # Update metrics
                self.metrics["total_batches"] += 1
                self.metrics["total_documents"] += len(batch.docs)
                if batch.chunks:
                    self.metrics["total_chunks"] += len(batch.chunks)
                if batch.has_errors:
                    self.metrics["errors"] += len(batch.errors)
                
                # Log batch completion
                logger.info(
                    f"Batch {batch.batch_id} completed: "
                    f"{len(batch.docs)} docs, "
                    f"{len(batch.chunks) if batch.chunks else 0} chunks, "
                    f"{len(batch.errors) if batch.has_errors else 0} errors"
                )
                
                # TODO: Write the batch to the database
                # For now, we just mark the task as done
                output_queue.task_done()
            except Exception as e:
                logger.error(f"Error in output worker: {str(e)}", exc_info=True)
                output_queue.task_done()
    
    async def create_batch_from_files(
        self, file_paths: List[Union[str, Path]]
    ) -> PipelineBatch:
        """
        Create a batch from a list of file paths.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            A batch containing the files
        """
        # Create document dictionaries
        docs = []
        for path in file_paths:
            path_str = str(path)
            docs.append({
                "id": f"doc_{uuid.uuid4().hex[:8]}",
                "path": path_str,
                "type": os.path.splitext(path_str)[1].lower().lstrip(".") or "text"
            })
        
        # Create the batch
        return PipelineBatch(
            batch_id=f"batch_{uuid.uuid4().hex[:8]}",
            docs=docs
        )
    
    async def process_directory(
        self, 
        directory: Union[str, Path], 
        recursive: bool = True,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Process all files in a directory.
        
        Args:
            directory: Directory to process
            recursive: Whether to process subdirectories
            batch_size: Override the batch size from the configuration
        """
        # Use the configured batch size if not specified
        if batch_size is None:
            batch_size = self.config.pipeline.batch_size
        
        # Get all files in the directory
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Collect all files
        files = []
        if recursive:
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    files.append(Path(root) / filename)
        else:
            files = [f for f in directory.iterdir() if f.is_file()]
        
        # Sort files for deterministic ordering
        files.sort()
        
        # Create batches
        batches = []
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            # Convert Path objects to strings to avoid type errors
            # Use a list that accepts both str and Path objects
            typed_files: List[Union[str, Path]] = [str(f) for f in batch_files]
            batch = await self.create_batch_from_files(typed_files)
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches from {len(files)} files in {directory}")
        
        # Process the batches
        for batch in batches:
            await self.queues["input"].put(batch)
    
    async def start(self) -> None:
        """
        Start the pipeline.
        """
        # Create stages and queues
        self.create_stages()
        self.create_queues()
        
        # Set running flag
        self.running = True
        
        # Start time
        self.metrics["start_time"] = time.time()
        
        # Create worker tasks
        stage_names = list(self.stages.keys())
        for i, stage_name in enumerate(stage_names):
            # Determine input and output queues
            if i == 0:
                input_queue = self.queues["input"]
            else:
                prev_stage = stage_names[i-1]
                input_queue = self.queues[f"{prev_stage}_to_{stage_name}"]
            
            if i == len(stage_names) - 1:
                output_queue = self.queues["output"]
            else:
                next_stage = stage_names[i+1]
                output_queue = self.queues[f"{stage_name}_to_{next_stage}"]
            
            # Create the worker task
            task = asyncio.create_task(
                self.stage_worker(stage_name, input_queue, output_queue)
            )
            self.tasks.append(task)
        
        # Create output worker
        output_task = asyncio.create_task(
            self.output_worker(self.queues["output"])
        )
        self.tasks.append(output_task)
        
        logger.info("Pipeline started")
    
    async def stop(self) -> None:
        """
        Stop the pipeline.
        """
        # Set running flag
        self.running = False
        
        # Send poison pills to all stages
        await self.queues["input"].put(None)
        
        # Wait for all tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # End time
        self.metrics["end_time"] = time.time()
        
        # Calculate elapsed time
        elapsed = self.metrics["end_time"] - self.metrics["start_time"]
        self.metrics["elapsed_seconds"] = elapsed
        
        # Log metrics
        logger.info(
            f"Pipeline stopped after {elapsed:.2f}s: "
            f"{self.metrics['total_batches']} batches, "
            f"{self.metrics['total_documents']} documents, "
            f"{self.metrics['total_chunks']} chunks, "
            f"{self.metrics['errors']} errors"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get pipeline metrics.
        
        Returns:
            A dictionary of metrics
        """
        return self.metrics


async def run_pipeline(
    input_dir: Union[str, Path],
    config_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    queue_depth: Optional[int] = None,
    recursive: bool = True
) -> Dict[str, Any]:
    """
    Run the pipeline on an input directory.
    
    Args:
        input_dir: Directory containing input files
        config_path: Path to the engine configuration file
        batch_size: Override the batch size from the configuration
        queue_depth: Override the queue depth from the configuration
        recursive: Whether to process subdirectories
        
    Returns:
        Pipeline metrics
    """
    # Create the orchestrator
    orchestrator = PipelineOrchestrator(
        config_path=config_path,
        batch_size=batch_size,
        queue_depth=queue_depth
    )
    
    # Start the pipeline
    await orchestrator.start()
    
    try:
        # Process the input directory
        await orchestrator.process_directory(
            directory=input_dir,
            recursive=recursive,
            batch_size=batch_size
        )
        
        # Wait for all queues to be empty
        input_queue = orchestrator.queues["input"]
        await input_queue.join()
        
        for queue_name, queue in orchestrator.queues.items():
            if queue_name != "input":
                await queue.join()
    finally:
        # Stop the pipeline
        await orchestrator.stop()
    
    # Return metrics
    return orchestrator.get_metrics()


if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the pipeline on an input directory")
    parser.add_argument("input_dir", help="Directory containing input files")
    parser.add_argument("--config", help="Path to the engine configuration file")
    parser.add_argument("--batch-size", type=int, help="Override the batch size from the configuration")
    parser.add_argument("--queue-depth", type=int, help="Override the queue depth from the configuration")
    parser.add_argument("--no-recursive", action="store_true", help="Do not process subdirectories")
    args = parser.parse_args()
    
    # Run the pipeline
    asyncio.run(
        run_pipeline(
            input_dir=args.input_dir,
            config_path=args.config,
            batch_size=args.batch_size,
            queue_depth=args.queue_depth,
            recursive=not args.no_recursive
        )
    )
