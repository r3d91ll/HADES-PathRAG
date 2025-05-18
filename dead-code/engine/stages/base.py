"""
Base stage module.

This module defines the base class for all pipeline stages.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Type

import torch
from torch.cuda import Stream as CudaStream

from ..batch_types import PipelineBatch


logger = logging.getLogger(__name__)


class StageBase(abc.ABC):
    """
    Base class for all pipeline stages.
    
    This class provides common functionality for all stages, including:
    - Device management
    - CUDA stream management
    - Timing and metrics
    - Error handling
    """
    
    name: str = "BaseStage"  # Override in subclasses
    
    def __init__(self, device: Union[str, torch.device]) -> None:
        """
        Initialize the stage.
        
        Args:
            device: The device to run this stage on (e.g., "cuda:0", "cuda:1")
        """
        self.device = torch.device(device)
        self.compute_stream: Optional[CudaStream] = None
        self.transfer_stream: Optional[CudaStream] = None
        self.metrics: Dict[str, List[float]] = {
            "batch_time": [],
            "batch_size": [],
        }
    
    def setup(self) -> None:
        """
        Set up the stage.
        
        This method is called once before processing begins.
        It should initialize any resources needed by the stage.
        """
        if self.device.type == "cuda":
            # Set the current device
            torch.cuda.set_device(self.device)
            
            # Create CUDA streams for compute and transfer
            self.compute_stream = torch.cuda.Stream(device=self.device)
            self.transfer_stream = torch.cuda.Stream(device=self.device)
            
            logger.info(f"Stage {self.name} set up on device {self.device}")
        else:
            logger.info(f"Stage {self.name} set up on CPU")
    
    def teardown(self) -> None:
        """
        Clean up the stage.
        
        This method is called once after processing is complete.
        It should release any resources used by the stage.
        """
        # Streams are automatically destroyed when they go out of scope
        self.compute_stream = None
        self.transfer_stream = None
        
        logger.info(f"Stage {self.name} torn down")
    
    @abc.abstractmethod
    async def process_batch(self, batch: PipelineBatch) -> PipelineBatch:
        """
        Process a batch.
        
        This method must be implemented by all subclasses.
        
        Args:
            batch: The batch to process
            
        Returns:
            The processed batch
        """
        raise NotImplementedError("Subclasses must implement process_batch")
    
    async def __call__(self, batch: PipelineBatch) -> PipelineBatch:
        """
        Process a batch and update metrics.
        
        This method wraps the process_batch method with timing and error handling.
        
        Args:
            batch: The batch to process
            
        Returns:
            The processed batch
        """
        start_time = time.time()
        
        try:
            # Process the batch
            result = await self.process_batch(batch)
            
            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["batch_time"].append(elapsed)
            self.metrics["batch_size"].append(len(batch.docs))
            
            logger.debug(f"Stage {self.name} processed batch {batch.batch_id} in {elapsed:.2f}s")
            
            return result
        
        except Exception as e:
            # Log the error
            logger.error(f"Error in stage {self.name}: {str(e)}", exc_info=True)
            
            # Add the error to the batch
            batch.add_error(self.name, e)
            
            # Return the batch with the error
            return batch
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for this stage.
        
        Returns:
            A dictionary of metrics
        """
        metrics: Dict[str, Any] = {
            "name": self.name,
            "device": str(self.device),
        }
        
        # Calculate statistics for numeric metrics
        for name, values in self.metrics.items():
            if values:
                metrics[f"{name}_mean"] = sum(values) / len(values)
                metrics[f"{name}_min"] = min(values)
                metrics[f"{name}_max"] = max(values)
                metrics[f"{name}_count"] = len(values)
        
        return metrics
