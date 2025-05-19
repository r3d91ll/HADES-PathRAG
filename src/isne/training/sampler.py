"""
Neighborhood sampling implementations for ISNE training.

This module provides efficient sampling strategies for training ISNE models
on large graphs, including node sampling, edge sampling, and neighborhood sampling.
"""

import torch
import numpy as np
from torch import Tensor
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set

# Set up logging
logger = logging.getLogger(__name__)

try:
    import torch_geometric
    from torch_geometric.utils import subgraph
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch Geometric not available. Install with: pip install torch-geometric")
    TORCH_GEOMETRIC_AVAILABLE = False


class NeighborSampler:
    """
    Neighbor sampling strategy for efficient ISNE training.
    
    This sampler provides methods for sampling nodes, edges, and neighborhoods
    to create mini-batches for training the ISNE model on large graphs.
    """
    
    def __init__(
        self,
        edge_index: Tensor,
        num_nodes: int,
        batch_size: int = 32,
        num_hops: int = 1,
        neighbor_size: int = 10,
        directed: bool = False,
        replace: bool = False,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the neighbor sampler.
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            num_nodes: Number of nodes in the graph
            batch_size: Batch size for node sampling
            num_hops: Number of hops to sample
            neighbor_size: Maximum number of neighbors to sample per node per hop
            directed: Whether the graph is directed
            replace: Whether to sample with replacement
            seed: Random seed for reproducibility
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.num_hops = num_hops
        self.neighbor_size = neighbor_size
        self.directed = directed
        self.replace = replace
        
        # Create adjacency list for efficient neighbor sampling
        self._create_adj_list()
        
        # Set random seed if provided
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    
    def _create_adj_list(self) -> None:
        """
        Create adjacency list from edge index for efficient neighbor access.
        """
        # Initialize empty lists
        self.adj_list = [[] for _ in range(self.num_nodes)]
        
        # Fill adjacency list
        for i in range(self.edge_index.size(1)):
            src = self.edge_index[0, i].item()
            dst = self.edge_index[1, i].item()
            
            # Add edge
            self.adj_list[src].append(dst)
            
            # If undirected, add reverse edge
            if not self.directed:
                self.adj_list[dst].append(src)
    
    def sample_nodes(self) -> Tensor:
        """
        Sample a batch of nodes for training.
        
        Returns:
            Batch of node indices [batch_size]
        """
        # Sample random node indices
        batch_idx = self.rng.choice(
            self.num_nodes, size=self.batch_size, replace=self.replace)
        
        # Convert to tensor
        return torch.tensor(batch_idx, dtype=torch.long, device=self.edge_index.device)
    
    def sample_neighbors(self, nodes: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Sample multi-hop neighborhoods for a batch of nodes.
        
        Args:
            nodes: Batch of node indices [batch_size]
            
        Returns:
            Tuple of (node_samples, edge_samples) for each hop:
                - node_samples: List of node indices for each hop
                - edge_samples: List of edge indices for each hop
        """
        # Convert to numpy for efficiency if not already
        if isinstance(nodes, torch.Tensor):
            nodes_np = nodes.cpu().numpy()
        else:
            nodes_np = np.array(nodes)
        
        # Lists to store results
        node_samples = []
        edge_samples = []
        
        # Set of nodes seen so far (for deduplication)
        seen_nodes = set(nodes_np.tolist())
        
        # Current frontier
        frontier = nodes_np
        
        # Sample for each hop
        for _ in range(self.num_hops):
            # Initialize arrays for this hop
            hop_nodes = []
            hop_edges_src = []
            hop_edges_dst = []
            
            # Process each node in the frontier
            for node in frontier:
                # Get neighbors
                neighbors = self.adj_list[node]
                
                if not neighbors:
                    continue
                
                # Sample neighbors (with or without replacement)
                if len(neighbors) > self.neighbor_size:
                    sampled_neighbors = self.rng.choice(
                        neighbors, size=self.neighbor_size, replace=self.replace)
                else:
                    sampled_neighbors = neighbors
                
                # Add sampled edges
                for neighbor in sampled_neighbors:
                    hop_edges_src.append(node)
                    hop_edges_dst.append(neighbor)
                    
                    # Add new nodes
                    if neighbor not in seen_nodes:
                        hop_nodes.append(neighbor)
                        seen_nodes.add(neighbor)
            
            # Update frontier with new nodes
            frontier = np.array(hop_nodes)
            
            # Convert to tensors and add to results
            if hop_edges_src:
                src_tensor = torch.tensor(hop_edges_src, dtype=torch.long, 
                                         device=self.edge_index.device)
                dst_tensor = torch.tensor(hop_edges_dst, dtype=torch.long,
                                         device=self.edge_index.device)
                
                # Add to results
                node_samples.append(torch.tensor(hop_nodes, dtype=torch.long,
                                               device=self.edge_index.device))
                edge_samples.append(torch.stack([src_tensor, dst_tensor], dim=0))
            else:
                # Empty hop
                node_samples.append(torch.tensor([], dtype=torch.long,
                                               device=self.edge_index.device))
                edge_samples.append(torch.zeros((2, 0), dtype=torch.long,
                                               device=self.edge_index.device))
        
        return node_samples, edge_samples
    
    def sample_subgraph(self, nodes: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Sample a subgraph around the given nodes.
        
        Args:
            nodes: Seed node indices
            
        Returns:
            Tuple of (subset_nodes, subgraph_edge_index):
                - subset_nodes: Node indices in the subgraph
                - subgraph_edge_index: Edge index of the subgraph
        """
        # Check that PyTorch Geometric is available
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric is required for subgraph sampling. "
                "Install with: pip install torch-geometric"
            )
        
        # Sample multi-hop neighborhoods
        node_samples, _ = self.sample_neighbors(nodes)
        
        # Combine all sampled nodes
        all_nodes = [nodes]
        all_nodes.extend(node_samples)
        
        subset_nodes = torch.cat([n for n in all_nodes if n.numel() > 0], dim=0)
        
        # Remove duplicates
        subset_nodes = torch.unique(subset_nodes)
        
        # Extract subgraph
        subgraph_edge_index, _ = subgraph(
            subset_nodes, self.edge_index, relabel_nodes=True)
        
        return subset_nodes, subgraph_edge_index
    
    def sample_triplets(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample triplets (anchor, positive, negative) for triplet loss.
        
        Args:
            batch_size: Number of triplets to sample (defaults to self.batch_size)
            
        Returns:
            Tuple of (anchor_indices, positive_indices, negative_indices)
        """
        batch_size = batch_size or self.batch_size
        
        # Lists to store results
        anchor_indices = []
        positive_indices = []
        negative_indices = []
        
        while len(anchor_indices) < batch_size:
            # Sample random anchor node
            anchor = self.rng.randint(0, self.num_nodes)
            
            # Skip if no neighbors
            if not self.adj_list[anchor]:
                continue
            
            # Sample positive neighbor
            positive = self.rng.choice(self.adj_list[anchor])
            
            # Sample negative node (not connected to anchor)
            while True:
                negative = self.rng.randint(0, self.num_nodes)
                
                # Check if not connected to anchor
                if negative != anchor and negative not in self.adj_list[anchor]:
                    break
            
            # Add triplet
            anchor_indices.append(anchor)
            positive_indices.append(positive)
            negative_indices.append(negative)
        
        # Convert to tensors
        device = self.edge_index.device
        return (
            torch.tensor(anchor_indices, dtype=torch.long, device=device),
            torch.tensor(positive_indices, dtype=torch.long, device=device),
            torch.tensor(negative_indices, dtype=torch.long, device=device)
        )
    
    def sample_positive_pairs(self, batch_size: Optional[int] = None) -> Tensor:
        """
        Sample positive pairs for contrastive loss.
        
        Args:
            batch_size: Number of pairs to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of positive pair indices [num_pairs, 2]
        """
        batch_size = batch_size or self.batch_size
        
        # Lists to store results
        pair_indices = []
        
        # Sample pairs of connected nodes
        while len(pair_indices) < batch_size:
            # Sample random node
            src = self.rng.randint(0, self.num_nodes)
            
            # Skip if no neighbors
            if not self.adj_list[src]:
                continue
            
            # Sample random neighbor
            dst = self.rng.choice(self.adj_list[src])
            
            # Add pair
            pair_indices.append([src, dst])
        
        # Convert to tensor
        return torch.tensor(pair_indices, dtype=torch.long, device=self.edge_index.device)
    
    def sample_negative_pairs(self, 
                             positive_pairs: Optional[Tensor] = None,
                             batch_size: Optional[int] = None) -> Tensor:
        """
        Sample negative pairs for contrastive loss.
        
        Args:
            positive_pairs: Optional tensor of positive pair indices to avoid
            batch_size: Number of pairs to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of negative pair indices [num_pairs, 2]
        """
        batch_size = batch_size or self.batch_size
        
        # Lists to store results
        pair_indices = []
        
        # Create set of existing edges for fast lookup
        edge_set = set()
        for i in range(self.edge_index.size(1)):
            src = self.edge_index[0, i].item()
            dst = self.edge_index[1, i].item()
            edge_set.add((src, dst))
            
            # If undirected, add reverse edge
            if not self.directed:
                edge_set.add((dst, src))
        
        # Add positive pairs to avoid if provided
        if positive_pairs is not None:
            for i in range(positive_pairs.size(0)):
                src = positive_pairs[i, 0].item()
                dst = positive_pairs[i, 1].item()
                edge_set.add((src, dst))
                
                # If undirected, add reverse edge
                if not self.directed:
                    edge_set.add((dst, src))
        
        # Sample pairs of disconnected nodes
        while len(pair_indices) < batch_size:
            # Sample random nodes
            src = self.rng.randint(0, self.num_nodes)
            dst = self.rng.randint(0, self.num_nodes)
            
            # Skip if self-loop or existing edge
            if src == dst or (src, dst) in edge_set:
                continue
            
            # Add pair
            pair_indices.append([src, dst])
        
        # Convert to tensor
        return torch.tensor(pair_indices, dtype=torch.long, device=self.edge_index.device)
