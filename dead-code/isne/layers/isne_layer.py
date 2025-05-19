"""
ISNE (Inductive Shallow Node Embedding) layer implementations.

This module contains the neural network layers used for the ISNE model,
primarily using PyTorch and PyTorch Geometric (if available).
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable, cast
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for type hinting
NodeFeatures = Tensor  # Node features tensor (num_nodes, feature_dim)
EdgeIndex = Tensor     # Edge indices tensor (2, num_edges)
EdgeWeight = Tensor    # Edge weights tensor (num_edges,)
GraphData = Tuple[NodeFeatures, EdgeIndex, Optional[EdgeWeight]]


class ISNEFeaturePropagation(nn.Module):
    """
    Feature propagation layer for ISNE implementation.
    
    This layer implements the message passing and feature aggregation
    functionality used in the ISNE model.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        bias: bool = True,
        dropout: float = 0.0,
        activation: Optional[Callable[[Tensor], Tensor]] = None
    ) -> None:
        """
        Initialize the ISNEFeaturePropagation layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias term
            dropout: Dropout probability
            activation: Activation function to use
        """
        super(ISNEFeaturePropagation, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.activation = activation
        
        # Define learnable parameters
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Reset the layer parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def scatter_mean(
        self, 
        src: Tensor, 
        index: Tensor, 
        dim: int = 0, 
        dim_size: Optional[int] = None
    ) -> Tensor:
        """
        Custom implementation of scatter_mean operation.
        
        This function aggregates features from source nodes to target nodes
        based on the provided indices, computing the mean of values.
        
        Args:
            src: Source tensor of features
            index: Index tensor specifying the mapping from src to output
            dim: Dimension along which to index
            dim_size: Size of the output tensor along dimension dim
            
        Returns:
            Tensor with aggregated features
        """
        if dim_size is None:
            dim_size = int(index.max()) + 1
        
        # Create output tensor
        out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
        
        # Compute counts for each target index for mean calculation
        ones = torch.ones(index.size(0), dtype=src.dtype, device=src.device)
        count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        count.scatter_add_(0, index, ones)
        
        # Sum the source features for each target index
        out.scatter_add_(0, index.view(-1, 1).expand(-1, src.size(1)), src)
        
        # Compute mean by dividing by counts (avoiding division by zero)
        count = torch.clamp(count.view(-1, 1), min=1)
        out = out / count
        
        return out
    
    def forward(
        self, 
        x: Tensor, 
        edge_index: Tensor, 
        edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for the feature propagation layer.
        
        Args:
            x: Node feature tensor [num_nodes, in_features]
            edge_index: Edge index tensor [2, num_edges]
            edge_weight: Optional edge weight tensor [num_edges]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Apply dropout to input features
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply linear transformation
        x = torch.matmul(x, self.weight)
        
        # Get source and target nodes
        src, dst = edge_index[0], edge_index[1]
        
        # Apply edge weights if provided
        if edge_weight is not None:
            x_j = x[src] * edge_weight.view(-1, 1)
        else:
            x_j = x[src]
        
        # Aggregate features (mean aggregation)
        out = self.scatter_mean(x_j, dst, dim=0, dim_size=x.size(0))
        
        # Apply bias if present
        if self.bias is not None:
            out = out + self.bias
        
        # Apply activation function if specified
        if self.activation is not None:
            out = self.activation(out)
            
        return out


class ISNELayer(nn.Module):
    """
    ISNE (Inductive Shallow Node Embedding) layer implementation.
    
    This layer combines feature transformation and propagation steps
    for inductive node embedding in graph neural networks.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        residual: bool = True,
        normalize: bool = True,
        bias: bool = True
    ) -> None:
        """
        Initialize the ISNELayer.
        
        Args:
            in_features: Number of input features
            hidden_features: Number of hidden features
            out_features: Number of output features
            num_layers: Number of propagation layers
            dropout: Dropout probability
            residual: Whether to use residual connections
            normalize: Whether to normalize output features
            bias: Whether to include bias term
        """
        super(ISNELayer, self).__init__()
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.normalize = normalize
        
        # Input projection layer
        self.input_projection = nn.Linear(in_features, hidden_features, bias=bias)
        
        # Create propagation layers
        self.propagation_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = hidden_features
            layer_out = hidden_features if i < num_layers - 1 else out_features
            activation = F.relu if i < num_layers - 1 else None
            
            self.propagation_layers.append(
                ISNEFeaturePropagation(
                    in_features=layer_in,
                    out_features=layer_out,
                    bias=bias,
                    dropout=dropout,
                    activation=activation
                )
            )
    
    def forward(
        self, 
        x: Tensor, 
        edge_index: Tensor, 
        edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for the ISNE layer.
        
        Args:
            x: Node feature tensor [num_nodes, in_features]
            edge_index: Edge index tensor [2, num_edges]
            edge_weight: Optional edge weight tensor [num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_features]
        """
        # Initial projection of features
        h = self.input_projection(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Store original features for residual connection
        h_prev = h
        
        # Apply propagation layers
        for i, prop_layer in enumerate(self.propagation_layers):
            h = prop_layer(h, edge_index, edge_weight)
            
            # Apply residual connection except for the last layer
            if self.residual and i < len(self.propagation_layers) - 1:
                h = h + h_prev
                h_prev = h
        
        # Normalize output embeddings if requested
        if self.normalize:
            h = F.normalize(h, p=2, dim=1)
            
        return h
