"""
ISNE model trainer implementation.

This module implements the trainer for the ISNE model, orchestrating the
training process including loss computation, optimization, and evaluation.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import logging
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.isne.models.isne_model import ISNEModel
from src.isne.losses.feature_loss import FeaturePreservationLoss
from src.isne.losses.structural_loss import StructuralPreservationLoss
from src.isne.losses.contrastive_loss import ContrastiveLoss
from src.isne.training.sampler import NeighborSampler

# Set up logging
logger = logging.getLogger(__name__)


class ISNETrainer:
    """
    Trainer for the ISNE model.
    
    This class implements the training procedure for the ISNE model as described
    in the original paper, including multi-objective loss computation, mini-batch
    training, and model evaluation.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        lambda_feat: float = 1.0,
        lambda_struct: float = 1.0,
        lambda_contrast: float = 0.5,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """
        Initialize the ISNE trainer.
        
        Args:
            embedding_dim: Dimensionality of input embeddings
            hidden_dim: Dimensionality of hidden representations
            output_dim: Dimensionality of output embeddings (defaults to hidden_dim)
            num_layers: Number of ISNE layers in the model
            num_heads: Number of attention heads in each layer
            dropout: Dropout probability for regularization
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            lambda_feat: Weight for feature preservation loss
            lambda_struct: Weight for structural preservation loss
            lambda_contrast: Weight for contrastive loss
            device: Device to use for training
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_feat = lambda_feat
        self.lambda_struct = lambda_struct
        self.lambda_contrast = lambda_contrast
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        logger.info(f"ISNE trainer initialized on device: {self.device}")
        
        # Initialize model, losses, and optimizer
        self.model = None
        self.feature_loss = None
        self.structural_loss = None
        self.contrastive_loss = None
        self.optimizer = None
        
        # Training statistics
        self.train_stats = {
            'epochs': 0,
            'total_loss': [],
            'feature_loss': [],
            'structural_loss': [],
            'contrastive_loss': [],
            'time_per_epoch': []
        }
    
    def _initialize_model(self) -> None:
        """
        Initialize the ISNE model.
        """
        self.model = ISNEModel(
            in_features=self.embedding_dim,
            hidden_features=self.hidden_dim,
            out_features=self.output_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            residual=True,
            normalize_features=True
        ).to(self.device)
    
    def _initialize_losses(self) -> None:
        """
        Initialize loss functions.
        """
        self.feature_loss = FeaturePreservationLoss(
            lambda_feat=self.lambda_feat
        )
        
        self.structural_loss = StructuralPreservationLoss(
            lambda_struct=self.lambda_struct,
            negative_samples=5
        )
        
        self.contrastive_loss = ContrastiveLoss(
            lambda_contrast=self.lambda_contrast
        )
    
    def _initialize_optimizer(self) -> None:
        """
        Initialize optimizer.
        """
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def prepare_model(self) -> None:
        """
        Prepare the model, loss functions, and optimizer for training.
        """
        self._initialize_model()
        self._initialize_losses()
        self._initialize_optimizer()
    
    def train(
        self,
        features: Tensor,
        edge_index: Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        num_hops: int = 1,
        neighbor_size: int = 10,
        eval_interval: int = 10,
        early_stopping_patience: int = 20,
        validation_data: Optional[Tuple[Tensor, Tensor]] = None,
        validation_metric: str = 'loss',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the ISNE model.
        
        Args:
            features: Node feature tensor [num_nodes, embedding_dim]
            edge_index: Edge index tensor [2, num_edges]
            epochs: Number of training epochs
            batch_size: Training batch size
            num_hops: Number of hops for neighborhood sampling
            neighbor_size: Maximum number of neighbors to sample per node per hop
            eval_interval: Interval for evaluation during training
            early_stopping_patience: Patience for early stopping
            validation_data: Optional validation data (features, edge_index)
            validation_metric: Metric for validation ('loss' or 'similarity')
            verbose: Whether to show progress bar
            
        Returns:
            Training statistics
        """
        # Prepare model if not already done
        if self.model is None:
            self.prepare_model()
        
        # Move data to device
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Initialize sampler
        num_nodes = features.size(0)
        sampler = NeighborSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=batch_size,
            num_hops=num_hops,
            neighbor_size=neighbor_size
        )
        
        # Training loop
        best_val_metric = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Set model to training mode
            self.model.train()
            
            # Sample batch of nodes
            batch_nodes = sampler.sample_nodes()
            
            # Sample neighborhood
            subset_nodes, subgraph_edge_index = sampler.sample_subgraph(batch_nodes)
            
            # Get features for subgraph
            subgraph_features = features[subset_nodes]
            
            # Forward pass
            self.optimizer.zero_grad()
            embeddings = self.model(subgraph_features, subgraph_edge_index)
            
            # Project features for feature loss
            projected_features = self.model.project_features(subgraph_features)
            
            # Compute feature preservation loss
            feat_loss = self.feature_loss(embeddings, projected_features)
            
            # Sample positive and negative pairs for contrastive loss
            pos_pairs = sampler.sample_positive_pairs()
            neg_pairs = sampler.sample_negative_pairs(pos_pairs)
            
            # Compute structural preservation loss
            struct_loss = self.structural_loss(embeddings, subgraph_edge_index)
            
            # Compute contrastive loss
            cont_loss = self.contrastive_loss(
                embeddings, 
                pos_pairs.to(self.device), 
                neg_pairs.to(self.device)
            )
            
            # Combine losses
            total_loss = feat_loss + struct_loss + cont_loss
            
            # Backward pass and optimization
            total_loss.backward()
            self.optimizer.step()
            
            # Record training statistics
            epoch_time = time.time() - epoch_start_time
            self.train_stats['total_loss'].append(total_loss.item())
            self.train_stats['feature_loss'].append(feat_loss.item())
            self.train_stats['structural_loss'].append(struct_loss.item())
            self.train_stats['contrastive_loss'].append(cont_loss.item())
            self.train_stats['time_per_epoch'].append(epoch_time)
            
            # Evaluate and check for early stopping
            if validation_data is not None and (epoch + 1) % eval_interval == 0:
                val_features, val_edge_index = validation_data
                val_features = val_features.to(self.device)
                val_edge_index = val_edge_index.to(self.device)
                
                val_metric = self.evaluate(val_features, val_edge_index, metric=validation_metric)
                
                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, "
                           f"Loss: {total_loss.item():.4f}, "
                           f"Time: {epoch_time:.2f}s")
        
        # Update epochs trained
        self.train_stats['epochs'] = epoch + 1
        
        return self.train_stats
    
    def evaluate(
        self,
        features: Tensor,
        edge_index: Tensor,
        metric: str = 'loss'
    ) -> float:
        """
        Evaluate the model on the given data.
        
        Args:
            features: Node feature tensor [num_nodes, embedding_dim]
            edge_index: Edge index tensor [2, num_edges]
            metric: Evaluation metric ('loss' or 'similarity')
            
        Returns:
            Evaluation score
        """
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            embeddings = self.model(features, edge_index)
            
            if metric == 'loss':
                # Project features for feature loss
                projected_features = self.model.project_features(features)
                
                # Compute losses
                feat_loss = self.feature_loss(embeddings, projected_features)
                struct_loss = self.structural_loss(embeddings, edge_index)
                
                # Total loss
                total_loss = feat_loss + struct_loss
                
                return total_loss.item()
            
            elif metric == 'similarity':
                # Compute average cosine similarity for connected nodes
                src, dst = edge_index
                src_emb = embeddings[src]
                dst_emb = embeddings[dst]
                
                sim = torch.nn.functional.cosine_similarity(src_emb, dst_emb)
                
                # Return negative similarity (since we're minimizing)
                return -sim.mean().item()
            
            else:
                raise ValueError(f"Unknown evaluation metric: {metric}")
    
    def get_embeddings(
        self,
        features: Tensor,
        edge_index: Tensor
    ) -> Tensor:
        """
        Get embeddings for the given nodes.
        
        Args:
            features: Node feature tensor [num_nodes, embedding_dim]
            edge_index: Edge index tensor [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Move data to device
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            embeddings = self.model(features, edge_index)
            
            return embeddings.cpu()
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Create directory if it doesn't exist
        path = Path(path) if isinstance(path, str) else path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'dropout': self.dropout
            },
            'train_stats': self.train_stats
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load the model from the specified path.
        
        Args:
            path: Path to load the model from
        """
        path = Path(path) if isinstance(path, str) else path
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update configuration
        config = checkpoint['config']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        
        # Initialize model
        self._initialize_model()
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize optimizer
        self._initialize_optimizer()
        
        # Load optimizer state if available
        if checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training statistics
        if 'train_stats' in checkpoint:
            self.train_stats = checkpoint['train_stats']
        
        logger.info(f"Model loaded from {path}")
    
    def visualize_embeddings(
        self,
        embeddings: Tensor,
        labels: Optional[Tensor] = None,
        method: str = 'tsne',
        n_components: int = 2,
        perplexity: int = 30,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Visualize embeddings using dimensionality reduction.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            labels: Optional node labels for coloring
            method: Dimensionality reduction method ('tsne', 'pca', 'umap')
            n_components: Number of components for visualization
            perplexity: Perplexity parameter for t-SNE
            random_state: Random state for reproducibility
            
        Returns:
            Low-dimensional representation of embeddings [num_nodes, n_components]
        """
        try:
            # Convert to numpy if tensor
            if isinstance(embeddings, torch.Tensor):
                embeddings_np = embeddings.cpu().numpy()
            else:
                embeddings_np = np.array(embeddings)
            
            if method == 'tsne':
                from sklearn.manifold import TSNE
                
                tsne = TSNE(
                    n_components=n_components,
                    perplexity=perplexity,
                    random_state=random_state
                )
                
                embeddings_2d = tsne.fit_transform(embeddings_np)
                
            elif method == 'pca':
                from sklearn.decomposition import PCA
                
                pca = PCA(n_components=n_components, random_state=random_state)
                embeddings_2d = pca.fit_transform(embeddings_np)
                
            elif method == 'umap':
                try:
                    import umap
                    
                    reducer = umap.UMAP(
                        n_components=n_components,
                        random_state=random_state
                    )
                    
                    embeddings_2d = reducer.fit_transform(embeddings_np)
                    
                except ImportError:
                    logger.warning("UMAP not installed. Falling back to t-SNE.")
                    from sklearn.manifold import TSNE
                    
                    tsne = TSNE(
                        n_components=n_components,
                        perplexity=perplexity,
                        random_state=random_state
                    )
                    
                    embeddings_2d = tsne.fit_transform(embeddings_np)
            else:
                raise ValueError(f"Unknown visualization method: {method}")
            
            return embeddings_2d
            
        except Exception as e:
            logger.error(f"Error visualizing embeddings: {e}")
            return np.array([])
