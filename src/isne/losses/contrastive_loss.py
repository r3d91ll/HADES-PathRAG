"""
Contrastive loss implementation for ISNE.

This module implements the contrastive loss component for the ISNE model, which
encourages similar nodes to have similar embeddings while pushing dissimilar nodes apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging
from typing import Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for ISNE training.
    
    This loss function encourages embeddings of similar nodes to be close together
    and embeddings of dissimilar nodes to be far apart in the embedding space,
    helping to create more discriminative node representations.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = 'mean',
        lambda_contrast: float = 1.0,
        distance_metric: str = 'cosine'
    ) -> None:
        """
        Initialize the contrastive loss.
        
        Args:
            margin: Margin for contrastive loss
            reduction: Reduction method ('none', 'mean', 'sum')
            lambda_contrast: Weight factor for the contrastive loss
            distance_metric: Distance metric to use ('cosine', 'euclidean')
        """
        super(ContrastiveLoss, self).__init__()
        
        self.margin = margin
        self.reduction = reduction
        self.lambda_contrast = lambda_contrast
        self.distance_metric = distance_metric
    
    def _compute_distance(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Compute distance between pairs of embeddings.
        
        Args:
            x1: First set of embeddings [batch_size, embedding_dim]
            x2: Second set of embeddings [batch_size, embedding_dim]
            
        Returns:
            Distance tensor [batch_size]
        """
        if self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            return 1.0 - F.cosine_similarity(x1, x2, dim=1)
        elif self.distance_metric == 'euclidean':
            # Euclidean distance (squared)
            return torch.sum((x1 - x2) ** 2, dim=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def forward(
        self,
        embeddings: Tensor,
        positive_pairs: Tensor,
        negative_pairs: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the contrastive loss.
        
        Args:
            embeddings: Node embeddings from the ISNE model [num_nodes, embedding_dim]
            positive_pairs: Tensor of positive pair indices [num_pos_pairs, 2]
            negative_pairs: Optional tensor of negative pair indices [num_neg_pairs, 2]
            
        Returns:
            Contrastive loss
        """
        # Get number of nodes from embeddings
        num_nodes = embeddings.size(0)
        
        # Validate and filter positive pairs (ensure indices are within bounds)
        valid_pos_mask = (positive_pairs[:, 0] < num_nodes) & (positive_pairs[:, 1] < num_nodes)
        if not torch.all(valid_pos_mask):
            logger.warning(f"Filtered {(~valid_pos_mask).sum().item()} out-of-bounds positive pairs")
            positive_pairs = positive_pairs[valid_pos_mask]
            
        # If no valid positive pairs, return zero loss
        if positive_pairs.size(0) == 0:
            logger.warning("No valid positive pairs found, returning zero loss")
            return torch.tensor(0.0, device=embeddings.device)
        
        # Extract positive pair embeddings
        pos_idx1, pos_idx2 = positive_pairs[:, 0], positive_pairs[:, 1]
        pos_emb1 = embeddings[pos_idx1]
        pos_emb2 = embeddings[pos_idx2]
        
        # Compute distance for positive pairs
        pos_distances = self._compute_distance(pos_emb1, pos_emb2)
        
        # Compute positive pair loss: encourage similar nodes to be close
        pos_loss = pos_distances
        
        # Process negative pairs if provided
        if negative_pairs is not None:
            # Validate and filter negative pairs (ensure indices are within bounds)
            valid_neg_mask = (negative_pairs[:, 0] < num_nodes) & (negative_pairs[:, 1] < num_nodes)
            if not torch.all(valid_neg_mask):
                logger.warning(f"Filtered {(~valid_neg_mask).sum().item()} out-of-bounds negative pairs")
                negative_pairs = negative_pairs[valid_neg_mask]
            
            # If no valid negative pairs after filtering, skip negative loss
            if negative_pairs.size(0) == 0:
                logger.warning("No valid negative pairs found, using only positive loss")
                # Only use positive loss if no valid negative pairs
                return self.lambda_contrast * pos_loss.mean()
            
            neg_idx1, neg_idx2 = negative_pairs[:, 0], negative_pairs[:, 1]
            neg_emb1 = embeddings[neg_idx1]
            neg_emb2 = embeddings[neg_idx2]
            
            # Compute distance for negative pairs
            neg_distances = self._compute_distance(neg_emb1, neg_emb2)
            
            # Compute negative pair loss: push dissimilar nodes apart
            neg_loss = F.relu(self.margin - neg_distances)
            
            # Combine positive and negative losses
            loss = pos_loss.mean() + neg_loss.mean()
        else:
            # Only use positive loss if no negative pairs provided
            loss = pos_loss.mean()
        
        # Apply weight factor
        weighted_loss = self.lambda_contrast * loss
        
        return weighted_loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for ISNE.
    
    This is a popular contrastive learning loss that treats each node as a class
    and tries to identify the positive samples from a set of negative samples.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        lambda_infonce: float = 1.0
    ) -> None:
        """
        Initialize the InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for scaling
            lambda_infonce: Weight factor for the InfoNCE loss
        """
        super(InfoNCELoss, self).__init__()
        
        self.temperature = temperature
        self.lambda_infonce = lambda_infonce
    
    def forward(
        self,
        anchor_embeddings: Tensor,
        positive_embeddings: Tensor,
        negative_embeddings: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the InfoNCE loss.
        
        Args:
            anchor_embeddings: Anchor node embeddings [batch_size, embedding_dim]
            positive_embeddings: Positive sample embeddings [batch_size, embedding_dim]
            negative_embeddings: Optional negative sample embeddings [num_negatives, embedding_dim]
            
        Returns:
            InfoNCE loss
        """
        batch_size = anchor_embeddings.size(0)
        
        # Normalize embeddings for cosine similarity
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        
        # Compute positive similarity (dot product)
        pos_sim = torch.sum(anchor_embeddings * positive_embeddings, dim=1) / self.temperature
        
        if negative_embeddings is not None:
            # If explicit negative samples are provided
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
            
            # Compute similarity with all negative samples
            neg_sim = torch.matmul(anchor_embeddings, negative_embeddings.t()) / self.temperature
            
            # Concatenate positive and negative similarities
            logits = torch.cat([pos_sim.view(-1, 1), neg_sim], dim=1)
            
            # Labels are zeros (positive samples are at index 0)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_embeddings.device)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, labels)
        else:
            # Use other samples in batch as negatives (similar to SimCLR)
            # Compute similarity matrix for all pairs
            sim_matrix = torch.matmul(anchor_embeddings, positive_embeddings.t()) / self.temperature
            
            # Labels are diagonal indices (each anchor matched with its own positive)
            labels = torch.arange(batch_size, device=anchor_embeddings.device)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(sim_matrix, labels)
        
        # Apply weight factor
        weighted_loss = self.lambda_infonce * loss
        
        return weighted_loss


class TripletLoss(nn.Module):
    """
    Triplet loss for ISNE.
    
    This loss takes triplets of (anchor, positive, negative) samples and
    encourages the anchor to be closer to the positive than to the negative
    by a specified margin.
    """
    
    def __init__(
        self,
        margin: float = 0.5,
        reduction: str = 'mean',
        lambda_triplet: float = 1.0,
        distance_metric: str = 'cosine'
    ) -> None:
        """
        Initialize the triplet loss.
        
        Args:
            margin: Margin for triplet loss
            reduction: Reduction method ('none', 'mean', 'sum')
            lambda_triplet: Weight factor for the triplet loss
            distance_metric: Distance metric to use ('cosine', 'euclidean')
        """
        super(TripletLoss, self).__init__()
        
        self.margin = margin
        self.reduction = reduction
        self.lambda_triplet = lambda_triplet
        self.distance_metric = distance_metric
    
    def _compute_distance(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Compute distance between pairs of embeddings.
        
        Args:
            x1: First set of embeddings [batch_size, embedding_dim]
            x2: Second set of embeddings [batch_size, embedding_dim]
            
        Returns:
            Distance tensor [batch_size]
        """
        if self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            return 1.0 - F.cosine_similarity(x1, x2, dim=1)
        elif self.distance_metric == 'euclidean':
            # Euclidean distance (squared)
            return torch.sum((x1 - x2) ** 2, dim=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def forward(
        self,
        embeddings: Tensor,
        anchor_indices: Tensor,
        positive_indices: Tensor,
        negative_indices: Tensor
    ) -> Tensor:
        """
        Compute the triplet loss.
        
        Args:
            embeddings: Node embeddings from the ISNE model [num_nodes, embedding_dim]
            anchor_indices: Indices of anchor nodes [batch_size]
            positive_indices: Indices of positive nodes [batch_size]
            negative_indices: Indices of negative nodes [batch_size]
            
        Returns:
            Triplet loss
        """
        # Extract triplet embeddings
        anchor_emb = embeddings[anchor_indices]
        positive_emb = embeddings[positive_indices]
        negative_emb = embeddings[negative_indices]
        
        # Compute distances
        pos_distances = self._compute_distance(anchor_emb, positive_emb)
        neg_distances = self._compute_distance(anchor_emb, negative_emb)
        
        # Compute triplet loss: d(anchor, positive) - d(anchor, negative) + margin
        loss = F.relu(pos_distances - neg_distances + self.margin)
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        # Apply weight factor
        weighted_loss = self.lambda_triplet * loss
        
        return weighted_loss
