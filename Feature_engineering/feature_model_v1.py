import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler
from Feature_engineering.feature_extraction import FeatureExtractor
import os
import glob

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    Adds positional information to the input embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize the positional encoding layer.
        
        Parameters:
        -----------
        d_model : int
            Dimension of the model
        max_len : int
            Maximum sequence length
        dropout : float
            Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (persistent but not model parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
        --------
        torch.Tensor
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MaskedTemporalEncoder(nn.Module):
    """
    Enhanced self-supervised transformer-based model for extracting features from financial time series data
    using masking and contrastive learning techniques.
    
    Uses an encoder-only transformer architecture with masking and contrastive loss functions
    to capture volatility regime characteristics without explicit reconstruction.
    """
    
    def __init__(
        self,
        input_size: int = 20,  # Using rich microstructure features instead of just 3 basic ones
        context_length: int = 50,
        d_model: int = 128,
        num_encoder_layers: int = 4,
        num_attention_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        mask_ratio: float = 0.15,
        temperature: float = 0.1,  # Temperature parameter for contrastive loss
        feature_groups: Optional[Dict[str, List[int]]] = None,  # Grouped feature indices
        device: Optional[str] = None,
        causal: bool = True  # Use causal attention mask for real-time applications
    ):
        """
        Initialize the masked temporal encoder for volatility regime feature extraction.
        
        Parameters:
        -----------
        input_size : int
            Number of input features (using rich microstructure features)
        context_length : int
            Length of context window for transformer
        d_model : int
            Dimension of transformer model
        num_encoder_layers : int
            Number of encoder layers
        num_attention_heads : int
            Number of attention heads
        dim_feedforward : int
            Dimension of feedforward network
        dropout : float
            Dropout rate
        attention_dropout : float
            Attention dropout rate
        activation : str
            Activation function
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        mask_ratio : float
            Ratio of tokens to mask during training
        temperature : float
            Temperature parameter for contrastive loss
        feature_groups : Dict[str, List[int]], optional
            Dictionary of feature group names to feature indices for group-based masking
        device : str, optional
            Device to use for computation
        causal : bool
            If True, use causal attention masking (for real-time applications)
            If False, use bidirectional attention (creates future leakage, research only)
        """
        super().__init__()
        
        # Device configuration
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Model hyperparameters
        self.input_size = input_size
        self.context_length = context_length
        self.d_model = d_model
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.num_attention_heads = num_attention_heads
        self.causal = causal  # Store causal mode parameter
        
        # Feature groups for structured masking
        if feature_groups is None:
            # Default grouping - can be overridden with domain knowledge
            self.feature_groups = {
                'volatility': list(range(input_size-3, input_size)),  # Assume last features are volatility-related
                'momentum': list(range(input_size//3, 2*input_size//3)),  # Middle features as momentum
                'price': list(range(input_size//3)),  # First third as price-related
            }
        else:
            self.feature_groups = feature_groups
            
        # Input projection: map input features to model dimension
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Time feature projection: map time features to model dimension
        self.time_projection = nn.Linear(8, d_model)  # Time features from feature_extraction.py
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Mask token embedding (learnable)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Activation function
        if activation == "gelu":
            activation_fn = nn.GELU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        else:
            activation_fn = nn.GELU()  # Default to GELU
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation_fn,
            batch_first=True  # Use batch-first convention (batch, seq, feature)
        )
        
        # Create encoder with multiple layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Feature projection for encoder outputs - create more compact representations
        self.feature_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(d_model // 4, d_model // 4),
            nn.BatchNorm1d(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 8)
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def _create_feature_masks(
        self, 
        batch_size: int, 
        seq_len: int,
        structured: bool = True
    ) -> torch.Tensor:
        """
        Create masks for input features based on masking strategy.
        
        Parameters:
        -----------
        batch_size : int
            Batch size
        seq_len : int
            Sequence length
        structured : bool
            Whether to use structured masking by feature groups
            
        Returns:
        --------
        torch.Tensor
            Binary mask of shape (batch_size, seq_len, input_size)
            1 means keep, 0 means mask
        """
        if structured:
            # Structured masking: mask entire feature groups together
            # Initialize mask with ones (keep all)
            mask = torch.ones(batch_size, seq_len, self.input_size, device=self.device)
            
            # For each sample in batch, randomly select feature groups to mask
            for b in range(batch_size):
                # Randomly select ~15% of time steps to apply masking
                time_mask_prob = torch.rand(seq_len, device=self.device)
                time_mask_indices = torch.where(time_mask_prob < self.mask_ratio)[0]
                
                if len(time_mask_indices) > 0:
                    # For each selected time step, randomly choose a feature group to mask
                    for t in time_mask_indices:
                        # Select a random feature group with higher probability for volatility-related
                        # We want to bias masking toward volatility features to make model focus on them
                        group_names = list(self.feature_groups.keys())
                        
                        # Assign higher sampling probability to volatility groups
                        group_probs = [0.6 if 'volatility' in g or 'jump' in g else 0.2 for g in group_names]
                        group_probs = torch.tensor(group_probs, device=self.device)
                        group_probs = group_probs / group_probs.sum()
                        
                        # Sample a group to mask
                        group_idx = torch.multinomial(group_probs, 1).item()
                        selected_group = group_names[group_idx]
                        feature_indices = self.feature_groups[selected_group]
                        
                        # Set mask to 0 for selected feature indices at time t
                        mask[b, t, feature_indices] = 0.0
        else:
            # Random masking: mask random individual features
            # Generate random mask based on mask_ratio
            mask_prob = torch.rand(batch_size, seq_len, self.input_size, device=self.device)
            mask = (mask_prob >= self.mask_ratio).float()
            
        return mask
    
    def forward(
        self, 
        values: torch.Tensor,
        time_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        apply_masking: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model with optional masking.
        
        Parameters:
        -----------
        values : torch.Tensor
            Input values of shape (batch_size, seq_len, input_size)
        time_features : torch.Tensor
            Time features of shape (batch_size, seq_len, time_features_size)
        mask : torch.Tensor, optional
            Optional mask of shape (batch_size, seq_len, input_size)
            If None, generate mask based on mask_ratio
        apply_masking : bool
            Whether to apply masking (set to False for inference)
        attention_mask : torch.Tensor, optional
            Optional attention mask for padding
        return_all_outputs : bool
            Whether to return all intermediate outputs
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Dictionary containing model outputs
        """
        batch_size, seq_len, _ = values.shape
        
        # Generate mask if not provided and applying masking
        if mask is None and apply_masking:
            mask = self._create_feature_masks(batch_size, seq_len)
        elif not apply_masking:
            # No masking during inference
            mask = torch.ones_like(values)
        
        # Store original input before masking for contrastive learning
        original_values = values.clone()
        
        # Apply mask to input values (mask=0 means feature is masked)
        masked_values = values * mask
        
        # Project input values to model dimension
        values_projected = self.input_projection(masked_values)  # (batch, seq, d_model)
        
        # Project time features to model dimension
        time_projected = self.time_projection(time_features)  # (batch, seq, d_model)
        
        # Indicate masked positions by adding mask token embedding
        if apply_masking:
            # Expand mask token to match batch and sequence dimensions
            mask_tokens = self.mask_token.expand(batch_size, seq_len, self.d_model)
            
            # Convert feature-level mask to sequence-level (1 if any feature is masked, 0 if all preserved)
            seq_mask = (mask.sum(dim=2) < self.input_size).float().unsqueeze(-1)
            
            # Add mask tokens only at positions where features are masked
            values_projected = values_projected * (1 - seq_mask) + mask_tokens * seq_mask
        
        # Combine value and time features
        combined_features = values_projected + time_projected
        
        # Add positional encoding
        encoded_features = self.positional_encoding(combined_features)
        
        # Store attention weights
        attention_weights = []
        
        # Hook to capture attention weights
        def hook_fn(module, input, output):
            # Get attention weights from the output
            if isinstance(output, tuple):
                current_output = output[0]
            else:
                current_output = output
            
            attn_output, attn_weights = module.self_attn(
                current_output, current_output, current_output,
                need_weights=True
            )
            attention_weights.append(attn_weights.detach())
            return output
        
        # Register hooks for all encoder layers
        hooks = []
        for layer in self.transformer_encoder.layers:
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Prepare and check attention mask format
        # PyTorch transformer expects mask in a specific format:
        # - For key_padding_mask: boolean tensor where True = ignored position
        src_key_padding_mask = None
        if attention_mask is not None:
            # Convert to Boolean where True means "ignore this position"
            # If attention_mask has 1s for valid positions, invert it
            if attention_mask.dtype != torch.bool:
                if (attention_mask == 1.0).any(): 
                    # Mask has 1s for valid positions, need to invert for PyTorch transformer
                    src_key_padding_mask = (attention_mask == 0.0).bool()
                else:
                    # Mask already has 0s for valid, 1s for masked positions
                    src_key_padding_mask = attention_mask.bool()
            else:
                src_key_padding_mask = attention_mask
            
            # Ensure the mask is 2D (batch, seq)
            if src_key_padding_mask.dim() == 3:
                # If we have a 3D mask, collapse to 2D
                src_key_padding_mask = src_key_padding_mask.any(dim=2)
        
        # For causal mode, create a causal attention mask
        # This prevents tokens from attending to future tokens
        attn_mask = None
        if self.causal:
            # Create a causal mask (lower triangular)
            # In this mask, each position can only attend to itself and previous positions
            # 0 means "can attend", -inf means "cannot attend"
            mask_shape = (seq_len, seq_len)
            attn_mask = torch.triu(
                torch.ones(mask_shape, device=self.device) * float('-inf'),
                diagonal=1
            )
        
        # Pass through transformer encoder
        latent_representations = self.transformer_encoder(
            encoded_features, 
            mask=attn_mask,  # Causal mask when in causal mode
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Remove the hooks
        for hook in hooks:
            hook.remove()
        
        # Project to lower dimensional space for compact feature representation
        projected_features = self.feature_projection(latent_representations)
        
        # Calculate attention entropy if we have attention weights
        mean_attention_entropy = None
        if attention_weights:
            # Stack attention weights
            attention_pattern = torch.stack(attention_weights, dim=0)  # (layers, batch, heads, seq, seq)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            attention_entropy = -torch.sum(
                attention_pattern * torch.log(attention_pattern + epsilon), 
                dim=-1
            )  # (layers, batch, heads, seq)
            
            # Mean entropy across sequence dimension
            mean_attention_entropy = attention_entropy.mean(dim=-1)  # (layers, batch, heads)
        
        # Apply projection head to get final contrastive representations
        # First, reshape to (batch*seq, feat_dim)
        batch_seq, feat_dim = batch_size * seq_len, projected_features.size(-1)
        reshaped_features = projected_features.reshape(batch_seq, feat_dim)
        
        # Apply projection head
        projection_output = self.projection_head(reshaped_features)
        
        # Reshape back to (batch, seq, proj_dim)
        projection_output = projection_output.reshape(batch_size, seq_len, -1)
        
        # Store outputs for loss calculation and feature extraction
        output = {
            "latent_representations": latent_representations,
            "projected_features": projected_features,
            "projection_output": projection_output,
            "attention_weights": attention_weights,
            "mask": mask,
            "original_values": original_values,
        }
        
        if mean_attention_entropy is not None:
            output["attention_entropy"] = mean_attention_entropy
            
        if return_all_outputs:
            output["encoded_features"] = encoded_features
            
        return output 

    def compute_contrastive_loss(
        self, 
        model_output: Dict[str, torch.Tensor],
        temporal_weight: float = 0.7,
        instance_weight: float = 0.3
    ) -> torch.Tensor:
        """
        Compute contrastive loss using both temporal and instance-based contrastive learning.
        
        Parameters:
        -----------
        model_output : Dict[str, torch.Tensor]
            Dictionary containing model outputs
        temporal_weight : float
            Weight for temporal contrastive loss
        instance_weight : float
            Weight for instance contrastive loss
            
        Returns:
        --------
        torch.Tensor
            Contrastive loss
        """
        projection_output = model_output["projection_output"]  # (batch, seq, proj_dim)
        mask = model_output["mask"]  # (batch, seq, input_size)
        
        batch_size, seq_len, proj_dim = projection_output.shape
        
        # Convert feature-level mask to sequence-level (1 if all features preserved, 0 if any masked)
        seq_mask = (mask.sum(dim=2) == self.input_size).float()  # (batch, seq)
        
        # Calculate temporal contrastive loss
        temporal_loss = self._compute_temporal_contrastive_loss(
            projection_output, seq_mask)
        
        # Calculate instance contrastive loss
        instance_loss = self._compute_instance_contrastive_loss(
            projection_output, seq_mask)
        
        # Combine losses with weights
        contrastive_loss = temporal_weight * temporal_loss + instance_weight * instance_loss
        
        return contrastive_loss
    
    def _compute_temporal_contrastive_loss(
        self, 
        projections: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal contrastive loss within each time series.
        
        Parameters:
        -----------
        projections : torch.Tensor
            Projection outputs of shape (batch, seq, proj_dim)
        mask : torch.Tensor
            Sequence mask of shape (batch, seq) where 1 means preserved, 0 means masked
            
        Returns:
        --------
        torch.Tensor
            Temporal contrastive loss
        """
        batch_size, seq_len, proj_dim = projections.shape
        
        # Check for NaN values in projections and replace with zeros
        if torch.isnan(projections).any():
            projections = torch.nan_to_num(projections, nan=0.0)
        
        # Normalize projections for cosine similarity
        # Add small epsilon to avoid division by zero
        proj_norm = F.normalize(projections, p=2, dim=2, eps=1e-8)  # (batch, seq, proj_dim)
        
        # Compute all pairwise similarities within each sequence
        # Reshape to (batch, seq, 1, proj_dim) and (batch, 1, seq, proj_dim)
        anchor_proj = proj_norm.unsqueeze(2)  # (batch, seq, 1, proj_dim)
        compare_proj = proj_norm.unsqueeze(1)  # (batch, 1, seq, proj_dim)
        
        # Compute similarity matrix for each batch: (batch, seq, seq)
        # Use a safe temperature factor to prevent overflow
        safe_temperature = max(0.1, self.temperature)  # Never allow temperature to be too small
        similarity = torch.sum(anchor_proj * compare_proj, dim=3) / safe_temperature
        
        # Clip similarity values to prevent extreme values
        similarity = torch.clamp(similarity, -20.0, 20.0)
        
        # Create mask for valid comparisons: only use unmasked tokens
        valid_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # (batch, seq, seq)
        
        # Create diagonal mask to exclude self-comparisons
        diag_mask = 1.0 - torch.eye(seq_len, device=self.device).unsqueeze(0)  # (1, seq, seq)
        valid_mask = valid_mask * diag_mask  # Remove self-comparisons
        
        # For each position, compute temporal contrastive loss
        # Use a "time window" concept - tokens closer in time should be more similar
        # Create time distance matrix
        time_dist = torch.abs(torch.arange(seq_len, device=self.device).unsqueeze(1) - 
                              torch.arange(seq_len, device=self.device).unsqueeze(0))  # (seq, seq)
        
        # Convert to weights: closer positions get higher weight 
        time_weights = torch.exp(-time_dist / 5.0)  # Decay with distance
        time_weights = time_weights.unsqueeze(0)  # (1, seq, seq)
        
        # Create positive mask: positions close in time are considered positive examples
        # The closer in time, the more positive they are (weighted by time_weights)
        pos_weights = time_weights * valid_mask
        
        # Numerator: exp(similarity) weighted by positional closeness
        # Use safe exponentiation to prevent overflow
        exp_similarity = torch.exp(similarity)
        
        # Check for NaN or inf in exp_similarity and handle it
        if torch.isnan(exp_similarity).any() or torch.isinf(exp_similarity).any():
            exp_similarity = torch.nan_to_num(exp_similarity, nan=0.0, posinf=1e6, neginf=0.0)
            
        weighted_similarity = exp_similarity * pos_weights
        
        # For positions with valid neighbors, compute loss
        valid_positions = (valid_mask.sum(dim=2) > 0).float()  # (batch, seq)
        
        # Sum weighted similarities for each position
        pos_sum = torch.sum(weighted_similarity, dim=2)  # (batch, seq)
        
        # Denominator: sum over all valid positions
        neg_sum = torch.sum(exp_similarity * valid_mask, dim=2)  # (batch, seq)
        
        # Add a larger epsilon for numerical stability
        epsilon = 1e-6
        
        # Compute loss for valid positions (avoid nan for positions without valid neighbors)
        # Use safe log to prevent -inf values
        safe_ratio = pos_sum / (neg_sum + epsilon) + epsilon
        per_pos_loss = -torch.log(safe_ratio) * valid_positions
        
        # Check for NaN or inf in loss and replace them
        if torch.isnan(per_pos_loss).any() or torch.isinf(per_pos_loss).any():
            per_pos_loss = torch.nan_to_num(per_pos_loss, nan=0.0, posinf=10.0, neginf=0.0)
        
        # Average over valid positions
        valid_count = valid_positions.sum()
        if valid_count > 0:
            loss = per_pos_loss.sum() / valid_count
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        return loss
    
    def _compute_instance_contrastive_loss(
        self, 
        projections: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute instance contrastive loss across different time series.
        
        Parameters:
        -----------
        projections : torch.Tensor
            Projection outputs of shape (batch, seq, proj_dim)
        mask : torch.Tensor
            Sequence mask of shape (batch, seq) where 1 means preserved, 0 means masked
            
        Returns:
        --------
        torch.Tensor
            Instance contrastive loss
        """
        batch_size, seq_len, proj_dim = projections.shape
        
        # Check for NaN values in projections and replace with zeros
        if torch.isnan(projections).any():
            projections = torch.nan_to_num(projections, nan=0.0)
        
        # Reshape to (batch*seq, proj_dim) for batch contrastive loss
        flat_projections = projections.reshape(-1, proj_dim)
        flat_mask = mask.reshape(-1)
        
        # Only use unmasked tokens for contrastive loss
        valid_indices = torch.where(flat_mask > 0)[0]
        
        if len(valid_indices) <= 1:
            # Not enough valid tokens for contrastive loss
            return torch.tensor(0.0, device=self.device)
        
        valid_projections = flat_projections[valid_indices]
        
        # Normalize for cosine similarity with epsilon for numerical stability
        valid_projections = F.normalize(valid_projections, p=2, dim=1, eps=1e-8)
        
        # Compute similarity matrix
        # Use a safe temperature factor to prevent overflow
        safe_temperature = max(0.1, self.temperature)  # Never allow temperature to be too small
        similarity = torch.matmul(valid_projections, valid_projections.t()) / safe_temperature
        
        # Clip similarity values to prevent extreme values
        similarity = torch.clamp(similarity, -20.0, 20.0)
        
        # Create label mask: identify which tokens belong to the same time series
        # Tokens from the same sequence are positives, different sequences are negatives
        seq_indices = torch.div(valid_indices, seq_len, rounding_mode='floor')
        pos_mask = (seq_indices.unsqueeze(1) == seq_indices.unsqueeze(0)).float()
        
        # Remove self-similarity
        self_mask = torch.eye(len(valid_indices), device=self.device)
        pos_mask = pos_mask - self_mask
        
        # Positive samples exist when multiple tokens from same sequence are valid
        # Count for each row how many positives exist
        valid_pos_count = pos_mask.sum(dim=1)
        valid_pos_indices = torch.where(valid_pos_count > 0)[0]
        
        if len(valid_pos_indices) == 0:
            # No valid positive pairs
            return torch.tensor(0.0, device=self.device)
        
        # Use InfoNCE loss for valid indices
        # For each anchor, positives are from same sequence, negatives from different sequences
        # Use safe exponentiation to prevent overflow
        exp_sim = torch.exp(similarity)
        
        # Check for NaN or inf in exp_sim and handle it
        if torch.isnan(exp_sim).any() or torch.isinf(exp_sim).any():
            exp_sim = torch.nan_to_num(exp_sim, nan=0.0, posinf=1e6, neginf=0.0)
        
        # Numerator: sum of exp(sim) for positives
        pos_sum = torch.sum(exp_sim * pos_mask, dim=1)
        
        # Denominator: sum of exp(sim) for all except self
        all_sum = torch.sum(exp_sim * (1 - self_mask), dim=1)
        
        # Use a larger epsilon for numerical stability
        epsilon = 1e-6
        
        # Compute loss only for indices with valid positives
        safe_ratio = pos_sum / (all_sum + epsilon) + epsilon
        per_anchor_loss = -torch.log(safe_ratio)
        
        # Handle NaN or inf values in loss
        if torch.isnan(per_anchor_loss).any() or torch.isinf(per_anchor_loss).any():
            per_anchor_loss = torch.nan_to_num(per_anchor_loss, nan=0.0, posinf=10.0, neginf=0.0)
        
        instance_loss = per_anchor_loss[valid_pos_indices].mean()
        
        return instance_loss
    
    def compute_entropy_regularization_loss(
        self, 
        model_output: Dict[str, torch.Tensor],
        n_clusters: int = 5,
        beta: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute entropy-based regularization loss to encourage formation of distinct regimes.
        
        Parameters:
        -----------
        model_output : Dict[str, torch.Tensor]
            Dictionary containing model outputs
        n_clusters : int
            Number of volatility regimes to identify
        beta : float
            Weight for between-cluster entropy term
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (total_entropy_loss, within_cluster_entropy, between_cluster_entropy)
        """
        # Get features from all sequences, averaging over time dimension
        projections = model_output["projected_features"]  # (batch, seq, feat_dim)
        
        # Use sequence mask to select only unmasked tokens
        if "mask" in model_output:
            mask = model_output["mask"]  # (batch, seq, input_size)
            seq_mask = (mask.sum(dim=2) == self.input_size).float()  # (batch, seq)
            seq_mask = seq_mask.unsqueeze(-1)  # (batch, seq, 1)
        else:
            seq_mask = torch.ones_like(projections[:, :, :1])
        
        # Average features over time, considering mask
        masked_features = projections * seq_mask
        masked_sum = masked_features.sum(dim=1)  # (batch, feat_dim)
        mask_sum = seq_mask.sum(dim=1)  # (batch, 1)
        
        # Avoid division by zero
        mask_sum = torch.clamp(mask_sum, min=1.0)
        
        # Compute average features per batch
        avg_features = masked_sum / mask_sum  # (batch, feat_dim)
        
        # Normalize features for cosine similarity
        norm_features = F.normalize(avg_features, p=2, dim=1)
        
        # Use soft clustering with learnable centroids or on-the-fly clustering
        # Here we use cosine similarity-based soft assignments
        
        # Compute all pairwise similarities to estimate cluster assignments
        similarity_matrix = torch.matmul(norm_features, norm_features.t())  # (batch, batch)
        
        # Apply softmax to get soft assignments
        temperature = 0.1  # Lower for harder assignments, higher for softer
        soft_assignments = F.softmax(similarity_matrix / temperature, dim=1)  # (batch, batch)
        
        # For each potential cluster, calculate within-cluster and between-cluster entropy
        within_cluster_entropy = 0.0
        between_cluster_entropy = 0.0
        
        batch_size = norm_features.shape[0]
        for i in range(batch_size):
            # Get soft assignment weights for this "cluster"
            weights = soft_assignments[:, i].unsqueeze(1)  # (batch, 1)
            
            # Calculate weighted mean of features (cluster centroid)
            centroid = (norm_features * weights).sum(dim=0) / (weights.sum() + 1e-6)
            centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1)  # (1, feat_dim)
            
            # Calculate distances to centroid
            distances = 1.0 - torch.matmul(norm_features, centroid.t())  # (batch, 1)
            
            # Calculate weighted within-cluster entropy
            # Want to minimize this - points in same cluster should be close
            within_entropy = (weights * distances).sum() / (weights.sum() + 1e-6)
            within_cluster_entropy += within_entropy
            
            # Calculate weighted between-cluster entropy
            # Want to maximize this - different clusters should be far apart
            other_weights = 1.0 - weights  # Weights for other clusters
            between_entropy = (other_weights * distances).sum() / (other_weights.sum() + 1e-6)
            between_cluster_entropy += between_entropy
        
        # Normalize by number of clusters analyzed
        within_cluster_entropy /= batch_size
        between_cluster_entropy /= batch_size
        
        # Final loss: minimize within-cluster entropy, maximize between-cluster entropy
        entropy_loss = within_cluster_entropy - beta * between_cluster_entropy
        
        return entropy_loss, within_cluster_entropy, between_cluster_entropy
    
    def compute_attention_diversity_loss(self, model_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute attention diversity loss: different attention heads should focus
        on different patterns.
        
        Parameters:
        -----------
        model_output : Dict[str, torch.Tensor]
            Dictionary containing model outputs
            
        Returns:
        --------
        torch.Tensor
            Attention diversity loss
        """
        # Get attention entropy if available
        if "attention_entropy" in model_output:
            attention_entropy = model_output["attention_entropy"]  # (layers, batch, heads)
            
            # Higher entropy means more diverse attention
            # We want to maximize entropy, so we minimize negative entropy
            diversity_loss = -attention_entropy.mean()
            
            return diversity_loss
        else:
            # If no attention entropy available, return zero loss
            return torch.tensor(0.0, device=self.device)
    
    def compute_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        values: Optional[torch.Tensor] = None,
        alpha: float = 0.5,  # Weight for contrastive loss
        beta: float = 0.3,   # Weight for entropy regularization
        gamma: float = 0.2   # Weight for attention diversity
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for the self-supervised model.
        
        Parameters:
        -----------
        model_output : Dict[str, torch.Tensor]
            Dictionary containing model outputs
        values : torch.Tensor, optional
            Original input values 
        alpha : float
            Weight for contrastive loss
        beta : float
            Weight for entropy regularization
        gamma : float
            Weight for attention diversity
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Dictionary containing individual and total losses
        """
        # Compute individual losses
        contrastive_loss = self.compute_contrastive_loss(model_output)
        
        entropy_loss, within_entropy, between_entropy = self.compute_entropy_regularization_loss(model_output)
        
        diversity_loss = self.compute_attention_diversity_loss(model_output)
        
        # Compute total loss
        total_loss = alpha * contrastive_loss + beta * entropy_loss + gamma * diversity_loss
        
        # Return loss dictionary
        loss_dict = {
            "contrastive_loss": contrastive_loss,
            "entropy_loss": entropy_loss,
            "within_cluster_entropy": within_entropy,
            "between_cluster_entropy": between_entropy,
            "attention_diversity_loss": diversity_loss,
            "total_loss": total_loss
        }
        
        return loss_dict 

    def train_step(
        self,
        values: torch.Tensor,
        time_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Parameters:
        -----------
        values : torch.Tensor
            Input values tensor
        time_features : torch.Tensor
            Time features tensor
        attention_mask : torch.Tensor, optional
            Attention mask
        alpha : float
            Weight for contrastive loss
        beta : float
            Weight for entropy regularization
        gamma : float
            Weight for attention diversity
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing loss values
        """
        self.optimizer.zero_grad()
        
        # Apply masking during training
        model_output = self.forward(
            values=values,
            time_features=time_features,
            apply_masking=True,
            attention_mask=attention_mask
        )
        
        # Compute losses
        loss_dict = self.compute_loss(
            model_output=model_output, 
            values=values, 
            alpha=alpha, 
            beta=beta, 
            gamma=gamma
        )
        
        # Backpropagate
        loss_dict["total_loss"].backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Convert tensors to float for logging
        return {k: v.item() for k, v in loss_dict.items()}
    
    def extract_features(
        self,
        values: torch.Tensor,
        time_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_projection_head: bool = True
    ) -> torch.Tensor:
        """
        Extract features from the model for downstream tasks.
        
        Parameters:
        -----------
        values : torch.Tensor
            Input values tensor
        time_features : torch.Tensor
            Time features tensor
        attention_mask : torch.Tensor, optional
            Attention mask
        use_projection_head : bool
            Whether to use the projection head for feature extraction
            
        Returns:
        --------
        torch.Tensor
            Extracted features
        """
        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Ensure attention mask is properly formatted (boolean)
            fixed_attention_mask = None
            if attention_mask is not None:
                # Create proper boolean mask - PyTorch 1.9+ transformer requires bool mask
                fixed_attention_mask = attention_mask.bool() if attention_mask.dtype != torch.bool else attention_mask
            
            # No masking during feature extraction
            model_output = self.forward(
                values=values,
                time_features=time_features,
                apply_masking=False,
                attention_mask=fixed_attention_mask
            )
            
            if use_projection_head:
                # Use the contrastive projection as features (usually better for downstream tasks)
                features = model_output["projection_output"].mean(dim=1)
            else:
                # Use the projected features as representation (more dimensions)
                features = model_output["projected_features"].mean(dim=1)
            
        # Set model back to training mode
        self.train()
        
        return features

# Helper functions for processing data with the enhanced model

def prepare_microstructure_data(
    df: pd.DataFrame,
    feature_extractor: FeatureExtractor,
    context_length: int = 50,
    timestamp_col: str = 'Timestamp',
    causal: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Prepare data for model training by creating tensors using the FeatureExtractor.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with raw market data
    feature_extractor : FeatureExtractor
        FeatureExtractor instance to extract microstructure features
    context_length : int
        Length of context window
    timestamp_col : str
        Name of timestamp column
    causal : bool
        If True, use causal mode (only past information) for real-time applications
        If False, use bidirectional context (WARNING: creates future leakage, research only)
        
    Returns:
    --------
    Dict[str, torch.Tensor]
        Dictionary containing model inputs
    """
    # Extract rich microstructure features
    feature_df = feature_extractor.extract_features(df, timestamp_col=timestamp_col)
    
    # Create tensors for time, features, and timedeltas
    time_tensor, features_tensor, timedelta_tensor = feature_extractor.create_tensors(
        feature_df, timestamp_col=timestamp_col, window_size=context_length, causal=causal
    )
    
    # Create attention mask (1 for valid tokens, 0 for padding/masked tokens)
    # In this case, all tokens are valid so we use ones
    batch_size, seq_len = features_tensor.shape[0], features_tensor.shape[1]
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.float32)
    
    # Return dictionary of model inputs
    return {
        "values": features_tensor,
        "time_features": time_tensor,
        "timedelta": timedelta_tensor,
        "attention_mask": attention_mask
    }


def train_masked_model(
    model: MaskedTemporalEncoder,
    train_data: Dict[str, torch.Tensor],
    num_epochs: int = 100,
    device: Optional[str] = None,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    warmup_epochs: int = 10,
    verbose: bool = True,
    early_stopping_patience: int = 15
) -> Dict[str, List[float]]:
    """
    Train the masked temporal encoder model with learning rate scheduling.
    
    Parameters:
    -----------
    model : MaskedTemporalEncoder
        Model to train
    train_data : Dict[str, torch.Tensor]
        Training data dictionary
    num_epochs : int
        Number of epochs to train
    device : str, optional
        Device to use for computation
    alpha : float
        Weight for contrastive loss
    beta : float
        Weight for entropy regularization
    gamma : float
        Weight for attention diversity
    warmup_epochs : int
        Number of warmup epochs for learning rate
    verbose : bool
        Whether to print progress
    early_stopping_patience : int
        Number of epochs to wait for improvement before early stopping
        
    Returns:
    --------
    Dict[str, List[float]]
        Training history
    """
    # Set device
    if device is None:
        device = model.device
    
    # Move data to device
    train_data = {k: v.to(device) for k, v in train_data.items()}
    
    # Training history
    history = {
        "contrastive_loss": [],
        "entropy_loss": [],
        "within_cluster_entropy": [],
        "between_cluster_entropy": [],
        "attention_diversity_loss": [],
        "total_loss": []
    }
    
    # Set model to training mode
    model.train()
    
    # Learning rate scheduler with warmup
    initial_lr = model.optimizer.param_groups[0]['lr']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
    )
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_metrics = {k: 0.0 for k in history.keys()}
        num_batches = 0
        
        # Apply learning rate warmup
        if epoch < warmup_epochs:
            lr = initial_lr * ((epoch + 1) / warmup_epochs)
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = lr
        
        # Generate random batch indices
        batch_size = model.batch_size
        num_samples = train_data["values"].shape[0]
        indices = np.random.permutation(num_samples)
        
        for batch_start in range(0, num_samples, batch_size):
            # Get batch indices
            batch_indices = indices[batch_start:min(batch_start+batch_size, num_samples)]
            
            # Extract batch data
            batch_values = train_data["values"][batch_indices]
            batch_time_features = train_data["time_features"][batch_indices]
            
            # Get attention mask if available
            batch_attention_mask = None
            if "attention_mask" in train_data:
                batch_attention_mask = train_data["attention_mask"][batch_indices]
            
            # Adjust loss weights over time
            if epoch < warmup_epochs:
                # More focus on contrastive loss in early epochs
                current_alpha = alpha * 1.5
                current_beta = beta * 0.5
                current_gamma = gamma * 0.5
            else:
                # Gradually increase entropy and diversity loss weights
                progress = min(1.0, (epoch - warmup_epochs) / (num_epochs - warmup_epochs))
                current_alpha = alpha * (1.0 - progress * 0.3)  # Slightly decrease
                current_beta = beta * (1.0 + progress)  # Increase
                current_gamma = gamma * (1.0 + progress * 0.5)  # Slightly increase
            
            # Perform training step
            batch_metrics = model.train_step(
                values=batch_values,
                time_features=batch_time_features,
                attention_mask=batch_attention_mask,
                alpha=current_alpha,
                beta=current_beta,
                gamma=current_gamma
            )
            
            # Update epoch metrics
            for k, v in batch_metrics.items():
                epoch_metrics[k] += v
                
            num_batches += 1
        
        # Calculate average metrics for the epoch
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= max(1, num_batches)  # Avoid division by zero
            history[k].append(epoch_metrics[k])
        
        # Update learning rate scheduler
        scheduler.step(epoch_metrics['total_loss'])
        
        # Early stopping check
        if epoch_metrics['total_loss'] < best_loss:
            best_loss = epoch_metrics['total_loss']
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().detach() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                # Restore best model
                if best_model_state is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                break
        
        # Print progress
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - " + 
                  f"Loss: {epoch_metrics['total_loss']:.4f} - " +
                  f"Contrastive: {epoch_metrics['contrastive_loss']:.4f} - " +
                  f"Entropy: {epoch_metrics['entropy_loss']:.4f} - " +
                  f"Diversity: {epoch_metrics['attention_diversity_loss']:.4f}")
    
    return history


def extract_features_from_df(
    model: MaskedTemporalEncoder,
    df: pd.DataFrame,
    feature_extractor: FeatureExtractor,
    context_length: int = 50,
    timestamp_col: str = 'Timestamp',
    use_projection_head: bool = True
) -> np.ndarray:
    """
    Extract features from a DataFrame using the trained model.
    
    Parameters:
    -----------
    model : MaskedTemporalEncoder
        Trained model
    df : pd.DataFrame
        Input dataframe with raw data
    feature_extractor : FeatureExtractor
        FeatureExtractor instance to extract microstructure features
    context_length : int
        Length of context window
    timestamp_col : str
        Name of timestamp column
    use_projection_head : bool
        Whether to use the projection head for feature extraction
        
    Returns:
    --------
    np.ndarray
        Extracted features
    """
    # Prepare data
    data = prepare_microstructure_data(
        df=df,
        feature_extractor=feature_extractor,
        context_length=context_length,
        timestamp_col=timestamp_col,
        causal=True
    )
    
    # Move data to model device
    data = {k: v.to(model.device) for k, v in data.items()}
    
    # Extract features
    features = model.extract_features(
        values=data["values"],
        time_features=data["time_features"],
        attention_mask=data["attention_mask"],
        use_projection_head=use_projection_head
    )
    
    # Convert to numpy array
    features_np = features.cpu().numpy()
    
    return features_np 

def save_model(model: MaskedTemporalEncoder, save_path: str, verbose: bool = True) -> None:
    """
    Save a model to a checkpoint file.
    
    Parameters:
    -----------
    model : MaskedTemporalEncoder
        Model to save
    save_path : str
        Path to save the model to
    verbose : bool
        Whether to print status messages
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare model configuration
    model_config = {
        'input_size': model.input_size,
        'context_length': model.context_length,
        'd_model': model.d_model,
        'num_encoder_layers': len(model.transformer_encoder.layers),
        'num_attention_heads': model.num_attention_heads,
        'batch_size': model.batch_size,
        'mask_ratio': model.mask_ratio,
        'temperature': model.temperature,
        'dim_feedforward': model.transformer_encoder.layers[0].linear1.out_features,
        'dropout': model.transformer_encoder.layers[0].dropout.p,
        'feature_groups': model.feature_groups
    }
    
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'model_config': model_config
    }, save_path)
    
    if verbose:
        print(f"Model saved to {save_path}")

def load_model(load_path: str, device: str = None, verbose: bool = True) -> MaskedTemporalEncoder:
    """
    Load a saved model from a checkpoint file.
    
    Parameters:
    -----------
    load_path : str
        Path to the saved model checkpoint
    device : str, optional
        Device to load the model on
    verbose : bool
        Whether to print status messages
        
    Returns:
    --------
    MaskedTemporalEncoder
        Loaded model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    
    # Get model configuration from checkpoint
    model_config = checkpoint['model_config']
    
    # Create model with the same configuration as the saved model
    model = MaskedTemporalEncoder(
        input_size=model_config.get('input_size', 20),
        context_length=model_config.get('context_length', 50),
        d_model=model_config.get('d_model', 128),
        num_encoder_layers=model_config.get('num_encoder_layers', 3),
        num_attention_heads=model_config.get('num_attention_heads', 8),
        dim_feedforward=model_config.get('dim_feedforward', 256),
        dropout=model_config.get('dropout', 0.1),
        feature_groups=model_config.get('feature_groups', None),
        device=device
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optionally load optimizer state
    if 'optimizer_state_dict' in checkpoint:
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.to(device)
    
    if verbose:
        print(f"Model loaded from {load_path}")
        
    return model


def save_features_as_dataframe(
    features: np.ndarray,
    original_df: pd.DataFrame,
    output_path: str,
    feature_prefix: str = 'regime_feature_',
    timestamp_col: str = 'Timestamp',
    value_col: str = 'Value',
    volume_col: str = 'Volume', 
    volatility_col: str = 'Volatility',
    include_original_metrics: bool = True,
    context_length: int = 50,
    causal: bool = True
) -> pd.DataFrame:
    """
    Save extracted features as a DataFrame with timestamps and original metrics.
    
    Parameters:
    -----------
    features : np.ndarray
        Extracted features array
    original_df : pd.DataFrame
        Original dataframe with timestamps and metrics
    output_path : str
        Path to save the features dataframe
    feature_prefix : str
        Prefix for feature column names
    timestamp_col : str
        Name of timestamp column
    value_col : str
        Name of price/value column
    volume_col : str
        Name of volume column
    volatility_col : str
        Name of volatility column
    include_original_metrics : bool
        Whether to include original metrics in output
    context_length : int
        Context length used for feature extraction
    causal : bool
        If True, map features to the last point in each window (for real-time applications)
        If False, map features to the center point (creates future leakage, research only)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with timestamps, original metrics, and features
    """
    # Create feature column names
    feature_columns = [f"{feature_prefix}{i+1}" for i in range(features.shape[1])]
    
    # Create dataframe with features
    features_df = pd.DataFrame(features, columns=feature_columns)
    
    # Calculate indices corresponding to the features
    if causal:
        # Causal mode: associate each feature with the last point in its window
        # For real-time applications (no future leakage)
        # In causal mode, we start at index context_length-1, and each feature
        # corresponds to the last (most recent) point in its window
        start_idx = context_length - 1
        feature_indices = range(start_idx, start_idx + len(features))
    else:
        # Bidirectional mode: associate each feature with the center of its window
        # WARNING: This creates future information leakage and should only be
        # used for research purposes, not for production or backtesting
        offset = context_length // 2
        feature_indices = range(offset, offset + len(features))
    
    # If we have fewer indices than features, extend the indices
    if len(feature_indices) < len(features):
        feature_indices = list(feature_indices)
        last_idx = feature_indices[-1]
        while len(feature_indices) < len(features):
            last_idx += 1
            feature_indices.append(min(last_idx, len(original_df)-1))
    
    # If we have more indices than features, truncate
    if len(feature_indices) > len(features):
        feature_indices = feature_indices[:len(features)]
    
    # Extract timestamps corresponding to these indices
    timestamps = original_df[timestamp_col].iloc[feature_indices].reset_index(drop=True)
    
    # Add timestamps to features dataframe
    features_df[timestamp_col] = timestamps.values
    
    # Add original metrics if requested
    if include_original_metrics:
        # Check which columns exist in the original dataframe
        for col_name, col_key in [
            (value_col, 'Value'), 
            (volume_col, 'Volume'), 
            (volatility_col, 'Volatility')
        ]:
            if col_name in original_df.columns:
                features_df[col_key] = original_df[col_name].iloc[feature_indices].reset_index(drop=True).values
    
    # Reorder columns to have timestamp and original metrics first
    core_cols = [timestamp_col]
    if include_original_metrics:
        for col_key in ['Value', 'Volume', 'Volatility']:
            if col_key in features_df.columns:
                core_cols.append(col_key)
    
    cols = core_cols + feature_columns
    features_df = features_df[cols]
    
    # Save to CSV
    features_df.to_csv(output_path, index=False)
    
    return features_df

def check_model_compatibility(model_path: str, input_size: int, verbose: bool = True) -> bool:
    """
    Check if a saved model is compatible with the current input data.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model checkpoint
    input_size : int
        Expected input size (feature count)
    verbose : bool
        Whether to print status messages
        
    Returns:
    --------
    bool
        True if model is compatible, False otherwise
    """
    try:
        # Load just the model config without loading the full model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model_config = checkpoint.get('model_config', {})
        
        # Check input size matches
        saved_input_size = model_config.get('input_size')
        if saved_input_size != input_size:
            if verbose:
                print(f"WARNING: Model input size ({saved_input_size}) doesn't match current data ({input_size})")
            return False
            
        return True
    except Exception as e:
        if verbose:
            print(f"Error checking model compatibility: {str(e)}")
        return False

def process_market_data(
    df: pd.DataFrame,
    model_dir: str = './saved_models',
    retrain: bool = True,
    num_epochs: int = 20,
    context_length: int = 50,
    num_attention_heads: int = 8,
    num_encoder_layers: int = 3,
    causal: bool = True,  # Use causal mode for real-time applications
    temperature: float = 0.5,
    grad_clip_norm: float = 1.0
) -> Tuple[pd.DataFrame, Optional[MaskedTemporalEncoder]]:
    """
    Process market data to extract volatility regime features.
    Can either train a new model or use a pre-trained one.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with raw market data
    model_dir : str
        Directory for saving/loading models
    retrain : bool
        Whether to train a new model (True) or use pre-trained (False)
    num_epochs : int
        Number of training epochs if retraining
    context_length : int
        Context window length for the model
    num_attention_heads : int
        Number of attention heads in the transformer model
    num_encoder_layers : int
        Number of transformer encoder layers
    causal : bool
        If True, use causal mode (only past information) for real-time applications
        If False, use bidirectional context (WARNING: creates future leakage, research only)
    temperature : float
        Temperature parameter for contrastive loss (lower = sharper contrasts)
    grad_clip_norm : float
        Maximum norm for gradient clipping
        
    Returns:
    --------
    Tuple[pd.DataFrame, Optional[MaskedTemporalEncoder]]
        DataFrame with extracted features and the model (if retrained)
    """
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    
    # Suppress all warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    # Assume timestamp column is 'Timestamp' by default
    timestamp_col = 'Timestamp'
    
    # Print minimal information (replace verbose parameter with minimal output)
    print(f"Model {'will be trained and saved to' if retrain else 'will be loaded from'}: {model_dir}")
    print(f"Using {'causal' if causal else 'bidirectional'} mode. " + 
          f"{'(Suitable for real-time applications)' if causal else '(WARNING: Contains future leakage! Research use only)'}")
    
    # Setup feature extractor
    print("Setting up feature extractor...")
    feature_extractor = FeatureExtractor()
    
    # Extract microstructure features
    print("Extracting microstructure features...")
    
    feature_df = feature_extractor.extract_features(df, timestamp_col=timestamp_col)
    
    print(f"Extracted {feature_extractor.get_feature_count()} microstructure features")
    
    # Prepare data tensors
    print("Preparing data tensors...")
    data = prepare_microstructure_data(
        df=df,
        feature_extractor=feature_extractor,
        context_length=context_length,
        timestamp_col=timestamp_col,
        causal=causal  # Pass causal mode to data preparation
    )
    
    print(f"Data shapes - Values: {data['values'].shape}, Time: {data['time_features'].shape}")
    
    # Handle NaN values
    if torch.isnan(data['values']).any() or torch.isnan(data['time_features']).any():
        print("WARNING: NaN values detected. Replacing with zeros.")
        data['values'] = torch.nan_to_num(data['values'], nan=0.0)
        data['time_features'] = torch.nan_to_num(data['time_features'], nan=0.0)
    
    # Define feature groups
    price_indices = [11, 12, 13, 14]      # price_range_* and log_return
    volume_indices = [2, 3, 4, 19]        # tick_imbalance, orderflow_imbalance, etc.
    momentum_indices = [0, 1, 8, 9, 10]   # trade_direction, is_buy, momentum_*
    volatility_indices = [15, 16, 17]     # bipower_var_*
    jump_indices = [5, 6, 7, 18]          # jump_* features
    
    feature_groups = {
        'price': price_indices,
        'volume': volume_indices, 
        'momentum': momentum_indices,
        'volatility': volatility_indices,
        'jumps': jump_indices,
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = None
    features = None
    
    # Either train new model or load existing one
    if retrain:
        # Create and train new model
        print("Training new model...")
        
        model = MaskedTemporalEncoder(
            input_size=data['values'].shape[2],
            context_length=context_length,
            d_model=128,
            num_encoder_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            dim_feedforward=256,
            batch_size=32,
            mask_ratio=0.15,
            temperature=temperature,
            feature_groups=feature_groups,
            device=device,
            causal=causal  # Pass causal mode to model
        )
        
        model.to(device)
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create proper attention mask (False values indicate valid positions)
        batch_size, seq_len = data['values'].shape[0], data['values'].shape[1]
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        data['attention_mask'] = attention_mask
        
        # Move data to device
        data = {k: v.to(device) for k, v in data.items()}
        
        # Train model
        train_masked_model(
            model=model,
            train_data=data,
            num_epochs=num_epochs,
            device=device,
            alpha=0.7,
            beta=0.2,
            gamma=0.1,
            warmup_epochs=min(5, num_epochs // 4),
            verbose=True
        )
        
        # Save model
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        model_path = f"{model_dir}/regime_model_{timestamp}_{'causal' if causal else 'bidirectional'}.pt"
        save_model(model, model_path, verbose=True)
        
    else:
        # Find and load most recent model
        print("Loading pre-trained model...")
        
        # Look for models with the right causal/bidirectional mode first
        mode_suffix = 'causal' if causal else 'bidirectional'
        model_files = glob.glob(f"{model_dir}/regime_model_*_{mode_suffix}.pt")
        
        # If no models with the right mode, try any model (will warn about mode mismatch)
        if not model_files:
            model_files = glob.glob(f"{model_dir}/regime_model_*.pt")
            
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        
        # Sort by modification time (newest first)
        newest_model = max(model_files, key=os.path.getmtime)
        print(f"Loading model from {newest_model}")
            
        # Check model compatibility
        input_size = data['values'].shape[2]
        if not check_model_compatibility(newest_model, input_size, verbose=True):
            print("WARNING: Model is not compatible with current data. Creating a new model...")
            
            # Create a new model with current data shape
            model = MaskedTemporalEncoder(
                input_size=input_size,
                context_length=context_length,
                d_model=128,
                num_encoder_layers=num_encoder_layers,
                num_attention_heads=num_attention_heads,
                dim_feedforward=256,
                batch_size=32,
                mask_ratio=0.15,
                temperature=temperature,
                feature_groups=feature_groups,
                device=device,
                causal=causal  # Pass causal mode to model
            )
            
            # Train the model with a few epochs
            batch_size, seq_len = data['values'].shape[0], data['values'].shape[1]
            attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
            data['attention_mask'] = attention_mask
            data = {k: v.to(device) for k, v in data.items()}
            
            print("Training model with new data shape...")
            
            train_masked_model(
                model=model,
                train_data=data,
                num_epochs=min(5, num_epochs),  # Shorter training
                device=device,
                alpha=0.7,
                beta=0.2,
                gamma=0.1,
                warmup_epochs=1,
                verbose=True
            )
            
            # Save the new model
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
            model_path = f"{model_dir}/regime_model_{timestamp}_{'causal' if causal else 'bidirectional'}_auto.pt"
            save_model(model, model_path, verbose=True)
        else:
            # Load existing compatible model
            model = load_model(newest_model, device, verbose=True)
            
            # Check if loaded model's causal setting matches requested mode
            if model.causal != causal:
                print(f"WARNING: Loaded model uses {'causal' if model.causal else 'bidirectional'} mode, " +
                      f"but you requested {'causal' if causal else 'bidirectional'} mode. " +
                      f"Using model's mode for consistency.")
    
    # Extract features
    print("Extracting features...")
    
    # Process in smaller batches to avoid memory issues
    batch_size = 32
    all_features = []
    num_samples = len(df) - context_length + 1
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_df = df.iloc[start_idx:end_idx + context_length - 1].copy()
        
        # Extract features for this batch
        batch_data = prepare_microstructure_data(
            df=batch_df,
            feature_extractor=feature_extractor,
            context_length=context_length,
            timestamp_col=timestamp_col,
            causal=causal  # Pass causal mode to data preparation
        )
        
        # Create explicit boolean attention mask (False = valid)
        batch_attention_mask = torch.zeros(
            (batch_data['values'].shape[0], batch_data['values'].shape[1]), 
            dtype=torch.bool, 
            device=device
        )
        
        # Move data to device
        batch_values = batch_data['values'].to(device)
        batch_time_features = batch_data['time_features'].to(device)
        
        # Extract features
        batch_features = model.extract_features(
            values=batch_values,
            time_features=batch_time_features,
            attention_mask=batch_attention_mask,
            use_projection_head=True
        )
        all_features.append(batch_features.cpu().numpy())
    
    # Combine all batches
    features = np.vstack(all_features)
    print(f"Extracted features shape: {features.shape}")
    
    # Save features as dataframe
    features_df = save_features_as_dataframe(
        features=features,
        original_df=df,
        output_path=f'{model_dir}/regime_features.csv',  # Save in model_dir instead of output_dir
        value_col='Value',
        volume_col='Volume',
        volatility_col='Volatility',
        include_original_metrics=True,
        context_length=context_length,
        causal=causal  # Pass causal mode to feature mapping
    )
    
    print(f"Features saved to {model_dir}/regime_features.csv")
    
    return features_df, model if retrain else None 