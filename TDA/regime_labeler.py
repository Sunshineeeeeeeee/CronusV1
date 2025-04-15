import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import math
import torch.utils.checkpoint as checkpoint
from tqdm import tqdm
import warnings
import logging


class HierarchicalRegimeDataset(Dataset):
    """
    Dataset for hierarchical regime classification.
    """
    def __init__(
        self, 
        df: pd.DataFrame,
        feature_cols: List[str],
        regime_col: str = 'regime',
        sub_regime_col: str = 'sub_regime',
        timestamp_col: Optional[str] = 'Timestamp',
        window_size: int = 50,
        stride: int = 1
    ):
        """
        Initialize dataset for hierarchical regime classification.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing features and labels
        feature_cols : List[str]
            List of feature column names
        regime_col : str
            Name of primary regime column
        sub_regime_col : str
            Name of sub-regime column
        timestamp_col : str, optional
            Name of timestamp column for sequential data
        window_size : int
            Size of sliding window for sequential data
        stride : int
            Stride for sliding window
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.regime_col = regime_col
        self.sub_regime_col = sub_regime_col
        self.timestamp_col = timestamp_col
        self.window_size = window_size
        self.stride = stride
        
        # Create label encoders for regimes and sub-regimes
        self.regime_encoder = LabelEncoder()
        self.sub_regime_encoder = LabelEncoder()
        
        # Fit label encoders
        self.regime_encoder.fit(self.df[regime_col].unique())
        self.sub_regime_encoder.fit(self.df[sub_regime_col].unique())
        
        # Create mapping dictionaries
        self.n_regimes = len(self.regime_encoder.classes_)
        self.n_sub_regimes = len(self.sub_regime_encoder.classes_)
        
        # Create regime to sub-regime mapping for consistency loss
        self.regime_to_sub_regimes = {}
        for regime in self.df[regime_col].unique():
            sub_regimes = self.df[self.df[regime_col] == regime][sub_regime_col].unique()
            self.regime_to_sub_regimes[regime] = list(sub_regimes)
            
        # Create windows if using sequential data
        self._create_windows()
        
    def _create_windows(self):
        """Create sliding windows for sequential data."""
        self.windows = []
        n_samples = len(self.df)
        
        for i in range(0, n_samples - self.window_size + 1, self.stride):
            # Get the window indices
            window_indices = list(range(i, i + self.window_size))
            
            # Get the regime and sub-regime for this window (using last point)
            regime = self.df.iloc[i + self.window_size - 1][self.regime_col]
            sub_regime = self.df.iloc[i + self.window_size - 1][self.sub_regime_col]
            
            # Add to windows list
            self.windows.append({
                'indices': window_indices,
                'regime': regime,
                'sub_regime': sub_regime
            })
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        indices = window['indices']
        
        # Get features for this window
        features = self.df.iloc[indices][self.feature_cols].values
        
        # Get encoded regime and sub-regime - handle unseen labels gracefully
        try:
            regime = self.regime_encoder.transform([window['regime']])[0]
        except ValueError:
            # Use most common regime for unseen values
            regime = 0  # Default to first regime
            warnings.warn(f"Encountered unseen regime: {window['regime']}, defaulting to {self.regime_encoder.classes_[0]}")
        
        try:
            sub_regime = self.sub_regime_encoder.transform([window['sub_regime']])[0]
        except ValueError:
            # Use most common sub-regime for unseen values
            sub_regime = 0  # Default to first sub-regime
            warnings.warn(f"Encountered unseen sub-regime: {window['sub_regime']}, defaulting to {self.sub_regime_encoder.classes_[0]}")
        
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        regime_tensor = torch.tensor(regime, dtype=torch.long)
        sub_regime_tensor = torch.tensor(sub_regime, dtype=torch.long)
        
        # Return a tuple to maintain consistent format expected by trainer
        return features_tensor, regime_tensor, sub_regime_tensor
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced classes."""
        # Count regimes
        regime_counts = self.df[self.regime_col].value_counts().to_dict()
        total_regimes = len(self.df)
        
        # Calculate regime weights (inverse frequency)
        regime_weights = {
            self.regime_encoder.transform([r])[0]: total_regimes / count 
            for r, count in regime_counts.items()
        }
        
        # Normalize regime weights
        regime_weight_sum = sum(regime_weights.values())
        regime_weights = {k: v / regime_weight_sum * len(regime_weights) 
                         for k, v in regime_weights.items()}
        
        # Do the same for sub-regimes
        sub_regime_counts = self.df[self.sub_regime_col].value_counts().to_dict()
        
        sub_regime_weights = {
            self.sub_regime_encoder.transform([sr])[0]: total_regimes / count 
            for sr, count in sub_regime_counts.items()
        }
        
        # Normalize sub-regime weights
        sub_regime_weight_sum = sum(sub_regime_weights.values())
        sub_regime_weights = {k: v / sub_regime_weight_sum * len(sub_regime_weights) 
                             for k, v in sub_regime_weights.items()}
        
        # Convert to tensors for easier use in loss function
        regime_weights_tensor = torch.zeros(self.n_regimes)
        for idx, weight in regime_weights.items():
            regime_weights_tensor[idx] = weight
            
        sub_regime_weights_tensor = torch.zeros(self.n_sub_regimes)
        for idx, weight in sub_regime_weights.items():
            sub_regime_weights_tensor[idx] = weight
            
        return {
            'regime_weights': regime_weights,
            'sub_regime_weights': sub_regime_weights,
            'regime_weights_tensor': regime_weights_tensor,
            'sub_regime_weights_tensor': sub_regime_weights_tensor
        }


class HierarchicalRegimeModel(nn.Module):
    """
    Enhanced hierarchical regime model with attention mechanisms and improved architecture.
    """
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=256, n_gru_layers=2, dropout=0.3, n_regimes=10, n_sub_regimes=11, use_gradient_checkpointing=False):
        """
        Initialize the hierarchical regime classification model with improved architecture
        
        Args:
            input_dim (int): Dimension of the input features
            hidden_dim (int): Hidden dimension size
            embedding_dim (int): Embedding dimension size
            n_gru_layers (int): Number of GRU layers
            dropout (float): Dropout rate
            n_regimes (int): Number of regime classes
            n_sub_regimes (int): Number of sub-regime classes
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing for memory efficiency
        """
        super(HierarchicalRegimeModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_gru_layers = n_gru_layers
        self.dropout = dropout
        self.n_regimes = n_regimes
        self.n_sub_regimes = n_sub_regimes
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Feature embedding layer
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # GRU layer with residual connections
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0,
            bidirectional=True
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),  # Bidirectional GRU output
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Regime classification head
        self.regime_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, n_regimes)
        )
        
        # Sub-regime classification head
        self.sub_regime_classifier = nn.Sequential(
            nn.Linear(embedding_dim + n_regimes, embedding_dim // 2),  # Concatenate regime logits
            nn.LayerNorm(embedding_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, n_sub_regimes)
        )
    
    def _gru_forward(self, x):
        # Apply GRU
        gru_out, _ = self.gru(x)
        return gru_out
        
    def forward(self, x):
        # Apply feature embedding
        x = self.feature_embedding(x)
        
        # Use gradient checkpointing to save memory during training if enabled
        if self.use_gradient_checkpointing and self.training:
            gru_out = checkpoint.checkpoint(self._gru_forward, x)
        else:
            # Apply GRU
            gru_out = self._gru_forward(x)
        
        # Apply feature fusion (use the last output from the sequence)
        features = self.feature_fusion(gru_out[:, -1, :])
        
        # Get regime logits
        regime_logits = self.regime_classifier(features)
        
        # Concatenate features with regime logits for sub-regime classification
        combined_features = torch.cat([features, regime_logits], dim=1)
        
        # Get sub-regime logits
        sub_regime_logits = self.sub_regime_classifier(combined_features)
        
        return regime_logits, sub_regime_logits


class HierarchicalLoss(nn.Module):
    """
    Enhanced hierarchical loss with adaptive margins and anti-collapse techniques.
    Combines focal loss, logit adjustment, label smoothing and consistency regularization.
    """
    def __init__(self, regime_to_sub_regimes, regime_encoder, sub_regime_encoder,
                 alpha=0.6, beta=0.3, gamma=0.1, device='cpu', 
                 focal_gamma=2.0, label_smoothing=0.1, temperature=1.0,
                 margin=0.05, class_freq_power=0.25):
        """
        Initialize the enhanced hierarchical loss with anti-collapse mechanisms.
        
        Args:
            regime_to_sub_regimes: Mapping from regimes to sub-regimes
            regime_encoder: Encoder for regime labels
            sub_regime_encoder: Encoder for sub-regime labels
            alpha: Weight for primary regime loss (0.6)
            beta: Weight for sub-regime loss (0.3) 
            gamma: Weight for consistency loss (0.1)
            device: Computation device
            focal_gamma: Focal loss gamma parameter (higher means more focus on hard examples)
            label_smoothing: Label smoothing factor to prevent overconfidence
            temperature: Temperature scaling for softmax
            margin: Margin for logit adjustment based on class frequency
            class_freq_power: Power factor for frequency-based adjustments
        """
        super(HierarchicalLoss, self).__init__()
        self.regime_to_sub_regimes = regime_to_sub_regimes
        self.regime_encoder = regime_encoder
        self.sub_regime_encoder = sub_regime_encoder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        self.margin = margin
        self.class_freq_power = class_freq_power
        
        # Get number of classes 
        self.n_regimes = len(regime_encoder.classes_)
        self.n_sub_regimes = len(sub_regime_encoder.classes_)
        
        # Create a mask of valid regime to sub-regime combinations
        self.valid_combinations = self._create_valid_combinations_mask()
        print(f"Valid combinations mask shape: {self.valid_combinations.shape}")
        print(f"Number of valid combinations: {self.valid_combinations.sum().item()}")
        print(f"Created HierarchicalLoss with {self.n_regimes} regimes and {self.n_sub_regimes} sub-regimes")
        
        # Print the mapping of regimes to sub-regimes
        mapping_str = {k: [sub_regime_encoder.classes_[i] for i, valid in 
                      enumerate(self.valid_combinations[regime_encoder.transform([k])[0]]) if valid] 
                      for k in regime_encoder.classes_}
        print(f"Regime to sub-regime mapping: {mapping_str}")
        
        # Initialize class frequency tracking for adaptive margin
        self.register_buffer('regime_freq', torch.ones(self.n_regimes, device=device))
        self.register_buffer('sub_regime_freq', torch.ones(self.n_sub_regimes, device=device))
        
        # Anti-collapse: initialize temperature schedule that will gradually increase
        self.initial_temp = temperature
        self.current_temp = temperature
        self.temp_step = 0.01  # Small increment for temperature
        self.max_temp = 2.0    # Maximum temperature
        
        # Track prediction diversity for anti-collapse monitoring
        self.register_buffer('regime_pred_history', torch.zeros(self.n_regimes, device=device))
        self.register_buffer('sub_regime_pred_history', torch.zeros(self.n_sub_regimes, device=device))
        self.history_momentum = 0.9  # EMA factor for prediction history
    
    def _create_valid_combinations_mask(self):
        """Create a binary mask of valid regime to sub-regime combinations."""
        mask = torch.zeros((self.n_regimes, self.n_sub_regimes), device=self.device)
        
        for regime, sub_regimes in self.regime_to_sub_regimes.items():
            try:
                regime_idx = self.regime_encoder.transform([regime])[0]
                for sub_regime in sub_regimes:
                    try:
                        sub_regime_idx = self.sub_regime_encoder.transform([sub_regime])[0]
                        mask[regime_idx, sub_regime_idx] = 1
                    except ValueError:
                        # Sub-regime not in encoder
                        pass
            except ValueError:
                # Regime not in encoder
                pass
        
        return mask
    
    def _update_class_frequencies(self, regime_targets, sub_regime_targets):
        """Update class frequency counters for adaptive margin adjustment"""
        # Update regime frequencies
        regime_counts = torch.bincount(regime_targets, minlength=self.n_regimes).float()
        self.regime_freq = 0.99 * self.regime_freq + 0.01 * regime_counts
        
        # Update sub-regime frequencies
        sub_regime_counts = torch.bincount(sub_regime_targets, minlength=self.n_sub_regimes).float()
        self.sub_regime_freq = 0.99 * self.sub_regime_freq + 0.01 * sub_regime_counts
        
        # Normalize frequencies
        if self.regime_freq.sum() > 0:
            self.regime_freq = self.regime_freq / (self.regime_freq.sum() + 1e-8)
        if self.sub_regime_freq.sum() > 0:
            self.sub_regime_freq = self.sub_regime_freq / (self.sub_regime_freq.sum() + 1e-8)
    
    def _get_adaptive_margins(self):
        """Calculate adaptive margins based on class frequencies"""
        # Compute adaptive margins based on frequency - rare classes get higher margins
        # Apply power scaling to prevent extreme values
        regime_margins = torch.pow(1.0 - self.regime_freq, self.class_freq_power) * self.margin
        sub_regime_margins = torch.pow(1.0 - self.sub_regime_freq, self.class_freq_power) * self.margin
        
        return regime_margins, sub_regime_margins
    
    def _calculate_focal_loss_with_adaptive_margin(self, logits, targets, margins=None, weights=None):
        """
        Enhanced focal loss with adaptive margins and logit adjustment
        
        Args:
            logits: Prediction logits
            targets: Ground truth targets
            margins: Per-class margins for logit adjustment
            weights: Optional class weights
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        # Apply temperature scaling
        logits = logits / self.current_temp
        
        # Apply logit adjustment based on margins if provided
        if margins is not None:
            # Create one-hot encoding of targets for margin application
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
            
            # Apply margins to adjust logits (give more margin to rare classes)
            # Subtract margin from non-target classes
            margin_weights = torch.ones_like(logits) * margins.unsqueeze(0)
            margin_weights = margin_weights * (1 - one_hot)  # Only apply to non-target classes
            logits = logits - margin_weights
        
        # Create target distribution with label smoothing
        if self.label_smoothing > 0:
            soft_targets = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1 - self.label_smoothing
            )
            # Add small probability for other classes
            soft_targets += self.label_smoothing / num_classes
        else:
            soft_targets = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        
        # Compute probabilities and log probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        # Compute focal weights
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = torch.pow(1 - p_t, self.focal_gamma)
        
        # Calculate per-sample loss
        loss = -torch.sum(soft_targets * log_probs, dim=1)
        
        # Apply focal and class weights
        if weights is not None:
            # Get weights for each sample based on its target class
            sample_weights = weights.gather(0, targets)
            focal_loss = focal_weight * loss * sample_weights
        else:
            focal_loss = focal_weight * loss
        
        # Return mean loss
        return focal_loss.mean()
    
    def _update_prediction_diversity(self, regime_preds, sub_regime_preds):
        """Update prediction diversity tracking for anti-collapse monitoring"""
        # Update regime prediction history with exponential moving average
        regime_counts = torch.bincount(regime_preds, minlength=self.n_regimes).float()
        regime_dist = regime_counts / (regime_counts.sum() + 1e-8)
        self.regime_pred_history = (self.history_momentum * self.regime_pred_history + 
                                   (1 - self.history_momentum) * regime_dist)
        
        # Update sub-regime prediction history
        sub_regime_counts = torch.bincount(sub_regime_preds, minlength=self.n_sub_regimes).float()
        sub_regime_dist = sub_regime_counts / (sub_regime_counts.sum() + 1e-8)
        self.sub_regime_pred_history = (self.history_momentum * self.sub_regime_pred_history + 
                                       (1 - self.history_momentum) * sub_regime_dist)
    
    def _check_for_collapse(self, regime_preds, sub_regime_preds):
        """Check for model collapse and adjust temperature if needed"""
        # Count unique predictions
        unique_regime_preds = torch.unique(regime_preds).size(0)
        unique_sub_regime_preds = torch.unique(sub_regime_preds).size(0)
        
        # Calculate entropy of prediction distribution
        regime_entropy = -(self.regime_pred_history * torch.log(self.regime_pred_history + 1e-10)).sum()
        sub_regime_entropy = -(self.sub_regime_pred_history * torch.log(self.sub_regime_pred_history + 1e-10)).sum()
        
        # Check for collapse - low entropy or few unique predictions
        is_collapsed = (
            unique_regime_preds < 3 or 
            unique_sub_regime_preds < 3 or
            regime_entropy < 0.7 * math.log(self.n_regimes) or
            sub_regime_entropy < 0.7 * math.log(self.n_sub_regimes)
        )
        
        # If collapsed, increase temperature to encourage exploration
        if is_collapsed:
            self.current_temp = min(self.max_temp, self.current_temp + self.temp_step)
            
            # Print warning only occasionally to avoid spam
            if torch.rand(1).item() < 0.1:  # 10% chance to print
                print(f"\nWARNING - MODEL COLLAPSE DETECTED:")
                print(f"  Unique regime predictions: {unique_regime_preds} (entropy={regime_entropy:.4f})")
                print(f"  Unique sub-regime predictions: {unique_sub_regime_preds} (entropy={sub_regime_entropy:.4f})")
                print(f"  Increasing temperature to {self.current_temp:.2f}")
        else:
            # Slowly decrease temperature if no collapse
            self.current_temp = max(self.initial_temp, self.current_temp - self.temp_step * 0.1)
    
    def forward(self, regime_logits, sub_regime_logits, regime_targets, sub_regime_targets, 
                regime_weights=None, sub_regime_weights=None):
        """Forward pass with enhanced hierarchical loss computation"""
        batch_size = regime_logits.size(0)
        
        # Get adaptive margins based on class frequency
        regime_margins, sub_regime_margins = self._get_adaptive_margins()
        
        # Update class frequency tracking
        self._update_class_frequencies(regime_targets, sub_regime_targets)
        
        # Primary regime loss with adaptive margins
        regime_loss = self._calculate_focal_loss_with_adaptive_margin(
            regime_logits, 
            regime_targets,
            margins=regime_margins,
            weights=regime_weights
        )
        
        # Sub-regime loss with adaptive margins
        sub_regime_loss = self._calculate_focal_loss_with_adaptive_margin(
            sub_regime_logits, 
            sub_regime_targets,
            margins=sub_regime_margins,
            weights=sub_regime_weights
        )
        
        # Clamp sub-regime loss to prevent extreme values
        sub_regime_loss = torch.clamp(sub_regime_loss, max=10.0)
        
        # Calculate consistency loss using KL divergence
        # Get probabilities from the logits
        regime_probs = F.softmax(regime_logits / self.current_temp, dim=1)
        
        # Compute expected sub-regime distribution
        expected_sub_regime_probs = []
        for i in range(batch_size):
            # Calculate expected sub-regime probabilities based on regime probabilities
            # and valid regime to sub-regime combinations
            regime_prob = regime_probs[i].unsqueeze(1)  # [n_regimes, 1]
            valid_sub_regimes = self.valid_combinations * regime_prob  # [n_regimes, n_sub_regimes]
            sub_regime_dist = valid_sub_regimes.sum(0)  # [n_sub_regimes]
            
            # Normalize if not zero
            norm_sum = sub_regime_dist.sum()
            if norm_sum > 1e-10:
                sub_regime_dist = sub_regime_dist / norm_sum
            expected_sub_regime_probs.append(sub_regime_dist)
        
        # Stack the distributions
        expected_sub_regime_dist = torch.stack(expected_sub_regime_probs)
        
        # Calculate KL divergence between predicted and expected sub-regime distributions
        # Use log_softmax for better numerical stability
        log_sub_regime_probs = F.log_softmax(sub_regime_logits / self.current_temp, dim=1)
        consistency_loss = F.kl_div(
            log_sub_regime_probs,
            expected_sub_regime_dist + 1e-10,
            reduction='batchmean',
            log_target=False
        )
        
        # Calculate predictions for monitoring
        with torch.no_grad():
            _, regime_preds = torch.max(regime_logits, 1)
            _, sub_regime_preds = torch.max(sub_regime_logits, 1)
            
            # Update prediction diversity tracking
            self._update_prediction_diversity(regime_preds, sub_regime_preds)
            
            # Check for model collapse and adjust temperature if needed
            self._check_for_collapse(regime_preds, sub_regime_preds)
        
        # Combine the losses with weights
        total_loss = self.alpha * regime_loss + self.beta * sub_regime_loss + self.gamma * consistency_loss
        
        # Return as dictionary for backward compatibility with trainer
        loss_dict = {
            'total_loss': total_loss,
            'regime_loss': regime_loss,
            'sub_regime_loss': sub_regime_loss,
            'consistency_loss': consistency_loss,
            'individual_losses': {
                'regime_loss': regime_loss.item(),
                'sub_regime_loss': sub_regime_loss.item(),
                'consistency_loss': consistency_loss.item()
            }
        }
        
        return loss_dict


class GradientManager:
    """Manages gradient clipping, scaling, and noise for stable training."""
    
    def __init__(self, model, max_norm=1.0, noise_scale=0.0, centralize=True, stats_window=100):
        """
        Initialize the gradient manager.
        
        Args:
            model: The model to manage gradients for
            max_norm: Maximum norm for gradient clipping
            noise_scale: Scale of noise to add to gradients (0.0 to disable)
            centralize: Whether to centralize gradients (zero mean)
            stats_window: Window size for tracking gradient statistics
        """
        self.model = model
        self.max_norm = max_norm
        self.noise_scale = noise_scale
        self.centralize = centralize
        
        # Statistics tracking
        self.grad_norms = []
        self.stats_window = stats_window
        self.iteration = 0
    
    def _centralize_gradients(self):
        """Centralize gradients to improve training stability."""
        for name, param in self.model.named_parameters():
            if param.grad is not None and len(param.shape) > 1:  # For weight matrices, not biases
                shape = param.grad.shape
                mean = param.grad.mean(dim=tuple(range(1, len(shape))), keepdim=True)
                param.grad.data.sub_(mean)  # Subtract mean for centralization
    
    def _add_gradient_noise(self):
        """Add small amount of Gaussian noise to gradients."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.noise_scale * (param.grad.abs().mean())
                param.grad.data.add_(noise)
    
    def clip_and_process(self):
        """Clip gradients and apply additional processing."""
        # Compute gradient norm for statistics
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) 
                         for p in self.model.parameters() if p.grad is not None]), 2)
        self.grad_norms.append(total_norm.item())
        
        # Keep fixed window of statistics
        if len(self.grad_norms) > self.stats_window:
            self.grad_norms.pop(0)
        
        # Clip gradients
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        
        # Centralize gradients
        if self.centralize:
            self._centralize_gradients()
        
        # Add gradient noise
        if self.noise_scale > 0 and self.iteration > 1000:  # Start noise after 1000 iterations
            self._add_gradient_noise()
        
        # Increment iteration counter
        self.iteration += 1
        
        # Log statistics periodically
        if self.iteration % 50 == 0:
            if len(self.grad_norms) > 0:
                avg_norm = sum(self.grad_norms) / len(self.grad_norms)
                print(f"Grad stats: avg_norm={avg_norm:.4f}, recent={self.grad_norms[-1]:.4f}")
    
    def update_max_norm(self, new_norm):
        """Update maximum gradient norm."""
        self.max_norm = new_norm
        
    def update_noise_scale(self, new_scale):
        """Update gradient noise scale."""
        self.noise_scale = new_scale


class WarmupScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing.
    Combines warmup with cosine annealing for better training stability.
    """
    
    def __init__(self, optimizer, warmup_steps=1000, max_lr=1e-3, min_lr=1e-6, 
                 patience=5, factor=0.5, total_steps=None):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            warmup_steps: Number of warmup steps
            max_lr: Maximum learning rate after warmup
            min_lr: Minimum learning rate
            patience: Patience for reducing learning rate
            factor: Factor for reducing learning rate
            total_steps: Total training steps (for cosine annealing)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.total_steps = total_steps
        
        # Variables for tracking
        self.step_count = 0
        self.best_loss = float('inf')
        self.bad_loss_count = 0
        
        # Initial learning rate
        self.base_lrs = [self.min_lr for _ in self.optimizer.param_groups]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
    
    def cosine_schedule(self, step):
        """Cosine annealing schedule."""
        if self.total_steps is None:
            return self.max_lr
        
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = max(0.0, min(1.0, progress))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
    
    def step(self):
        """Update learning rate based on warmup schedule."""
        self.step_count += 1
        
        # During warmup
        if self.step_count < self.warmup_steps:
            progress = self.step_count / self.warmup_steps
            lr = self.min_lr + progress * (self.max_lr - self.min_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        # After warmup, use cosine schedule
        else:
            lr = self.cosine_schedule(self.step_count)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def step_with_loss(self, loss):
        """Update learning rate based on loss (for plateau detection)."""
        self.step()  # First do regular step
        
        # Then check for plateau
        if loss < self.best_loss:
            self.best_loss = loss
            self.bad_loss_count = 0
        else:
            self.bad_loss_count += 1
            
        # If plateau detected, reduce learning rate
        if self.bad_loss_count >= self.patience:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(self.min_lr, param_group['lr'] * self.factor)
            self.bad_loss_count = 0
            print(f"Reducing learning rate to {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class RegimeLabelTrainer:
    """
    Trainer for hierarchical regime classification model.
    Handles training, evaluation, and prediction with advanced training features.
    """
    
    def __init__(
        self, 
        model, 
        loss_fn,
        optimizer=None,
        device=None,
        grad_clip_norm=1.0,
        lr=1e-3,
        warmup_steps=1000,
        scheduler_patience=5,
        use_gradient_centralization=True,
        gradient_noise_scale=0.01,
        log_interval=10,
        debug=False
    ):
        """
        Initialize the RegimeLabelTrainer.
        
        Args:
            model: HierarchicalRegimeModel
            loss_fn: Loss function (HierarchicalLoss)
            optimizer: Optional torch optimizer (if None, Adam will be used)
            device: Device to use for training
            grad_clip_norm: Maximum gradient norm for clipping
            lr: Learning rate
            warmup_steps: Warmup steps for scheduler
            scheduler_patience: Patience for learning rate reduction
            use_gradient_centralization: Whether to use gradient centralization
            gradient_noise_scale: Scale of noise to add to gradients (0 to disable)
            log_interval: Interval for logging during training
            debug: Whether to print debug information
        """
        self.model = model
        self.loss_fn = loss_fn
        
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
            
        # Initialize scheduler and gradient manager
        self.scheduler = WarmupScheduler(
            self.optimizer, 
            warmup_steps=warmup_steps,
            max_lr=lr,
            min_lr=lr/100,
            patience=scheduler_patience
        )
        
        self.grad_manager = GradientManager(
            self.model,
            max_norm=grad_clip_norm,
            noise_scale=gradient_noise_scale,
            centralize=use_gradient_centralization
        )
        
        # Training settings
        self.log_interval = log_interval
        self.debug = debug
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Print setup information
        print(f"Training on device: {self.device}")
        print(f"Model parameter count: {self._count_parameters():,}")
    
    def _count_parameters(self):
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _prepare_batch(self, batch):
        """
        Prepare batch data for training or evaluation.
        Handles different batch formats and transfers to device.
        
        Args:
            batch: The batch data from dataloader or dictionary

        Returns:
            Tuple of (features, regime_targets, sub_regime_targets)
        """
        try:
            # Handle different batch formats
            if isinstance(batch, dict):
                # Dictionary format
                X = batch.get('features', batch.get('X'))
                regime_targets = batch.get('regime', batch.get('regimes', batch.get('regime_targets')))
                sub_regime_targets = batch.get('sub_regime', batch.get('sub_regimes', batch.get('sub_regime_targets')))
            elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
                # DataLoader format with unpacked tuple
                X, regime_targets, sub_regime_targets = batch[:3]
            else:
                # Try to handle other formats
                print(f"Warning: Unexpected batch format: {type(batch)}")
                # If it's a simple tensor, treat it as features only (prediction mode)
                if isinstance(batch, torch.Tensor):
                    X = batch
                    regime_targets = None
                    sub_regime_targets = None
                else:
                    # Last resort - try to convert to list and unpack
                    try:
                        batch_list = list(batch)
                        if len(batch_list) >= 3:
                            X, regime_targets, sub_regime_targets = batch_list[:3]
                        elif len(batch_list) >= 1:
                            X = batch_list[0]
                            regime_targets = None
                            sub_regime_targets = None
                        else:
                            raise ValueError("Batch contains no elements")
                    except:
                        raise ValueError(f"Unsupported batch format: {type(batch)}")
            
            # Check for None or invalid values
            if X is None:
                raise ValueError("Features (X) is missing from batch")
            
            # Move to device if tensor
            if isinstance(X, torch.Tensor):
                X = X.to(self.device)
            else:
                X = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            # Handle regime targets - could be None in prediction mode
            if regime_targets is not None:
                if isinstance(regime_targets, torch.Tensor):
                    regime_targets = regime_targets.to(self.device)
                else:
                    regime_targets = torch.tensor(regime_targets, dtype=torch.long).to(self.device)
                
            # Handle sub-regime targets - could be None in prediction mode
            if sub_regime_targets is not None:
                if isinstance(sub_regime_targets, torch.Tensor):
                    sub_regime_targets = sub_regime_targets.to(self.device)
                else:
                    sub_regime_targets = torch.tensor(sub_regime_targets, dtype=torch.long).to(self.device)
            
            return X, regime_targets, sub_regime_targets
        
        except Exception as e:
            print(f"Error in _prepare_batch: {e}")
            print(f"Batch type: {type(batch)}")
            if isinstance(batch, (list, tuple)):
                print(f"Batch length: {len(batch)}")
                for i, item in enumerate(batch):
                    print(f"  Item {i} type: {type(item)}")
            elif isinstance(batch, dict):
                print(f"Batch keys: {batch.keys()}")
            # Return empty tensors as fallback
            dummy_X = torch.zeros((1, 1, self.model.input_dim), device=self.device)
            dummy_regime = torch.zeros(1, dtype=torch.long, device=self.device)
            dummy_sub_regime = torch.zeros(1, dtype=torch.long, device=self.device)
            return dummy_X, dummy_regime, dummy_sub_regime
    
    def train_epoch(self, train_loader, regime_weights=None, sub_regime_weights=None):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            regime_weights: Optional weights for regime classes (dictionary or tensor)
            sub_regime_weights: Optional weights for sub-regime classes (dictionary or tensor)
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        total_regime_loss = 0
        total_sub_regime_loss = 0
        total_consistency_loss = 0
        correct_regimes = 0
        correct_sub_regimes = 0
        total_samples = 0
        
        # Convert dictionary weights to tensor if needed
        if isinstance(regime_weights, dict):
            # Convert dict to tensor, ordered by class indices
            regime_classes = sorted(regime_weights.keys())
            regime_weight_tensor = torch.tensor([regime_weights[c] for c in regime_classes], 
                                              device=self.device, dtype=torch.float32)
        else:
            regime_weight_tensor = regime_weights
            
        if isinstance(sub_regime_weights, dict):
            # Convert dict to tensor, ordered by class indices
            sub_regime_classes = sorted(sub_regime_weights.keys())
            sub_regime_weight_tensor = torch.tensor([sub_regime_weights[c] for c in sub_regime_classes],
                                                  device=self.device, dtype=torch.float32)
        else:
            sub_regime_weight_tensor = sub_regime_weights
        
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            try:
                X, regime_targets, sub_regime_targets = self._prepare_batch(batch)
                
                # Skip this batch if targets are missing
                if regime_targets is None or sub_regime_targets is None:
                    print(f"Skipping batch {batch_idx} due to missing targets")
                    continue
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                regime_logits, sub_regime_logits = self.model(X)
                
                # Compute loss
                loss_dict = self.loss_fn(
                    regime_logits, 
                    sub_regime_logits, 
                    regime_targets, 
                    sub_regime_targets,
                    regime_weights=regime_weight_tensor,
                    sub_regime_weights=sub_regime_weight_tensor
                )
                
                # Backward pass - handle errors safely
                try:
                    loss_dict['total_loss'].backward()
                    
                    # Process and clip gradients
                    self.grad_manager.clip_and_process()
                    
                    # Update weights
                    self.optimizer.step()
                except RuntimeError as e:
                    print(f"Error in backward pass for batch {batch_idx}: {e}")
                    # Zero gradients to recover from error
                    self.optimizer.zero_grad()
                    continue
                
                # Update learning rate
                self.scheduler.step()
                
                # Update metrics
                batch_size = X.size(0)
                total_loss += loss_dict['total_loss'].item() * batch_size
                total_regime_loss += loss_dict['regime_loss'].item() * batch_size
                total_sub_regime_loss += loss_dict['sub_regime_loss'].item() * batch_size
                total_consistency_loss += loss_dict['consistency_loss'].item() * batch_size
                total_samples += batch_size
                
                # Calculate accuracy
                _, regime_preds = torch.max(regime_logits, 1)
                _, sub_regime_preds = torch.max(sub_regime_logits, 1)
                correct_regimes += (regime_preds == regime_targets).sum().item()
                correct_sub_regimes += (sub_regime_preds == sub_regime_targets).sum().item()
                
                # Log progress
                if self.debug and batch_idx % self.log_interval == 0:
                    print(f"Train Batch: {batch_idx}/{len(train_loader)} "
                        f"Loss: {loss_dict['total_loss'].item():.4f} "
                        f"LR: {self.scheduler.get_lr():.6f}")
                    
                    # Print distribution of predictions
                    if batch_idx % (self.log_interval * 5) == 0:
                        regime_pred_counts = torch.bincount(regime_preds, minlength=regime_logits.size(1))
                        sub_regime_pred_counts = torch.bincount(sub_regime_preds, minlength=sub_regime_logits.size(1))
                        print(f"Regime pred distribution: {regime_pred_counts}")
                        print(f"Sub-regime pred distribution: {sub_regime_pred_counts}")
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Check if any samples were processed
        if total_samples == 0:
            print("Warning: No valid samples processed during training epoch!")
            return {
                'loss': float('inf'),
                'regime_loss': float('inf'),
                'sub_regime_loss': float('inf'),
                'consistency_loss': float('inf'),
                'regime_accuracy': 0.0,
                'sub_regime_accuracy': 0.0,
                'learning_rate': self.scheduler.get_lr()
            }
        
        # Calculate average metrics
        avg_loss = total_loss / total_samples
        avg_regime_loss = total_regime_loss / total_samples
        avg_sub_regime_loss = total_sub_regime_loss / total_samples
        avg_consistency_loss = total_consistency_loss / total_samples
        regime_accuracy = correct_regimes / total_samples
        sub_regime_accuracy = correct_sub_regimes / total_samples
        
        # Update learning rate based on loss
        self.scheduler.step_with_loss(avg_loss)
        
        # Return metrics
        return {
            'loss': avg_loss,
            'regime_loss': avg_regime_loss,
            'sub_regime_loss': avg_sub_regime_loss,
            'consistency_loss': avg_consistency_loss,
            'regime_accuracy': regime_accuracy,
            'sub_regime_accuracy': sub_regime_accuracy,
            'learning_rate': self.scheduler.get_lr()
        }
    
    def evaluate(self, test_loader, regime_weights=None, sub_regime_weights=None):
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data
            regime_weights: Optional weights for regime classes (dictionary or tensor)
            sub_regime_weights: Optional weights for sub-regime classes (dictionary or tensor)
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_regime_loss = 0
        total_sub_regime_loss = 0
        total_consistency_loss = 0
        correct_regimes = 0
        correct_sub_regimes = 0
        correct_hierarchical = 0
        total_samples = 0
        all_regime_preds = []
        all_sub_regime_preds = []
        all_regime_targets = []
        all_sub_regime_targets = []
        
        # Convert dictionary weights to tensor if needed
        if isinstance(regime_weights, dict):
            regime_classes = sorted(regime_weights.keys())
            regime_weight_tensor = torch.tensor([regime_weights[c] for c in regime_classes], 
                                              device=self.device, dtype=torch.float32)
        else:
            regime_weight_tensor = regime_weights
            
        if isinstance(sub_regime_weights, dict):
            sub_regime_classes = sorted(sub_regime_weights.keys())
            sub_regime_weight_tensor = torch.tensor([sub_regime_weights[c] for c in sub_regime_classes],
                                                  device=self.device, dtype=torch.float32)
        else:
            sub_regime_weight_tensor = sub_regime_weights
        
        # Evaluation loop
        with torch.no_grad():
            # Instead of iterating through the dataloader which might have unseen classes,
            # we'll process a single batch at a time with proper error handling
            for batch_idx, batch in enumerate(test_loader):
                try:
                    X, regime_targets, sub_regime_targets = self._prepare_batch(batch)
                    
                    # Skip this batch if targets are missing
                    if regime_targets is None or sub_regime_targets is None:
                        print(f"Skipping eval batch {batch_idx} due to missing targets")
                        continue
                    
                    # Forward pass
                    regime_logits, sub_regime_logits = self.model(X)
                    
                    # Compute loss
                    loss_dict = self.loss_fn(
                        regime_logits, 
                        sub_regime_logits, 
                        regime_targets, 
                        sub_regime_targets,
                        regime_weights=regime_weight_tensor,
                        sub_regime_weights=sub_regime_weight_tensor
                    )
                    
                    # Update metrics
                    batch_size = X.size(0)
                    total_loss += loss_dict['total_loss'].item() * batch_size
                    total_regime_loss += loss_dict['regime_loss'].item() * batch_size
                    total_sub_regime_loss += loss_dict['sub_regime_loss'].item() * batch_size
                    total_consistency_loss += loss_dict['consistency_loss'].item() * batch_size
                    total_samples += batch_size
                    
                    # Calculate accuracy
                    _, regime_preds = torch.max(regime_logits, 1)
                    _, sub_regime_preds = torch.max(sub_regime_logits, 1)
                    correct_regimes += (regime_preds == regime_targets).sum().item()
                    correct_sub_regimes += (sub_regime_preds == sub_regime_targets).sum().item()
                    
                    # Calculate hierarchical accuracy (both predictions correct)
                    correct_hierarchical += ((regime_preds == regime_targets) & 
                                            (sub_regime_preds == sub_regime_targets)).sum().item()
                    
                    # Store predictions and targets for detailed metrics
                    all_regime_preds.append(regime_preds.cpu())
                    all_sub_regime_preds.append(sub_regime_preds.cpu())
                    all_regime_targets.append(regime_targets.cpu())
                    all_sub_regime_targets.append(sub_regime_targets.cpu())
                except Exception as e:
                    print(f"Error processing eval batch {batch_idx}: {e}")
                    continue
        
        # Check if any samples were processed
        if total_samples == 0:
            print("Warning: No valid samples processed during evaluation!")
            return {
                'loss': float('inf'),
                'regime_loss': float('inf'),
                'sub_regime_loss': float('inf'),
                'consistency_loss': float('inf'),
                'regime_accuracy': 0.0,
                'sub_regime_accuracy': 0.0,
                'hierarchical_accuracy': 0.0,
                'regime_confusion_matrix': None,
                'sub_regime_confusion_matrix': None
            }
        
        # Convert list of tensors to single tensors if any predictions were made
        if all_regime_preds:
            all_regime_preds = torch.cat(all_regime_preds)
            all_sub_regime_preds = torch.cat(all_sub_regime_preds)
            all_regime_targets = torch.cat(all_regime_targets)
            all_sub_regime_targets = torch.cat(all_sub_regime_targets)
        else:
            all_regime_preds = torch.tensor([])
            all_sub_regime_preds = torch.tensor([])
            all_regime_targets = torch.tensor([])
            all_sub_regime_targets = torch.tensor([])
        
        # Calculate average metrics
        avg_loss = total_loss / total_samples
        avg_regime_loss = total_regime_loss / total_samples
        avg_sub_regime_loss = total_sub_regime_loss / total_samples
        avg_consistency_loss = total_consistency_loss / total_samples
        regime_accuracy = correct_regimes / total_samples
        sub_regime_accuracy = correct_sub_regimes / total_samples
        hierarchical_accuracy = correct_hierarchical / total_samples
        
        # If this is the best model so far, save state
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_model_state = self.model.state_dict().copy()
            print(f"New best model saved with val loss: {avg_loss:.4f}")
        
        # Compute confusion matrix and per-class metrics (if sklearn is imported) when there's data
        regime_cm = None
        sub_regime_cm = None
        if len(all_regime_preds) > 0:
            try:
                from sklearn.metrics import confusion_matrix, classification_report
                
                # Compute confusion matrices
                regime_cm = confusion_matrix(all_regime_targets, all_regime_preds)
                sub_regime_cm = confusion_matrix(all_sub_regime_targets, all_sub_regime_preds)
                
                # Print classification reports
                if self.debug:
                    print("\nRegime Classification Report:")
                    print(classification_report(all_regime_targets, all_regime_preds, zero_division=0))
                    print("\nSub-regime Classification Report:")
                    print(classification_report(all_sub_regime_targets, all_sub_regime_preds, zero_division=0))
            except ImportError:
                pass
            except Exception as e:
                print(f"Error computing classification metrics: {e}")
        
        # Return metrics
        return {
            'loss': avg_loss,
            'regime_loss': avg_regime_loss,
            'sub_regime_loss': avg_sub_regime_loss,
            'consistency_loss': avg_consistency_loss,
            'regime_accuracy': regime_accuracy,
            'sub_regime_accuracy': sub_regime_accuracy,
            'hierarchical_accuracy': hierarchical_accuracy,
            'regime_confusion_matrix': regime_cm,
            'sub_regime_confusion_matrix': sub_regime_cm
        }
    
    def train(self, 
             train_loader, 
             test_loader=None, 
             num_epochs=10, 
             regime_weights=None, 
             sub_regime_weights=None,
             early_stopping_patience=10):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data (optional)
            num_epochs: Number of epochs to train
            regime_weights: Optional weights for regime classes
            sub_regime_weights: Optional weights for sub-regime classes
            early_stopping_patience: Number of epochs to wait before stopping if no improvement
            
        Returns:
            Dictionary with training history
        """
        print(f"Starting training for {num_epochs} epochs")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_regime_acc': [],
            'val_regime_acc': [],
            'train_sub_regime_acc': [],
            'val_sub_regime_acc': [],
            'val_hierarchical_acc': [],
            'learning_rate': []
        }
        
        no_improvement_count = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, regime_weights, sub_regime_weights)
            
            # Update history for training metrics
            history['train_loss'].append(train_metrics['loss'])
            history['train_regime_acc'].append(train_metrics['regime_accuracy'])
            history['train_sub_regime_acc'].append(train_metrics['sub_regime_accuracy'])
            history['learning_rate'].append(train_metrics['learning_rate'])
            
            # Evaluate if test_loader is provided
            if test_loader is not None:
                try:
                    val_metrics = self.evaluate(test_loader, regime_weights, sub_regime_weights)
                    
                    # Update history for validation metrics
                    history['val_loss'].append(val_metrics['loss'])
                    history['val_regime_acc'].append(val_metrics['regime_accuracy'])
                    history['val_sub_regime_acc'].append(val_metrics['sub_regime_accuracy'])
                    history['val_hierarchical_acc'].append(val_metrics['hierarchical_accuracy'])
                    
                    # Print metrics with validation
                    print(f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Val Loss: {val_metrics['loss']:.4f}")
                    print(f"Train Regime Acc: {train_metrics['regime_accuracy']:.4f}, "
                          f"Val Regime Acc: {val_metrics['regime_accuracy']:.4f}")
                    print(f"Train Sub-regime Acc: {train_metrics['sub_regime_accuracy']:.4f}, "
                          f"Val Sub-regime Acc: {val_metrics['sub_regime_accuracy']:.4f}")
                    print(f"Val Hierarchical Acc: {val_metrics['hierarchical_accuracy']:.4f}, "
                          f"LR: {train_metrics['learning_rate']:.6f}")
                    
                    # Early stopping based on validation loss
                    if val_metrics['loss'] < getattr(self, 'best_val_loss', float('inf')):
                        self.best_val_loss = val_metrics['loss']
                        self.best_model_state = self.model.state_dict().copy()
                        no_improvement_count = 0
                        print(f"New best model saved with val loss: {self.best_val_loss:.4f}")
                    else:
                        no_improvement_count += 1
                except Exception as e:
                    # Handle validation errors gracefully
                    print(f"Warning: Validation error - {str(e)}")
                    print("Continuing training without validation...")
                    # Use training metrics for model selection
                    if train_metrics['loss'] < getattr(self, 'best_val_loss', float('inf')):
                        self.best_val_loss = train_metrics['loss']
                        self.best_model_state = self.model.state_dict().copy()
                        no_improvement_count = 0
            else:
                # Print metrics without validation
                print(f"Train Loss: {train_metrics['loss']:.4f}")
                print(f"Train Regime Acc: {train_metrics['regime_accuracy']:.4f}")
                print(f"Train Sub-regime Acc: {train_metrics['sub_regime_accuracy']:.4f}")
                print(f"LR: {train_metrics['learning_rate']:.6f}")
                
                # Use training metrics for model selection
                if train_metrics['loss'] < getattr(self, 'best_val_loss', float('inf')):
                    self.best_val_loss = train_metrics['loss']
                    self.best_model_state = self.model.state_dict().copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            
            # Early stopping check
            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Restore best model
        if hasattr(self, 'best_model_state') and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Restored best model from training")
        
        return history
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input features (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (regime_predictions, sub_regime_predictions, regime_probs, sub_regime_probs)
        """
        self.model.eval()
        
        # Handle different input formats
        if isinstance(X, list):
            X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        X = X.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            regime_logits, sub_regime_logits = self.model(X)
            
            # Convert to probabilities
            regime_probs = F.softmax(regime_logits, dim=1)
            sub_regime_probs = F.softmax(sub_regime_logits, dim=1)
            
            # Get predictions
            _, regime_preds = torch.max(regime_probs, 1)
            _, sub_regime_preds = torch.max(sub_regime_probs, 1)
        
        return (
            regime_preds.cpu().numpy(),
            sub_regime_preds.cpu().numpy(),
            regime_probs.cpu().numpy(),
            sub_regime_probs.cpu().numpy()
        )
    
    def save(self, path):
        """
        Save model, optimizer, scheduler state to file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_model_state': self.best_model_state,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load model, optimizer, scheduler state from file.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_model_state = checkpoint['best_model_state']
        print(f"Model loaded from {path}")

def count_model_parameters(model, verbose=True):
    """
    Count the number of parameters in a PyTorch model and calculate its complexity.
    
    Args:
        model: PyTorch model
        verbose: Whether to print parameter counts
        
    Returns:
        Dictionary with parameter statistics
    """
    # Get total parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters by layer type
    layer_counts = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            layer_type = module.__class__.__name__
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                if layer_type not in layer_counts:
                    layer_counts[layer_type] = {'count': 0, 'params': 0}
                layer_counts[layer_type]['count'] += 1
                layer_counts[layer_type]['params'] += params
    
    # Calculate complexity statistics
    param_stats = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'trainable_percentage': (trainable_params / total_params * 100) if total_params > 0 else 0,
        'layer_counts': layer_counts
    }
    
    if verbose:
        print(f"\nModel Parameter Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({param_stats['trainable_percentage']:.2f}%)")
        print(f"Non-trainable parameters: {param_stats['non_trainable_params']:,}")
        
        print("\nParameters by layer type:")
        for layer_type, stats in sorted(layer_counts.items(), key=lambda x: x[1]['params'], reverse=True):
            print(f"  {layer_type}: {stats['count']} layers with {stats['params']:,} parameters " +
                  f"({stats['params']/total_params*100:.2f}%)")
    
    return param_stats 