import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import json
from datetime import datetime
import math
from typing import Dict, Tuple, List, Optional, Union, Any
from scipy.stats import norm


class BayesianLinear(nn.Module):
    """Bayesian Linear Layer with Gaussian weight uncertainty."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 prior_sigma_1: float = 1.0, prior_sigma_2: float = 0.1, 
                 prior_pi: float = 0.5):
        super(BayesianLinear, self).__init__()
        
        # Layer dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # Weight priors
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize mean of weights
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        
        # Initialize rho of weights (controls variance)
        nn.init.constant_(self.weight_rho, -6.0)  # Start with small variance for stability
        
        if self.bias:
            nn.init.zeros_(self.bias_mu)
            nn.init.constant_(self.bias_rho, -6.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample weights during training
        weight = self.sample_weights()
        
        # Forward pass
        if self.bias:
            bias = self.bias_mu
            if self.training:
                bias = self.sample_bias()
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, weight)
    
    def sample_weights(self) -> torch.Tensor:
        """Sample weights using the reparameterization trick."""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_epsilon = torch.randn_like(weight_sigma)
        
        # Only sample during training
        if self.training:
            return self.weight_mu + weight_epsilon * weight_sigma
        else:
            return self.weight_mu
    
    def sample_bias(self) -> torch.Tensor:
        """Sample bias using the reparameterization trick."""
        if self.bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_epsilon = torch.randn_like(bias_sigma)
            return self.bias_mu + bias_epsilon * bias_sigma
        else:
            return None
    
    def kl_loss(self) -> torch.Tensor:
        """Calculate the KL divergence between the posterior and prior of weights."""
        # Calculate KL for weights
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_sigma2 = weight_sigma.pow(2)
        
        kl_loss = self._kl_divergence(
            self.weight_mu, weight_sigma2,
            torch.zeros_like(self.weight_mu), 
            self.prior_sigma_1**2, self.prior_sigma_2**2, self.prior_pi
        )
        
        # Add KL for bias if present
        if self.bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_sigma2 = bias_sigma.pow(2)
            
            kl_loss += self._kl_divergence(
                self.bias_mu, bias_sigma2,
                torch.zeros_like(self.bias_mu), 
                self.prior_sigma_1**2, self.prior_sigma_2**2, self.prior_pi
            )
        
        return kl_loss
    
    def _kl_divergence(self, mu_q, sigma_q, mu_p, sigma_p1, sigma_p2, pi):
        """Calculate the KL divergence between a Gaussian mixture posterior and prior."""
        # Avoid numerical instability with clipping
        sigma_q = torch.clamp(sigma_q, min=1e-8)
        
        # Convert scalar values to tensors if needed
        if not torch.is_tensor(sigma_p1):
            sigma_p1 = torch.tensor(sigma_p1, device=mu_q.device)
        if not torch.is_tensor(sigma_p2):
            sigma_p2 = torch.tensor(sigma_p2, device=mu_q.device)
        if not torch.is_tensor(pi):
            pi = torch.tensor(pi, device=mu_q.device)
        
        # Calculate KL for mixture prior
        term1 = torch.log(pi * torch.sqrt(sigma_p1) / torch.sqrt(sigma_q) + 
                         (1 - pi) * torch.sqrt(sigma_p2) / torch.sqrt(sigma_q))
        
        mu_diff1 = (mu_q - mu_p).pow(2)
        term2 = -0.5 + 0.5 * (pi * (sigma_q + mu_diff1) / sigma_p1 + 
                            (1 - pi) * (sigma_q + mu_diff1) / sigma_p2)
        
        kl = term1 + term2
        return kl.sum()


class DirectionPredictor:
    """
    Adaptive direction predictor for determining trend direction based on signals.
    Uses dynamic thresholding for more balanced predictions.
    """
    
    def __init__(self, threshold=0.2, adaptive=True, up_threshold=None, down_threshold=None):
        """
        Initialize the direction predictor.
        
        Args:
            threshold: Base threshold value for direction determination
            adaptive: Whether to use adaptive thresholding
            up_threshold: Explicit threshold for up direction (default: same as base)
            down_threshold: Explicit threshold for down direction (default: same as base)
        """
        self.base_threshold = threshold
        self.adaptive = adaptive
        # Allow for asymmetric thresholds to handle class imbalance
        self.up_threshold = up_threshold if up_threshold is not None else threshold
        self.down_threshold = down_threshold if down_threshold is not None else threshold
        self.history = {'up': [], 'down': [], 'neutral': []}
        self.calibration_window = 100  # Size of window for dynamic calibration
    
    def update_thresholds(self, true_directions, pred_means):
        """
        Update thresholds based on observed data distribution.
        
        Args:
            true_directions: Ground truth directions
            pred_means: Predicted mean values
        """
        if not self.adaptive or len(true_directions) == 0:
            return
            
        # Count directions
        up_count = (true_directions > 0).sum().item()
        down_count = (true_directions < 0).sum().item()
        neutral_count = (true_directions == 0).sum().item()
        total = len(true_directions)
        
        # Calculate class weights (inverse frequency)
        weights = {}
        weights['up'] = total / max(up_count, 1)
        weights['down'] = total / max(down_count, 1)
        weights['neutral'] = total / max(neutral_count, 1)
        
        # Normalize weights
        total_weight = weights['up'] + weights['down'] + weights['neutral']
        for k in weights:
            weights[k] /= total_weight
            
        # Calculate threshold adjustments based on class imbalance
        # If one class is under-represented, reduce its threshold
        base = self.base_threshold
        if weights['up'] > weights['down']:
            # Up class is underrepresented, make up predictions easier
            self.up_threshold = base * (1.0 - 0.5 * (weights['up'] - weights['down']))
            self.down_threshold = base * (1.0 + 0.3 * (weights['up'] - weights['down']))
        elif weights['down'] > weights['up']:
            # Down class is underrepresented, make down predictions easier
            self.up_threshold = base * (1.0 + 0.3 * (weights['down'] - weights['up']))
            self.down_threshold = base * (1.0 - 0.5 * (weights['down'] - weights['up']))
        else:
            # Balanced
            self.up_threshold = self.down_threshold = base
            
        # Ensure thresholds are reasonable
        self.up_threshold = max(0.05, min(0.5, self.up_threshold))
        self.down_threshold = max(0.05, min(0.5, self.down_threshold))
    
    def predict_direction(self, trend_mean, trend_var=None, calibrate=False, true_directions=None):
        """
        Predict trend direction using adaptive threshold approach.
        
        Args:
            trend_mean: Predicted trend strength tensor
            trend_var: Predicted trend variance tensor (optional)
            calibrate: Whether to update thresholds using this batch
            true_directions: True directions if calibrating
            
        Returns:
            Tuple of (direction, confidence)
            - direction: -1 (down), 0 (neutral), 1 (up)
            - confidence: Value between 0 and 1 indicating prediction confidence
        """
        device = trend_mean.device
        
        # Update thresholds if calibrating
        if calibrate and true_directions is not None:
            self.update_thresholds(true_directions, trend_mean)
        
        # Apply asymmetric thresholds to determine direction
        direction = torch.zeros_like(trend_mean)
        direction[trend_mean > self.up_threshold] = 1.0      # Up trend
        direction[trend_mean < -self.down_threshold] = -1.0  # Down trend
        
        # Calculate confidence based on signal strength and variance
        if trend_var is not None:
            # Higher confidence for stronger signals with lower variance
            # Use appropriate threshold for each direction
            confidence = torch.zeros_like(trend_mean)
            up_mask = (trend_mean > 0)
            down_mask = (trend_mean < 0)
            neutral_mask = ~(up_mask | down_mask)
            
            # Calculate confidence for each direction type
            if up_mask.any():
                confidence[up_mask] = torch.abs(trend_mean[up_mask]) / (self.up_threshold * (1.0 + torch.sqrt(trend_var[up_mask])))
            
            if down_mask.any():
                confidence[down_mask] = torch.abs(trend_mean[down_mask]) / (self.down_threshold * (1.0 + torch.sqrt(trend_var[down_mask])))
            
            if neutral_mask.any():
                # For neutral, confidence is inverse of distance to nearest threshold
                threshold_distance = torch.minimum(
                    torch.abs(self.up_threshold - trend_mean[neutral_mask]),
                    torch.abs(-self.down_threshold - trend_mean[neutral_mask])
                )
                confidence[neutral_mask] = 1.0 - threshold_distance / max(self.up_threshold, self.down_threshold)
        else:
            # Simpler confidence calculation without variance
            confidence = torch.zeros_like(trend_mean)
            up_mask = (trend_mean > 0)
            down_mask = (trend_mean < 0)
            
            if up_mask.any():
                confidence[up_mask] = torch.abs(trend_mean[up_mask]) / self.up_threshold
            
            if down_mask.any():
                confidence[down_mask] = torch.abs(trend_mean[down_mask]) / self.down_threshold
            
        confidence = torch.clamp(confidence, 0.0, 1.0)
        
        return direction, confidence


class FocalLoss(nn.Module):
    """Focal Loss for better handling of class imbalance in direction prediction."""
    
    def __init__(self, gamma=3.0, alpha=0.25, auto_weight=True):
        """
        Initialize focal loss.
        
        Args:
            gamma: Focusing parameter for hard examples (higher = more focus)
            alpha: Weighting factor for positive class
            auto_weight: Whether to automatically adjust class weights based on frequency
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.auto_weight = auto_weight
        self.class_weights = {'up': 1.0, 'neutral': 1.0, 'down': 1.0}
        self.eps = 1e-8  # Small value to avoid numerical issues
        
    def update_weights(self, true_direction):
        """
        Update weights based on class distribution.
        
        Args:
            true_direction: True direction labels (-1, 0, 1)
        """
        if not self.auto_weight:
            return
            
        # Count each class
        up_count = (true_direction > 0).sum().item()
        down_count = (true_direction < 0).sum().item()
        neutral_count = (true_direction == 0).sum().item()
        total = len(true_direction)
        
        # Handle edge cases
        if total == 0:
            return
            
        if up_count == 0:
            up_count = 1
        if down_count == 0:
            down_count = 1
        if neutral_count == 0:
            neutral_count = 1
            
        # Calculate inverse frequency weights
        self.class_weights['up'] = total / up_count
        self.class_weights['down'] = total / down_count
        self.class_weights['neutral'] = total / neutral_count
        
        # Normalize weights to sum to class count (3)
        weight_sum = self.class_weights['up'] + self.class_weights['down'] + self.class_weights['neutral']
        self.class_weights['up'] = 3 * self.class_weights['up'] / weight_sum
        self.class_weights['down'] = 3 * self.class_weights['down'] / weight_sum
        self.class_weights['neutral'] = 3 * self.class_weights['neutral'] / weight_sum
    
    def forward(self, pred_means, true_direction, sample_weights=None):
        """
        Calculate focal loss for direction prediction.
        
        Args:
            pred_means: Predicted trend means
            true_direction: True direction labels (-1, 0, 1)
            sample_weights: Optional tensor of weights per sample
            
        Returns:
            Focal loss
        """
        # Update weights if using auto-weighting
        if self.auto_weight:
            self.update_weights(true_direction)
            
        # Ensure inputs have correct dimensions
        if pred_means.dim() > 1 and pred_means.size(1) == 1:
            pred_means = pred_means.squeeze(1)
            
        if true_direction.dim() > 1 and true_direction.size(1) == 1:
            true_direction = true_direction.squeeze(1)
            
        if sample_weights is not None and sample_weights.dim() > 1 and sample_weights.size(1) == 1:
            sample_weights = sample_weights.squeeze(1)
            
        # Convert direction to class indices (0, 1, 2)
        target = (true_direction.long() + 1)  # Map [-1, 0, 1] to [0, 1, 2]
        
        # Convert predictions to directional probabilities
        # Using sigmoid to map trend strength to [0, 1]
        pos_probs = torch.sigmoid(pred_means)  # Probability of uptrend
        neg_probs = torch.sigmoid(-pred_means)  # Probability of downtrend
        neutral_probs = 1.0 - pos_probs - neg_probs + self.eps  # Probability of neutral trend
        neutral_probs = torch.clamp(neutral_probs, min=0.0, max=1.0)  # Ensure valid probabilities
        
        # Stack probabilities into one tensor [batch_size, 3]
        probs = torch.stack([neg_probs, neutral_probs, pos_probs], dim=1)
        
        # One-hot encode targets
        target_one_hot = torch.zeros_like(probs)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Apply class weights
        weight_tensor = torch.ones_like(probs)
        weight_tensor[:, 0] = self.class_weights['down']  # Down class (0)
        weight_tensor[:, 1] = self.class_weights['neutral']  # Neutral class (1)
        weight_tensor[:, 2] = self.class_weights['up']  # Up class (2)
        
        # Apply focal weights (focus more on hard examples)
        focal_weights = (1 - probs).pow(self.gamma) * target_one_hot + probs.pow(self.gamma) * (1 - target_one_hot)
        
        # Combine all weighting factors
        weights = weight_tensor * focal_weights
        
        # Apply sample weights if provided
        if sample_weights is not None:
            weights = weights * sample_weights.unsqueeze(1)
        
        # Calculate loss (cross-entropy with weights)
        loss = -weights * (target_one_hot * torch.log(probs + self.eps) + 
                           (1 - target_one_hot) * torch.log(1 - probs + self.eps))
        
        # Sum across classes and average across batch
        return loss.sum(1).mean()


class LayerNorm(nn.Module):
    """
    Layer Normalization with optional conditional scaling and shifting.
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class ResidualBlock(nn.Module):
    """
    Residual block with layer normalization and dropout.
    """
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(dim)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.gelu(self.linear1(x))
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = residual + x
        return self.norm2(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with layer normalization and residual connection.
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(dim)
        
    def forward(self, x, return_attention=False):
        # Input shape: [batch_size, seq_len, dim]
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Apply normalization before attention
        x = self.norm(x)
        
        # Project queries, keys, values
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape output
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        # Apply residual connection
        output = residual + self.dropout(output)
        
        if return_attention:
            return output, attention
        return output


class TradingOpportunityModel(nn.Module):
    """
    Enhanced model for predicting trading opportunities with residual connections,
    improved normalization, and better handling of class imbalance.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128,
        regime_dim: int = 16,
        subregime_dim: int = 8,
        time_dim: int = 16,
        n_regimes: int = 10,
        n_subregimes: int = 10,
        n_heads: int = 4,
        dropout: float = 0.1,
        batch_norm: bool = True,
        residual: bool = True,
        device: torch.device = None
    ):
        super(TradingOpportunityModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.batch_norm = batch_norm
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize embeddings for regime and subregime
        self.regime_embedding = nn.Embedding(n_regimes, regime_dim)
        self.subregime_embedding = nn.Embedding(n_subregimes, subregime_dim)
        
        # Time projection layer for continuous time features
        self.time_projection = nn.Linear(2, time_dim)  # Assuming 2 time features (position, hour)
        
        # Feature encoder with residual connections and normalization
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout)
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            dim=hidden_dim + regime_dim + subregime_dim + time_dim,
            num_heads=n_heads,
            dropout=dropout
        )
        
        # Batch normalization for feature processing
        if batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(hidden_dim + regime_dim + subregime_dim + time_dim)
        
        # Trend prediction layers (Bayesian)
        self.trend_predictor = nn.Sequential(
            LayerNorm(hidden_dim + regime_dim + subregime_dim + time_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + regime_dim + subregime_dim + time_dim, hidden_dim),
            nn.GELU(),
            LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.trend_mean = BayesianLinear(hidden_dim, 1)
        self.trend_var = BayesianLinear(hidden_dim, 1)
        
        # Direction predictor
        self.direction_predictor = DirectionPredictor(threshold=0.1)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Focal loss for handling class imbalance
        self.focal_loss = FocalLoss(gamma=3.0, alpha=0.25)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move to device
        self.to(self.device)
    
    def _init_weights(self, module):
        """Initialize weights for better training stability"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, features, regimes=None, subregimes=None, time_features=None, return_attention=False):
        """
        Forward pass through the model.
        
        Args:
            features: Input features [batch_size, seq_len, feature_dim]
            regimes: Optional regime indices [batch_size, seq_len]
            subregimes: Optional subregime indices [batch_size, seq_len]
            time_features: Optional time features [batch_size, seq_len, time_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (trend_mean, trend_var, kl_loss, [projected_features], [attention_weights])
        """
        # Process input features through feature encoder
        x = self.feature_encoder(features)
        batch_size, seq_len, _ = x.shape
        
        # Process regime information if available
        if regimes is not None and hasattr(self, 'regime_embedding'):
            # Ensure regime indices are valid (clamp to valid range)
            valid_regimes = torch.clamp(regimes, 0, self.regime_embedding.num_embeddings - 1)
            regime_emb = self.regime_embedding(valid_regimes)
            x = torch.cat([x, regime_emb], dim=-1)
            
        # Process subregime information if available
        if subregimes is not None and hasattr(self, 'subregime_embedding'):
            # Ensure subregime indices are valid (clamp to valid range)
            valid_subregimes = torch.clamp(subregimes, 0, self.subregime_embedding.num_embeddings - 1)
            subregime_emb = self.subregime_embedding(valid_subregimes)
            x = torch.cat([x, subregime_emb], dim=-1)
            
        # Process time features if available
        if time_features is not None:
            # Handle time features directly (no embedding needed for continuous values)
            if hasattr(self, 'time_projection'):
                # Use a projection layer for time features
                time_proj = self.time_projection(time_features)
                x = torch.cat([x, time_proj], dim=-1)
            else:
                # Just concatenate the time features directly
                x = torch.cat([x, time_features], dim=-1)
            
        # Apply attention
        if return_attention:
            x, attention_weights = self.attention(x, return_attention=True)
        else:
            x = self.attention(x, return_attention=False)
            attention_weights = None
        
        # Apply batch normalization if enabled
        if hasattr(self, 'batch_norm_layer') and self.batch_norm_layer is not None:
            # Reshape for batch norm
            orig_shape = x.shape
            x = x.reshape(-1, x.size(-1))
            x = self.batch_norm_layer(x)
            x = x.reshape(orig_shape)
            
        # Global averaging to get sequence embedding
        x = torch.mean(x, dim=1)
        
        # Apply trend predictor
        x = self.trend_predictor(x)
        
        # Store encoded features for later use
        self.encoded_features = x
        
        # Project features for contrastive learning
        projected_features = self.projection_head(x)
        
        # Apply Bayesian prediction layers
        trend_mean = self.trend_mean(x)
        trend_logvar = self.trend_var(x)
        trend_var = torch.exp(trend_logvar)
        
        # Return KL loss of 0 since we're not using Bayesian layers anymore
        kl_loss = torch.tensor(0.0, device=self.device)
        
        # Store attention weights if needed
        if return_attention:
            self.attention_weights = attention_weights
            return trend_mean, trend_var, kl_loss, projected_features, attention_weights
        
        return trend_mean, trend_var, kl_loss, projected_features
    
    def predict(self, features, regimes=None, subregimes=None, time_features=None, return_details=False, direction_threshold=None, calibrate=False, true_direction=None):
        """
        Make predictions with the model.
        
        Args:
            features: Input features of shape [batch_size, seq_len, feature_dim]
            regimes: Optional regime information of shape [batch_size, seq_len]
            subregimes: Optional sub-regime information of shape [batch_size, seq_len]
            time_features: Optional time features of shape [batch_size, seq_len, time_dim]
            return_details: Whether to return additional details beyond predictions
            direction_threshold: Optional custom threshold for direction prediction
            calibrate: Whether to calibrate the direction predictor
            true_direction: True directions for calibration (if calibrating)
            
        Returns:
            Dictionary with predictions
        """
        self.eval()
        batch_size = features.size(0)
        
        with torch.no_grad():
            # Get trend predictions
            trend_mean, trend_var, _, projected_features = self.forward(features, regimes, subregimes, time_features)
            
            # Create or retrieve direction predictor (store it as an attribute if needed)
            if not hasattr(self, 'direction_predictor') or self.direction_predictor is None:
                base_threshold = direction_threshold or 0.2
                self.direction_predictor = DirectionPredictor(threshold=base_threshold, adaptive=True)
            elif direction_threshold is not None:
                # Update threshold if a new one is provided
                self.direction_predictor.base_threshold = direction_threshold
                if not self.direction_predictor.adaptive:
                    self.direction_predictor.up_threshold = direction_threshold
                    self.direction_predictor.down_threshold = direction_threshold
            
            # Calculate trend direction using DirectionPredictor
            direction, confidence = self.direction_predictor.predict_direction(
                trend_mean, 
                trend_var,
                calibrate=calibrate, 
                true_directions=true_direction
            )
            
            # Convert to numpy arrays for easier handling
            trend_mean_np = trend_mean.cpu().numpy()
            trend_std_np = torch.sqrt(trend_var).cpu().numpy()
            direction_np = direction.cpu().numpy()
            confidence_np = confidence.cpu().numpy()
            
            # Create prediction dictionary
            predictions = {
                'trend_mean': trend_mean_np,
                'trend_std': trend_std_np,
                'direction': direction_np,
                'confidence': confidence_np,
            }
            
            # Add threshold information
            predictions['up_threshold'] = self.direction_predictor.up_threshold
            predictions['down_threshold'] = self.direction_predictor.down_threshold
            
            if return_details:
                # Add additional details like encoded features
                predictions['encoded_features'] = self.encoded_features.cpu().numpy()
                
                # Add attention weights if available
                if hasattr(self, 'attention_weights') and self.attention_weights is not None:
                    predictions['attention_weights'] = self.attention_weights.cpu().numpy()
            
            return predictions
    
    def kl_loss(self) -> torch.Tensor:
        """
        Calculate the total KL divergence for all Bayesian layers.
            
        Returns:
            Total KL divergence
        """
        total_kl = 0.0
        
        # Iterate through modules to find Bayesian layers
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                total_kl += module.kl_loss()
        
        # Convert to tensor if it's a scalar
        if not isinstance(total_kl, torch.Tensor):
            total_kl = torch.tensor(total_kl, device=self.device)
        
        return total_kl
    
    def nt_xent_loss(self, embeddings, labels=None, temperature=0.5):
        """
        Compute the NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.
        
        Args:
            embeddings: Feature embeddings [batch_size, embedding_dim]
            labels: Optional labels for supervised contrastive learning [batch_size]
            temperature: Temperature parameter for scaling (default: 0.5)
            
        Returns:
            NT-Xent loss value
        """
        # Normalize embeddings for cosine similarity (dot product of normalized vectors)
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.transpose(0, 1)) / temperature
        
        # Mask out self-similarity
        batch_size = similarity_matrix.size(0)
        mask = torch.eye(batch_size, device=self.device)
        
        # Convert similarities to a masked version
        # Fill diagonal with large negative value to exclude self-pairs
        similarity_matrix = similarity_matrix * (1 - mask) - 10.0 * mask
        
        # If labels are provided, use them to define positive pairs (supervised)
        if labels is not None and not torch.all(labels == 0):
            # For each sample, positive pairs have the same label
            # Expand labels for comparison
            labels_expand = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())
            
            # Remove self-pairs
            labels_expand = labels_expand & (1 - mask).bool()
            
            # Compute loss for each positive pair
            pos_similarities = torch.exp(similarity_matrix) * labels_expand.float()
            neg_similarities = torch.exp(similarity_matrix) * (1 - labels_expand.float() - mask) 
            
            # Avoid division by zero
            pos_sum = torch.sum(pos_similarities, dim=1, keepdim=True)
            neg_sum = torch.sum(neg_similarities, dim=1, keepdim=True)
            
            # Only compute loss for samples with positive pairs
            valid_mask = (pos_sum > 0).float().squeeze()
            
            # Compute loss only for valid samples
            if valid_mask.sum() > 0:
                valid_pos_sum = torch.clamp(pos_sum, min=1e-6)
                valid_neg_sum = torch.clamp(neg_sum + pos_sum, min=1e-6)  # Include positives in denominator
                loss_per_sample = -torch.log(valid_pos_sum / valid_neg_sum)
                loss = (loss_per_sample.squeeze() * valid_mask).sum() / valid_mask.sum()
            else:
                # If no valid supervised pairs found, fall back to unsupervised
                loss = self._unsupervised_nt_xent(similarity_matrix)
        else:
            # Use unsupervised contrastive loss (all non-self pairs are negatives)
            loss = self._unsupervised_nt_xent(similarity_matrix)
            
        return loss
    
    def _unsupervised_nt_xent(self, similarity_matrix):
        """
        Compute unsupervised NT-Xent loss where all non-self pairs are negatives.
        
        Args:
            similarity_matrix: Similarity matrix with self-similarity masked out
            
        Returns:
            Unsupervised NT-Xent loss
        """
        batch_size = similarity_matrix.size(0)
        
        # For each row, compute softmax over all other samples
        exp_sim = torch.exp(similarity_matrix)
        
        # Sum over all samples to get denominator
        exp_sim_sum = torch.sum(exp_sim, dim=1, keepdim=True)
        
        # Compute loss: -log(exp(z_i · z_j) / sum_k exp(z_i · z_k))
        # Since all pairs are negative in unsupervised case, we minimize all similarities
        loss = -torch.log(exp_sim_sum.clamp(min=1e-6))
        
        # Average over all samples
        return loss.mean()
    
    def directional_contrastive_loss(self, embeddings, trend_mean, margin=0.5):
        """
        Compute a directional contrastive loss that pulls similar trade signals together.
        
        Args:
            embeddings: Feature embeddings [batch_size, embedding_dim]
            trend_mean: Predicted trend means [batch_size, 1]
            margin: Margin for the contrastive loss (default: 0.5)
            
        Returns:
            Directional contrastive loss value
        """
        # Create direction labels: 1 for long, -1 for short, 0 for neutral
        batch_size = trend_mean.size(0)
        directions = torch.zeros(batch_size, device=self.device)
        
        # Threshold for long and short decisions
        threshold = 0.2
        
        # Assign directions based on trend mean
        directions[trend_mean.squeeze() > threshold] = 1.0     # Long
        directions[trend_mean.squeeze() < -threshold] = -1.0   # Short
        
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances (1 - cosine similarity)
        distances = 1.0 - torch.matmul(embeddings_norm, embeddings_norm.transpose(0, 1))
        
        # Create mask for pairs with same direction (exclude neutral-neutral)
        direction_matrix = directions.unsqueeze(1) * directions.unsqueeze(0)
        
        # Create masks for different pair types
        positive_mask = ((direction_matrix > 0) & (directions.unsqueeze(1) != 0) & 
                          (directions.unsqueeze(0) != 0)).float()
        negative_mask = ((direction_matrix < 0) & (directions.unsqueeze(1) != 0) & 
                          (directions.unsqueeze(0) != 0)).float()
        
        # Remove self-pairs
        self_mask = torch.eye(batch_size, device=self.device)
        positive_mask = positive_mask * (1 - self_mask)
        
        # Count valid pairs
        num_positives = positive_mask.sum()
        num_negatives = negative_mask.sum()
        
        loss = 0.0
        
        # Compute contrastive loss only if we have both positive and negative pairs
        if num_positives > 0 and num_negatives > 0:
            # Positive loss: minimize distance between same-direction pairs
            positive_loss = (distances * positive_mask).sum() / num_positives
            
            # Negative loss: encourage distance between opposite-direction pairs
            # Using margin: max(0, margin - distance)
            hinge_component = (torch.clamp(margin - distances, min=0.0) * negative_mask).sum() / num_negatives
            
            loss = positive_loss + hinge_component
            
        elif num_positives > 0:
            # Only positive pairs
            loss = (distances * positive_mask).sum() / num_positives
            
        # Return zero loss if no valid pairs
        return loss
    
    def loss_function(
            self, 
            trend_mean: torch.Tensor, 
            trend_var: torch.Tensor, 
            projected_features: torch.Tensor,
            trend_true: torch.Tensor,
            kl_weight: float = 0.01,
            contrastive_weight: float = 0.5,
            directional_weight: float = 0.3,
            focal_weight: float = 1.0,
            sample_weights: Optional[torch.Tensor] = None
        ) -> Dict[str, torch.Tensor]:
        """
        Calculate the combined loss function with contrastive components and focal loss.
        
        Args:
            trend_mean: Mean of trend strength prediction [batch_size, 1]
            trend_var: Variance of trend strength prediction [batch_size, 1]
            projected_features: Features from projection head [batch_size, projection_dim]
            trend_true: True trend strength values [batch_size, 1]
            kl_weight: Weight for KL divergence term
            contrastive_weight: Weight for contrastive loss component
            directional_weight: Weight for directional contrastive loss
            focal_weight: Weight for focal loss component
            sample_weights: Optional tensor of weights per sample for loss weighting
            
        Returns:
            Dictionary containing:
                - loss: Total loss
                - trend_loss: Negative log likelihood for trend
                - contrastive_loss: NT-Xent contrastive loss
                - directional_loss: Directional contrastive loss
                - focal_loss: Focal loss for direction prediction
                - kl_div: KL divergence
        """
        # Trend strength loss (negative log likelihood of Gaussian)
        # Avoid numerical issues
        trend_var = torch.clamp(trend_var, min=1e-6)
        trend_diff = trend_true - trend_mean
        trend_loss_per_sample = 0.5 * torch.log(2 * math.pi * trend_var) + 0.5 * (trend_diff.pow(2) / trend_var)
        
        # Apply sample weights if provided
        if sample_weights is not None:
            trend_loss_per_sample = trend_loss_per_sample * sample_weights
        
        trend_loss = trend_loss_per_sample.mean()
        
        # Create direction labels for supervised contrastive learning
        direction_labels = torch.sign(trend_true).long() + 1  # Map [-1, 0, 1] to [0, 1, 2]
        
        # Compute contrastive loss (NT-Xent)
        contrastive_loss = self.nt_xent_loss(projected_features, labels=direction_labels)
        
        # Compute directional contrastive loss
        directional_loss = self.directional_contrastive_loss(projected_features, trend_mean)
        
        # Compute focal loss for direction prediction with sample weights
        if hasattr(self, 'focal_loss') and self.focal_loss is not None:
            focal_loss = self.focal_loss(trend_mean, torch.sign(trend_true), sample_weights)
        else:
            # Simple direction prediction loss if focal loss not available
            direction_pred = torch.sign(trend_mean)
            direction_true = torch.sign(trend_true)
            direction_loss = F.mse_loss(direction_pred.float(), direction_true.float())
            focal_loss = direction_loss
        
        # KL divergence
        kl_div = self.kl_loss()
        
        # Combined loss
        total_loss = (
            trend_loss +  # Base loss always included
            contrastive_weight * contrastive_loss +
            directional_weight * directional_loss +
            focal_weight * focal_loss +
            kl_weight * kl_div
        )
        
        # Return all loss components
        return {
            "loss": total_loss,
            "trend_loss": trend_loss,
            "contrastive_loss": contrastive_loss,
            "directional_loss": directional_loss,
            "focal_loss": focal_loss,
            "kl_div": kl_div,
            "total_loss": total_loss
        }
    
    def save_model(self, filepath: str, metadata: Dict = None):
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model to
            metadata: Additional metadata to save with the model
        """
        # Create model parameters dictionary
        model_params = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'regime_dim': self.regime_embedding.weight.size(1) if hasattr(self, 'regime_embedding') else 16,
            'subregime_dim': self.subregime_embedding.weight.size(1) if hasattr(self, 'subregime_embedding') else 8,
            'time_dim': self.time_projection.weight.size(0) if hasattr(self, 'time_projection') else 16,
            'n_regimes': self.regime_embedding.num_embeddings if hasattr(self, 'regime_embedding') else 10,
            'n_subregimes': self.subregime_embedding.num_embeddings if hasattr(self, 'subregime_embedding') else 10,
            'n_heads': self.attention.num_heads if hasattr(self, 'attention') else 4,
            'dropout': 0.1,  # Fixed value since we can't extract from nn.Dropout
        }
        
        # Save model state and parameters
        save_data = {
            'model_params': model_params,
            'state_dict': self.state_dict(),
            'metadata': metadata or {}
        }
        
        # Add mappings if available
        if hasattr(self, 'regime_mapping') and self.regime_mapping is not None:
            save_data['metadata']['regime_mapping'] = self.regime_mapping
            
        if hasattr(self, 'subregime_mapping') and self.subregime_mapping is not None:
            save_data['metadata']['subregime_mapping'] = self.subregime_mapping
            
        # Save data to file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(save_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device=None) -> Tuple['TradingOpportunityModel', Dict]:
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            device: Device to load model to
            
        Returns:
            Tuple of (model, metadata)
        """
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Load saved data
        save_data = torch.load(filepath, map_location=device)
        
        # Extract model parameters and metadata
        model_params = save_data.get('model_params', {})
        metadata = save_data.get('metadata', {})
        state_dict = save_data.get('state_dict', {})
        
        # Create a new model instance
        try:
            # Try to create model with saved parameters
            model = cls(
                input_dim=model_params.get('input_dim', 32),
                hidden_dim=model_params.get('hidden_dim', 128),
                regime_dim=model_params.get('regime_dim', 16),
                subregime_dim=model_params.get('subregime_dim', 8),
                time_dim=model_params.get('time_dim', 16),
                n_regimes=model_params.get('n_regimes', 10),
                n_subregimes=model_params.get('n_subregimes', 10),
                n_heads=model_params.get('n_heads', 4),
                dropout=model_params.get('dropout', 0.1),
                device=device
            )
        except (TypeError, ValueError) as e:
            # If creating the model fails, try with default parameters
            print(f"Warning: Error loading model with saved parameters: {e}")
            print("Creating model with default parameters instead")
            
            model = cls(
                input_dim=32,
                hidden_dim=128,
                regime_dim=16,
                subregime_dim=8,
                time_dim=16,
                n_regimes=10,
                n_subregimes=10,
                n_heads=4,
                dropout=0.1,
                device=device
            )
        
        # Load state dict
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")
            print("Model will use initialized weights")
        
        return model, metadata
    
    def generate_adversarial_examples(self, features, regimes=None, subregimes=None, time_features=None, 
                                      trend_true=None, epsilon=0.01):
        """
        Generate adversarial examples using Fast Gradient Sign Method (FGSM)
        
        Args:
            features: Input features tensor
            regimes: Regime labels (optional)
            subregimes: Subregime labels (optional)
            time_features: Time features (optional)
            trend_true: True trend labels for calculating loss
            epsilon: Perturbation size for adversarial examples
            
        Returns:
            Adversarial examples
        """
        # Create a copy of features with gradients enabled
        features_adv = features.detach().clone()
        features_adv.requires_grad = True
        
        # Forward pass with adversarial features
        outputs = self(features_adv, regimes, subregimes, time_features)
        trend_mean, trend_var = outputs[0], outputs[1]
        kl_loss = outputs[2]
        projected_features = outputs[3] if len(outputs) > 3 else None
        
        if trend_true is None:
            # If no true labels provided, create dummy targets to maximize loss
            # This pushes the model away from its current prediction
            trend_true = -torch.sign(trend_mean.detach())
        
        # Calculate loss
        loss_dict = self.loss_function(
            trend_mean=trend_mean,
            trend_var=trend_var,
            projected_features=projected_features,
            trend_true=trend_true,
            kl_weight=0.01,
            contrastive_weight=0.5,
            directional_weight=0.3,
            focal_weight=1.0
        )
        
        # Extract the loss
        loss = loss_dict["loss"]
        
        # Compute gradients
        loss.backward()
        
        # Generate adversarial examples using FGSM
        # Use sign of gradients to perturb in direction that increases loss
        grad_sign = features_adv.grad.sign()
        features_adv = features_adv.detach() + epsilon * grad_sign
        
        # Ensure adversarial examples are within valid range of original features
        features_adv = torch.clamp(features_adv, features.min(), features.max())
        
        return features_adv.detach()