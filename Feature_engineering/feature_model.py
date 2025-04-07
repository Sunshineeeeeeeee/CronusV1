import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler


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


class VolatilityRegimeFeatureExtractor(nn.Module):
    """
    Custom transformer-based model for extracting features from financial time series data
    that can be used to identify distinct volatility regimes.
    
    Uses an encoder-only transformer architecture with custom loss functions designed
    to capture volatility regime characteristics.
    """
    
    def __init__(
        self,
        input_size: int = 3,  # Value, Volume, Volatility
        context_length: int = 50,
        d_model: int = 64,
        num_encoder_layers: int = 3,
        num_attention_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
    ):
        """
        Initialize the volatility regime feature extractor.
        
        Parameters:
        -----------
        input_size : int
            Number of input features (Value, Volume, Volatility)
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
        device : str, optional
            Device to use for computation
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
        self.num_attention_heads = num_attention_heads
        
        # Input projection: map input features to model dimension
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Time feature projection: map time features to model dimension
        self.time_projection = nn.Linear(1, d_model)  # Assuming 1D time feature
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
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
        
        # Feature projection for encoder outputs
        self.feature_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(
        self, 
        values: torch.Tensor,
        time_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model to get the latent representations.
        
        Parameters:
        -----------
        values : torch.Tensor
            Input values of shape (batch_size, seq_len, input_size)
        time_features : torch.Tensor
            Time features of shape (batch_size, seq_len, time_features_size)
        attention_mask : torch.Tensor, optional
            Not used in current implementation as all sequences are valid
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Dictionary containing latent representations and attention weights
        """
        batch_size, seq_len, _ = values.shape
        
        # Project input values to model dimension
        values_projected = self.input_projection(values)  # (batch, seq, d_model)
        
        # Project time features to model dimension
        time_projected = self.time_projection(time_features)  # (batch, seq, d_model)
        
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
        
        # Pass through transformer encoder without any masks
        latent_representations = self.transformer_encoder(encoded_features)
        
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
        
        # Store outputs for loss calculation and feature extraction
        output = {
            "latent_representations": latent_representations,
            "projected_features": projected_features,
            "attention_weights": attention_weights,
        }
        
        if mean_attention_entropy is not None:
            output["attention_entropy"] = mean_attention_entropy
        
        return output
    
    def compute_reconstruction_loss(self, model_output: Dict[str, torch.Tensor], values: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss: the model should be able to reconstruct the input
        values from the latent representations.
        
        Parameters:
        -----------
        model_output : Dict[str, torch.Tensor]
            Dictionary containing model outputs
        values : torch.Tensor
            Input values of shape (batch_size, seq_len, input_size)
            
        Returns:
        --------
        torch.Tensor
            Reconstruction loss
        """
        # Get latent representations
        latent_reps = model_output["latent_representations"]
        
        # Simple linear projection to reconstruct original values
        decoder = nn.Linear(self.d_model, self.input_size).to(self.device)
        reconstructed = decoder(latent_reps)
        
        # Compute MSE loss between original and reconstructed values
        # Focus on the last 25% of the sequence which is most relevant for regime
        seq_len = values.shape[1]
        focus_start = int(seq_len * 0.75)
        
        orig_values = values[:, focus_start:, :]
        recon_values = reconstructed[:, focus_start:, :]
        
        mse_loss = F.mse_loss(recon_values, orig_values)
        
        return mse_loss
    
    def compute_temporal_consistency_loss(self, model_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute temporal consistency loss: similar temporal patterns should have
        similar representations.
        
        Parameters:
        -----------
        model_output : Dict[str, torch.Tensor]
            Dictionary containing model outputs
            
        Returns:
        --------
        torch.Tensor
            Temporal consistency loss
        """
        # Get latent representations
        latent_reps = model_output["projected_features"]  # (batch, seq_len, d_model)
        
        # Compute temporal consistency: adjacent time steps should be similar
        # We compare each time step with the next one
        current_reps = latent_reps[:, :-1, :]  # (batch, seq_len-1, d_model)
        next_reps = latent_reps[:, 1:, :]      # (batch, seq_len-1, d_model)
        
        # Compute cosine similarity for consecutive time steps
        # Normalize the representations
        current_norms = torch.norm(current_reps, dim=2, keepdim=True)
        next_norms = torch.norm(next_reps, dim=2, keepdim=True)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        current_normalized = current_reps / (current_norms + epsilon)
        next_normalized = next_reps / (next_norms + epsilon)
        
        # Compute cosine similarity
        cosine_sim = torch.sum(current_normalized * next_normalized, dim=2)  # (batch, seq_len-1)
        
        # Temporal consistency loss: higher similarity (closer to 1) = lower loss
        consistency_loss = 1.0 - cosine_sim.mean()
        
        return consistency_loss
    
    def compute_volatility_sensitivity_loss(
        self, 
        model_output: Dict[str, torch.Tensor], 
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute volatility sensitivity loss: the model should be more sensitive
        to changes in volatility than other features.
        
        Parameters:
        -----------
        model_output : Dict[str, torch.Tensor]
            Dictionary containing model outputs
        values : torch.Tensor
            Input values of shape (batch_size, seq_len, input_size)
            
        Returns:
        --------
        torch.Tensor
            Volatility sensitivity loss
        """
        # Get latent representations
        latent_reps = model_output["latent_representations"]  # (batch, seq_len, d_model)
        
        # Assuming the last feature is volatility
        volatility = values[:, :, -1]  # (batch, seq_len)
        
        # Compute volatility changes
        volatility_changes = torch.abs(volatility[:, 1:] - volatility[:, :-1])  # (batch, seq_len-1)
        
        # Compute representation changes
        rep_changes = torch.norm(
            latent_reps[:, 1:, :] - latent_reps[:, :-1, :], 
            dim=2
        )  # (batch, seq_len-1)
        
        # Normalize both changes to [0, 1] range
        vol_changes_norm = volatility_changes / (volatility_changes.max() + 1e-8)
        rep_changes_norm = rep_changes / (rep_changes.max() + 1e-8)
        
        # Correlation loss: representation changes should correlate with volatility changes
        # We want to maximize correlation, so we minimize negative correlation
        corr_loss = F.mse_loss(rep_changes_norm, vol_changes_norm)
        
        return corr_loss
    
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
        alpha: float = 0.1,
        beta: float = 0.1,
        gamma: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for the model.
        
        Parameters:
        -----------
        model_output : Dict[str, torch.Tensor]
            Dictionary containing model outputs
        values : torch.Tensor, optional
            Input values tensor for reconstruction and volatility sensitivity loss
        alpha : float
            Weight for temporal consistency loss
        beta : float
            Weight for attention diversity loss
        gamma : float
            Weight for volatility sensitivity loss
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Dictionary containing individual and total losses
        """
        # Compute individual losses
        recon_loss = torch.tensor(0.0, device=self.device)
        vol_loss = torch.tensor(0.0, device=self.device)
        
        if values is not None:
            recon_loss = self.compute_reconstruction_loss(model_output, values)
            vol_loss = self.compute_volatility_sensitivity_loss(model_output, values)
            
        temp_loss = self.compute_temporal_consistency_loss(model_output)
        div_loss = self.compute_attention_diversity_loss(model_output)
        
        # Compute total loss
        total_loss = recon_loss + alpha * temp_loss + beta * div_loss + gamma * vol_loss
        
        # Return loss dictionary
        loss_dict = {
            "reconstruction_loss": recon_loss,
            "temporal_consistency_loss": temp_loss,
            "attention_diversity_loss": div_loss,
            "volatility_sensitivity_loss": vol_loss,
            "total_loss": total_loss
        }
        
        return loss_dict
    
    def train_step(
        self,
        values: torch.Tensor,
        time_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        alpha: float = 0.1,
        beta: float = 0.1,
        gamma: float = 0.5
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
            Weight for temporal consistency loss
        beta : float
            Weight for attention diversity loss
        gamma : float
            Weight for volatility sensitivity loss
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing loss values
        """
        self.optimizer.zero_grad()
        
        model_output = self.forward(
            values=values,
            time_features=time_features,
            attention_mask=attention_mask
        )
        
        loss_dict = self.compute_loss(model_output, values, alpha=alpha, beta=beta, gamma=gamma)
        loss_dict["total_loss"].backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Convert tensors to float for logging
        return {k: v.item() for k, v in loss_dict.items()}
    
    def extract_features(
        self,
        values: torch.Tensor,
        time_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract features from the model.
        
        Parameters:
        -----------
        values : torch.Tensor
            Input values tensor
        time_features : torch.Tensor
            Time features tensor
        attention_mask : torch.Tensor, optional
            Attention mask
            
        Returns:
        --------
        torch.Tensor
            Extracted features
        """
        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            model_output = self.forward(
                values=values,
                time_features=time_features,
                attention_mask=attention_mask
            )
            
            # Use the projected features as final representation
            # Average over sequence dimension for a single vector per instance
            features = model_output["projected_features"].mean(dim=1)
            
        # Set model back to training mode
        self.train()
        
        return features


def prepare_data_for_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    context_length: int,
    time_feature_cols: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Prepare data for model training by creating sliding windows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    feature_cols : List[str]
        List of feature column names to use as inputs
    context_length : int
        Length of context window
    time_feature_cols : List[str], optional
        List of time feature column names
        
    Returns:
    --------
    Dict[str, torch.Tensor]
        Dictionary containing model inputs
    """
    # Default time feature if none provided
    if time_feature_cols is None:
        time_feature_cols = ["time_idx"]
        if "time_idx" not in df.columns:
            df["time_idx"] = np.arange(len(df)) / len(df)  # Normalized index
    
    # Normalize features using standard scaler
    scaler = StandardScaler()
    feature_values = scaler.fit_transform(df[feature_cols].values)
    
    # Get time features
    time_features = df[time_feature_cols].values
    
    # Create sliding windows
    windows_values = []
    windows_time_features = []
    
    for i in range(len(df) - context_length + 1):
        # Extract window
        window_values = feature_values[i:i+context_length]
        window_time = time_features[i:i+context_length]
        
        windows_values.append(window_values)
        windows_time_features.append(window_time)
    
    # Convert to numpy arrays
    windows_values = np.array(windows_values)
    windows_time_features = np.array(windows_time_features)
    
    # Create attention mask (1 for valid tokens, 0 for padding/masked tokens)
    # In this case, all tokens are valid so we use ones
    attention_mask = np.ones((len(windows_values), context_length))
    
    # Convert to PyTorch tensors
    values_tensor = torch.tensor(windows_values, dtype=torch.float32)
    time_features_tensor = torch.tensor(windows_time_features, dtype=torch.float32)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.float32)
    
    # Return dictionary of model inputs
    return {
        "values": values_tensor,
        "time_features": time_features_tensor,
        "attention_mask": attention_mask_tensor
    }


def train_model(
    model: VolatilityRegimeFeatureExtractor,
    train_data: Dict[str, torch.Tensor],
    num_epochs: int = 50,
    device: Optional[str] = None,
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.2,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train the volatility regime feature extractor model.
    
    Parameters:
    -----------
    model : VolatilityRegimeFeatureExtractor
        Model to train
    train_data : Dict[str, torch.Tensor]
        Training data dictionary
    num_epochs : int
        Number of epochs to train
    device : str, optional
        Device to use for computation
    alpha : float
        Weight for temporal consistency loss
    beta : float
        Weight for attention diversity loss
    gamma : float
        Weight for volatility sensitivity loss
    verbose : bool
        Whether to print progress
        
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
        "reconstruction_loss": [],
        "temporal_consistency_loss": [],
        "attention_diversity_loss": [],
        "volatility_sensitivity_loss": [],
        "total_loss": []
    }
    
    # Set model to training mode
    model.train()
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_metrics = {k: 0.0 for k in history.keys()}
        num_batches = 0
        
        # Generate random batch indices
        batch_size = model.batch_size
        num_samples = train_data["values"].shape[0]
        indices = np.random.permutation(num_samples)
        
        for batch_start in range(0, num_samples, batch_size):
            # Get batch indices
            batch_indices = indices[batch_start:batch_start+batch_size]
            
            # Extract batch data
            batch_values = train_data["values"][batch_indices]
            batch_time_features = train_data["time_features"][batch_indices]
            batch_attention_mask = train_data["attention_mask"][batch_indices]
            
            # Perform training step
            batch_metrics = model.train_step(
                values=batch_values,
                time_features=batch_time_features,
                attention_mask=batch_attention_mask,
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )
            
            # Update epoch metrics
            for k, v in batch_metrics.items():
                epoch_metrics[k] += v
                
            num_batches += 1
        
        # Calculate average metrics for the epoch
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= max(1, num_batches)  # Avoid division by zero
            history[k].append(epoch_metrics[k])
        
        # Print progress
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - " + 
                  f"Loss: {epoch_metrics['total_loss']:.4f} - " +
                  f"Recon: {epoch_metrics['reconstruction_loss']:.4f} - " +
                  f"Temp: {epoch_metrics['temporal_consistency_loss']:.4f} - " +
                  f"Div: {epoch_metrics['attention_diversity_loss']:.4f} - " +
                  f"Vol: {epoch_metrics['volatility_sensitivity_loss']:.4f}")
    
    return history


def extract_features_from_df(
    model: VolatilityRegimeFeatureExtractor,
    df: pd.DataFrame,
    feature_cols: List[str],
    context_length: int,
    time_feature_cols: Optional[List[str]] = None
) -> np.ndarray:
    """
    Extract features from a DataFrame using the trained model.
    
    Parameters:
    -----------
    model : VolatilityRegimeFeatureExtractor
        Trained model
    df : pd.DataFrame
        Input dataframe with features
    feature_cols : List[str]
        List of feature column names to use as inputs
    context_length : int
        Length of context window
    time_feature_cols : List[str], optional
        List of time feature column names
        
    Returns:
    --------
    np.ndarray
        Extracted features
    """
    # Prepare data
    data = prepare_data_for_model(
        df=df,
        feature_cols=feature_cols,
        context_length=context_length,
        time_feature_cols=time_feature_cols
    )
    
    # Move data to model device
    data = {k: v.to(model.device) for k, v in data.items()}
    
    # Extract features
    features = model.extract_features(
        values=data["values"],
        time_features=data["time_features"],
        attention_mask=data["attention_mask"]
    )
    
    # Convert to numpy array
    features_np = features.cpu().numpy()
    
    return features_np 