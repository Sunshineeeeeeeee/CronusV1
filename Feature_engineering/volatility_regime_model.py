import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig
from Feature_engineering.feature_extraction import FeatureExtractor
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler
import warnings
import logging
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VolatilityFeatureTransformer:
    """
    A model for learning latent representations of market microstructure features
    that are useful for volatility analysis. Uses a transformer encoder to learn
    complex temporal patterns in the data.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        context_length: int = 50,
        latent_dim: int = 32,
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        """
        Initialize the volatility feature transformer.
        
        Parameters:
        -----------
        d_model : int
            The dimension of the transformer model
        n_heads : int
            Number of attention heads
        n_encoder_layers : int
            Number of transformer encoder layers
        dim_feedforward : int
            Dimension of the feedforward network
        dropout : float
            Dropout rate
        context_length : int
            Length of the context window
        latent_dim : int
            Dimension of the learned latent representation
        feature_extractor : FeatureExtractor, optional
            Custom feature extractor instance
        """
        self.context_length = context_length
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # Initialize feature extractor
        self.feature_extractor = feature_extractor if feature_extractor else FeatureExtractor()
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.volatility_scaler = StandardScaler()
        
        logger.info("Initializing VolatilityFeatureTransformer...")
        logger.info(f"Context length: {context_length}")
        logger.info(f"Model dimension (d_model): {d_model}")
        logger.info(f"Number of heads: {n_heads}")
        logger.info(f"Number of encoder layers: {n_encoder_layers}")
        
        # Configure the transformer model
        self.config = TimeSeriesTransformerConfig(
            prediction_length=1,
            context_length=context_length,
            input_size=1,  # Set to 1 as we'll handle the projection ourselves
            d_model=d_model,
            encoder_layers=n_encoder_layers,
            encoder_attention_heads=n_heads,
            decoder_layers=0,  # We only use the encoder
            encoder_ffn_dim=dim_feedforward,
            dropout=dropout,
            is_encoder_decoder=False,
            use_cache=False
        )
        
        logger.info("Transformer config:")
        logger.info(f"Input size: {self.config.input_size}")
        logger.info(f"d_model: {self.config.d_model}")
        
        # Initialize the transformer model
        self.model = TimeSeriesTransformerModel(self.config)
        
        # Override the value embedding with our own projection
        self.model.encoder.value_embedding = nn.Identity()
        
        # These will be initialized in prepare_data
        self.input_projection = None
        self.feature_names = None
        
        # Add latent feature projection layer
        self.feature_projector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, latent_dim)
        )
        
        # Add volatility reconstruction head
        self.volatility_reconstructor = nn.Sequential(
            nn.Linear(latent_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'Timestamp',
        price_col: str = 'Value',
        volume_col: str = 'Volume',
        volatility_col: str = 'Volatility'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for the model with detailed logging of feature extraction process.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with tick data
        timestamp_col : str
            Column name for timestamp
        price_col : str
            Column name for price values
        volume_col : str
            Column name for volume values
        volatility_col : str
            Column name for volatility values
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            features_tensor, volatility_tensor
        """
        logger.info("\n=== Data Preparation Start ===")
        logger.info(f"Input DataFrame shape: {df.shape}")
        logger.info(f"Input DataFrame columns: {df.columns.tolist()}")
        
        # Extract features
        feature_df = self.feature_extractor.extract_features(
            df,
            timestamp_col=timestamp_col,
            price_col=price_col,
            volume_col=volume_col,
            volatility_col=volatility_col
        )
        
        # Get feature columns
        feature_cols = [col for col in feature_df.columns 
                       if col not in [timestamp_col, price_col, volume_col, volatility_col]]
        
        logger.info("\n=== Feature Information ===")
        logger.info(f"Number of features extracted: {len(feature_cols)}")
        logger.info(f"Feature columns: {feature_cols}")
        
        # Scale features
        features = self.feature_scaler.fit_transform(feature_df[feature_cols])
        logger.info(f"Scaled features shape: {features.shape}")
        
        # Create sliding windows
        X = []
        for i in range(len(features) - self.context_length + 1):
            window = features[i:i + self.context_length]
            X.append(window)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(np.array(X))
        
        # Scale volatility values
        volatility = df[volatility_col].values
        volatility_scaled = self.volatility_scaler.fit_transform(volatility.reshape(-1, 1))
        
        # Create volatility tensor for self-supervised learning
        volatility_tensor = torch.FloatTensor(volatility_scaled[self.context_length-1:])
        
        logger.info("\n=== Tensor Shapes ===")
        logger.info(f"Features tensor shape: {features_tensor.shape}")
        logger.info(f"Volatility tensor shape: {volatility_tensor.shape}")
        
        # Initialize input projection if not already done
        if self.input_projection is None:
            input_feat_dim = features_tensor.shape[-1]
            logger.info(f"\n=== Input Projection Layer ===")
            logger.info(f"Initializing projection: {input_feat_dim} -> {self.config.d_model}")
            self.input_projection = nn.Linear(input_feat_dim, self.config.d_model)
        
        return features_tensor, volatility_tensor
    
    def train(
        self,
        train_features: torch.Tensor,
        train_volatility: torch.Tensor,
        num_epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> Dict[str, List[float]]:
        """
        Train the model using self-supervised learning.
        The model learns to extract features that are useful for reconstructing volatility.
        """
        logger.info("\n=== Training Start ===")
        logger.info(f"Device: {device}")
        logger.info(f"Input features shape: {train_features.shape}")
        logger.info(f"Input volatility shape: {train_volatility.shape}")
        
        self.model = self.model.to(device)
        self.input_projection = self.input_projection.to(device)
        self.feature_projector = self.feature_projector.to(device)
        self.volatility_reconstructor = self.volatility_reconstructor.to(device)
        
        optimizer = torch.optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.input_projection.parameters()},
            {'params': self.feature_projector.parameters()},
            {'params': self.volatility_reconstructor.parameters()}
        ], lr=learning_rate)
        
        criterion = nn.MSELoss()
        history = {'loss': [], 'batch_losses': []}
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            batch_losses = []
            num_batches = len(train_features) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_features = train_features[start_idx:end_idx].to(device)
                batch_volatility = train_volatility[start_idx:end_idx].to(device)
                
                # Project input features
                batch_size, seq_len, feat_dim = batch_features.shape
                batch_features_flat = batch_features.view(batch_size * seq_len, feat_dim)
                projected_features = self.input_projection(batch_features_flat)
                projected_features = projected_features.view(batch_size, seq_len, self.d_model)
                
                # Forward pass through encoder
                encoder_outputs = self.model.encoder(
                    inputs_embeds=projected_features,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True
                )
                
                # Get all hidden states and project each timestep
                hidden_states = encoder_outputs.last_hidden_state  # [batch_size, seq_len, d_model]
                hidden_states_flat = hidden_states.reshape(-1, self.d_model)
                latent_features = self.feature_projector(hidden_states_flat)
                latent_features = latent_features.view(batch_size, seq_len, self.latent_dim)
                
                # Use the center point's latent features for volatility reconstruction
                center_features = latent_features[:, seq_len//2, :]  # Take middle timestep
                reconstructed_volatility = self.volatility_reconstructor(center_features)
                
                loss = criterion(reconstructed_volatility, batch_volatility)
                batch_losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            history['loss'].append(avg_loss)
            history['batch_losses'].extend(batch_losses)
            
            # Log summary statistics for the epoch
            logger.info(f"\n=== Epoch {epoch+1}/{num_epochs} Summary ===")
            logger.info(f"Average Loss: {avg_loss:.6f}")
            logger.info(f"Min Batch Loss: {min(batch_losses):.6f}")
            logger.info(f"Max Batch Loss: {max(batch_losses):.6f}")
            logger.info(f"Std Dev Loss: {np.std(batch_losses):.6f}")
            
            # Log progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                logger.info("\n=== Training Progress ===")
                logger.info(f"Completed {epoch+1}/{num_epochs} epochs")
                logger.info(f"Current Avg Loss: {avg_loss:.6f}")
                logger.info(f"Best Loss So Far: {min(history['loss']):.6f}")
        
        logger.info("\n=== Training Complete ===")
        logger.info(f"Final Average Loss: {history['loss'][-1]:.6f}")
        logger.info(f"Best Loss Achieved: {min(history['loss']):.6f}")
        
        return history
    
    def extract_features(
        self,
        features: torch.Tensor,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> np.ndarray:
        """
        Extract latent features from the input data.
        Returns a feature vector for each timestep.
        """
        self.model.eval()
        self.input_projection.eval()
        self.feature_projector.eval()
        
        latent_features = []
        
        with torch.no_grad():
            batch_size = 128
            num_batches = len(features) // batch_size + (1 if len(features) % batch_size != 0 else 0)
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(features))
                
                batch_features = features[start_idx:end_idx].to(device)
                
                # Project input features
                batch_size, seq_len, feat_dim = batch_features.shape
                batch_features_flat = batch_features.view(batch_size * seq_len, feat_dim)
                projected_features = self.input_projection(batch_features_flat)
                projected_features = projected_features.view(batch_size, seq_len, self.d_model)
                
                # Forward pass through encoder
                encoder_outputs = self.model.encoder(
                    inputs_embeds=projected_features,
                    return_dict=True
                )
                
                # Get all hidden states and project each timestep
                hidden_states = encoder_outputs.last_hidden_state  # [batch_size, seq_len, d_model]
                hidden_states_flat = hidden_states.reshape(-1, self.d_model)
                batch_latent = self.feature_projector(hidden_states_flat)
                batch_latent = batch_latent.view(batch_size, seq_len, self.latent_dim)
                
                # Only keep the center point's features for each window
                center_features = batch_latent[:, seq_len//2, :].cpu().numpy()
                latent_features.extend(center_features)
        
        return np.array(latent_features)
    
    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_projection_state_dict': self.input_projection.state_dict(),
            'feature_projector_state_dict': self.feature_projector.state_dict(),
            'volatility_reconstructor_state_dict': self.volatility_reconstructor.state_dict(),
            'feature_scaler': self.feature_scaler,
            'volatility_scaler': self.volatility_scaler,
            'config': self.config
        }, path)
    
    def load_model(self, path: str):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.input_projection.load_state_dict(checkpoint['input_projection_state_dict'])
        self.feature_projector.load_state_dict(checkpoint['feature_projector_state_dict'])
        self.volatility_reconstructor.load_state_dict(checkpoint['volatility_reconstructor_state_dict'])
        self.feature_scaler = checkpoint['feature_scaler']
        self.volatility_scaler = checkpoint['volatility_scaler']
        self.config = checkpoint['config'] 

    def _extract_features(self, window):
        """
        Extract features from a window of data with logging.
        """
        features = {}
        
        # Basic price features
        features['returns'] = (window['Value'].pct_change().fillna(0)).values[-1]
        features['log_returns'] = (np.log(window['Value']).diff().fillna(0)).values[-1]
        features['rolling_std'] = window['Value'].rolling(5).std().fillna(0).values[-1]
        
        # Volume features
        features['volume'] = window['Volume'].values[-1]
        features['volume_ma'] = window['Volume'].rolling(5).mean().fillna(0).values[-1]
        features['volume_std'] = window['Volume'].rolling(5).std().fillna(0).values[-1]
        
        # Price movement features
        price_diff = window['Value'].diff().fillna(0)
        features['price_acceleration'] = price_diff.diff().fillna(0).values[-1]
        features['price_momentum'] = price_diff.rolling(5).mean().fillna(0).values[-1]
        
        # Add more features as needed...
        
        return features
    
    def forward(self, x):
        """
        Forward pass with logging of tensor shapes.
        """
        batch_size, seq_len, feat_dim = x.shape
        logger.info(f"Input tensor shape: {x.shape}")
        
        # Reshape for linear projection
        batch_features = x.view(batch_size * seq_len, feat_dim)
        logger.info(f"Reshaped for projection: {batch_features.shape}")
        
        # Project features
        projected_features = self.input_projection(batch_features)
        logger.info(f"After input projection: {projected_features.shape}")
        
        # Reshape back for transformer
        projected_features = projected_features.view(batch_size, seq_len, -1)
        logger.info(f"Reshaped for transformer: {projected_features.shape}")
        
        # Pass through transformer
        transformer_output = self.model(inputs_embeds=projected_features).last_hidden_state
        logger.info(f"Transformer output shape: {transformer_output.shape}")
        
        return transformer_output
    
    def train_model(self, features, volatility, batch_size=32, num_epochs=20, learning_rate=1e-4):
        """
        Train the model with detailed logging.
        """
        logger.info("Starting model training...")
        logger.info(f"Training data shapes - Features: {features.shape}, Volatility: {volatility.shape}")
        
        dataset = TensorDataset(features, volatility)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        logger.info(f"Created DataLoader with batch size: {batch_size}")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            total_loss = 0
            batch_count = 0
            
            for batch_features, batch_volatility in dataloader:
                optimizer.zero_grad()
                
                logger.info(f"Epoch {epoch+1}, Batch {batch_count+1}")
                logger.info(f"Batch features shape: {batch_features.shape}")
                
                # Forward pass
                outputs = self(batch_features)
                logger.info(f"Model output shape: {outputs.shape}")
                
                # Calculate loss
                loss = criterion(outputs[:, -1, :], batch_volatility)
                logger.info(f"Batch loss: {loss.item():.6f}")
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
            
        logger.info("Training completed!") 