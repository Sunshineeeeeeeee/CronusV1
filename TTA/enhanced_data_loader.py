import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from typing import Dict, List, Tuple, Optional, Union
import datetime
import torch.nn.functional as F

# Import our signal processing module
from CronusV1.TTA.advanced_signal_processing_v1 import SignalProcessor, extract_advanced_features


class EnhancedTradingDataset(Dataset):
    """Dataset for trading data with enhanced trend strength features."""
    
    def __init__(self, features, trend_strengths, profits, regimes=None, subregimes=None, time_features=None):
        """
        Initialize the trading dataset.
        
        Args:
            features (np.ndarray): Feature matrix of shape (n_samples, window_size, n_features)
            trend_strengths (np.ndarray): Trend strength values of shape (n_samples,)
            profits (np.ndarray): Profit values of shape (n_samples,)
            regimes (np.ndarray, optional): Regime labels of shape (n_samples, window_size)
            subregimes (np.ndarray, optional): Sub-regime labels of shape (n_samples, window_size)
            time_features (np.ndarray, optional): Time features of shape (n_samples, window_size, 2)
        """
        self.features = features
        self.trend_strengths = trend_strengths
        self.profits = profits
        self.regimes = regimes
        self.subregimes = subregimes
        self.time_features = time_features
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = {
            'features': self.features[idx],
            'trend_strength': self.trend_strengths[idx],
            'profit': self.profits[idx]
        }
        
        if self.regimes is not None:
            sample['regimes'] = self.regimes[idx]
            
        if self.subregimes is not None:
            sample['subregimes'] = self.subregimes[idx]
            
        if self.time_features is not None:
            sample['time_features'] = self.time_features[idx]
            
        return sample


def prepare_enhanced_trading_data(data_path, lookback_window=30, prediction_window=7, 
                               batch_size=64, val_ratio=0.15, test_ratio=0.15,
                               use_regime_feature=True, use_time_feature=True, device='cuda'):
    """
    Load and prepare enhanced trading data for training, validation, and testing.
    Optimized for GPU acceleration.
    
    Parameters:
        data_path (str): Path to the data file
        lookback_window (int): Number of days to look back for features
        prediction_window (int): Number of days to predict ahead
        batch_size (int): Batch size for DataLoader
        val_ratio (float): Ratio of data for validation set
        test_ratio (float): Ratio of data for test set
        use_regime_feature (bool): Whether to use regime feature
        use_time_feature (bool): Whether to use time feature
        device (str): Device to use for tensor operations ('cuda' or 'cpu')
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, feature_scaler)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Extract feature columns
    feature_cols = [col for col in df.columns if col not in ['date', 'close']]
    
    # Initialize StandardScaler on CPU
    feature_scaler = StandardScaler()
    
    # Scale features on CPU (StandardScaler doesn't work on GPU)
    features_np = df[feature_cols].values
    scaled_features_np = feature_scaler.fit_transform(features_np)
    
    # Move to device only after preprocessing
    features = torch.tensor(scaled_features_np, dtype=torch.float32, device=device)
    prices = torch.tensor(df['close'].values, dtype=torch.float32, device=device)
    
    # Calculate future prices and percentage changes in batch on GPU
    with torch.no_grad():  # Disable gradient tracking for efficiency
        future_prices = torch.roll(prices, shifts=-prediction_window, dims=0)
        future_prices[-prediction_window:] = float('nan')
        price_change_pct = (future_prices - prices) / prices * 100
    
    # Remove the last prediction_window rows that don't have valid targets
    features = features[:-prediction_window]
    prices = prices[:-prediction_window]
    price_change_pct = price_change_pct[:-prediction_window]
    
    # Optional: Prepare regime features
    if use_regime_feature:
        with torch.no_grad():
            # Use efficient 1D convolution for moving averages
            prices_1d = prices.view(1, 1, -1)
            short_ma = torch.nn.functional.avg_pool1d(prices_1d, kernel_size=5, stride=1).view(-1)
            long_ma = torch.nn.functional.avg_pool1d(prices_1d, kernel_size=20, stride=1).view(-1)
            
            # Pad the beginning of the short and long MAs
            short_ma_padded = torch.cat([torch.full((4,), float('nan'), device=device), short_ma])[:prices.shape[0]]
            long_ma_padded = torch.cat([torch.full((19,), float('nan'), device=device), long_ma])[:prices.shape[0]]
            
            # Vectorized regime calculation
            regime = torch.where(short_ma_padded > long_ma_padded, 
                                torch.ones(1, device=device),
                                torch.full((1,), -1.0, device=device))
            
            # Handle NaN values
            regime_mask = torch.isnan(short_ma_padded) | torch.isnan(long_ma_padded)
            regime = torch.where(regime_mask, torch.zeros(1, device=device), regime)
            
            # Add regime as a feature
            regime = regime.view(-1, 1)
            features = torch.cat([features, regime], dim=1)
    
    # Optional: Prepare time features
    if use_time_feature:
        # Process dates on CPU
        dates = pd.to_datetime(df['date'][:-prediction_window])
        
        # Create normalized time features and move to device
        day_of_week = torch.tensor(dates.dt.dayofweek.values / 6, dtype=torch.float32, device=device).view(-1, 1)
        day_of_month = torch.tensor(dates.dt.day.values / 31, dtype=torch.float32, device=device).view(-1, 1)
        month = torch.tensor((dates.dt.month - 1) / 11, dtype=torch.float32, device=device).view(-1, 1)
        
        # Concatenate time features on GPU
        features = torch.cat([features, day_of_week, day_of_month, month], dim=1)
    
    # Create sliding windows efficiently on GPU
    feature_windows = create_sliding_windows(features, lookback_window)
    
    # Align targets with the windows
    targets = price_change_pct[lookback_window-1:]
    
    # Filter out windows with NaN targets
    valid_indices = ~torch.isnan(targets)
    feature_windows = feature_windows[valid_indices]
    targets = targets[valid_indices]
    
    # Calculate dataset sizes
    total_samples = feature_windows.size(0)
    test_size = int(total_samples * test_ratio)
    val_size = int(total_samples * val_ratio)
    train_size = total_samples - test_size - val_size
    
    # Split datasets efficiently on GPU
    train_features = feature_windows[:train_size]
    train_targets = targets[:train_size]
    
    val_features = feature_windows[train_size:train_size+val_size]
    val_targets = targets[train_size:train_size+val_size]
    
    test_features = feature_windows[train_size+val_size:]
    test_targets = targets[train_size+val_size:]
    
    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(train_features, train_targets.view(-1, 1))
    val_dataset = torch.utils.data.TensorDataset(val_features, val_targets.view(-1, 1))
    test_dataset = torch.utils.data.TensorDataset(test_features, test_targets.view(-1, 1))
    
    # Create optimized DataLoaders for GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=(device == 'cuda'),  # Pin memory for faster transfers
        num_workers=4 if device == 'cuda' else 0,  # Parallelize data loading when using GPU
        persistent_workers=True if device == 'cuda' else False,  # Keep workers alive between batches
        prefetch_factor=2 if device == 'cuda' else None  # Prefetch batches
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=(device == 'cuda'),
        num_workers=4 if device == 'cuda' else 0,
        persistent_workers=True if device == 'cuda' else False,
        prefetch_factor=2 if device == 'cuda' else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=(device == 'cuda'),
        num_workers=4 if device == 'cuda' else 0,
        persistent_workers=True if device == 'cuda' else False,
        prefetch_factor=2 if device == 'cuda' else None
    )
    
    return train_loader, val_loader, test_loader, feature_scaler


def create_sliding_windows(data, window_size):
    """
    Create sliding windows of features for sequential data.
    Optimized for GPU processing with efficient tensor operations.
    
    Parameters:
        data (torch.Tensor): Input tensor with shape [sequence_length, feature_dim]
        window_size (int): Size of the sliding window
    
    Returns:
        torch.Tensor: Tensor with sliding windows [num_windows, window_size, feature_dim]
    """
    # Get sequence length and feature dimension
    seq_len, feat_dim = data.shape
    
    # Calculate number of windows
    num_windows = seq_len - window_size + 1
    
    # Using efficient unfold operation for creating sliding windows
    # This avoids explicit loops and is optimized for GPU execution
    windows = data.unfold(0, window_size, 1)
    
    # No need to create a new tensor with cat operations
    # The unfold operation directly returns the correct shape [num_windows, window_size, feature_dim]
    
    return windows


def collate_batch(batch):
    """Collate function for DataLoader that handles the enhanced trend-based data."""
    # Extract all features, regimes, etc. from the batch
    features = [item['features'] for item in batch]
    trend_strengths = [item['trend_strength'] for item in batch]
    profits = [item['profit'] for item in batch]
    
    # Extract regimes, subregimes, and time features if available
    regimes = [item.get('regimes', None) for item in batch]
    subregimes = [item.get('subregimes', None) for item in batch]
    time_features = [item.get('time_features', None) for item in batch]
    
    # Convert to tensors
    features_tensor = torch.tensor(np.stack(features), dtype=torch.float32)
    trend_strengths_tensor = torch.tensor(trend_strengths, dtype=torch.float32).unsqueeze(-1)
    profits_tensor = torch.tensor(profits, dtype=torch.float32).unsqueeze(-1)
    
    # Handle regimes and subregimes (might be None)
    regimes_tensor = None
    if all(r is not None for r in regimes):
        regimes_tensor = torch.tensor(np.stack(regimes), dtype=torch.long)
    
    subregimes_tensor = None
    if all(s is not None for s in subregimes):
        subregimes_tensor = torch.tensor(np.stack(subregimes), dtype=torch.long)
    
    time_features_tensor = None
    if all(t is not None for t in time_features):
        time_features_tensor = torch.tensor(np.stack(time_features), dtype=torch.float32)
    
    return {
        'features': features_tensor,
        'trend_strength': trend_strengths_tensor,
        'profit': profits_tensor,
        'regimes': regimes_tensor,
        'subregimes': subregimes_tensor,
        'time_features': time_features_tensor
    }


def build_regime_feature(features, prices, window_sizes=[5, 10, 20, 50, 100, 200]):
    """
    Build market regime features using efficient GPU operations.
    
    Parameters:
        features (torch.Tensor): Feature tensor
        prices (torch.Tensor): Price tensor
        window_sizes (list): List of window sizes for moving averages
    
    Returns:
        torch.Tensor: Combined features including regime indicators
    """
    device = features.device
    
    # Calculate market regimes based on moving averages
    with torch.no_grad():
        # Prepare convolution layer for efficient moving average calculation
        regime_features = []
        
        # Create 1D convolution kernels for each window size
        for window_size in window_sizes:
            # Create convolution kernel (equal weights)
            kernel = torch.ones(1, 1, window_size, device=device) / window_size
            
            # Pad the price tensor to handle boundary cases
            padded_prices = F.pad(prices.view(1, 1, -1), (window_size-1, 0))
            
            # Apply convolution to calculate moving average
            ma = F.conv1d(padded_prices, kernel).view(-1)
            
            # Calculate regime indicator (price relative to moving average)
            regime = torch.zeros_like(prices, device=device)
            valid_indices = torch.arange(prices.size(0), device=device)
            
            # Vectorized comparison
            regime = torch.where(prices > ma, torch.tensor(1.0, device=device), 
                               torch.where(prices < ma, torch.tensor(-1.0, device=device), 
                                         torch.tensor(0.0, device=device)))
            
            regime_features.append(regime)
    
    # Efficiently stack all regime features
    all_regime_features = torch.stack(regime_features, dim=1)
    
    # Combine with original features
    combined_features = torch.cat([features, all_regime_features], dim=1)
    
    return combined_features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Enhanced Trading Data Loader")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to process")
    args = parser.parse_args()
    
    # Test data preparation with limited samples
    train_loader, val_loader, test_loader, metadata = prepare_enhanced_trading_data(
        csv_path=args.data_path,
        window_size=50,
        max_samples=args.max_samples,
        visualize=True
    )
    
    # Print metadata
    print("\nMetadata:")
    for key, value in metadata.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list) and len(value) > 10:
            print(f"{key}: [List with {len(value)} elements]")
        else:
            print(f"{key}: {value}")
    
    # Fetch a batch and print shapes
    batch = next(iter(train_loader))
    print("\nBatch structure:")
    for key, value in batch.items():
        if value is not None:
            print(f"{key} shape: {value.shape}")
        else:
            print(f"{key}: None") 