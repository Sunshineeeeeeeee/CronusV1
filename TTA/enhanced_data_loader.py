import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from typing import Dict, List, Tuple, Optional, Union
import datetime

# Import our signal processing module
from advanced_signal_processing import SignalProcessor, extract_advanced_features


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


def prepare_enhanced_trading_data(
    csv_path: str,
    window_size: int = 50,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    target_horizon: int = 50,         # 50 ticks ahead
    profit_cap: float = 0.05,         # Cap profit at 5%
    max_samples: Optional[int] = None,
    random_seed: int = 42,
    visualize: bool = False,
    signal_window_sizes: List[int] = [8, 16, 32, 64],
    results_dir: Optional[str] = None,
    regime_mapping: Optional[dict] = None,
    subregime_mapping: Optional[dict] = None,
    disable_direction_correction: bool = False,
    use_balanced_sampling: bool = False,
    balance_threshold: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Prepare enhanced trading data with trend features, regimes, and time features.
    
    Args:
        csv_path: Path to CSV data file
        window_size: Size of sliding window for sequence data
        batch_size: Batch size for data loader
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        target_horizon: Prediction horizon in ticks
        profit_cap: Cap profit at this value
        max_samples: Maximum number of samples to use (for testing)
        random_seed: Random seed for reproducibility
        visualize: Whether to generate visualizations
        signal_window_sizes: Window sizes for signal processing
        results_dir: Directory to save results
        regime_mapping: Mapping from regime values to embedding indices
        subregime_mapping: Mapping from subregime values to embedding indices
        disable_direction_correction: Whether to disable automatic correction of trend direction
        use_balanced_sampling: Whether to use class-balanced sampling for training
        balance_threshold: Threshold for determining trend direction classes
        
    Returns:
        Tuple of DataLoaders for train, validation, and test sets, and metadata dictionary
    """
    print("\nLoading data...")
    print(f"Loading data from {csv_path}...")
    
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Read data
    df = pd.read_csv(csv_path)
    
    if max_samples is not None:
        print(f"Limiting to {max_samples} samples for testing")
        df = df.iloc[:max_samples + window_size + target_horizon]
    
    print(f"Data shape: {df.shape}")
    
    # Get the price column (typically 'close' or similar)
    price_cols = [col for col in df.columns if col.lower() in ['close', 'price', 'last', 'value']]
    if not price_cols:
        raise ValueError("No price column found in data. Expected 'close', 'price', 'last', or 'value'")
    price_col = price_cols[0]
    print(f"Using '{price_col}' as the price column")
    
    # Calculate advanced trend features
    print("Calculating advanced trend features...")
    processor = SignalProcessor(window_sizes=signal_window_sizes)
    trend_features = processor.calculate_multi_timeframe_trend(
        df, price_col, disable_direction_correction=disable_direction_correction
    )
    
    # Add trend features to dataframe
    df = pd.concat([df, trend_features], axis=1)
    
    # Create the target: future return over the horizon
    df['future_return'] = df[price_col].pct_change(target_horizon).shift(-target_horizon)
    
    # Create profit target: capped absolute return
    df['profit'] = df['future_return'].clip(lower=-profit_cap, upper=profit_cap)
    
    # Drop rows with NaN values
    df = df.dropna(subset=['future_return', 'profit', 'weighted_trend_strength'])
    
    print(f"Data shape after dropping NaN: {df.shape}")
    
    # Identify feature columns (excluding target columns and metadata)
    exclude_cols = ['profit', 'future_return', 'timestamp', 'datetime', 'date', 'time']
    exclude_cols += ['trend_strength', 'trend_agreement', 'weighted_trend_strength']
    exclude_cols += [f'trend_{w}' for w in signal_window_sizes]
    
    feature_cols = [col for col in df.columns if not any(excl in col.lower() for excl in exclude_cols)]
    
    # Standardize features (fit only on training data)
    total_rows = len(df)
    train_end_idx = int(total_rows * train_ratio)
    train_df = df.iloc[:train_end_idx]
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = [col for col in feature_cols if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    if numerical_cols:
        # Fit scaler on training data
        scaler.fit(train_df[numerical_cols].values)
        # Apply to all data
        df[numerical_cols] = scaler.transform(df[numerical_cols].values)
    
    # Store numerical features as arrays
    df['features'] = df[numerical_cols].values.tolist()
    
    # Identify regime columns if available
    regime_cols = [col for col in df.columns if 'regime' in col.lower() and 'sub_regime' not in col.lower()]
    sub_regime_cols = [col for col in df.columns if 'sub_regime' in col.lower()]
    
    # Create trend direction column using threshold
    df['trend_direction'] = 0
    df.loc[df['weighted_trend_strength'] > balance_threshold, 'trend_direction'] = 1
    df.loc[df['weighted_trend_strength'] < -balance_threshold, 'trend_direction'] = -1
    
    # Print stats about the dataset
    print(f"Total samples after preparation: {len(df)}")
    print(f"Total features: {len(numerical_cols)}")
    print(f"Trend strength range: {df['weighted_trend_strength'].min():.4f} to {df['weighted_trend_strength'].max():.4f}, "
          f"mean: {df['weighted_trend_strength'].mean():.4f}")
    
    # Count direction classes
    direction_counts = df['trend_direction'].value_counts()
    print(f"Direction class distribution: Up={direction_counts.get(1, 0)}, "
          f"Neutral={direction_counts.get(0, 0)}, Down={direction_counts.get(-1, 0)}")
    
    # Visualize the data distribution if requested
    if visualize:
        results_dir = results_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Trend strength distribution
        plt.subplot(2, 2, 1)
        plt.hist(df['weighted_trend_strength'], bins=50)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Weighted Trend Strength Distribution')
        plt.xlabel('Trend Strength')
        plt.ylabel('Count')
        
        # Plot 2: Profit distribution
        plt.subplot(2, 2, 2)
        plt.hist(df['profit'], bins=50, alpha=0.75)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Profit Distribution')
        plt.xlabel('Profit')
        plt.ylabel('Count')
        
        # Plot 3: Trend strength vs profit
        plt.subplot(2, 2, 3)
        plt.scatter(df['weighted_trend_strength'], df['profit'], alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Trend Strength vs Profit')
        plt.xlabel('Trend Strength')
        plt.ylabel('Profit')
        
        # Plot 4: Trend agreement distribution
        plt.subplot(2, 2, 4)
        plt.hist(df['trend_agreement'], bins=20)
        plt.title('Trend Agreement Distribution')
        plt.xlabel('Agreement Level')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"trend_data_distribution_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
        plt.close()
    
    # Create sliding windows for sequential data
    window_data = create_sliding_windows(df, window_size, 
                                         feature_key='features',
                                         trend_key='weighted_trend_strength',
                                         regime_cols=regime_cols, 
                                         sub_regime_cols=sub_regime_cols,
                                         regime_mapping=regime_mapping,
                                         subregime_mapping=subregime_mapping)
    
    # Split the windowed data into train, val, and test sets
    total_windows = len(window_data['features'])
    train_end_idx = int(total_windows * train_ratio)
    val_end_idx = train_end_idx + int(total_windows * val_ratio)
    
    train_indices = range(0, train_end_idx)
    val_indices = range(train_end_idx, val_end_idx)
    test_indices = range(val_end_idx, total_windows)
    
    # Apply class-balanced sampling for training data if requested
    if use_balanced_sampling:
        print("\nApplying class-balanced sampling for training data...")
        
        # Get trend directions for all windows
        trend_true = window_data['trend_strengths']
        trend_directions = np.zeros_like(trend_true)
        trend_directions[trend_true > balance_threshold] = 1
        trend_directions[trend_true < -balance_threshold] = -1
        
        # Separate indices by direction class for the training set only
        up_indices = np.where((trend_directions == 1) & (np.arange(len(trend_directions)) < train_end_idx))[0]
        neutral_indices = np.where((trend_directions == 0) & (np.arange(len(trend_directions)) < train_end_idx))[0]
        down_indices = np.where((trend_directions == -1) & (np.arange(len(trend_directions)) < train_end_idx))[0]
        
        # Print class counts
        print(f"Original train class distribution: Up={len(up_indices)}, Neutral={len(neutral_indices)}, Down={len(down_indices)}")
        
        # Determine target count (min of all classes, but at least 100 samples)
        min_class_count = min(len(up_indices), len(down_indices), len(neutral_indices))
        target_count = max(min_class_count, min(100, len(up_indices), len(down_indices), len(neutral_indices)))
        
        # Sample from each class with replacement if needed
        if target_count > 0:
            balanced_up_indices = np.random.choice(up_indices, size=target_count, replace=(len(up_indices) < target_count))
            balanced_neutral_indices = np.random.choice(neutral_indices, size=target_count, replace=(len(neutral_indices) < target_count))
            balanced_down_indices = np.random.choice(down_indices, size=target_count, replace=(len(down_indices) < target_count))
            
            # Combine indices and shuffle
            balanced_train_indices = np.concatenate([balanced_up_indices, balanced_neutral_indices, balanced_down_indices])
            np.random.shuffle(balanced_train_indices)
            
            # Replace original train indices with balanced indices
            train_indices = balanced_train_indices
            
            print(f"Balanced train class distribution: {target_count} samples per class, total {len(train_indices)} samples")
    
    # Create datasets
    train_dataset = EnhancedTradingDataset(
        features=window_data['features'][train_indices],
        trend_strengths=window_data['trend_strengths'][train_indices],
        profits=window_data['profits'][train_indices],
        regimes=window_data.get('regimes', None)[train_indices] if 'regimes' in window_data else None,
        subregimes=window_data.get('subregimes', None)[train_indices] if 'subregimes' in window_data else None,
        time_features=window_data.get('time_features', None)[train_indices] if 'time_features' in window_data else None
    )
    
    val_dataset = EnhancedTradingDataset(
        features=window_data['features'][val_indices],
        trend_strengths=window_data['trend_strengths'][val_indices],
        profits=window_data['profits'][val_indices],
        regimes=window_data.get('regimes', None)[val_indices] if 'regimes' in window_data else None,
        subregimes=window_data.get('subregimes', None)[val_indices] if 'subregimes' in window_data else None,
        time_features=window_data.get('time_features', None)[val_indices] if 'time_features' in window_data else None
    )
    
    test_dataset = EnhancedTradingDataset(
        features=window_data['features'][test_indices],
        trend_strengths=window_data['trend_strengths'][test_indices],
        profits=window_data['profits'][test_indices],
        regimes=window_data.get('regimes', None)[test_indices] if 'regimes' in window_data else None,
        subregimes=window_data.get('subregimes', None)[test_indices] if 'subregimes' in window_data else None,
        time_features=window_data.get('time_features', None)[test_indices] if 'time_features' in window_data else None
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch
    )
    
    # Create metadata for the loaders
    metadata = {
        'dataset_size': total_windows,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'feature_dim': window_data['features'][0].shape[-1] if len(window_data['features']) > 0 else 0,
        'window_size': window_size,
        'target_horizon': target_horizon,
        'signal_processing': {
            'window_sizes': signal_window_sizes
        },
        'balance_threshold': balance_threshold,
        'balanced_sampling': use_balanced_sampling
    }
    
    return train_loader, val_loader, test_loader, metadata


def create_sliding_windows(df: pd.DataFrame, window_size: int, 
                           feature_key: str = 'features',
                           trend_key: str = 'weighted_trend_strength',
                           regime_cols: List[str] = None, 
                           sub_regime_cols: List[str] = None,
                           regime_mapping: dict = None,
                           subregime_mapping: dict = None) -> Dict[str, np.ndarray]:
    """
    Create sliding windows for sequential data.
    
    Args:
        df: DataFrame containing data
        window_size: Size of each window
        feature_key: Column containing feature vectors
        trend_key: Column containing trend strength values
        regime_cols: Columns containing regime information
        sub_regime_cols: Columns containing sub-regime information
        regime_mapping: Dictionary mapping regime values to indices
        subregime_mapping: Dictionary mapping subregime values to indices
    
    Returns:
        Dictionary containing windowed data arrays
    """
    # Convert features from list of arrays to numpy array
    features = np.array(df[feature_key].tolist())
    
    # Get total samples and feature dimension
    n_samples = len(df) - window_size
    feature_dim = features.shape[1]
    
    # Initialize arrays to store windowed data
    windowed_features = np.zeros((n_samples, window_size, feature_dim), dtype=np.float32)
    trend_strengths = np.zeros(n_samples, dtype=np.float32)
    profits = np.zeros(n_samples, dtype=np.float32)
    
    # Create time features (relative position and hour of day)
    time_features = np.zeros((n_samples, window_size, 2), dtype=np.float32)
    
    # Create windows for regimes if available
    regimes = None
    subregimes = None
    
    if regime_cols and len(regime_cols) > 0:
        regimes = np.zeros((n_samples, window_size), dtype=np.int32)
    
    if sub_regime_cols and len(sub_regime_cols) > 0:
        subregimes = np.zeros((n_samples, window_size), dtype=np.int32)
    
    # Create sliding windows
    for i in range(n_samples):
        # Extract window
        window_slice = slice(i, i + window_size)
        
        # Features - combine window of input features
        for j in range(window_size):
            windowed_features[i, j] = features[i + j]
        
        # Target values (from the last point in the window)
        trend_strengths[i] = df[trend_key].iloc[i + window_size - 1]
        profits[i] = df['profit'].iloc[i + window_size - 1]
        
        # Time features
        # 1. Relative position in window (0 to 1)
        time_features[i, :, 0] = np.linspace(0, 1, window_size)
        
        # 2. Hour of day (if timestamp is available)
        if 'timestamp' in df.columns or 'datetime' in df.columns:
            timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'datetime'
            
            try:
                # Try to convert to datetime if it's not already
                timestamps = pd.to_datetime(df[timestamp_col].iloc[window_slice])
                
                # Extract hour and normalize to [0, 1]
                hours = timestamps.dt.hour + timestamps.dt.minute / 60.0
                time_features[i, :, 1] = hours / 24.0
            except:
                # If conversion fails, use zeros
                time_features[i, :, 1] = 0
        
        # Extract regime information if available
        if regime_cols and len(regime_cols) > 0:
            raw_regimes = df[regime_cols[0]].iloc[window_slice].values
            # Map regimes to indices if mapping is provided
            if regime_mapping is not None:
                for j, r in enumerate(raw_regimes):
                    regimes[i, j] = regime_mapping.get(r, 0)  # Default to 0 if not found
            else:
                regimes[i] = raw_regimes
        
        # Extract subregime information if available
        if sub_regime_cols and len(sub_regime_cols) > 0:
            raw_subregimes = df[sub_regime_cols[0]].iloc[window_slice].values
            # Map subregimes to indices if mapping is provided
            if subregime_mapping is not None:
                for j, sr in enumerate(raw_subregimes):
                    subregimes[i, j] = subregime_mapping.get(sr, 0)  # Default to 0 if not found
            else:
                subregimes[i] = raw_subregimes
    
    # Create output dictionary
    windowed_data = {
        'features': windowed_features,
        'trend_strengths': trend_strengths,
        'profits': profits,
        'time_features': time_features
    }
    
    if regimes is not None:
        windowed_data['regimes'] = regimes
    
    if subregimes is not None:
        windowed_data['subregimes'] = subregimes
    
    return windowed_data


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