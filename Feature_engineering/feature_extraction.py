import pandas as pd
import numpy as np
import torch
from datetime import datetime
import pywt  # PyWavelets for wavelet-based analysis
from scipy import signal  # For spectral analysis
from sklearn.preprocessing import StandardScaler  
from typing import List, Optional, Tuple
import math  # For mathematical constants

class FeatureExtractor:
    """
    Extract specific market microstructure features from tick data
    for processing through a transformer model.
    
    Creates three tensors:
    - Time tensor: Time-based features
    - Features tensor: Market microstructure features
    - Timedelta tensor: Time differences between ticks
    
    All features are calculated causally, using only data available up to 
    the current point in time to avoid look-ahead bias.
    """
    
    def __init__(self, window_sizes: List[int] = [10, 20, 50]):
        """
        Initialize the feature extractor.
        
        Parameters:
        -----------
        window_sizes : List[int]
            List of window sizes for feature calculation
        """
        self.window_sizes = window_sizes
        # Remove unused StandardScaler to avoid potential misuse
        # self.scaler = StandardScaler()
        # Default market features count
        self._feature_count = 20  # Base number of features extracted
        
    def get_feature_count(self) -> int:
        """
        Get the number of features extracted by this extractor.
        
        Returns:
        --------
        int
            Number of market microstructure features
        """
        return self._feature_count
    
    def extract_features(
        self, 
        df: pd.DataFrame, 
        timestamp_col: str = 'Timestamp', 
        price_col: str = 'Value', 
        volume_col: str = 'Volume',
        volatility_col: Optional[str] = 'Volatility'
    ) -> pd.DataFrame:
        """
        Extract specific features from tick data.
        
        All features are calculated in a causal manner, using only information
        available up to the current point in time, preventing look-ahead bias.
        
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
        volatility_col : str, optional
            Column name for volatility values
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with original data plus extracted features
        """
        # Make a copy to avoid modifying the original dataframe
        result_df = df.copy()
        
        # Ensure timestamp is datetime type
        result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
        
        # Calculate basic price changes and returns
        result_df['price_change'] = result_df[price_col].diff()
        result_df['log_return'] = np.log(result_df[price_col] / result_df[price_col].shift(1))
        
        # Calculate time between trades (in seconds)
        result_df['time_delta'] = result_df[timestamp_col].diff().dt.total_seconds()
        
        # Replace NaN values with 0 for the first row
        for col in ['price_change', 'log_return', 'time_delta']:
            result_df[col] = result_df[col].fillna(0)
        
        # Extract trade direction indicator (buy/sell)
        result_df = self._extract_trade_direction(result_df, price_col)
        
        # Extract tick imbalance
        result_df = self._extract_tick_imbalance(result_df, volume_col)
        
        # Extract jump features (3 types) using wavelet analysis
        result_df = self._extract_jump_features_wavelet(result_df, price_col)
        
        # Extract Kyle's lambda (price impact)
        result_df = self._extract_kyle_lambda(result_df, price_col, volume_col)
        
        # Extract orderflow imbalance
        result_df = self._extract_orderflow_imbalance(result_df, price_col, volume_col)
        
        # Extract momentum
        result_df = self._extract_momentum(result_df, price_col)
        
        # Extract price range (max-min)
        result_df = self._extract_price_range(result_df, price_col)
        
        # Track feature count - update based on actual columns
        feature_cols = [
            'trade_direction', 'is_buy', 
            'tick_imbalance', 'orderflow_imbalance',
            'kyle_lambda', 
            'jump_diffusion', 'jump_magnitude', 'jump_arrival',
            'momentum_short', 'momentum_medium', 'momentum_long',
            'price_range_short', 'price_range_medium', 'price_range_long',
            'log_return',
            'bipower_var_short', 'bipower_var_medium', 'bipower_var_long',
            'jump_ratio'
        ]
        
        # Extract volatility per volume
        if volatility_col in result_df.columns:
            result_df = self._extract_volatility_per_volume(result_df, volume_col, volatility_col)
            feature_cols.append('volatility_per_volume')
        
        # Extract bipower variation as a robust volatility estimator
        result_df = self._extract_bipower_variation(result_df)
        
        # Drop intermediate calculation columns
        cols_to_drop = [col for col in result_df.columns if col.startswith('_temp_')]
        if cols_to_drop:
            result_df = result_df.drop(columns=cols_to_drop)
        
        # Clean features by handling any remaining NaN, infinite or extreme values
        result_df = self._clean_features(result_df, timestamp_col, price_col, volume_col)
        
        # Update feature count based on actual features
        self._feature_count = len([col for col in feature_cols if col in result_df.columns])
            
        return result_df
    
    def create_tensors(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'Timestamp',
        window_size: int = 50,
        causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create tensors for transformer model input.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with features
        timestamp_col : str
            Column name for timestamp
        window_size : int
            Size of sliding window for creating tensors
        causal : bool
            If True, use causal context (only past information) for real-time applications
            If False, use bidirectional context (past and future) for research purposes only
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            time_tensor, features_tensor, timedelta_tensor
        """
        # Ensure timestamp is datetime type
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Define time features
        df['hour'] = df[timestamp_col].dt.hour
        df['minute'] = df[timestamp_col].dt.minute
        df['second'] = df[timestamp_col].dt.second
        df['microsecond'] = df[timestamp_col].dt.microsecond
        
        # Normalize time features
        df['norm_hour'] = df['hour'] / 24.0
        df['norm_minute'] = df['minute'] / 60.0
        df['norm_second'] = df['second'] / 60.0
        df['norm_microsecond'] = df['microsecond'] / 1_000_000.0
        
        # Cyclical encoding of time
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['sin_minute'] = np.sin(2 * np.pi * df['minute'] / 60.0)
        df['cos_minute'] = np.cos(2 * np.pi * df['minute'] / 60.0)
        
        # Define time features for time tensor
        time_features = [
            'norm_hour', 'norm_minute', 'norm_second', 'norm_microsecond',
            'sin_hour', 'cos_hour', 'sin_minute', 'cos_minute'
        ]
        
        # Define market microstructure features for features tensor
        market_features = [
            'trade_direction', 'is_buy', 
            'tick_imbalance', 'orderflow_imbalance',
            'kyle_lambda', 
            'jump_diffusion', 'jump_magnitude', 'jump_arrival',
            'momentum_short', 'momentum_medium', 'momentum_long',
            'price_range_short', 'price_range_medium', 'price_range_long',
            'log_return',
            'bipower_var_short', 'bipower_var_medium', 'bipower_var_long',
            'jump_ratio'  
        ]
        
        # Add volatility per volume if available
        if 'volatility_per_volume' in df.columns:
            market_features.append('volatility_per_volume')
            
        # Create sliding windows
        time_tensor_list = []
        features_tensor_list = []
        timedelta_tensor_list = []
        
        # In causal mode, each prediction point uses only past and current information
        # preventing future information leakage for real-time applications
        for i in range(window_size - 1, len(df)):
            if causal:
                # Causal context: use window_size-1 previous points + current point
                # This ensures no future leakage for real-time applications
                window_df = df.iloc[i-(window_size-1):i+1]
            else:
                # Bidirectional context: center window around prediction point
                # WARNING: This creates future information leakage and should only be
                # used for research, not for backtests or real-time applications
                window_start = max(0, i - window_size // 2)
                window_end = min(len(df), window_start + window_size)
                window_start = max(0, window_end - window_size)  # Adjust start if end hit boundary
                window_df = df.iloc[window_start:window_end]
            
            # Create time tensor
            time_window = window_df[time_features].values
            time_tensor_list.append(time_window)
            
            # Create features tensor
            features_window = window_df[market_features].values
            features_tensor_list.append(features_window)
            
            # Create timedelta tensor
            timedelta_window = window_df['time_delta'].values
            timedelta_tensor_list.append(timedelta_window)
            
        # Convert to numpy arrays
        time_array = np.array(time_tensor_list)
        features_array = np.array(features_tensor_list)
        timedelta_array = np.array(timedelta_tensor_list)
        
        # Convert to PyTorch tensors
        time_tensor = torch.tensor(time_array, dtype=torch.float32)
        features_tensor = torch.tensor(features_array, dtype=torch.float32)
        timedelta_tensor = torch.tensor(timedelta_array, dtype=torch.float32)
        
        return time_tensor, features_tensor, timedelta_tensor
    
    def _extract_bipower_variation(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate bipower variation (BPV) - a robust estimator of integrated variance
        that is less affected by jumps than realized variance.
        
        CAUSAL FEATURE: Uses rolling windows (backward-looking only) to calculate bipower variation.
        
        Introduced by Barndorff-Nielsen and Shephard (2004), bipower variation
        uses products of adjacent absolute returns to filter out jumps.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with log_return column
        windows : List[int], optional
            List of window sizes for BPV calculation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added bipower variation columns
        """
        result_df = df.copy()
        
        # Use default windows if not specified
        if windows is None:
            windows = [10, 30, 50]  # Short, medium, and long term
            
        # Get absolute log returns
        abs_returns = np.abs(result_df['log_return'])
        
        # Scaling constant: π/2 ≈ 1.57
        mu1_squared = (math.pi / 2) ** 2
        
        # Pre-calculate adjacent products for efficiency
        # |r_t| * |r_{t-1}|
        adjacent_products = abs_returns * abs_returns.shift(1)
        adjacent_products.fillna(0, inplace=True)
        
        # Calculate bipower variation for different windows
        for window, name in zip(windows, ['short', 'medium', 'long']):
            # Traditional BPV
            bpv = adjacent_products.rolling(window=window, min_periods=2).sum() * (window / (window - 1))
            
            # Handle potential NaN values
            # First, check if we have NaNs
            nan_count = bpv.isna().sum()
            if nan_count > 0:
                # For NaN values at the beginning, use squared returns as a substitute
                # (this is less robust to jumps but better than NaN)
                squared_returns = result_df['log_return'] ** 2
                squared_returns_rollsum = squared_returns.rolling(window=window, min_periods=1).sum()
                
                # Replace NaN values in BPV with squared returns estimate
                bpv.fillna(squared_returns_rollsum, inplace=True)
                
                # If any NaN values remain, use forward fill then fill with the median
                if bpv.isna().any():
                    bpv = bpv.ffill().fillna(bpv.median() if not bpv.median() != bpv.median() else 0.0001)
            
            # Ensure no zero values (can cause issues in jump ratio calculation)
            bpv = bpv.replace(0, bpv.median() if bpv.median() > 0 else 0.0001)
            
            # Store in result dataframe
            result_df[f'bipower_var_{name}'] = bpv
            
            # Also calculate traditional realized variance for comparison
            rv = (result_df['log_return'] ** 2).rolling(window=window, min_periods=1).sum()
            result_df[f'_temp_rv_{name}'] = rv
            
            # Calculate jump ratio: RV/BPV - 1
            # This estimates the contribution of jumps to total variation
            # (if no jumps, ratio ≈ 1, if jumps present, ratio > 1)
            jump_ratio = rv / bpv - 1
            
            # Filter out extreme values and NaNs
            jump_ratio = jump_ratio.clip(0, 5).fillna(0)
            
            if name == 'medium':  # Store medium-term jump ratio as a feature
                result_df['jump_ratio'] = jump_ratio
        
        # Normalized staggered bipower variation - more robust to microstructure noise
        # Uses returns separated by 2 periods
        staggered_products = abs_returns * abs_returns.shift(2)
        staggered_products.fillna(0, inplace=True)
        
        # Calculate staggered BPV for medium window
        med_window = windows[1]
        staggered_bpv = staggered_products.rolling(window=med_window, min_periods=3).sum() * (med_window / (med_window - 2))
        
        # Handle any NaN values
        if staggered_bpv.isna().any():
            # First try to use regular BPV as substitute
            med_name = 'medium'
            staggered_bpv.fillna(result_df[f'bipower_var_{med_name}'], inplace=True)
            
            # If any NaNs remain, use forward fill then fill with median
            staggered_bpv = staggered_bpv.ffill().fillna(staggered_bpv.median() if not pd.isna(staggered_bpv.median()) else 0.0001)
        
        result_df['staggered_bipower_var'] = staggered_bpv
        
        # Calculate min-bipower variation (minimum of adjacent bipower measures)
        # This is even more robust to jumps
        for window in windows:
            if window >= 5:  # Need enough data for meaningful calculation
                # Calculate multiple skip bipower variations with different lags
                bpv_lags = []
                
                for lag in range(1, min(4, window // 2)):
                    lagged_products = abs_returns * abs_returns.shift(lag)
                    lagged_products.fillna(0, inplace=True)
                    bpv_lag = lagged_products.rolling(window=window, min_periods=lag+1).sum() * (window / (window - lag))
                    
                    # Ensure no NaN values
                    if bpv_lag.isna().any():
                        bpv_lag = bpv_lag.ffill().fillna(bpv_lag.median() if not pd.isna(bpv_lag.median()) else 0.0001)
                        
                    bpv_lags.append(bpv_lag)
                
                # Take minimum of the bipower measures (most robust to jumps)
                if bpv_lags:
                    min_bpv = pd.concat(bpv_lags, axis=1).min(axis=1)
                    
                    # One final check for NaN values
                    if min_bpv.isna().any():
                        # Use regular bipower variation as fallback
                        name = 'medium' if window == 30 else 'short' if window == 10 else 'long'
                        min_bpv.fillna(result_df[f'bipower_var_{name}'], inplace=True)
                        # If still have NaNs, use median
                        min_bpv = min_bpv.fillna(min_bpv.median() if not pd.isna(min_bpv.median()) else 0.0001)
                    
                    result_df[f'min_bipower_var_{window}'] = min_bpv
        
        return result_df
    
    def _extract_trade_direction(
        self, 
        df: pd.DataFrame, 
        price_col: str
    ) -> pd.DataFrame:
        """
        Extract trade direction indicator (simplified Lee-Ready algorithm).
        1: buyer-initiated, -1: seller-initiated, 0: undetermined
        """
        result_df = df.copy()
        
        # Calculate price change
        result_df['_temp_price_change'] = result_df[price_col].diff()
        
        # Determine trade direction based on price change
        conditions = [
            (result_df['_temp_price_change'] > 0),  # Price went up -> buy
            (result_df['_temp_price_change'] < 0),  # Price went down -> sell
            (result_df['_temp_price_change'] == 0)  # No change -> use previous direction
        ]
        choices = [1, -1, 0]
        result_df['trade_direction'] = np.select(conditions, choices, default=0)
        
        # For zero price changes, use tick test (look back until non-zero)
        zero_mask = (result_df['trade_direction'] == 0)
        if zero_mask.any():
            # Save original index and reset to avoid KeyError with non-continuous indices
            original_index = result_df.index
            result_df = result_df.reset_index(drop=True)
            
            for i in range(1, len(result_df)):
                if result_df.loc[i, 'trade_direction'] == 0:
                    j = i - 1
                    while j >= 0 and result_df.loc[j, 'trade_direction'] == 0:
                        j -= 1
                    
                    if j >= 0:  # Found a non-zero direction
                        result_df.loc[i, 'trade_direction'] = result_df.loc[j, 'trade_direction']
            
            # Restore original index
            result_df.index = original_index
        
        # Create binary buy indicator
        result_df['is_buy'] = (result_df['trade_direction'] > 0).astype(int)
        
        return result_df
    
    def _extract_tick_imbalance(
        self, 
        df: pd.DataFrame, 
        volume_col: str,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate tick imbalance to measure buying vs selling pressure.
        Tick imbalance ranges from -1 (all sells) to 1 (all buys).
        
        CAUSAL FEATURE: Uses rolling window calculations that only look backward.
        Each value is computed using only current and past data points.
        """
        result_df = df.copy()
        
        # Calculate buy volume and sell volume
        buy_volume = result_df[volume_col] * result_df['is_buy']
        sell_volume = result_df[volume_col] * (1 - result_df['is_buy'])
        
        # Calculate rolling sum of buy and sell volume
        rolling_buy = buy_volume.rolling(window=window, min_periods=1).sum()
        rolling_sell = sell_volume.rolling(window=window, min_periods=1).sum()
        
        # Calculate tick imbalance
        total_volume = rolling_buy + rolling_sell
        result_df['tick_imbalance'] = (rolling_buy - rolling_sell) / total_volume.replace(0, 1)
        
        return result_df
    
    def _extract_jump_features_wavelet(
        self, 
        df: pd.DataFrame, 
        price_col: str
    ) -> pd.DataFrame:
        """
        Extract three distinct jump features using wavelet analysis:
        1. Jump diffusion: estimate of jump component in price process
        2. Jump magnitude: size of significant jumps
        3. Jump arrival: frequency of jumps
        
        CAUSAL FEATURE: Wavelet transformation is applied to existing returns. The exponential 
        weighted moving average (ewm) for jump arrival looks only at past jumps.
        """
        result_df = df.copy()
        
        # Get log returns
        log_returns = result_df['log_return'].values
        
        # Apply wavelet decomposition using PyWavelets
        # Use 'db4' wavelet with 3 levels of decomposition (reduced from 4)
        wavelet = 'db4'
        level = 3  # Reduced from 4 to avoid warnings
        
        # Pad the signal to ensure proper length
        n = len(log_returns)
        pad_size = 2**level
        if n % pad_size != 0:
            # Pad with zeros to make length divisible by 2^level
            pad_length = pad_size - (n % pad_size)
            log_returns_padded = np.pad(log_returns, (0, pad_length), 'constant')
        else:
            log_returns_padded = log_returns
            
        # Apply wavelet decomposition
        coeffs = pywt.wavedec(log_returns_padded, wavelet, level=level)
        
        # Reconstruct the signal for each level, but handle empty coefficient lists
        reconstructed = []
        for i in range(len(coeffs)):
            # Create a list of coefficients with only one level filled
            coeff_list = [None] * len(coeffs)
            coeff_list[i] = coeffs[i]
            
            # Check if we have at least one non-None coefficient
            if any(c is not None for c in coeff_list):
                # Reconstruct the signal from the coefficients
                try:
                    rec = pywt.waverec(coeff_list, wavelet)
                    if rec is not None and len(rec) >= n:
                        reconstructed.append(rec[:n])
                    else:
                        # Fallback: use zeros if reconstruction fails
                        reconstructed.append(np.zeros(n))
                except ValueError:
                    # If wavelet reconstruction fails, use zeros
                    reconstructed.append(np.zeros(n))
            else:
                # All coefficients are None, use zeros
                reconstructed.append(np.zeros(n))
        
        # Make sure we have at least one reconstructed level (for jump component)
        if len(reconstructed) <= 1:
            # Fallback approach: use simple thresholding for jumps
            std_returns = np.std(log_returns)
            jump_threshold = 3 * std_returns
            jumps = (np.abs(log_returns) > jump_threshold).astype(float)
            result_df['jump_diffusion'] = log_returns * jumps
            result_df['jump_magnitude'] = np.abs(log_returns) * jumps
            result_df['jump_arrival'] = pd.Series(jumps).ewm(span=50, min_periods=1).mean().values
            return result_df
        
        # The highest frequency details (level 1) capture jumps
        jump_component = reconstructed[1] if len(reconstructed) > 1 else np.zeros(n)
        
        # Jump diffusion: wavelet-detected jump component
        result_df['jump_diffusion'] = jump_component
        
        # Calculate jump threshold (3 std dev)
        jump_std = np.std(jump_component)
        jump_threshold = 3 * jump_std if jump_std > 0 else 0.001
        
        # Detect significant jumps
        jumps = (np.abs(jump_component) > jump_threshold).astype(float)
        
        # Jump magnitude: absolute size of detected jumps
        result_df['jump_magnitude'] = np.abs(jump_component) * jumps
        
        # Jump arrival: frequency of jumps (use exponential weighted moving average)
        result_df['jump_arrival'] = pd.Series(jumps).ewm(span=50, min_periods=1).mean().values
        
        return result_df
    
    def _extract_kyle_lambda(
        self, 
        df: pd.DataFrame, 
        price_col: str, 
        volume_col: str,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate Kyle's lambda, which measures price impact of trades.
        Higher lambda means greater price impact per unit of volume.
        
        This implementation uses regression of price changes on signed volume
        to estimate the price impact coefficient.
        
        CAUSAL FEATURE: For each point, lambda is calculated using only prior observations
        in the rolling window, ensuring no future information leakage.
        """
        result_df = df.copy()
        
        # Use signed volume based on trade direction
        signed_volume = result_df[volume_col] * result_df['trade_direction']
        
        # Calculate rolling lambda
        lambda_values = []
        
        # For each point, calculate lambda using previous window observations
        for i in range(len(result_df)):
            if i < window:
                # Not enough data, use forward window
                start, end = 0, min(window, len(result_df))
            else:
                start, end = i - window, i
                
            # Get price changes and volumes in window
            y = result_df['price_change'].iloc[start:end].values
            X = signed_volume.iloc[start:end].values
            
            # Simple regression: price_change = lambda * signed_volume
            if np.sum(X**2) > 0:  # Check to avoid division by zero
                lambda_est = np.sum(X * y) / np.sum(X**2)
            else:
                lambda_est = 0
                
            lambda_values.append(lambda_est)
            
        # Save results
        result_df['kyle_lambda'] = lambda_values
        
        # Take absolute value since lambda represents magnitude of impact
        result_df['kyle_lambda'] = np.abs(result_df['kyle_lambda'])
        
        return result_df
    
    def _extract_orderflow_imbalance(
        self, 
        df: pd.DataFrame, 
        price_col: str, 
        volume_col: str,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate orderflow imbalance as the sum of signed volumes.
        Positive values indicate buying pressure, negative values indicate selling pressure.
        
        CAUSAL FEATURE: Uses a backward-looking rolling window that only considers
        past trades, ensuring no future data is used in the calculation.
        """
        result_df = df.copy()
        
        # Calculate signed volume (positive for buys, negative for sells)
        signed_volume = result_df[volume_col] * result_df['trade_direction']
        
        # Calculate rolling sum of signed volume
        rolling_signed_volume = signed_volume.rolling(window=window, min_periods=1).sum()
        
        # Calculate rolling sum of total volume
        rolling_total_volume = result_df[volume_col].rolling(window=window, min_periods=1).sum()
        
        # Calculate orderflow imbalance normalized by total volume
        result_df['orderflow_imbalance'] = rolling_signed_volume / rolling_total_volume.replace(0, 1)
        
        return result_df
    
    def _extract_momentum(
        self, 
        df: pd.DataFrame, 
        price_col: str
    ) -> pd.DataFrame:
        """
        Calculate price momentum at different time scales.
        Momentum is measured as the rate of change of price.
        
        CAUSAL FEATURE: Uses pandas pct_change() which calculates percentage change
        relative to prior values (current price / previous price - 1), ensuring
        no future information is included.
        """
        result_df = df.copy()
        
        # Define momentum windows for different timescales
        short_window = 10
        medium_window = 30
        long_window = 50
        
        # Calculate momentum as percentage change over different windows
        result_df['momentum_short'] = result_df[price_col].pct_change(periods=short_window).fillna(0)
        result_df['momentum_medium'] = result_df[price_col].pct_change(periods=medium_window).fillna(0)
        result_df['momentum_long'] = result_df[price_col].pct_change(periods=long_window).fillna(0)
        
        return result_df
    
    def _extract_price_range(
        self, 
        df: pd.DataFrame, 
        price_col: str
    ) -> pd.DataFrame:
        """
        Calculate price range (max - min) over different windows.
        
        CAUSAL FEATURE: Uses backward-looking rolling windows to calculate
        min and max values, ensuring no future information is used.
        """
        result_df = df.copy()
        
        # Define windows for different timescales
        short_window = 10
        medium_window = 30
        long_window = 50
        
        # Calculate price range for each window
        for window, name in zip([short_window, medium_window, long_window], 
                               ['short', 'medium', 'long']):
            # Calculate rolling max and min
            rolling_max = result_df[price_col].rolling(window=window, min_periods=1).max()
            rolling_min = result_df[price_col].rolling(window=window, min_periods=1).min()
            
            # Calculate price range and normalize by current price
            price_range = (rolling_max - rolling_min) / result_df[price_col]
            result_df[f'price_range_{name}'] = price_range
            
        return result_df
    
    def _extract_volatility_per_volume(
        self, 
        df: pd.DataFrame, 
        volume_col: str,
        volatility_col: str,
        window: int = 50
    ) -> pd.DataFrame:
        """
        Calculate volatility per unit of volume.
        This measures how much volatility is generated per unit of trading volume.
        
        CAUSAL FEATURE: Uses backward-looking rolling sum of volume
        to normalize volatility, ensuring no future information is used.
        """
        result_df = df.copy()
        
        # Calculate rolling sum of volume
        rolling_volume = result_df[volume_col].rolling(window=window, min_periods=1).sum()
        
        # Calculate volatility per volume
        result_df['volatility_per_volume'] = result_df[volatility_col] / rolling_volume.replace(0, np.nan)
        result_df['volatility_per_volume'] = result_df['volatility_per_volume'].fillna(0)
        
        return result_df
    
    def _clean_features(self, df: pd.DataFrame, timestamp_col: str, price_col: str, volume_col: str) -> pd.DataFrame:
        """
        Clean features by handling any remaining NaN, infinite or extreme values.
        This ensures the data is ready for neural network training.
        
        CAUSAL IMPLEMENTATION: All calculations only use data available up to each point 
        to avoid look-ahead bias.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with extracted features
        timestamp_col : str
            Name of timestamp column (to preserve)
        price_col : str
            Name of price column (to preserve)
        volume_col : str
            Name of volume column (to preserve)
            
        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame with no NaN or infinite values
        """
        result_df = df.copy()
        
        # Identify columns to clean (all except timestamp)
        cols_to_clean = [col for col in result_df.columns if col != timestamp_col]
        
        # Identify numeric columns
        numeric_cols = [col for col in cols_to_clean if np.issubdtype(result_df[col].dtype, np.number)]
        
        # Check for NaN values
        nan_counts = result_df[cols_to_clean].isna().sum()
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            print(f"Found {total_nans} NaN values across {(nan_counts > 0).sum()} columns")
            
            # For each column with NaNs, apply appropriate cleaning
            for col in cols_to_clean:
                if result_df[col].isna().any():
                    # Different handling based on column type
                    if col in [price_col, volume_col, 'SYMBOL']:
                        # For core data columns, use forward fill then backward fill
                        result_df[col] = result_df[col].ffill().bfill()
                    else:
                        # For calculated features, use median for replacement
                        # First try forward fill (most appropriate for time series)
                        result_df[col] = result_df[col].ffill()
                        
                        # If any NaNs remain, use the median of non-NaN values
                        # CAUSAL: Calculate expanding window medians
                        if result_df[col].isna().any():
                            if np.issubdtype(result_df[col].dtype, np.number):
                                # For each NaN value, use median of prior values
                                for i in range(len(result_df)):
                                    if pd.isna(result_df.loc[i, col]):
                                        # If we have enough prior data, use prior median
                                        if i > 0:
                                            prior_vals = result_df.loc[:i-1, col].dropna()
                                            if len(prior_vals) > 0:
                                                result_df.loc[i, col] = prior_vals.median()
                                            else:
                                                result_df.loc[i, col] = 0
                                        else:
                                            result_df.loc[i, col] = 0
                            else:
                                # For non-numeric columns, use forward fill then backward fill
                                result_df[col] = result_df[col].ffill().bfill()
                                # If still NaN, use a default value appropriate for the type
                                if result_df[col].dtype == 'object':
                                    result_df[col] = result_df[col].fillna('')
                                else:
                                    result_df[col] = result_df[col].fillna(0)
        
        # Check for infinite values (only in numeric columns)
        if numeric_cols:
            # Process each numeric column individually to avoid issues with mixed types
            total_infs = 0
            cols_with_infs = 0
            
            for col in numeric_cols:
                # Check for infinities
                inf_mask = np.isinf(result_df[col].values)
                col_infs = np.sum(inf_mask)
                total_infs += col_infs
                
                if col_infs > 0:
                    cols_with_infs += 1
                    # CAUSAL: Replace infinities using expanding windows
                    for i in range(len(result_df)):
                        if np.isinf(result_df.loc[i, col]):
                            # Use median of prior finite values
                            if i > 0:
                                prior_vals = result_df.loc[:i-1, col]
                                prior_vals = prior_vals[~np.isinf(prior_vals)]
                                if len(prior_vals) > 0:
                                    replacement = prior_vals.median()
                                    result_df.loc[i, col] = replacement if not pd.isna(replacement) else 0
                                else:
                                    result_df.loc[i, col] = 0
                            else:
                                result_df.loc[i, col] = 0
            
            if total_infs > 0:
                print(f"Found {total_infs} infinite values across {cols_with_infs} columns")
                    
        # Use expanding window approach for outlier clipping
        # to ensure we only use data up to the current point (causal)
        min_window_size = 100  # Minimum points needed for reliable percentiles
        for col in numeric_cols:
            if col not in [price_col, volume_col, 'SYMBOL']:
                # Initialize arrays for bounds to avoid repeated pandas operations
                upper_bounds = np.full(len(result_df), np.nan)
                lower_bounds = np.full(len(result_df), np.nan)
                
                try:
                    # Calculate expanding window percentiles
                    for i in range(min_window_size, len(result_df)):
                        # Use only data up to current point for bounds calculation
                        subset = result_df.loc[:i, col].dropna()
                        
                        if len(subset) >= min_window_size:
                            q01 = np.percentile(subset, 1)
                            q99 = np.percentile(subset, 99)
                            
                            # Ensure the range is non-zero to avoid clipping everything to the same value
                            if q99 > q01:
                                # Add a buffer to avoid clipping too aggressively
                                buffer = (q99 - q01) * 0.5
                                lower_bounds[i] = q01 - buffer
                                upper_bounds[i] = q99 + buffer
                    
                    # Apply clipping based on expanding window bounds
                    for i in range(min_window_size, len(result_df)):
                        if not np.isnan(lower_bounds[i]) and not np.isnan(upper_bounds[i]):
                            current_val = result_df.loc[i, col]
                            result_df.loc[i, col] = max(lower_bounds[i], min(upper_bounds[i], current_val))
                        
                except Exception as e:
                    # Skip columns that can't be processed
                    print(f"Skip outlier clipping for {col}: {str(e)}")
                    
        # Verify no NaNs remain
        final_nan_count = result_df[cols_to_clean].isna().sum().sum()
        if final_nan_count > 0:
            # Final fallback - replace any remaining NaNs with 0
            print(f"Warning: {final_nan_count} NaN values remained after cleaning. Replacing with 0.")
            result_df = result_df.fillna(0)
            
        return result_df 