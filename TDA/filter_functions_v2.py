import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Union, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional libraries
try:
    import pywt  # PyWavelets for wavelet-based filters
    WAVELETS_AVAILABLE = True
except ImportError:
    WAVELETS_AVAILABLE = False
    logger.warning("PyWavelets not installed. Wavelet-based filters will not be available.")

try:
    import umap  # UMAP for better dimensionality reduction
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not installed. Will use PCA for dimensionality reduction.")


class FinancialFeatures:
    """Extract financial-specific features from market data windows."""
    
    
    @staticmethod
    def wavelet_decomposition(series: np.ndarray, 
                            wavelet: str = 'db4', 
                            level: int = 3) -> List[np.ndarray]:
        """
        Perform wavelet decomposition of time series.
        
        Args:
            series: Time series data
            wavelet: Wavelet function
            level: Decomposition level
            
        Returns:
            List of wavelet coefficients [cA, cD_n, ..., cD_1]
        """
        if not WAVELETS_AVAILABLE:
            logger.warning("PyWavelets not available. Cannot perform wavelet decomposition.")
            return [series]
            
        # Ensure power of 2 length for better decomposition
        orig_len = len(series)
        power = int(np.ceil(np.log2(orig_len)))
        pad_len = 2**power
        
        padded_series = np.pad(series, (0, pad_len - orig_len), 'constant')
        
        try:
            coeffs = pywt.wavedec(padded_series, wavelet, level=level)
            return coeffs
        except Exception as e:
            logger.error(f"Error in wavelet decomposition: {e}")
            return [series]
    

class FinancialLensFactory:
    """Factory for creating financial-specific lens functions for TDA."""
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize with optional DataFrame.
        
        Args:
            df: Optional DataFrame with market data
        """
        self.df = df
        self.scaler = MinMaxScaler()
        self.feature_calculator = FinancialFeatures()
        self._validate_dataframe()
        
    def _validate_dataframe(self):
        """Validate that the dataframe has necessary financial data columns."""
        if self.df is None:
            logger.info("No DataFrame provided. Will work with direct window inputs.")
            return
            
        # Check for essential financial columns
        financial_columns = ['Value', 'Volatility']
        missing_columns = [col for col in financial_columns if col not in self.df.columns]
        
        if missing_columns:
            logger.warning(f"DataFrame missing important financial columns: {missing_columns}")
            logger.warning("Some financial lens functions may not work properly.")
    
    def create_windows(self, window_size: int, feature_columns: List[str]) -> List[np.ndarray]:
        """
        Create sliding windows from DataFrame.
        
        Args:
            window_size: Size of each window
            feature_columns: List of columns to include
            
        Returns:
            List of windows
        """
        if self.df is None:
            raise ValueError("No DataFrame provided")
        
        # Validate feature columns
        missing_cols = [col for col in feature_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
        # Extract features
        feature_data = self.df[feature_columns].values
        
        # Calculate number of windows
        n_samples = len(self.df)
        n_windows = n_samples - window_size + 1
        
        if n_windows <= 0:
            raise ValueError(f"Window size {window_size} larger than data length {n_samples}")
        
        # Create windows
        windows = []
        for i in range(n_windows):
            window = feature_data[i:i+window_size]
            windows.append(window)
        
        logger.info(f"Created {len(windows)} windows of size {window_size} with {len(feature_columns)} features")
        return windows


    def wavelet_lens(self, windows: List[np.ndarray], n_components: int = 2, 
                   wavelet: str = 'db4', max_level: int = 3) -> np.ndarray:
        """
        Wavelet-based lens optimized for financial regime detection with HDBSCAN.
        
        Args:
            windows: List of time series windows
            n_components: Number of components in output
            wavelet: Wavelet function to use
            max_level: Maximum decomposition level
            
        Returns:
            Array of shape (n_windows, n_components)
        """
        if not WAVELETS_AVAILABLE:
            logger.warning("PyWavelets not available. Using fallback approach.")
            # Create a simple fallback based on statistical features
            return self._statistical_lens_fallback(windows, n_components)
        
        n_windows = len(windows)
        if n_windows == 0:
            return np.array([])
        
        # Determine optimal wavelet for financial data
        if wavelet not in ['db4', 'sym4', 'db6', 'sym6']:
            logger.info(f"Optimizing wavelet choice: changing {wavelet} to db4 (optimal for financial data)")
            wavelet = 'db4'  # Consistently use db4 which works best for financial data
        
        # Preallocate features array for better performance
        # Use a fixed feature size that captures essential wavelet characteristics
        n_features = 8
        features_array = np.zeros((n_windows, n_features))
        
        # Process each window
        for i, window in enumerate(windows):
            # Extract price series (first column)
            if len(window.shape) > 1:
                prices = window[:, 0]
                # If volatility is available as second column, use it for weighting
                volatility = window[:, 1] if window.shape[1] > 1 else None
            else:
                prices = window
                volatility = None
            
            # Skip if not enough points
            if len(prices) < 4:
                continue
            
            try:
                # Determine appropriate level based on data length
                level = min(max_level, int(np.log2(len(prices))) - 1)
                level = max(1, level)  # Ensure at least level 1
                
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(prices, wavelet, level=level)
                
                # Extract optimized features for regime detection
                
                # 1. Energy distribution across scales - crucial for regime transitions
                energies = [np.sum(coeff**2) for coeff in coeffs]
                total_energy = sum(energies)
                
                if total_energy > 0:
                    # Feature 1-3: Energy distribution across first 3 levels (normalized)
                    for j in range(min(3, len(energies))):
                        features_array[i, j] = energies[j] / total_energy
                    
                    # Feature 4: Low/high frequency ratio - key for regime identification
                    low_energy = energies[0]  # Approximation coefficients
                    high_energy = sum(energies[1:]) if len(energies) > 1 else 1.0
                    features_array[i, 3] = min(10.0, low_energy / (high_energy + 1e-10))
                    
                    # Feature 5: Wavelet entropy - measures randomness/disorder
                    p = np.array([e / total_energy for e in energies])
                    entropy = -np.sum(p * np.log2(p + 1e-10))
                    features_array[i, 4] = entropy / np.log2(len(energies) + 1e-10)  # Normalized
                    
                    # Features 6-7: Detail coefficients sparsity - captures jumps/regime shifts
                    for j in range(min(2, len(coeffs)-1)):
                        if len(coeffs[j+1]) > 0:
                            # Normalized count of significant coefficients
                            threshold = np.std(coeffs[j+1]) * 0.2
                            sparsity = np.sum(np.abs(coeffs[j+1]) > threshold) / len(coeffs[j+1])
                            features_array[i, 5+j] = sparsity
                    
                    # Feature 8: Temporal persistence - helps with regime stability
                    if volatility is not None:
                        # Use volatility as a regime persistence indicator
                        features_array[i, 7] = np.mean(volatility) / (np.std(volatility) + 1e-10)
                    else:
                        # Calculate persistence using autocorrelation
                        diff = np.diff(prices)
                        if len(diff) > 1:
                            autocorr = np.correlate(diff[:-1], diff[1:], mode='valid')[0] / (np.var(diff) * len(diff))
                            features_array[i, 7] = np.abs(autocorr)
            
            except Exception as e:
                logger.warning(f"Error in wavelet decomposition: {str(e)[:100]}...")
                # Leave as zeros for this window
        
        # Handle NaN/Inf values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=10.0, neginf=0.0)
        
        # Scale features to [0,1] range for better HDBSCAN performance
        # Wasserstein distance works better with normalized data
        scaled_features = self.scaler.fit_transform(features_array)
        
        # Reduce dimensionality to specified components
        if n_components < scaled_features.shape[1]:
            try:
                # PCA works better with HDBSCAN than UMAP for this specific application
                pca = PCA(n_components=n_components, random_state=42)
                reduced_features = pca.fit_transform(scaled_features)
                
                # Log explained variance for diagnostics
                explained_var = np.sum(pca.explained_variance_ratio_)
                logger.info(f"Wavelet lens PCA explained variance: {explained_var:.2f}")
                
                return reduced_features
            except Exception as e:
                logger.error(f"Error in dimensionality reduction: {e}")
                # If reduction fails, return subset of features
                return scaled_features[:, :n_components]
        
        return scaled_features[:, :n_components]
    
    def _statistical_lens_fallback(self, windows: List[np.ndarray], n_components: int = 2) -> np.ndarray:
        """
        Fallback lens using statistical features when wavelets are not available.
        
        Args:
            windows: List of time series windows
            n_components: Number of components in output
            
        Returns:
            Array of shape (n_windows, n_components)
        """
        n_windows = len(windows)
        if n_windows == 0:
            return np.array([])
        
        # Use 5 statistical features
        features_array = np.zeros((n_windows, 5))
        
        for i, window in enumerate(windows):
            # Extract price series (first column)
            if len(window.shape) > 1:
                prices = window[:, 0]
            else:
                prices = window
                
            if len(prices) < 2:
                continue
                
            # Calculate statistical features
            features_array[i, 0] = np.mean(prices)
            features_array[i, 1] = np.std(prices)
            features_array[i, 2] = np.median(prices)
            
            # Calculate returns
            returns = np.diff(prices) / (prices[:-1] + 1e-10)
            if len(returns) > 0:
                features_array[i, 3] = np.mean(np.abs(returns))
                features_array[i, 4] = np.std(returns)
        
        # Handle NaN/Inf values
        features_array = np.nan_to_num(features_array)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features_array)
        
        # Reduce dimensionality if needed
        if n_components < scaled_features.shape[1]:
            pca = PCA(n_components=n_components)
            return pca.fit_transform(scaled_features)
        
        return scaled_features[:, :n_components]


    def create_financial_lens(self, 
                            windows: List[np.ndarray] = None,
                            lens_type: str = 'comprehensive',
                            window_size: int = 50,
                            feature_columns: List[str] = None,
                            n_components: int = 2,
                            **kwargs) -> np.ndarray:
        """
        Create financial lens projection for TDA.
        
        Args:
            windows: List of time series windows (optional)
            lens_type: Type of lens ('volatility', 'wavelet', 'comprehensive')
            window_size: Size of sliding windows (if windows not provided)
            feature_columns: Feature columns to use (if windows not provided)
            n_components: Number of components in lens
            **kwargs: Additional parameters for specific lens functions
            
        Returns:
            Lens projection of shape (n_windows, n_components)
        """
        
        # Apply selected lens function
        if lens_type == 'wavelet':
            return self.wavelet_lens(
                windows, 
                n_components, 
                wavelet=kwargs.get('wavelet', 'db4'),
                max_level=kwargs.get('max_level', 3)
            )
        else:
            logger.warning(f"Unknown lens type: {lens_type}, using comprehensive lens")
            return self.comprehensive_financial_lens(windows, n_components) 