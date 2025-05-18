import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Union, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import logging
import time
import warnings
from scipy import stats
from statsmodels.nonparametric.kernel_regression import KernelReg
import scipy
from sklearn.manifold import TSNE
import numpy.typing as npt
import os

# Import shared logging configuration - try both relative and absolute imports
try:
    from .distance_metrics_v2 import SilentFilter
except ImportError:
    try:
        from distance_metrics_v2 import SilentFilter
    except ImportError:
        # Create a minimal SilentFilter implementation if import fails
        class SilentFilter(logging.Filter):
            def filter(self, record):
                return False

# Configure module logger 
logger = logging.getLogger(__name__)

# Apply SilentFilter to completely silence this module's logger
class ModuleSilentFilter(SilentFilter):
    pass

logger.addFilter(ModuleSilentFilter())
logger.setLevel(logging.ERROR)

# Force silence warnings
warnings.filterwarnings("ignore")

# Check for GPU availability and import CuPy if available
try:
    import cupy as cp
    from cupy.cuda import Device
    # Test if we can actually use CUDA
    cp_available = cp.cuda.is_available()
    if cp_available:
        # Get GPU info for logging
        dev = Device()
        gpu_mem_total = dev.mem_info[0] / (1024**3)  # Total memory in GB
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        logger.info(f"GPU acceleration enabled: {gpu_name} with {gpu_mem_total:.2f} GB memory")
    else:
        logger.warning("CUDA is not available despite CuPy being installed")
except ImportError:
    cp = None
    cp_available = False
    logger.warning("CuPy not installed. GPU acceleration will not be available.")

# Try to import cuSignal for GPU-accelerated signal processing
try:
    import cusignal
    CUSIGNAL_AVAILABLE = True and cp_available
except ImportError:
    CUSIGNAL_AVAILABLE = False
    logger.warning("cusignal not installed. GPU signal processing will not be available.")

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

# GPU memory management utilities
def _gpu_mem_usage():
    """Get current GPU memory usage in GB."""
    if not cp_available:
        return 0.0
    
    try:
        mem_used = cp.cuda.memory_allocated() / (1024 ** 3)  # GB
        return mem_used
    except Exception as e:
        logger.error(f"Error getting GPU memory usage: {e}")
        return 0.0

def _ensure_gpu_memory(required_gb=1.0, force_cpu=False):
    """Check if there's enough GPU memory available, fall back to CPU if not."""
    if force_cpu or not cp_available:
        return False
    
    try:
        # Get available memory
        device = cp.cuda.Device()
        total_mem = device.mem_info[0] / (1024 ** 3)  # GB
        used_mem = _gpu_mem_usage()
        available_mem = total_mem - used_mem
        
        if available_mem < required_gb:
            logger.warning(f"Not enough GPU memory available ({available_mem:.2f}GB < {required_gb:.2f}GB required)")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking GPU memory: {e}")
        return False

def to_gpu(arr, force_copy=False):
    """Safely transfer a numpy array to GPU memory."""
    if not cp_available:
        return arr
    
    try:
        # If it's already a cupy array, return it
        if isinstance(arr, cp.ndarray):
            return arr.copy() if force_copy else arr
        
        # Estimate memory requirements and check availability
        mem_needed = arr.nbytes / (1024 ** 3)  # GB
        if not _ensure_gpu_memory(mem_needed * 1.5):  # Add 50% safety margin
            return arr
        
        # Transfer to GPU
        return cp.asarray(arr)
    except Exception as e:
        logger.warning(f"Failed to transfer array to GPU: {e}")
        return arr

def to_cpu(arr):
    """Safely transfer a cupy array to CPU memory."""
    if not cp_available or not isinstance(arr, cp.ndarray):
        return arr
    
    try:
        return cp.asnumpy(arr)
    except Exception as e:
        logger.warning(f"Failed to transfer array to CPU: {e}")
        return arr

class FinancialFeatures:
    """Extract financial-specific features from market data windows."""
    
    @staticmethod
    def wavelet_decomposition(series: np.ndarray, 
                            wavelet: str = 'db4', 
                            level: int = 3,
                            use_gpu: bool = None) -> List[np.ndarray]:
        """
        Perform wavelet decomposition of time series.
        
        Args:
            series: Time series data
            wavelet: Wavelet function
            level: Decomposition level
            use_gpu: Whether to use GPU acceleration (None for auto-detection)
            
        Returns:
            List of wavelet coefficients [cA, cD_n, ..., cD_1]
        """
        # Determine if we should use GPU
        if use_gpu is None:
            use_gpu = cp_available and CUSIGNAL_AVAILABLE
        
        # Use GPU implementation if requested and available
        if use_gpu and cp_available and CUSIGNAL_AVAILABLE:
            try:
                return FinancialFeatures._gpu_wavelet_decomposition(series, wavelet, level)
            except Exception as e:
                logger.warning(f"GPU wavelet decomposition failed: {e}, falling back to CPU")
                # Fall back to CPU implementation on failure
                
        # CPU implementation
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
            logger.error(f"Error in CPU wavelet decomposition: {e}")
            return [series]
    
    @staticmethod
    def _gpu_wavelet_decomposition(series: np.ndarray, 
                                 wavelet: str = 'db4', 
                                 level: int = 3) -> List[np.ndarray]:
        """
        Perform GPU-accelerated wavelet decomposition of time series.
        
        Args:
            series: Time series data
            wavelet: Wavelet function
            level: Decomposition level
            
        Returns:
            List of wavelet coefficients [cA, cD_n, ..., cD_1]
        """
        # Ensure power of 2 length for better decomposition
        orig_len = len(series)
        power = int(np.ceil(np.log2(orig_len)))
        pad_len = 2**power
        
        # Transfer to GPU
        try:
            # Calculate required memory
            mem_needed = (series.nbytes * 5) / (1024**3)  # Approximate 5x for coefficients and padding
            if not _ensure_gpu_memory(mem_needed):
                raise MemoryError("Not enough GPU memory for wavelet decomposition")
            
            # Convert to GPU array
            gpu_series = to_gpu(series)
            
            # Pad to power of 2
            gpu_padded = cp.pad(gpu_series, (0, pad_len - orig_len), 'constant')
            
            # Use cusignal's wavelet functions
            # cusignal's wavedec has a slightly different API than PyWavelets
            gpu_coeffs = cusignal.cwt(gpu_padded, wavelet, level)
            
            # Convert results format to match PyWavelets
            coeffs = []
            for i in range(level + 1):
                coeffs.append(to_cpu(gpu_coeffs[i]))
            
            # Clean up GPU memory
            del gpu_series, gpu_padded, gpu_coeffs
            cp.get_default_memory_pool().free_all_blocks()
            
            return coeffs
            
        except Exception as e:
            logger.error(f"Error in GPU wavelet decomposition: {e}")
            # Fall back to CPU implementation
            if WAVELETS_AVAILABLE:
                padded_series = np.pad(series, (0, pad_len - orig_len), 'constant')
                return pywt.wavedec(padded_series, wavelet, level=level)
            else:
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
                   wavelet: str = 'db4', max_level: int = 3, 
                   use_gpu: bool = None) -> np.ndarray:
        """
        Wavelet-based lens optimized for financial regime detection with HDBSCAN.
        Uses GPU acceleration when available for significantly faster processing.
        
        Args:
            windows: List of time series windows
            n_components: Number of components in output
            wavelet: Wavelet function to use
            max_level: Maximum decomposition level
            use_gpu: Whether to use GPU acceleration (None for auto-detection)
            
        Returns:
            Array of shape (n_windows, n_components)
        """
        # Determine if we should use GPU
        if use_gpu is None:
            use_gpu = cp_available
        
        # Check for available wavelet implementations
        use_gpu_wavelets = use_gpu and cp_available and CUSIGNAL_AVAILABLE
        
        if not WAVELETS_AVAILABLE and not use_gpu_wavelets:
            logger.warning("Neither PyWavelets nor cuSignal available. Using fallback approach.")
            # Create a simple fallback based on statistical features
            return self._statistical_lens_fallback(windows, n_components)
        
        start_time = time.time()
        n_windows = len(windows)
        if n_windows == 0:
            return np.array([])
        
        # Logging
        if use_gpu_wavelets:
            logger.info(f"Using GPU-accelerated wavelet lens processing for {n_windows} windows")
        
        # Determine optimal wavelet for financial data
        if wavelet not in ['db4', 'sym4', 'db6', 'sym6']:
            logger.info(f"Optimizing wavelet choice: changing {wavelet} to db4 (optimal for financial data)")
            wavelet = 'db4'  # Consistently use db4 which works best for financial data
        
        # Preallocate features array for better performance
        # Use a fixed feature size that captures essential wavelet characteristics
        n_features = 8
        
        # Choose CPU or GPU implementation for features array
        if use_gpu and cp_available:
            try:
                features_array = cp.zeros((n_windows, n_features), dtype=cp.float32)
                using_gpu_array = True
            except Exception as e:
                logger.warning(f"Failed to allocate GPU memory for features array: {e}")
                features_array = np.zeros((n_windows, n_features))
                using_gpu_array = False
        else:
            features_array = np.zeros((n_windows, n_features))
            using_gpu_array = False
        
        # Batch processing for larger datasets
        batch_size = 1000 if use_gpu else 5000
        num_batches = (n_windows + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_windows)
            
            if num_batches > 1 and batch_idx % 5 == 0:
                logger.info(f"Processing wavelet batch {batch_idx+1}/{num_batches}")
            
            # Process windows in this batch
            for i in range(start_idx, end_idx):
                window_idx = i - start_idx  # Index within batch
                window = windows[i]
                
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
                    
                    # Perform wavelet decomposition (CPU or GPU)
                    coeffs = self.feature_calculator.wavelet_decomposition(
                        prices, wavelet, level, use_gpu=use_gpu_wavelets
                    )
                    
                    # Rest of feature extraction - CPU/GPU depending on data location
                    if use_gpu and cp_available and all(isinstance(c, np.ndarray) for c in coeffs):
                        # Transfer coeffs to GPU if they're not already there
                        gpu_coeffs = [to_gpu(coeff) for coeff in coeffs]
                        
                        # Extract features on GPU
                        self._extract_wavelet_features_gpu(
                            i, gpu_coeffs, features_array, 
                            volatility=volatility if volatility is not None else None,
                            using_gpu_array=using_gpu_array
                        )
                        
                        # Clean up GPU memory
                        del gpu_coeffs
                    else:
                        # CPU extraction
                        self._extract_wavelet_features_cpu(
                            i, coeffs, features_array, 
                            volatility=volatility if volatility is not None else None,
                            using_gpu_array=using_gpu_array
                        )
                
                except Exception as e:
                    logger.warning(f"Error in wavelet processing for window {i}: {str(e)[:100]}...")
                    # Leave as zeros for this window
            
            # Explicitly free memory after batch
            if use_gpu and cp_available:
                cp.get_default_memory_pool().free_all_blocks()
        
        # Make sure features_array is on CPU for further processing
        if using_gpu_array:
            features_array = to_cpu(features_array)
        
        # Handle NaN/Inf values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=10.0, neginf=0.0)
        
        # Scale features to [0,1] range for better HDBSCAN performance
        # Wasserstein distance works better with normalized data
        scaled_features = self.scaler.fit_transform(features_array)
        
        # Perform dimensionality reduction (PCA)
        if n_components < scaled_features.shape[1]:
            try:
                if use_gpu and cp_available and CUML_AVAILABLE:
                    try:
                        # Use GPU-accelerated PCA if available
                        from cuml.decomposition import PCA as cuPCA
                        gpu_scaled = to_gpu(scaled_features)
                        pca = cuPCA(n_components=n_components, random_state=42)
                        reduced_features = to_cpu(pca.fit_transform(gpu_scaled))
                        explained_var = float(cp.sum(pca.explained_variance_ratio_))
                        logger.info(f"GPU Wavelet lens PCA explained variance: {explained_var:.2f}")
                        
                        # Clean up GPU memory
                        del gpu_scaled
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception as e:
                        logger.warning(f"GPU PCA failed: {e}, falling back to CPU PCA")
                        # Fall back to CPU PCA
                        pca = PCA(n_components=n_components, random_state=42)
                        reduced_features = pca.fit_transform(scaled_features)
                        explained_var = np.sum(pca.explained_variance_ratio_)
                        logger.info(f"CPU Wavelet lens PCA explained variance: {explained_var:.2f}")
                else:
                    # Use CPU PCA
                    pca = PCA(n_components=n_components, random_state=42)
                    reduced_features = pca.fit_transform(scaled_features)
                    explained_var = np.sum(pca.explained_variance_ratio_)
                    logger.info(f"CPU Wavelet lens PCA explained variance: {explained_var:.2f}")
                
                end_time = time.time()
                logger.info(f"Wavelet lens computation completed in {end_time - start_time:.2f} seconds")
                return reduced_features
            except Exception as e:
                logger.error(f"Error in dimensionality reduction: {e}")
                # If reduction fails, return subset of features
                return scaled_features[:, :n_components]
        
        end_time = time.time()
        logger.info(f"Wavelet lens computation completed in {end_time - start_time:.2f} seconds")
        return scaled_features[:, :n_components]
    
    def _extract_wavelet_features_cpu(self, i, coeffs, features_array, volatility=None, using_gpu_array=False):
        """Extract wavelet features using CPU."""
        # 1. Energy distribution across scales - crucial for regime transitions
        energies = [np.sum(coeff**2) for coeff in coeffs]
        total_energy = sum(energies)
        
        if total_energy > 0:
            # Feature 1-3: Energy distribution across first 3 levels (normalized)
            for j in range(min(3, len(energies))):
                value = energies[j] / total_energy
                if using_gpu_array:
                    features_array[i, j] = value
                else:
                    features_array[i, j] = value
            
            # Feature 4: Low/high frequency ratio - key for regime identification
            low_energy = energies[0]  # Approximation coefficients
            high_energy = sum(energies[1:]) if len(energies) > 1 else 1.0
            value = min(10.0, low_energy / (high_energy + 1e-10))
            if using_gpu_array:
                features_array[i, 3] = value
            else:
                features_array[i, 3] = value
            
            # Feature 5: Wavelet entropy - measures randomness/disorder
            p = np.array([e / total_energy for e in energies])
            entropy = -np.sum(p * np.log2(p + 1e-10))
            value = entropy / np.log2(len(energies) + 1e-10)  # Normalized
            if using_gpu_array:
                features_array[i, 4] = value
            else:
                features_array[i, 4] = value
            
            # Features 6-7: Detail coefficients sparsity - captures jumps/regime shifts
            for j in range(min(2, len(coeffs)-1)):
                if len(coeffs[j+1]) > 0:
                    # Normalized count of significant coefficients
                    threshold = np.std(coeffs[j+1]) * 0.2
                    sparsity = np.sum(np.abs(coeffs[j+1]) > threshold) / len(coeffs[j+1])
                    if using_gpu_array:
                        features_array[i, 5+j] = sparsity
                    else:
                        features_array[i, 5+j] = sparsity
            
            # Feature 8: Temporal persistence - helps with regime stability
            if volatility is not None:
                # Use volatility as a regime persistence indicator
                value = np.mean(volatility) / (np.std(volatility) + 1e-10)
                if using_gpu_array:
                    features_array[i, 7] = value
                else:
                    features_array[i, 7] = value
            else:
                # Calculate persistence using autocorrelation
                prices = coeffs[0]  # Use approximation coefficients as prices
                diff = np.diff(prices)
                if len(diff) > 1:
                    autocorr = np.correlate(diff[:-1], diff[1:], mode='valid')[0] / (np.var(diff) * len(diff))
                    value = np.abs(autocorr)
                    if using_gpu_array:
                        features_array[i, 7] = value
                    else:
                        features_array[i, 7] = value
    
    def _extract_wavelet_features_gpu(self, i, coeffs, features_array, volatility=None, using_gpu_array=False):
        """Extract wavelet features using GPU."""
        # 1. Energy distribution across scales - crucial for regime transitions
        energies = [cp.sum(coeff**2) for coeff in coeffs]
        total_energy = sum(energies)
        
        if total_energy > 0:
            # Feature 1-3: Energy distribution across first 3 levels (normalized)
            for j in range(min(3, len(energies))):
                value = float(energies[j] / total_energy)
                if using_gpu_array:
                    features_array[i, j] = value
                else:
                    features_array[i, j] = value
            
            # Feature 4: Low/high frequency ratio - key for regime identification
            low_energy = energies[0]  # Approximation coefficients
            high_energy = sum(energies[1:]) if len(energies) > 1 else 1.0
            value = float(min(10.0, low_energy / (high_energy + 1e-10)))
            if using_gpu_array:
                features_array[i, 3] = value
            else:
                features_array[i, 3] = value
            
            # Feature 5: Wavelet entropy - measures randomness/disorder
            p = cp.array([float(e / total_energy) for e in energies])
            entropy = -cp.sum(p * cp.log2(p + 1e-10))
            value = float(entropy / cp.log2(len(energies) + 1e-10))  # Normalized
            if using_gpu_array:
                features_array[i, 4] = value
            else:
                features_array[i, 4] = value
            
            # Features 6-7: Detail coefficients sparsity - captures jumps/regime shifts
            for j in range(min(2, len(coeffs)-1)):
                if coeffs[j+1].size > 0:
                    # Normalized count of significant coefficients
                    threshold = float(cp.std(coeffs[j+1]) * 0.2)
                    sparsity = float(cp.sum(cp.abs(coeffs[j+1]) > threshold) / coeffs[j+1].size)
                    if using_gpu_array:
                        features_array[i, 5+j] = sparsity
                    else:
                        features_array[i, 5+j] = sparsity
            
            # Feature 8: Temporal persistence - helps with regime stability
            if volatility is not None:
                volatility_gpu = to_gpu(volatility)
                # Use volatility as a regime persistence indicator
                value = float(cp.mean(volatility_gpu) / (cp.std(volatility_gpu) + 1e-10))
                if using_gpu_array:
                    features_array[i, 7] = value
                else:
                    features_array[i, 7] = value
            else:
                # Calculate persistence using autocorrelation
                prices = coeffs[0]  # Use approximation coefficients as prices
                diff = cp.diff(prices)
                if diff.size > 1:
                    # GPU implementation of autocorrelation
                    diff_padded = cp.pad(diff, (0, diff.size - 1))
                    diff_reversed = cp.pad(cp.flip(diff), (0, diff.size - 1))
                    corr = cp.fft.ifft(cp.fft.fft(diff_padded) * cp.fft.fft(diff_reversed)).real
                    corr = corr[:diff.size]
                    var_diff = cp.var(diff)
                    if var_diff > 0:
                        autocorr = corr[1] / (var_diff * diff.size)
                        value = float(cp.abs(autocorr))
                        if using_gpu_array:
                            features_array[i, 7] = value
                        else:
                            features_array[i, 7] = value
    
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
                            use_gpu: bool = None,
                            **kwargs) -> np.ndarray:
        """
        Create financial lens projection for TDA.
        
        Args:
            windows: List of time series windows (optional)
            lens_type: Type of lens ('volatility', 'wavelet', 'comprehensive')
            window_size: Size of sliding windows (if windows not provided)
            feature_columns: Feature columns to use (if windows not provided)
            n_components: Number of components in lens
            use_gpu: Whether to use GPU acceleration (None for auto-detection)
            **kwargs: Additional parameters for specific lens functions
            
        Returns:
            Lens projection of shape (n_windows, n_components)
        """
        # Determine if we should use GPU
        if use_gpu is None:
            use_gpu = cp_available
            
        # If windows not provided, create them from dataframe
        if windows is None:
            if self.df is None:
                raise ValueError("No DataFrame or windows provided")
            
            if feature_columns is None:
                # Use default columns
                if 'Value' in self.df.columns and 'Volatility' in self.df.columns:
                    feature_columns = ['Value', 'Volatility']
                else:
                    # Use first available column
                    feature_columns = [self.df.columns[0]]
                    logger.warning(f"Using default feature column: {feature_columns[0]}")
            
            windows = self.create_windows(window_size, feature_columns)
        
        # Apply selected lens function
        if lens_type == 'wavelet':
            return self.wavelet_lens(
                windows, 
                n_components, 
                wavelet=kwargs.get('wavelet', 'db4'),
                max_level=kwargs.get('max_level', 3),
                use_gpu=use_gpu
            )
        elif lens_type == 'comprehensive':
            return self.comprehensive_financial_lens(
                windows, 
                n_components, 
                use_gpu=use_gpu,
                **kwargs
            )
        else:
            logger.warning(f"Unknown lens type: {lens_type}, using comprehensive lens")
            return self.comprehensive_financial_lens(
                windows, 
                n_components,
                use_gpu=use_gpu
            )
    
    def comprehensive_financial_lens(self, windows: List[np.ndarray], n_components: int = 2, 
                                   use_gpu: bool = None, **kwargs) -> np.ndarray:
        """
        Comprehensive financial lens combining multiple feature types.
        This lens combines wavelet features with statistical features for best regime detection.
        
        Args:
            windows: List of time series windows
            n_components: Number of components in output
            use_gpu: Whether to use GPU acceleration (None for auto-detection)
            **kwargs: Additional parameters for feature extraction
            
        Returns:
            Array of shape (n_windows, n_components)
        """
        # By default, just use the wavelet lens which already has good regime detection
        # This implementation can be extended with additional features in the future
        return self.wavelet_lens(
            windows, 
            n_components, 
            wavelet=kwargs.get('wavelet', 'db4'),
            max_level=kwargs.get('max_level', 3),
            use_gpu=use_gpu
        ) 