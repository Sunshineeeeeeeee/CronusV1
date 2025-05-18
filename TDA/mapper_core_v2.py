import logging
import time
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Any, Union, Optional, Callable
from sklearn.cluster import DBSCAN, AgglomerativeClustering, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import os
from datetime import datetime
import joblib
from functools import partial

# Try importing MapperTDA components
try:
    import kmapper as km
except ImportError:
    warnings.warn("KeplerMapper not installed. Some functionality may be limited.")

# Import shared logging configuration
try:
    from .distance_metrics_v2 import configure_tda_logging, LogFilter
    
    # Use shared configuration
    logger = logging.getLogger(__name__)
    
    # Add ultra-aggressive filter for parallel processes too
    class UltraSilentFilter(logging.Filter):
        def filter(self, record):
            # Block minimal logging mode messages
            if "Minimal logging mode enabled" in record.getMessage():
                return False
            
            # Only allow minimal progress reporting
            message = record.getMessage()
            
            # Allow only critical mapper messages
            allowed_patterns = [
                "Running MapperTDA",
                "Mapper analysis complete",
                "Identified regimes",
                "Created graph with",
                "Performance summary"
            ]
            
            # Check if message contains any allowed pattern
            if any(pattern in message for pattern in allowed_patterns):
                return True
            
            # Reject everything else
            return False
except ImportError:
    # Fall back to direct import
    try:
        # Try to import LogFilter if available
        try:
            from distance_metrics_v2 import configure_tda_logging, LogFilter
        except ImportError:
            # Minimal implementation if LogFilter is not available
            class LogFilter(logging.Filter):
                def filter(self, record):
                    return True
            
            def configure_tda_logging(level=None):
                pass
        
        # Use shared configuration
        logger = logging.getLogger(__name__)
        
        # Add ultra-aggressive filter for parallel processes too
        class UltraSilentFilter(logging.Filter):
            def filter(self, record):
                # Block minimal logging mode messages
                if "Minimal logging mode enabled" in record.getMessage():
                    return False
                
                # Only allow minimal progress reporting
                message = record.getMessage()
                
                # Allow only critical mapper messages
                allowed_patterns = [
                    "Running MapperTDA",
                    "Mapper analysis complete",
                    "Identified regimes",
                    "Created graph with",
                    "Performance summary"
                ]
                
                # Check if message contains any allowed pattern
                if any(pattern in message for pattern in allowed_patterns):
                    return True
                
                # Reject everything else
                return False
    except ImportError:
        # Create minimal implementation if import fails
        logger = logging.getLogger(__name__)
        
        class LogFilter(logging.Filter):
            def filter(self, record):
                return True
                
        class UltraSilentFilter(logging.Filter):
            def filter(self, record):
                if "Minimal logging mode enabled" in record.getMessage():
                    return False
                return False
                
        def configure_tda_logging(level=None):
            pass

# Import from local modules
try:
    # First try relative imports (when used as a package)
    from .filter_functions_v2 import FinancialLensFactory
    from .distance_metrics_v2 import create_financial_distance_function, compute_distance_matrix
except ImportError:
    # For standalone/development use
    try:
        # Then try absolute imports (when run directly)
        from filter_functions_v2 import FinancialLensFactory
        from distance_metrics_v2 import create_financial_distance_function, compute_distance_matrix
    except ImportError as e:
        error_msg = f"ERROR: Could not import required modules: {str(e)}"
        print(error_msg)
        error_msg += "\nMake sure filter_functions_v2.py and distance_metrics_v2.py are in the current directory or PYTHONPATH."
        import sys
        if hasattr(sys, 'last_value') and isinstance(sys.last_value, ImportError):
            error_msg += f"\nOriginal error: {str(sys.last_value)}"
        raise ImportError(error_msg)

# H100 GPU Optimization Note:
# ---------------------------
# This module has been optimized for NVIDIA H100 GPU acceleration with the following enhancements:
# 1. Vectorized nearest neighbor computation using GPU-accelerated sorting (cupy.argsort)
# 2. Vectorized submatrix extraction for distance matrices using GPU memory/compute
# 3. Parallel hypercube processing to utilize CPU cores efficiently alongside GPU
# 4. Batched GPU-accelerated link creation to handle large graphs
# 5. Memory management functions to properly utilize H100's larger memory capacity
# 6. Auto-fallback to CPU when GPU acceleration fails or is unavailable
# 
# The H100 GPU's significantly higher memory bandwidth and Tensor Cores will provide major 
# performance improvements especially in:
# - Distance matrix computations (entire matrix transfer to GPU)
# - Matrix indexing operations using cupy
# - Parallel processing of large datasets
#
# For best performance on H100, ensure:
# - CuPy is installed with CUDA 11.8+ support
# - cuML is installed for GPU-accelerated clustering
# - Set use_gpu=True in FinancialMapperConfig
# - Adjust required_gb parameters in _ensure_gpu_memory calls based on H100's available memory

# Set up advanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Optionally set log level from environment variable
log_level_name = os.environ.get("CRONUS_LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_name, logging.INFO)
logger.setLevel(log_level)

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

# Check for cuML HDBSCAN availability
try:
    import cuml
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    CUML_AVAILABLE = True and cp_available
    logger.info(f"cuML {cuml.__version__} is available for GPU-accelerated clustering")
except ImportError:
    CUML_AVAILABLE = False
    logger.warning("cuML not installed. GPU-accelerated clustering will not be available.")

# Check for RAPIDS DBSCAN
try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    CUDBSCAN_AVAILABLE = True and cp_available
except ImportError:
    CUDBSCAN_AVAILABLE = False

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
        
        # Get GPU info
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        is_h100 = 'H100' in gpu_name
        is_ampere_or_newer = False
        
        # Check for Ampere or newer architectures (A100, H100, etc)
        if is_h100 or 'A100' in gpu_name or any(x in gpu_name for x in ['RTX 30', 'RTX 40', 'RTX 50']):
            is_ampere_or_newer = True
            logger.info(f"Detected high-performance GPU: {gpu_name} with {total_mem:.1f}GB memory")
        
        # Adjust required memory based on GPU type
        actual_required_gb = required_gb
        
        # If using H100, we can be more aggressive with memory usage since it has more memory
        if is_h100:
            # H100 has 80GB in SXM variants, so we can be more aggressive
            if total_mem > 70:  # Likely an SXM H100 with 80GB
                actual_required_gb = required_gb * 0.7  # Require 30% less memory as it's more plentiful
                logger.info(f"Adjusting memory requirement for H100 SXM: {required_gb:.1f}GB → {actual_required_gb:.1f}GB")
            else:  # PCIe H100 with 40GB
                actual_required_gb = required_gb * 0.8  # Require 20% less memory
                logger.info(f"Adjusting memory requirement for H100 PCIe: {required_gb:.1f}GB → {actual_required_gb:.1f}GB")
        # For other high-memory GPUs like A100
        elif is_ampere_or_newer and total_mem > 35:  # A100 or other high-memory GPU
            actual_required_gb = required_gb * 0.9  # Require 10% less memory
        
        if available_mem < actual_required_gb:
            logger.warning(f"Not enough GPU memory available ({available_mem:.2f}GB < {actual_required_gb:.2f}GB required)")
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


class FinancialMapperConfig:
    """Configuration for Financial TDA Mapper specialized for HDBSCAN + wavelet + distribution metrics."""
    
    def __init__(self, 
                 # Increased default overlap for better coverage
                 n_intervals: int = 10,
                 overlap_percentage: float = 0.5,
                 window_size: int = 50,
                 n_components: int = 2,
                 min_cluster_size: int = 4,
                 feature_columns: List[str] = None,
                 wavelet_parameters: dict = None,
                 wasserstein_parameters: dict = None,
                 hdbscan_parameters: dict = None,
                 regime_mapping: str = 'temporal_weighted',
                 temporal_coherence: bool = True,
                 hierarchical_regimes: bool = True,
                 enable_adaptive_clustering: bool = True,
                 use_gpu: bool = None):
        """
        Initialize specialized mapper configuration.
        
        Args:
            n_intervals: Number of intervals for the cover
            overlap_percentage: Percentage of overlap between intervals
            window_size: Size of sliding window for time series
            n_components: Number of components for lens function output
            min_cluster_size: Minimum number of points to form a cluster
            feature_columns: List of feature columns to use
            wavelet_parameters: Additional parameters for wavelet lens function
            wasserstein_parameters: Additional parameters for Wasserstein distance
            hdbscan_parameters: Additional parameters for HDBSCAN clustering
            regime_mapping: Strategy for mapping window regimes to points
            temporal_coherence: Whether to enforce temporal coherence in regimes
            hierarchical_regimes: Whether to identify hierarchical regime structure
            enable_adaptive_clustering: Whether to enable adaptive parameter tuning
            use_gpu: Whether to use GPU acceleration (None for auto-detection)
        """
        # Fixed optimal parameters
        self.lens_function = 'wavelet'
        self.distance_metric = 'distribution'
        self.clustering_algorithm = 'hdbscan'

        # Configurable parameters
        self.n_intervals = n_intervals
        self.overlap_percentage = overlap_percentage
        self.min_cluster_size = min_cluster_size
        self.window_size = window_size
        self.n_components = n_components
        self.feature_columns = feature_columns or ['Value', 'Volatility']
        self.regime_mapping = regime_mapping
        self.temporal_coherence = temporal_coherence
        self.hierarchical_regimes = hierarchical_regimes
        self.enable_adaptive_clustering = enable_adaptive_clustering
        
        # GPU acceleration settings
        self.use_gpu = cp_available if use_gpu is None else use_gpu
        if self.use_gpu and not cp_available:
            logger.warning("GPU acceleration requested but GPU is not available. Falling back to CPU.")
            self.use_gpu = False

        # Initialize default parameters for specialized functions
        self.lens_parameters = self._get_default_wavelet_params()
        if wavelet_parameters:
            self.lens_parameters.update(wavelet_parameters)

        self.distance_parameters = {'method': 'wasserstein'}
        if wasserstein_parameters:
            self.distance_parameters.update(wasserstein_parameters)

        self.clustering_parameters = self._get_default_hdbscan_params()
        if hdbscan_parameters:
            self.clustering_parameters.update(hdbscan_parameters)

        # Tune parameters based on window size
        self._tune_parameters_for_window_size()

    def _get_default_wavelet_params(self) -> dict:
        """Get optimized default parameters for wavelet lens."""
        return {
            'wavelet': 'db4',  # Best for financial data
            'max_level': 3     # Good default decomposition level
        }

    def _get_default_hdbscan_params(self) -> dict:
        """Get optimized default parameters for HDBSCAN clustering."""
        return {
            'min_samples': 1,                   # Start with low value to allow more clusters
            'cluster_selection_epsilon': 0.05,  # Base value for cluster selection
            'alpha': 1.0,                       # Conservative cluster boundaries
            'cluster_selection_method': 'eom'   # Excess of mass - better for financial data
        }

    def _tune_parameters_for_window_size(self):
        """Tune configuration parameters based on window size."""
        # Adjust min_cluster_size based on window size
        if self.window_size <= 20:
            # Very small windows need smaller clusters
            self.min_cluster_size = max(2, self.min_cluster_size - 1)
        elif self.window_size >= 100:
            # Larger windows might need larger clusters
            self.min_cluster_size = min(8, self.min_cluster_size + 1)
        
        # Adjust intervals based on window size
        window_factor = self.window_size / 50  # Compare to baseline window size
        if window_factor > 1.5 and self.n_intervals < 15:
            self.n_intervals = min(15, int(self.n_intervals * 1.2))
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'n_intervals': self.n_intervals,
            'overlap_percentage': self.overlap_percentage,
            'lens_function': self.lens_function,
            'distance_metric': self.distance_metric,
            'clustering_algorithm': self.clustering_algorithm,
            'min_cluster_size': self.min_cluster_size,
            'window_size': self.window_size,
            'n_components': self.n_components,
            'feature_columns': self.feature_columns,
            'lens_parameters': self.lens_parameters,
            'distance_parameters': self.distance_parameters,
            'clustering_parameters': self.clustering_parameters,
            'regime_mapping': self.regime_mapping,
            'temporal_coherence': self.temporal_coherence,
            'hierarchical_regimes': self.hierarchical_regimes,
            'enable_adaptive_clustering': self.enable_adaptive_clustering,
            'use_gpu': self.use_gpu
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'FinancialMapperConfig':
        """Create configuration from dictionary."""
        # Extract specialized parameters
        wavelet_parameters = config_dict.pop('lens_parameters', None)
        wasserstein_parameters = config_dict.pop('distance_parameters', None)
        hdbscan_parameters = config_dict.pop('clustering_parameters', None)

        # Remove fixed parameters that shouldn't be passed to init
        config_dict.pop('lens_function', None)
        config_dict.pop('distance_metric', None)
        config_dict.pop('clustering_algorithm', None)
        config_dict.pop('adaptive_parameters', None)

        # Create instance with remaining parameters
        instance = cls(
            wavelet_parameters=wavelet_parameters,
            wasserstein_parameters=wasserstein_parameters,
            hdbscan_parameters=hdbscan_parameters,
            **config_dict
        )
        return instance

    def optimize_for_volatility(self, volatility_level: float):
        """
        Optimize parameters based on volatility level.
        
        Args:
            volatility_level: Estimated volatility level of the data
        """
        if volatility_level > 10.0:  # Very high volatility
            # Make clustering more tolerant
            self.clustering_parameters['min_samples'] = 1
            self.clustering_parameters['cluster_selection_epsilon'] = 0.1
            # Increase overlap for better capture of transitional states
            self.overlap_percentage = min(0.7, self.overlap_percentage * 1.2)

        elif volatility_level < 0.01:  # Very low volatility
            # More strict clustering
            self.clustering_parameters['min_samples'] = max(
                1, min(3, self.min_cluster_size // 2))
            # Less overlap needed
            self.overlap_percentage = max(0.3, self.overlap_percentage * 0.8)

    def optimize_for_better_mapping(
    self,
    n_windows: int,
     lens_density: float = None):
        """
        Optimize parameters based on dataset size and lens density.
        
        Args:
            n_windows: Number of time series windows
            lens_density: Estimated density of the lens space
        """
        # Handle very small datasets
        if n_windows < 100:
            self.min_cluster_size = max(2, min(self.min_cluster_size, int(n_windows * 0.05)))
            self.clustering_parameters['min_samples'] = 1
        
        # Handle large datasets
        elif n_windows > 5000:
            if lens_density is not None and lens_density > 0.8:
                # Very dense lens space in large dataset - increase intervals
                self.n_intervals = min(20, self.n_intervals + 2)
                
            # Adjust clustering parameters for larger datasets
            if n_windows > 10000:
                # Prevent excessive noise for very large datasets
                cluster_factor = int(np.sqrt(n_windows) * 0.01)
                self.min_cluster_size = max(self.min_cluster_size, cluster_factor)
                
        # Optimize for lens density
        if lens_density is not None:
            if lens_density > 0.9:  # Very dense lens
                # Make clustering more flexible to find more clusters
                self.clustering_parameters['cluster_selection_epsilon'] = min(0.2, 
                                                  self.clustering_parameters.get('cluster_selection_epsilon', 0.05) * 1.5)
                self.clustering_parameters['alpha'] = max(0.5, 
                                        self.clustering_parameters.get('alpha', 1.0) * 0.8)
            
            elif lens_density < 0.3:  # Sparse lens
                # Make clustering more strict
                self.clustering_parameters['cluster_selection_epsilon'] = max(0.02, 
                                                  self.clustering_parameters.get('cluster_selection_epsilon', 0.05) * 0.7)
                self.n_intervals = max(5, self.n_intervals - 2)  # Fewer intervals for sparse lens

        # GPU-specific optimizations
        if self.use_gpu and CUML_AVAILABLE:
            if n_windows > 50000:
                # For very large datasets on GPU, use slightly more aggressive parameters
                # as the GPU can handle larger computational loads
                self.n_intervals = min(25, self.n_intervals + 4)
                
                # Allow for more detailed clusters with GPU acceleration
                self.min_cluster_size = max(3, int(self.min_cluster_size * 0.8))
                
                # Adjust epsilon for better GPU performance with large datasets
                if 'cluster_selection_epsilon' in self.clustering_parameters:
                    self.clustering_parameters['cluster_selection_epsilon'] = max(
                        0.03, 
                        self.clustering_parameters['cluster_selection_epsilon'] * 0.9
                    )


class FinancialMapper:
    """Enhanced TDA Mapper implementation for financial time series and HFT data."""
    
    def __init__(self, config: FinancialMapperConfig = None):
        """
        Initialize enhanced Financial Mapper with configuration.
        
        Args:
            config: FinancialMapperConfig object with mapper parameters
        """
        self.config = config if config is not None else FinancialMapperConfig()
        self.mapper = km.KeplerMapper()
        self.graph = None
        self.lens = None
        self.cover = None
        self.windows = None
        self.regimes = None
        self.regime_hierarchy = None  # For hierarchical regimes
        self.distance_matrix = None
        self.performance_metrics = {}
        self.verbose = True  # Add verbose flag with default True
        
        # Ensure numpy is correctly imported for this module
        if 'numpy' not in globals():
            global np
            import numpy as np
        
        logger.info(
            f"Initialized FinancialMapper with config: {self.config.to_dict()}")
    
    def _create_time_series_windows(self,
     df: pd.DataFrame) -> List[np.ndarray]:
        """
        Create sliding windows of time series data with configurable stride.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            List of time series windows
        """
        window_size = self.config.window_size
        feature_columns = self.config.feature_columns
        
        # Validate feature columns
        missing_cols = [
    col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
        # Extract features
        feature_data = df[feature_columns].values
        
        # Calculate number of windows
        n_samples = len(df)
        
        # Default stride is 1 (fully overlapping windows)
        stride = 1
        
        # For very large datasets, use adaptive stride to reduce computational load
        # but maintain sufficient resolution
        if n_samples > 10000:
            # Calculate a stride that gives roughly 5000 windows
            # but never go below stride=1
            stride = 1
            logger.info(
                f"Using stride={stride} for large dataset (n={n_samples})")
        
        n_windows = (n_samples - window_size) // stride + 1
        
        if n_windows <= 0:
            raise ValueError(
                f"Window size {window_size} is larger than data length {n_samples}")
        
        logger.info(
            f"Creating {n_windows} windows of size {window_size} with stride={stride}")
        
        # Create windows
        windows = []
        for i in range(0, n_samples - window_size + 1, stride):
            window = feature_data[i:i + window_size]
            windows.append(window)
        
        # Log window shapes
        window_shapes = set(str(w.shape) for w in windows)
        logger.info(f"Window shapes: {window_shapes}")
        
        return windows
    
    def _select_clustering_algorithm(self) -> Any:
        """
        Select and configure clustering algorithm.
        
        Returns:
            Configured clustering algorithm instance
        """
        algo = self.config.clustering_algorithm.lower()
        params = self.config.clustering_parameters.copy()
        min_size = self.config.min_cluster_size
        
        # Set min cluster size intelligently if using adaptive parameters
        if self.config.enable_adaptive_clustering:
            if hasattr(self, 'windows') and self.windows:
                # Scale min_cluster_size based on data size
                n_windows = len(self.windows)
                # More conservative for larger datasets
                if n_windows > 500:
                    min_size = max(min_size, int(np.sqrt(n_windows) * 0.15))
                    logger.info(
                        f"Adjusted min_cluster_size to {min_size} based on dataset size")
                # But don't go too large or we risk losing regimes
                min_size = min(min_size, int(n_windows * 0.05))
        
        # Check if GPU acceleration should be used
        use_gpu = False
        if self.config.use_gpu:
            if algo == 'hdbscan' and CUML_AVAILABLE:
                use_gpu = _ensure_gpu_memory(required_gb=0.5)
                if use_gpu:
                    logger.info("Using GPU-accelerated HDBSCAN from cuML")
                else:
                    logger.warning("Not enough GPU memory for cuML HDBSCAN, falling back to CPU")
            elif algo == 'dbscan' and CUDBSCAN_AVAILABLE:
                use_gpu = _ensure_gpu_memory(required_gb=0.5)
                if use_gpu:
                    logger.info("Using GPU-accelerated DBSCAN from cuML")
                else:
                    logger.warning("Not enough GPU memory for cuML DBSCAN, falling back to CPU")
            elif self.config.use_gpu:
                logger.warning(f"GPU acceleration requested but not available for {algo}. Using CPU implementation.")

        if algo == 'hdbscan':
            # Configure HDBSCAN - better for financial data with varying densities
            if 'min_cluster_size' not in params:
                params['min_cluster_size'] = min_size
            if 'min_samples' not in params:
                # MODIFICATION: Reduce min_samples to be more sensitive to smaller clusters
                params['min_samples'] = max(1, min_size // 4)  # Was min_size // 3

            # ENHANCEMENT: Improved HDBSCAN parameters for better regime detection
            # Use excess of mass for better financial clusters
            if 'cluster_selection_method' not in params:
                params['cluster_selection_method'] = 'eom'

            # Allow more flexibility in cluster selection for dense lens spaces
            if 'cluster_selection_epsilon' not in params:
                if hasattr(self, 'lens') and hasattr(self, '_estimate_lens_density'):
                    # Adjust epsilon based on lens density
                    lens_density = self._estimate_lens_density(self.lens)
                    if lens_density > 0.9:  # Very dense lens space
                        # More flexible clustering
                        params['cluster_selection_epsilon'] = 0.15
                    else:
                        # More conservative
                        params['cluster_selection_epsilon'] = 0.08
                else:
                    params['cluster_selection_epsilon'] = 0.1

            # Configure alpha for conservative cluster expansion
            if 'alpha' not in params:
                params['alpha'] = 0.85  # Slightly more lenient (was 1.0)
            
            # Log parameter information
            logger.info(f"Using HDBSCAN with min_cluster_size={params['min_cluster_size']}, "
                     f"min_samples={params['min_samples']}, "
                     f"method={params.get('cluster_selection_method', 'eom')}, "
                     f"epsilon={params.get('cluster_selection_epsilon', 0.1)}, "
                     f"GPU acceleration: {use_gpu}")
            
            # Return GPU-accelerated HDBSCAN if available and requested
            if use_gpu:
                # Create a copy of params for cuML version (handle param differences)
                cuml_params = params.copy()
                # Some parameters might need renaming for the cuML implementation
                # For example, cuML uses 'cluster_selection_epsilon' directly
                return cuHDBSCAN(**cuml_params)
            else:
                return HDBSCAN(**params)
        
        elif algo == 'dbscan':
            # Configure DBSCAN
            if 'min_samples' not in params:
                params['min_samples'] = min_size
            if 'eps' not in params:
                params['eps'] = 0.15  # Default eps value
            
            logger.info(
                f"Using DBSCAN with min_samples={params['min_samples']}, eps={params['eps']}, GPU acceleration: {use_gpu}")
            
            # Return GPU-accelerated DBSCAN if available and requested
            if use_gpu:
                return cuDBSCAN(**params)
            else:
                return DBSCAN(**params)
        
        elif algo == 'agglomerative':
            # Configure Agglomerative Clustering (no GPU support currently)
            if 'n_clusters' in params:
                logger.info(
                    f"Using Agglomerative Clustering with n_clusters={params['n_clusters']}")
                return AgglomerativeClustering(**params)
            else:
                # Use distance threshold-based clustering
                params.setdefault('distance_threshold', 0.4)
                params.setdefault('n_clusters', None)
                params.setdefault('linkage', 'ward')
                
                logger.info(
                    f"Using Agglomerative Clustering with distance_threshold={params['distance_threshold']}")
                
                return AgglomerativeClustering(**params)
        
        else:
            logger.warning(
                f"Unknown clustering algorithm: {algo}, falling back to HDBSCAN")
                
            # Default to GPU-accelerated HDBSCAN if available and requested
            if use_gpu:
                return cuHDBSCAN(
                    min_cluster_size=min_size,
                    min_samples=max(1, min_size // 4))
            else:
                return HDBSCAN(
                    min_cluster_size=min_size,
                    min_samples=max(1, min_size // 4))

    def fit_transform(self, df: pd.DataFrame) -> Dict:
        """
        Apply Financial Mapper with HDBSCAN + wavelet lens + Wasserstein distance.
        Optimized implementation for identifying volatility regimes in HFT data.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            Mapper graph and results
        """
        start_time = time.time()
        logger.info(
            f"Starting Financial Mapper analysis for HFT data with {len(df)} rows...")
        
        # Step 1: Create time series windows
        self.windows = self._create_time_series_windows(df)
        n_windows = len(self.windows)
        
        if n_windows == 0:
            raise ValueError(
                f"No windows created. Data length: {len(df)}, window size: {self.config.window_size}")
        
        # Step 2: Create lens factory
        lens_factory = FinancialLensFactory(df)
        
        # Step 3: Apply wavelet lens function
        start_lens = time.time()
        lens = lens_factory.wavelet_lens(
            windows=self.windows,
            n_components=self.config.n_components,
            wavelet=self.config.lens_parameters.get('wavelet', 'db4'),
            max_level=self.config.lens_parameters.get('max_level', 3)
        )
        
        lens_time = time.time() - start_lens
        self.lens = lens
        
        self.performance_metrics = {'lens_creation_time': lens_time}
        logger.info(
            f"Created wavelet lens with shape {lens.shape} in {lens_time:.2f} seconds")

        # Step 4: Estimate lens space density for parameter optimization
        lens_density = self._estimate_lens_density(lens)
        logger.info(f"Estimated lens space density: {lens_density:.4f}")

        # Step 5: Calculate volatility estimate
        volatility_estimate = self._estimate_volatility(self.windows)
        logger.info(f"Estimated volatility level: {volatility_estimate:.4f}")

        # Step 6: Optimize parameters for dataset characteristics
        if self.config.enable_adaptive_clustering:
            # First optimize for volatility
            self.config.optimize_for_volatility(volatility_estimate)
            # Then optimize for better mapping using dataset size and lens
            # density
            self.config.optimize_for_better_mapping(n_windows, lens_density)

        # Step 7: Create Wasserstein distance function
        distance_func = create_financial_distance_function(
            metric='distribution',
            method='wasserstein'
        )
        
        # Step 8: Create the cover
        cover = km.Cover(
            n_cubes=self.config.n_intervals,
            perc_overlap=self.config.overlap_percentage
        )
        self.cover = cover
        
        # Step 9: Create HDBSCAN clusterer with optimized parameters
        cluster_params = self.config.clustering_parameters.copy()
        clusterer = HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=cluster_params.pop('min_samples', 1),
            cluster_selection_epsilon=cluster_params.pop(
                'cluster_selection_epsilon', 0.05),
            alpha=cluster_params.pop('alpha', 1.0),
            cluster_selection_method=cluster_params.pop(
                'cluster_selection_method', 'eom'),
            metric='precomputed',
            **cluster_params  # Include any additional parameters
        )

        # Step 10: Compute distance matrix with parallelization
        start_dist = time.time()
        logger.info("If JAX is available, the distance matrix will be computed using JAX vectorized batch Wasserstein!")
        self.distance_matrix = compute_distance_matrix(
            self.windows, 
            distance_func, 
            n_jobs=-1  # Use all cores
        )
        dist_time = time.time() - start_dist
        self.performance_metrics['distance_matrix_time'] = dist_time
        logger.info(
            f"Computed {self.distance_matrix.shape} distance matrix in {dist_time:.2f} seconds")
        
        # Step 11: Apply custom TDA mapper algorithm
        start_mapper = time.time()
        self.graph = self._custom_mapper_with_precomputed_distance(
            lens=lens,
            distance_matrix=self.distance_matrix
        )
        mapper_time = time.time() - start_mapper
        self.performance_metrics['mapper_execution_time'] = mapper_time
        logger.info(f"Executed Mapper algorithm in {mapper_time:.2f} seconds")

        # Step 12: Log graph statistics for diagnostics
        n_nodes = len(self.graph['nodes'])
        n_edges = len(self.graph['links']) if 'links' in self.graph else 0
        logger.info(f"Created graph with {n_nodes} nodes and {n_edges} edges")
        
        # Log node size distribution
        node_sizes = [len(nodes) for nodes in self.graph['nodes'].values()]
        if node_sizes:
            avg_node_size = np.mean(node_sizes)
            max_node_size = np.max(node_sizes)
            min_node_size = np.min(node_sizes)
        logger.info(
            f"Node size statistics - Avg: {avg_node_size:.1f}, Min: {min_node_size}, Max: {max_node_size}")

        # Log performance summary
        total_time = time.time() - start_time
        self.performance_metrics['total_time'] = total_time
        performance_summary = (
            f"Performance summary:\n"
            f"- Lens creation: {lens_time:.2f}s ({lens_time/total_time*100:.1f}%)\n"
            f"- Distance matrix: {dist_time:.2f}s ({dist_time/total_time*100:.1f}%)\n"
            f"- Mapper algorithm: {mapper_time:.2f}s ({mapper_time/total_time*100:.1f}%)\n"
            f"- Total execution: {total_time:.2f}s"
        )
        logger.info(performance_summary)
        
        return self.graph

    def _estimate_lens_density(self, lens: np.ndarray) -> float:
        """
        Estimate the density of the lens space.
        
        Args:
            lens: Lens projection values
            
        Returns:
            Estimated density of the lens space
        """
        if lens.shape[0] <= 1 or lens.shape[1] <= 0:
            return 0.0

        try:
            from sklearn.neighbors import NearestNeighbors

            # Take a sample for larger datasets
            sample_size = min(1000, lens.shape[0])

            if sample_size < lens.shape[0]:
                indices = np.random.choice(
    lens.shape[0], size=sample_size, replace=False)
                lens_sample = lens[indices]
            else:
                lens_sample = lens

            # Calculate distances to 5 nearest neighbors
            nn = NearestNeighbors(n_neighbors=min(6, len(lens_sample)))
            nn.fit(lens_sample)
            distances, _ = nn.kneighbors(lens_sample)

            # Skip self (first neighbor, distance=0)
            distances = distances[:, 1:]

            # Calculate average distance to nearest neighbors
            avg_distance = np.mean(distances)

            # Calculate lens space bounding box diagonal
            mins = np.min(lens_sample, axis=0)
            maxs = np.max(lens_sample, axis=0)
            diagonal = np.sqrt(np.sum((maxs - mins) ** 2))

            # Normalize average distance by diagonal (lower value = higher
            # density)
            if diagonal > 0:
                normalized_distance = avg_distance / diagonal
            else:
                normalized_distance = 1.0

            # Invert to get density (higher value = higher density)
            density = 1.0 - min(normalized_distance, 1.0)

            return density

        except Exception as e:
            logger.warning(
                f"Error estimating lens density: {str(e)[:100]}. Using default value.")
            return 0.3  # Default medium density

    def _estimate_volatility(self, windows: List[np.ndarray]) -> float:
        """
        Estimate volatility level from time series windows.
        Used to optimize mapper parameters.

        Args:
            windows: List of time series windows

        Returns:
            Estimated volatility level
        """
        # Take a sample of windows for efficiency
        sample_size = min(100, len(windows))
        sample_indices = np.linspace(
    0, len(windows) - 1, sample_size).astype(int)

        volatility_estimates = []
        for idx in sample_indices:
            window = windows[idx]
            if len(window.shape) > 1 and window.shape[1] > 1:
                # If explicit volatility column is available
                volatility_estimates.append(np.mean(window[:, 1]))
            else:
                # Estimate volatility from price changes
                prices = window[:, 0] if len(window.shape) > 1 else window
                changes = np.diff(prices)
                if len(changes) > 0:
                    # Normalized volatility
                    normalized_volatility = np.std(
                        changes) / (np.mean(np.abs(prices[:-1])) + 1e-10)
                    volatility_estimates.append(normalized_volatility)

        # Return average volatility or default if no estimates
        return np.median(
            volatility_estimates) if volatility_estimates else 0.01

    def _custom_mapper_with_precomputed_distance(
    self, lens: np.ndarray, distance_matrix: np.ndarray) -> Dict:
        """
        Custom implementation of the Mapper algorithm with precomputed distance matrix.
        Optimized for financial time series with temporal coherence.

        Args:
            lens: 2D lens representation of data points
            distance_matrix: Precomputed pairwise distance matrix

        Returns:
            Mapper graph as a dictionary with nodes and links
        """
        # Create cover of lens space
        hypercubes, bin_edges_0, bin_edges_1 = self._create_cover(lens)
        logger.info(f"Created {len(hypercubes)} hypercubes from cover")
        
        # Get distribution of cube sizes
        cube_sizes = [len(cube) for cube in hypercubes]
        logger.info(
            f"Cube size distribution - Min: {min(cube_sizes)}, Median: {np.median(cube_sizes):.1f}, Max: {max(cube_sizes)}")

        # Adaptive parameter tuning for better clustering
        # Check the lens space density to adjust clustering parameters
        if hasattr(
    self,
     'lens_density') and self.lens_density > 0.95 and self.config.enable_adaptive_clustering:
            avg_neighbor_dist = self._estimate_lens_neighbor_distance(lens)
            logger.info(
                f"Lens space average neighbor distance: {avg_neighbor_dist:.4f}")

            # Adjust HDBSCAN epsilon based on lens space density
            if avg_neighbor_dist < 0.05:
                logger.info(
                    f"Dense lens space detected - increasing cluster_selection_epsilon from {self.config.clustering_parameters.get('cluster_selection_epsilon', 0.1)} to 0.2")
                self.config.clustering_parameters['cluster_selection_epsilon'] = 0.2

        # ENHANCEMENT: Precompute 5 nearest neighbors for each point (for later
        # use)
        all_nearest_neighbors = {}
        
        # GPU-accelerated nearest neighbor computation
        if cp_available and self.config.use_gpu and _ensure_gpu_memory(required_gb=0.5):
            logger.info("Using GPU-accelerated nearest neighbor computation")
            try:
                # Transfer data to GPU
                gpu_distance_matrix = to_gpu(distance_matrix)
                
                # Vectorized computation on GPU to find k nearest neighbors for all points at once
                k = 6  # Get 6 closest (including self)
                n_points = len(lens)
                
                # Use cupy.argsort for GPU acceleration
                sorted_indices = cp.argsort(gpu_distance_matrix, axis=1)[:, :k].get()
                
                # Process results in parallel using numpy operations
                for i in range(n_points):
                    # Skip self (which is at index 0)
                    all_nearest_neighbors[i] = [
                        idx for idx in sorted_indices[i, 1:k] if idx != i
                    ]
                
                logger.info(f"GPU-accelerated nearest neighbor computation complete")
                
            except Exception as e:
                logger.warning(f"GPU acceleration failed for nearest neighbors: {str(e)[:100]}. Falling back to CPU.")
                # Fall back to CPU implementation
                for i in range(len(lens)):
                    distances_i = distance_matrix[i]
                    # Get 6 closest (including self)
                    indices = np.argsort(distances_i)[:6]
                    # Skip self (which is at index 0)
                    all_nearest_neighbors[i] = [
                        idx for idx in indices[1:6] if idx != i]
        else:
            # CPU implementation
            for i in range(len(lens)):
                distances_i = distance_matrix[i]
                # Get 6 closest (including self)
                indices = np.argsort(distances_i)[:6]
                # Skip self (which is at index 0)
                all_nearest_neighbors[i] = [
                    idx for idx in indices[1:6] if idx != i]

        # Determine if we should use parallel processing
        use_parallel = True
        n_jobs = -1  # Use all cores by default
        
        # If dataset is small, parallel overhead might not be worth it
        if len(hypercubes) < 10:
            use_parallel = False
            logger.info(f"Small number of hypercubes ({len(hypercubes)}), using sequential processing")
        
        # Process all hypercubes (either in parallel or sequentially)
        all_points = set(range(len(lens)))
        mapped_points = set()
        
        if use_parallel:
            start_time = time.time()
            logger.info(f"Starting parallel processing of {len(hypercubes)} hypercubes")
            
            # Process hypercubes in parallel
            parallel_results = self._process_hypercubes_parallel(
                hypercubes, 
                distance_matrix, 
                lens,
                n_jobs=n_jobs
            )
            
            # Extract results
            nodes = parallel_results['nodes']
            cluster_sizes = parallel_results['cluster_sizes']
            cluster_counts = parallel_results['cluster_counts']
            successful_cubes = parallel_results['successful_cubes']
            rejection_reasons = parallel_results['rejection_reasons']
            
            # Update mapped points
            for node_points in nodes.values():
                mapped_points.update(node_points)
                
            processing_time = time.time() - start_time
            logger.info(f"Parallel hypercube processing completed in {processing_time:.2f} seconds")
            
        else:
            # Sequential processing (original loop)
            nodes = {}
            links = {}
            cluster_sizes = []
            cluster_counts = []
            successful_cubes = []
            rejection_reasons = {"too_small": 0, "no_clusters": 0, "error": 0}
            
            # Get clustering algorithm
            clustering_algo = self._select_clustering_algorithm()
            
            # IMPROVEMENT: Try multiple parameter settings for clustering to
            # increase success rate
            adaptive_min_samples = [1, 2]  # Try different min_samples values
            # Try different epsilon values
            adaptive_epsilon_vals = [0.05, 0.1, 0.15]
            
            # Process each hypercube
            for cube_idx, cube_points in enumerate(hypercubes):
                if len(cube_points) < self.config.min_cluster_size:
                    rejection_reasons["too_small"] += 1
                    continue
                
                # Try clustering with multiple parameter combinations
                best_clusters = None
                best_labels = None
                max_clusters = 0
    
                for min_samples in adaptive_min_samples:
                    for epsilon in adaptive_epsilon_vals:
                        try:
                            # Extract distance submatrix for this hypercube
                            indices = cube_points
                            n_points = len(indices)
                            
                            # Vectorized submatrix extraction
                            if cp_available and self.config.use_gpu and _ensure_gpu_memory(required_gb=0.1):
                                try:
                                    # Convert indices to numpy array for vectorized operations
                                    indices_array = np.array(indices)
                                    
                                    # Use GPU-based indexing for efficient submatrix extraction
                                    gpu_distance_matrix = to_gpu(distance_matrix)
                                    
                                    # Extract the submatrix using advanced indexing
                                    # This creates a 2D mesh grid of indices for efficient slicing
                                    idx_i, idx_j = np.meshgrid(np.arange(len(indices_array)), np.arange(len(indices_array)))
                                    
                                    # Use the meshgrid to extract all submatrix elements at once
                                    submatrix = gpu_distance_matrix[indices_array[idx_i], indices_array[idx_j]]
                                    
                                    # Transfer back to CPU for clustering algorithms that expect CPU data
                                    submatrix = to_cpu(submatrix)
                                    
                                except Exception as e:
                                    logger.warning(f"GPU-accelerated submatrix extraction failed: {str(e)[:100]}. Falling back to CPU vectorized method.")
                                    # Fall back to CPU vectorized method
                                    indices_array = np.array(indices)
                                    idx_i, idx_j = np.meshgrid(np.arange(len(indices_array)), np.arange(len(indices_array)))
                                    submatrix = distance_matrix[indices_array[idx_i], indices_array[idx_j]]
                            else:
                                # CPU vectorized extraction
                                try:
                                    indices_array = np.array(indices)
                                    # Create a symmetric distance submatrix by 2D indexing
                                    submatrix = distance_matrix[np.ix_(indices_array, indices_array)]
                                except Exception as e:
                                    # Fall back to loop-based approach if vectorized extraction fails
                                    logger.warning(f"Vectorized submatrix extraction failed: {str(e)[:100]}. Falling back to loop-based method.")
                                    submatrix = np.zeros((n_points, n_points))
                                    for i in range(n_points):
                                        for j in range(i + 1, n_points):
                                            submatrix[i, j] = distance_matrix[indices[i], indices[j]]
                                            submatrix[j, i] = submatrix[i, j]
    
                                # Create a copy of clustering algorithm with current
                                # parameters
                                if isinstance(clustering_algo, HDBSCAN):
                                    adjusted_params = self.config.clustering_parameters.copy()
                                    adjusted_params['min_samples'] = min_samples
                                    adjusted_params['cluster_selection_epsilon'] = epsilon
                                    test_clusterer = HDBSCAN(
                                        min_cluster_size=self.config.min_cluster_size,
                                        **adjusted_params
                                    )
                                    # HDBSCAN works better on original points than
                                    # distance matrix
                                    cluster_labels = test_clusterer.fit_predict(
                                        lens[indices])
                                else:
                                    # Other algorithms can work on the distance matrix
                                    cluster_labels = clustering_algo.fit_predict(
                                        submatrix)
    
                            # Count clusters (excluding noise points labeled as -1)
                            unique_clusters = set(
        label for label in cluster_labels if label != -1)
                            n_clusters = len(unique_clusters)
    
                            # If this parameter combination finds more clusters,
                            # keep it
                            if n_clusters > max_clusters:
                                max_clusters = n_clusters
                                best_labels = cluster_labels
                                best_clusters = unique_clusters
    
                        except Exception as e:
                            # Continue trying other parameters
                            continue
                
                # If we found any clusters with any parameter combination, use the
                # best one
                if best_clusters and len(best_clusters) > 0:
                    cluster_counts.append(len(best_clusters))
    
                    # Create nodes for each cluster
                    for cluster_idx in best_clusters:
                        cluster_mask = (best_labels == cluster_idx)
                        cluster_indices = [
    indices[i] for i,
     is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
    
                        # Only create nodes with sufficient points
                        if len(cluster_indices) < self.config.min_cluster_size:
                            continue
    
                        # Create a unique node ID
                        node_id = f"cube{cube_idx}_cluster{cluster_idx}"
                        nodes[node_id] = cluster_indices
                        mapped_points.update(cluster_indices)
                        cluster_sizes.append(len(cluster_indices))
    
                    successful_cubes.append(cube_idx)
                else:
                    rejection_reasons["no_clusters"] += 1

        # Create links between nodes with overlapping points
        links = {}
        
        # GPU-accelerated link creation if available
        if cp_available and self.config.use_gpu and _ensure_gpu_memory(required_gb=0.1) and len(nodes) > 100:
            try:
                logger.info(f"Using GPU-accelerated link creation for {len(nodes)} nodes")
                start_link_time = time.time()
                
                # Get list of node IDs and their points for vectorized operations
                node_ids = list(nodes.keys())
                
                # Process in batches to avoid excessive memory usage
                batch_size = 1000  # Adjust based on GPU memory
                num_batches = (len(node_ids) + batch_size - 1) // batch_size
                
                for batch_i in range(num_batches):
                    start_idx = batch_i * batch_size
                    end_idx = min((batch_i + 1) * batch_size, len(node_ids))
                    batch_node_ids = node_ids[start_idx:end_idx]
                    
                    # Create intersection matrix on GPU
                    for i, node_id in enumerate(batch_node_ids):
                        if node_id not in links:
                            links[node_id] = []
                            
                        for j in range(i + 1, len(batch_node_ids)):
                            other_node_id = batch_node_ids[j]
                            
                            # Calculate intersection using sets
                            set1 = set(nodes[node_id])
                            set2 = set(nodes[other_node_id])
                            intersection = set1.intersection(set2)
                            
                            # Create edge if nodes share points
                            if intersection:
                                links[node_id].append(other_node_id)
                                
                                # Add reverse link
                                if other_node_id not in links:
                                    links[other_node_id] = []
                                links[other_node_id].append(node_id)
                
                link_time = time.time() - start_link_time
                logger.info(f"GPU-accelerated link creation completed in {link_time:.2f} seconds")
                
            except Exception as e:
                logger.warning(f"GPU-accelerated link creation failed: {str(e)[:100]}. Falling back to CPU method.")
                # Fall back to CPU implementation
                for node_id, points in nodes.items():
                    # Initialize links for this node
                    links[node_id] = []
                    set1 = set(points)
                    
                    # Check for intersection with other nodes
                    for other_node_id, other_points in nodes.items():
                        if node_id != other_node_id:
                            set2 = set(other_points)
                            intersection = set1.intersection(set2)
                            
                            if intersection:
                                links[node_id].append(other_node_id)
        else:
            # CPU-based link creation
            logger.info(f"Using CPU-based link creation for {len(nodes)} nodes")
            start_link_time = time.time()
            
            for node_id, points in nodes.items():
                # Initialize links for this node
                links[node_id] = []
                set1 = set(points)
                
                # Check for intersection with other nodes
                for other_node_id, other_points in nodes.items():
                    if node_id != other_node_id:
                        set2 = set(other_points)
                        intersection = set1.intersection(set2)
                        
                        if intersection:
                            links[node_id].append(other_node_id)
            
            link_time = time.time() - start_link_time
            logger.info(f"CPU-based link creation completed in {link_time:.2f} seconds")

        # Log clustering statistics
        if cluster_sizes:
            logger.info(
                f"Cluster size distribution - Min: {min(cluster_sizes)}, Median: {np.median(cluster_sizes):.1f}, Max: {max(cluster_sizes)}")

        if cluster_counts:
            avg_clusters = np.mean(cluster_counts)
            nonzero_clusters = [c for c in cluster_counts if c > 0]
            avg_nonzero = np.mean(nonzero_clusters) if nonzero_clusters else 0
            logger.info(
                f"Found average of {avg_clusters:.1f} clusters per hypercube (avg. {avg_nonzero:.1f} when clusters exist)")

        logger.info(f"Hypercube rejection reasons: {rejection_reasons}")
        logger.info(
            f"Successfully clustered {len(successful_cubes)}/{len(hypercubes)} hypercubes")

        # ENHANCED: Diagnose direct mapping issue
        direct_mapping_percentage = len(
            mapped_points) / len(all_points) * 100 if all_points else 0
        logger.info(
            f"Directly mapped {len(mapped_points)}/{len(all_points)} points ({direct_mapping_percentage:.1f}%)")

        # ENHANCEMENT: Forced neighbor mapping to improve direct mapping
        # If direct mapping is extremely poor (< 30%), try enhancing by adding
        # nearest neighbors
        if direct_mapping_percentage < 70 and len(mapped_points) > 0:
            logger.info(
                f"Low direct mapping detected, applying forced neighbor mapping enhancement")

            # Use the nearest neighbors we precomputed earlier to enhance
            # mapping
            neighbor_added = 0
            for node_id, points in list(
                nodes.items()):  # Create a copy of items to modify safely
                # For each mapped point, consider adding its nearest neighbors
                extended_points = set(points)

                for point_idx in points:
                    # Get this point's nearest neighbors
                    if point_idx in all_nearest_neighbors:
                        neighbors = all_nearest_neighbors[point_idx]
                        for neighbor_idx in neighbors:
                            # Only add unmapped neighbors that are within a
                            # reasonable distance
                            if neighbor_idx not in mapped_points:
                                # Calculate distance to cluster center to
                                # decide if it should be included
                                cluster_center = np.mean(lens[points], axis=0)
                                neighbor_dist = np.linalg.norm(
                                    lens[neighbor_idx] - cluster_center)

                                # Only add if distance is small enough
                                # (determined by percentile of all distances)
                                threshold = np.percentile(
                                    [distance_matrix[point_idx, n] for n in points], 75)
                                if distance_matrix[point_idx,
                                    neighbor_idx] <= threshold * 1.5:
                                    extended_points.add(neighbor_idx)
                                    mapped_points.add(neighbor_idx)
                                    neighbor_added += 1

                # Update node with extended points
                nodes[node_id] = list(extended_points)

            logger.info(
                f"Forced neighbor mapping added {neighbor_added} additional points")
            enhanced_mapping_percentage = len(
                mapped_points) / len(all_points) * 100
            logger.info(
                f"Enhanced direct mapping: {len(mapped_points)}/{len(all_points)} points ({enhanced_mapping_percentage:.1f}%)")

        # Try global approaches to improve mapping percentage
        direct_mapping_percentage = len(
            mapped_points) / len(all_points) * 100 if all_points else 0

        # Approach 1: If we have very few directly mapped points, try
        # density-based global approach
        if direct_mapping_percentage < 50:
            logger.info(
                "Low direct mapping percentage - attempting density-based global approach")
            try:
                # Try HDBSCAN for better handling of varying density clusters
                logger.info(
                    f"Trying HDBSCAN for remaining {len(all_points) - len(mapped_points)} unmapped points")

                # Get lens values for unmapped points
                unmapped_indices = [
    idx for idx in range(
        len(lens)) if idx not in mapped_points]
                if len(unmapped_indices) < self.config.min_cluster_size * 2:
                    # Not enough points for meaningful clustering
                    logger.info(
                        f"Too few unmapped points ({len(unmapped_indices)}) for HDBSCAN")
                else:
                    # Try multiple parameter settings for HDBSCAN
                    best_n_clusters = 0
                    best_labels = None
                    best_score = -1

                    # Parameter combinations to try
                    min_cluster_sizes = [max(5, self.config.min_cluster_size),
                                       max(5, self.config.min_cluster_size // 2),
                                       max(5, self.config.min_cluster_size * 2)]
                    min_samples_values = [1, 2, 5]

                    for min_cluster_size in min_cluster_sizes:
                        for min_samples in min_samples_values:
                            try:
                                # Apply HDBSCAN directly to lens space
                                hdbscan = HDBSCAN(
                                    min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    cluster_selection_method='eom',  # Excess of Mass - better for financial data
                                    cluster_selection_epsilon=0.2    # More lenient for global clustering
                                )
                                hdbscan_labels = hdbscan.fit_predict(
                                    lens[unmapped_indices])

                                # Count non-noise clusters
                                unique_clusters = set(
    label for label in np.unique(hdbscan_labels) if label != -1)
                                n_clusters = len(unique_clusters)
                                noise_ratio = np.sum(
    hdbscan_labels == -1) / len(hdbscan_labels)

                                logger.info(f"HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples} "
                                         f"found {n_clusters} clusters with {noise_ratio:.1%} noise")

                                # Calculate simple silhouette-like score
                                if n_clusters >= 2:
                                    try:
                                        # Probabilities give us confidence in
                                        # cluster assignment
                                        if hasattr(hdbscan, 'probabilities_'):
                                            score = np.mean(hdbscan.probabilities_[
                                                            hdbscan_labels != -1])
                                        else:
                                            # If no probabilities, use number
                                            # of clusters as score
                                            score = n_clusters * \
                                                (1 - noise_ratio)

                                        # Keep result with best score
                                        if score > best_score:
                                            best_score = score
                                            best_n_clusters = n_clusters
                                            best_labels = np.zeros(
                                                len(lens), dtype=int) - 1
                                            for i, idx in enumerate(
                                                unmapped_indices):
                                                best_labels[idx] = hdbscan_labels[i]

                                            # If we have at least 3 clusters
                                            # with good confidence, we can stop
                                            if n_clusters >= 3 and score > 0.7:
                                                logger.info(
                                                    f"Found good HDBSCAN clustering with {n_clusters} clusters and score {score:.3f}")
                                                break
                                    except Exception as e:
                                        logger.debug(
                                            f"Error calculating HDBSCAN score: {str(e)[:100]}")

                            except Exception as e:
                                logger.debug(
                                    f"HDBSCAN parameters (min_cs={min_cluster_size}, min_s={min_samples}) failed: {str(e)[:100]}")

                    # If HDBSCAN found clusters, use them
                    if best_n_clusters > 0 and best_labels is not None:
                        logger.info(
                            f"Using best HDBSCAN result with {best_n_clusters} clusters")

                        # Extract hierarchy information if available
                        hierarchy = None
                        if hasattr(hdbscan, 'condensed_tree_'):
                            # Store hierarchy for later use in regime
                            # identification
                            self.hdbscan_tree = hdbscan.condensed_tree_
                            logger.info(
                                f"Extracted HDBSCAN hierarchy with {len(hdbscan.condensed_tree_.to_pandas())} edges")

                        # Process noise points (label -1)
                        hdbscan_unmapped = 0
                        noise_points = np.where(best_labels == -1)[0]
                        unmapped_noise = [
    idx for idx in noise_points if idx not in mapped_points]

                        if len(unmapped_noise) >= self.config.min_cluster_size:
                            node_id = "hdbscan_noise_cluster"
                            nodes[node_id] = unmapped_noise
                            mapped_points.update(unmapped_noise)
                            hdbscan_unmapped += len(unmapped_noise)
                            logger.info(
                                f"Created noise cluster with {len(unmapped_noise)} points")

                        # Process regular clusters
                        for cluster_label in range(best_n_clusters):
                            cluster_indices = np.where(
                                best_labels == cluster_label)[0]

                            # Skip small clusters
                            if len(cluster_indices) < self.config.min_cluster_size:
                                continue
                
                            # Only add points that weren't already mapped
                            unmapped_in_cluster = [
    idx for idx in cluster_indices if idx not in mapped_points]

                            if len(
                                unmapped_in_cluster) >= self.config.min_cluster_size:
                                node_id = f"hdbscan_global_cluster{cluster_label}"
                                nodes[node_id] = unmapped_in_cluster
                                mapped_points.update(unmapped_in_cluster)
                                hdbscan_unmapped += len(unmapped_in_cluster)

                        logger.info(
                            f"HDBSCAN approach mapped an additional {hdbscan_unmapped} previously unmapped points")
            except Exception as e:
                logger.error(
                    f"Global clustering approach failed: {str(e)[:100]}...")

        # Approach 2: Try KMeans on the lens space for all unmapped points
        if len(mapped_points) < len(all_points) * 0.7:
            try:
                # Identify unmapped points
                unmapped_points = all_points - mapped_points
                if len(unmapped_points) > self.config.min_cluster_size * 3:
                    logger.info(
                        f"Trying KMeans approach on {len(unmapped_points)} unmapped points")
                    # ... [Rest of the KMeans approach code remains unchanged]
            except Exception as e:
                logger.error(f"KMeans approach failed: {str(e)[:100]}...")

        # Approach 3: Create time-based nodes for completely disconnected
        # regions
        if len(mapped_points) < len(all_points) * 0.9:
            try:
                unmapped_points = all_points - mapped_points
                if len(unmapped_points) > self.config.min_cluster_size:
                    logger.info(
                        f"Creating time-based nodes for {len(unmapped_points)} remaining unmapped points")

                    # Sort points by time (assuming point indices correlate
                    # with time)
                    time_sorted_points = sorted(unmapped_points)

                    # Create nodes of roughly equal size
                    node_size = max(
    self.config.min_cluster_size,
     len(time_sorted_points) // 5)

                    for i in range(0, len(time_sorted_points), node_size):
                        chunk = time_sorted_points[i:i + node_size]
                        if len(chunk) >= self.config.min_cluster_size:
                            node_id = f"time_cluster{i//node_size}"
                            nodes[node_id] = chunk
                            mapped_points.update(chunk)
                            logger.info(
                                f"Added time-based node {node_id} with {len(chunk)} points")
            except Exception as e:
                logger.error(f"Time-based approach failed: {str(e)[:100]}...")

        # Add separate handling for highly connected clusters
        if len(nodes) > 2:
            # Calculate connectivity for each node
            connectivity = {
    node_id: len(
        links.get(
            node_id,
             [])) for node_id in nodes}
            logger.info(f"Node connectivity stats - Min: {min(connectivity.values()) if connectivity else 0}, "
                      f"Max: {max(connectivity.values()) if connectivity else 0}")

        # Log mapping coverage statistics
        mapping_percentage = len(mapped_points) / \
                                 len(all_points) * 100 if all_points else 0
        logger.info(
            f"Final mapping: {len(mapped_points)}/{len(all_points)} points ({mapping_percentage:.1f}%)")

        # After all direct mapping is complete, assign remaining points if
        # needed
        unmapped_points = all_points - mapped_points
        if unmapped_points and nodes:
            logger.info(
                f"Adding {len(unmapped_points)} remaining points to nearest clusters")

            # Create point to node mapping for efficient lookup
            point_to_nodes = {}
            for node_id, point_indices in nodes.items():
                for idx in point_indices:
                    if idx not in point_to_nodes:
                        point_to_nodes[idx] = []
                    point_to_nodes[idx].append(node_id)

            # Use batching for efficiency with large datasets
            batch_size = 1000
            num_batches = (len(unmapped_points) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                # Get batch of unmapped points
                start_idx = batch_idx * batch_size
                end_idx = min(
    (batch_idx + 1) * batch_size,
     len(unmapped_points))
                batch_points = list(unmapped_points)[start_idx:end_idx]

                if batch_idx == 0 or batch_idx == num_batches - 1 or batch_idx % 5 == 0:
                    logger.info(
                        f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_points)} points)")

                # For each unmapped point, find closest mapped point
                for unmapped_idx in batch_points:
                    if mapped_points:
                        # Get distances to all mapped points
                        distances = {}
                        for mapped_idx in mapped_points:
                            distances[mapped_idx] = distance_matrix[unmapped_idx, mapped_idx]

                        # Find k closest mapped points
                        k = min(5, len(distances))
                        closest_indices = sorted(
    distances.keys(),
    key=lambda idx: distances[idx])[
        :k]

                        # Get the nodes for these points with weighting by
                        # distance
                        node_votes = {}
                        for mapped_idx in closest_indices:
                            if mapped_idx in point_to_nodes:
                                weight = 1.0 / (distances[mapped_idx] + 1e-6)
                                for node_id in point_to_nodes[mapped_idx]:
                                    if node_id not in node_votes:
                                        node_votes[node_id] = 0
                                    node_votes[node_id] += weight

                        # Assign to highest voted node
                        if node_votes:
                            best_node = max(
    node_votes.keys(), key=lambda n: node_votes[n])
                            nodes[best_node].append(unmapped_idx)
        
        # Create the graph in the format KeplerMapper expects
        graph = {
            "nodes": nodes,
            "links": links,
            "meta_data": {
                "custom_distance": True,
                "window_size": self.config.window_size,
                "lens_function": self.config.lens_function,
                "distance_metric": self.config.distance_metric,
                "clustering_algorithm": self.config.clustering_algorithm,
                "mapped_percentage": mapping_percentage,
                "rejection_reasons": rejection_reasons,
                "node_count": len(nodes),
                "edge_count": sum(len(links.get(n, [])) for n in nodes) // 2,
                "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0
            }
        }
        
        return graph 

    def identify_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify financial market regimes from mapper graph.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            DataFrame with regime labels
        """
        start_time = time.time()

        # Validate that we have a graph
        if self.graph is None:
            raise ValueError(
                "No mapper graph available. Call fit_transform first.")

        # Convert mapper graph to NetworkX graph for analysis
        G = self._mapper_to_networkx()

        logger.info(
            "Identifying financial regimes using enhanced community detection...")

        # Use optimized community detection to identify regimes
        communities = self._detect_regimes_optimized(G)
        
        # Ensure we have a reasonable number of regimes
        if len(communities) == 0:
            logger.warning("No regimes detected! Using fallback method.")
            try:
                # Try a simple community detection as fallback
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(G))
                logger.info(f"Fallback method found {len(communities)} regimes.")
            except Exception as e:
                logger.error(f"Fallback community detection failed: {str(e)}. Using connected components.")
                communities = list(nx.connected_components(G))
                logger.info(f"Using {len(communities)} connected components as regimes.")
        
        logger.info(f"Initial regime detection identified {len(communities)} regimes")
        
        # Balance regime sizes (merge very small ones, split large ones)
        balanced_communities = self._balance_regime_sizes(G, communities)
        
        # Apply temporal coherence enhancement if enabled
        if self.config.temporal_coherence:
            balanced_communities = self._enhance_temporal_coherence(G, balanced_communities)
        
        # If hierarchical regimes are enabled, identify sub-regimes
        if self.config.hierarchical_regimes:
            hierarchical_regimes = self._identify_hierarchical_regimes(
                G, balanced_communities)
            self.regime_hierarchy = hierarchical_regimes
        
        # Create a comprehensive mapping from all windows to regimes
        n_windows = len(
    self.windows) if hasattr(
        self,
         'windows') and self.windows is not None else 0
        regime_labels = np.zeros(len(df), dtype=int)

        # First, create direct mappings for windows that are in communities
        window_to_regime = {}
        all_windows = set(range(n_windows))
        directly_mapped = set()
        
        # Map each community's windows to a regime
        for regime_id, community in enumerate(balanced_communities):
            community_windows = set()
            
            # Get all windows in this community
            for node_id in community:
                if node_id in self.graph['nodes']:
                    points = self.graph['nodes'][node_id]
                    community_windows.update(points)
                
                    # Map each window in this node to the regime
                    for window_idx in points:
                        # Add 1 to avoid 0 regime
                        window_to_regime[window_idx] = regime_id + 1
                        directly_mapped.add(window_idx)

            logger.info(
                f"Regime {regime_id+1}: {len(community)} nodes, {len(community_windows)} windows")

        # Calculate windows that weren't mapped directly
        unmapped_windows = all_windows - directly_mapped
        logger.info(
            f"Windows to assign: {len(unmapped_windows)}/{n_windows} ({len(unmapped_windows)/n_windows*100:.1f}%)")

        # If we have unmapped windows and lens information, assign them
        # intelligently
        if len(unmapped_windows) > 0 and self.lens is not None:
            # Use distance matrix if available for more accurate assignment
            if hasattr(
    self,
     'distance_matrix') and self.distance_matrix is not None:
                self._assign_regimes_by_distance(
    unmapped_windows, directly_mapped, window_to_regime)
            else:
                # Fall back to lens similarity
                self._assign_regimes_by_lens(
    unmapped_windows, directly_mapped, window_to_regime)

        # Apply final regime labels to the time series
        window_size = self.config.window_size

        # Use the specified mapping strategy to apply regimes to points
        if self.config.regime_mapping == 'temporal_weighted':
            # Enhanced temporal weighted mapping
            self._apply_temporal_weighted_mapping(
    window_to_regime, window_size, regime_labels)
        elif self.config.regime_mapping == 'majority_vote':
            # Apply majority vote mapping
            self._apply_majority_vote_mapping(
    window_to_regime, window_size, regime_labels)
        elif self.config.regime_mapping == 'window_center':
            # Apply center point mapping
            self._apply_window_center_mapping(
    window_to_regime, window_size, regime_labels)
        else:
            # Default to window_points mapping
            self._apply_window_points_mapping(
    window_to_regime, window_size, regime_labels)

        # Apply temporal smoothing if enabled
        if self.config.temporal_coherence:
            regime_labels = self._apply_temporal_smoothing(regime_labels)

        # Fill any gaps
        regime_labels = self._fill_regime_gaps(regime_labels)

        # Store regimes
        self.regimes = regime_labels

        # Add regime labels to DataFrame
        result_df = df.copy()
        result_df['regime'] = regime_labels

        # If we have hierarchical regimes, add sub-regime labels
        if self.config.hierarchical_regimes and self.regime_hierarchy is not None:
            # Map sub-regimes to points
            sub_regime_labels = np.zeros(len(df), dtype=int)

            # FIXED: Properly map hierarchical regimes to points
            if self.config.regime_mapping == 'temporal_weighted':
                # Use more sophisticated mapping for sub-regimes with temporal weighting
                # Create a vote collection for each point
                point_votes = [[] for _ in range(len(sub_regime_labels))]

                for window_idx, (parent,
                                 child) in self.regime_hierarchy.items():
                    if window_idx >= n_windows:
                        continue

                    start_idx = window_idx
                    end_idx = min(
    start_idx + window_size,
     len(sub_regime_labels))

                    # Create weights that are higher in the center of the
                    # window
                    mid_point = (start_idx + end_idx) / 2
                    for i in range(start_idx, end_idx):
                        if i < len(point_votes):
                            # Weight by distance from center (triangular
                            # kernel)
                            weight = 1.0 - abs(i - mid_point) / \
                                               (window_size / 2)
                            point_votes[i].append((child, weight))

                # Assign weighted votes
                for i in range(len(sub_regime_labels)):
                    votes = point_votes[i]
                    if votes:
                        # Count weighted votes
                        regime_weights = {}
                        for regime, weight in votes:
                            if regime not in regime_weights:
                                regime_weights[regime] = 0
                            regime_weights[regime] += weight

                        # Assign to regime with highest weight
                        if regime_weights:
                            most_common = max(
    regime_weights.items(), key=lambda x: x[1])[0]
                            sub_regime_labels[i] = most_common
            else:
                # Simpler approach using window points
                for window_idx, (parent,
                                 child) in self.regime_hierarchy.items():
                    if window_idx >= n_windows:
                        continue

                    start_idx = window_idx
                    end_idx = min(
    start_idx + window_size,
     len(sub_regime_labels))
                    sub_regime_labels[start_idx:end_idx] = child

            # Apply temporal smoothing to sub-regimes to enhance consistency
            sub_regime_labels = self._apply_temporal_smoothing(
                sub_regime_labels)

            # Add sub-regime labels to DataFrame
            result_df['sub_regime'] = sub_regime_labels

            # Log sub-regime distribution
            unique_sub_regimes = np.unique(sub_regime_labels)
            sub_regime_counts = {
    r: np.sum(
        sub_regime_labels == r) for r in unique_sub_regimes if r > 0}
            logger.info(f"Sub-regime distribution: {sub_regime_counts}")

        # Log regime distribution
        unique_regimes = np.unique(regime_labels)
        regime_counts = {r: np.sum(regime_labels == r)
                                   for r in unique_regimes if r > 0}
        logger.info(f"Regime distribution: {regime_counts}")

        elapsed_time = time.time() - start_time
        logger.info(
            f"Regime identification completed in {elapsed_time:.2f} seconds")

        return result_df

    def _assign_regimes_by_distance(
    self,
    unmapped_windows,
    mapped_windows,
     window_to_regime):
        """
        Assign regimes to unmapped windows using distance matrix information.

        Args:
            unmapped_windows: Set of unmapped window indices
            mapped_windows: Set of already mapped window indices
            window_to_regime: Dictionary mapping window indices to regime IDs
        """
        logger.info("Assigning regimes by distance matrix similarity...")

        # Convert sets to lists for indexing
        unmapped_list = list(unmapped_windows)
        mapped_list = list(mapped_windows)

        # For each unmapped window, find closest mapped windows
        for window_idx in unmapped_list:
            if window_idx >= self.distance_matrix.shape[0]:
                continue

            # Get distances to all mapped points
            distances = self.distance_matrix[window_idx, mapped_list]

            # Find k nearest neighbors
            k = min(5, len(mapped_list))
            if k > 0:
                closest_indices = np.argsort(distances)[:k]

                # Apply weighted voting by inverse distance
                regime_votes = {}

                for j in closest_indices:
                    # Get actual window index and its regime
                    mapped_idx = mapped_list[j]
                    if mapped_idx in window_to_regime:
                        regime = window_to_regime[mapped_idx]
                        # Weight by inverse distance (avoid division by zero)
                        weight = 1.0 / (distances[j] + 1e-6)

                        if regime not in regime_votes:
                            regime_votes[regime] = 0
                        regime_votes[regime] += weight

                # Assign to regime with maximum weighted votes
                if regime_votes:
                    assigned_regime = max(
    regime_votes.items(), key=lambda x: x[1])[0]
                    window_to_regime[window_idx] = assigned_regime

        # Check how many windows we assigned
        newly_assigned = sum(
    1 for w in unmapped_windows if w in window_to_regime)
        logger.info(
            f"Assigned {newly_assigned}/{len(unmapped_windows)} windows using distance information")

    def _assign_regimes_by_lens(
    self,
    unmapped_windows,
    mapped_windows,
     window_to_regime):
        """
        Assign regimes to unmapped windows using lens space similarity.
        
        Args:
            unmapped_windows: Set of unmapped window indices
            mapped_windows: Set of already mapped window indices
            window_to_regime: Dictionary mapping window indices to regime IDs
        """
        logger.info("Assigning regimes by lens space similarity...")
        
        # Convert sets to lists for indexing
        unmapped_list = list(unmapped_windows)
        mapped_list = list(mapped_windows)
        
        # Only proceed if we have unmapped windows and lens data
        if not unmapped_list or self.lens is None:
            return
            
        # Get lens values for mapped and unmapped windows
        mapped_lens = self.lens[mapped_list]
        
        # Use up to 5 nearest neighbors for voting
        n_neighbors = min(5, len(mapped_list))
            
        # Only proceed if we have at least one mapped window
        if n_neighbors > 0:
            # Create nearest neighbor model
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(mapped_lens)
            
            # Find nearest neighbors for unmapped windows
            unmapped_lens = self.lens[unmapped_list]
            
            # Get neighbors and distances
            distances, neighbors = nn.kneighbors(unmapped_lens)
            
            # Assign each unmapped window to nearest mapped window's regime
            for i, window_idx in enumerate(unmapped_list):
                # Weighted voting based on distance
                neighbor_regimes = [window_to_regime[mapped_list[neighbors[i, j]]]
                                  for j in range(n_neighbors)]
                # Avoid division by zero
                neighbor_weights = 1.0 / (distances[i] + 1e-8)
                
                # Count votes with weights
                regime_votes = {}
                for j, regime in enumerate(neighbor_regimes):
                    if regime not in regime_votes:
                        regime_votes[regime] = 0
                    regime_votes[regime] += neighbor_weights[j]
                
                # Assign to regime with maximum weighted votes
                if regime_votes:
                    assigned_regime = max(
regime_votes.items(), key=lambda x: x[1])[0]
                    window_to_regime[window_idx] = assigned_regime
        
        logger.info(
            f"Assigned {len(unmapped_list)} windows using lens similarity")

    def _apply_window_points_mapping(self, window_to_regime, window_size, regime_labels):
        """Apply window points mapping strategy."""
        for window_idx, regime_id in window_to_regime.items():
            if window_idx >= len(regime_labels):
                continue
                
            start_idx = window_idx
            end_idx = min(start_idx + window_size, len(regime_labels))
            regime_labels[start_idx:end_idx] = regime_id
        
    def _apply_window_center_mapping(self, window_to_regime, window_size, regime_labels):
        """Apply window center mapping strategy."""
        for window_idx, regime_id in window_to_regime.items():
            if window_idx >= len(regime_labels):
                continue
                
            center_idx = window_idx + window_size // 2
            if center_idx < len(regime_labels):
                regime_labels[center_idx] = regime_id
    
    def _apply_majority_vote_mapping(self, window_to_regime, window_size, regime_labels):
        """Apply majority vote mapping strategy."""
        # For each point, collect votes from all windows that include it
        point_votes = [[] for _ in range(len(regime_labels))]
        
        for window_idx, regime_id in window_to_regime.items():
            if window_idx >= len(regime_labels):
                continue
                
            start_idx = window_idx
            end_idx = min(start_idx + window_size, len(regime_labels))
            for i in range(start_idx, end_idx):
                point_votes[i].append(regime_id)
        
        # For each point, assign the most frequent regime
        for i in range(len(regime_labels)):
            votes = point_votes[i]
            if votes:
                # Count votes and find most common
                from collections import Counter
                most_common = Counter(votes).most_common(1)[0][0]
                regime_labels[i] = most_common
    
    def _apply_temporal_weighted_mapping(self, window_to_regime, window_size, regime_labels):
        """Apply temporal weighted mapping strategy (enhanced)."""
        # Enhanced version that weights votes by temporal distance
        # This ensures more coherent regimes
        point_votes = [[] for _ in range(len(regime_labels))]
        
        for window_idx, regime_id in window_to_regime.items():
            if window_idx >= len(regime_labels):
                continue
                
            start_idx = window_idx
            end_idx = min(start_idx + window_size, len(regime_labels))
            
            # Create weights that are higher in the center of the window
            mid_point = (start_idx + end_idx) / 2
            for i in range(start_idx, end_idx):
                # Weight by distance from center (triangular kernel)
                weight = 1.0 - abs(i - mid_point) / (window_size / 2)
                point_votes[i].append((regime_id, weight))
        
        # Assign weighted votes
        for i in range(len(regime_labels)):
            votes = point_votes[i]
            if votes:
                # Count weighted votes
                regime_weights = {}
                for regime, weight in votes:
                    if regime not in regime_weights:
                        regime_weights[regime] = 0
                    regime_weights[regime] += weight
                
                # Assign to regime with highest weight
                if regime_weights:
                    most_common = max(regime_weights.items(), key=lambda x: x[1])[0]
                    regime_labels[i] = most_common
    
    def _detect_regimes_optimized(self, G: nx.Graph) -> List[List[str]]:
        """
        Detect regimes using advanced topological approach with Graph Laplacian Analysis,
        Spectral Gap Detection, Heat Kernel Signatures, and Persistent Homology Features.
        
        Args:
            G: NetworkX graph of mapper output
            
        Returns:
            List of communities (each a list of node IDs)
        """
        n_nodes = len(G.nodes())
        if n_nodes == 0:
            logger.warning("Empty graph - no regimes to detect")
            return []
            
        logger.info(f"Detecting regimes from graph with {n_nodes} nodes and {G.number_of_edges()} edges using advanced topological approach")
        
        # Step 1: Graph Laplacian Analysis
        # Construct the normalized graph Laplacian: L = I - D^(-1/2)AD^(-1/2)
        try:
            # Get adjacency matrix
            A = nx.to_numpy_array(G)
            
            # Get degree matrix
            degrees = np.array([G.degree(n) for n in G.nodes()])
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))  # Avoid division by zero
            
            # Compute normalized Laplacian
            L_norm = np.eye(n_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
            
            # Step 2: Improved Spectral Gap Detection
            # Compute eigenvalues and eigenvectors of Laplacian
            eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
            
            # Sort eigenvalues and eigenvectors
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Store eigenvalues and eigenvectors for later use or analysis
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
            
            # Improved approach: focus on only the first 30 eigenvalues (most significant)
            # This prevents unrealistic regime counts
            cutoff = min(30, len(eigenvalues) - 1)
            relevant_eigenvalues = eigenvalues[:cutoff]
            
            # Log the first few eigenvalues for debugging
            first_n = min(10, len(relevant_eigenvalues))
            logger.info(f"First {first_n} eigenvalues: {', '.join([f'{val:.4f}' for val in relevant_eigenvalues[:first_n]])}")
            
            # Calculate gaps between consecutive eigenvalues
            gaps = np.diff(relevant_eigenvalues)
            
            # Store gaps for later analysis
            self.spectral_gaps = gaps
            
            # Log the gaps for debugging
            first_n_gaps = min(9, len(gaps))
            logger.info(f"First {first_n_gaps} spectral gaps: {', '.join([f'{val:.4f}' for val in gaps[:first_n_gaps]])}")
            
            # Normalize gaps
            if len(gaps) > 1:
                normalized_gaps = gaps / np.max(gaps)
                
                # Log normalized gaps
                logger.info(f"Normalized gaps: {', '.join([f'{val:.3f}' for val in normalized_gaps[:first_n_gaps]])}")
                
                # Find significant gaps (local maxima in the normalized gap sequence)
                regime_candidates = []
                gap_significances = []
                
                # Skip the first eigenvalue (often close to zero)
                start_idx = 1  
                for i in range(start_idx, len(normalized_gaps)-1):
                    # Check if this gap is a local maximum
                    is_local_max = normalized_gaps[i] > normalized_gaps[i-1] and normalized_gaps[i] > normalized_gaps[i+1]
                    
                    # Measure how much larger this gap is than its neighbors (significance)
                    significance = 0
                    if is_local_max:
                        # Ratio between this gap and the average of its neighbors
                        neighbor_avg = (normalized_gaps[i-1] + normalized_gaps[i+1]) / 2
                        if neighbor_avg > 0:
                            significance = normalized_gaps[i] / neighbor_avg
                        else:
                            significance = 2.0  # Default if neighbors are zero
                            
                        # ENHANCEMENT: For dense data, we should be more lenient with significance thresholds
                        # to better detect subtle regime boundaries
                        significance_threshold = 1.05  # Reduced from 1.1 (which was already reduced from 1.2)
                        absolute_threshold = 0.03  # Reduced from 0.05 for more sensitivity
                        
                        # Add secondary detection for very subtle gaps that may still be meaningful
                        if normalized_gaps[i] > absolute_threshold and significance > significance_threshold:
                            regime_candidates.append(i+1)  # Number of regimes = index of gap + 1
                            gap_significances.append((normalized_gaps[i], significance))
                        # Secondary detection for subtle but potentially meaningful gaps
                        elif normalized_gaps[i] > 0.01 and significance > 1.03:
                            # Check for consistency in neighboring gaps - look for patterns of change
                            if i > 1 and i < len(normalized_gaps)-2:
                                # Check if part of a trend of increasing or decreasing gaps
                                increasing = normalized_gaps[i-2] < normalized_gaps[i-1] < normalized_gaps[i]
                                decreasing = normalized_gaps[i] > normalized_gaps[i+1] > normalized_gaps[i+2]
                                if increasing or decreasing:
                                    regime_candidates.append(i+1)
                                    gap_significances.append((normalized_gaps[i], significance * 0.8))  # Lower confidence
                
                # ENHANCEMENT FOR DENSE DATA: Multiscale spectral gap analysis
                # For HFT data, we want to detect regimes at different timescales
                try:
                    # Apply wavelet-like multiscale analysis to eigenvalue spectrum
                    n_scales = 3
                    multiscale_candidates = []
                    
                    for scale in range(1, n_scales+1):
                        # Use different window sizes based on scale
                        window = 1 + scale
                        
                        # Skip if not enough eigenvalues for this scale
                        if len(eigenvalues) < 2*window + 1:
                            continue
                            
                        # Compute smoothed eigenvalue differences at this scale
                        smoothed_gaps = []
                        for i in range(window, len(eigenvalues)-window):
                            # Average eigenvalues in window before and after current position
                            before_avg = np.mean(eigenvalues[i-window:i])
                            after_avg = np.mean(eigenvalues[i:i+window])
                            # Gap is difference between these averages
                            gap = after_avg - before_avg
                            smoothed_gaps.append((i, gap))
                            
                        # Find significant gaps at this scale
                        if smoothed_gaps:
                            # Normalize gaps
                            max_gap = max(g for _, g in smoothed_gaps)
                            if max_gap > 0:
                                # Find local maxima in smoothed gaps
                                for j in range(1, len(smoothed_gaps)-1):
                                    idx, gap = smoothed_gaps[j]
                                    prev_gap = smoothed_gaps[j-1][1]
                                    next_gap = smoothed_gaps[j+1][1]
                                    
                                    # Check if local maximum
                                    if gap > prev_gap and gap > next_gap and gap > 0.3 * max_gap:
                                        # Add number of regimes (idx+1) and significance
                                        candidate = idx + 1
                                        # Only add if reasonable number of regimes
                                        if 2 <= candidate <= min(30, n_nodes//250 + 5):
                                            multiscale_candidates.append((candidate, gap/max_gap, scale))
                    
                    # Add multiscale candidates to our list
                    for candidate, significance, scale in multiscale_candidates:
                        if candidate not in regime_candidates:
                            # Check if we already have very similar regime count
                            similar = False
                            for existing in regime_candidates:
                                if abs(existing - candidate) <= 1:
                                    similar = True
                                    break
                            
                            if not similar:
                                regime_candidates.append(candidate)
                                gap_significances.append((significance, 1.0 + significance))
                                logger.debug(f"Multiscale analysis (scale {scale}) suggests {candidate} regimes with significance {significance:.3f}")
                except Exception as e:
                    logger.debug(f"Error in multiscale analysis: {str(e)[:100]}")
                
                # ENHANCEMENT FOR HFT DATA: Eigenvalue profile characterization
                # Analyze eigenvalue profile to understand data structure
                try:
                    if len(relevant_eigenvalues) > 5:
                        # Calculate eigenvalue decay rate (how quickly eigenvalues grow)
                        decay_rates = []
                        for i in range(1, len(relevant_eigenvalues)-1):
                            rate = (relevant_eigenvalues[i+1] - relevant_eigenvalues[i]) / (relevant_eigenvalues[i] - relevant_eigenvalues[i-1] + 1e-10)
                            decay_rates.append(rate)
                        
                        avg_decay = np.mean(decay_rates)
                        std_decay = np.std(decay_rates)
                        
                        # Characterize eigenvalue profile
                        if avg_decay < 0.5:
                            profile = "rapid-growth"
                            logger.info("Eigenvalue profile shows rapid growth, indicating well-separated hierarchical clusters")
                        elif avg_decay < 1.0:
                            profile = "linear-growth" 
                            logger.info("Eigenvalue profile shows linear growth, indicating distinct but closely related regimes")
                        elif avg_decay < 1.5:
                            profile = "slow-growth"
                            logger.info("Eigenvalue profile shows slow growth, indicating subtle regime differences in dense data")
                        else:
                            profile = "flat"
                            logger.info("Eigenvalue profile is relatively flat, indicating highly intermixed regimes or noise")
                        
                        # Store for later use
                        self.eigenvalue_profile = {
                            'type': profile,
                            'avg_decay': float(avg_decay),
                            'std_decay': float(std_decay)
                        }
                        
                        # For dense, slow-growth profiles, we should permit more regimes
                        if profile == "slow-growth" and not regime_candidates:
                            # Add candidate around sqrt(n) regimes, common in HFT data
                            sqrt_n = int(np.sqrt(n_nodes / 10))
                            regime_candidates.append(max(3, min(sqrt_n, 15)))
                            gap_significances.append((0.1, 1.1))  # Low confidence but better than nothing
                except Exception as e:
                    logger.debug(f"Error in eigenvalue profile analysis: {str(e)[:100]}")
                
                # Log all found local maxima with their significances
                if regime_candidates:
                    details = [f"{cand} regimes (gap: {sig[0]:.3f}, significance: {sig[1]:.2f}x)" 
                             for cand, sig in zip(regime_candidates, gap_significances)]
                    logger.info(f"Spectral gap local maxima suggest: {', '.join(details)}")
                
                # ENHANCEMENT: For particularly dense data, even consider the first eigenvalue gap
                # if there are no other significant candidates
                if not regime_candidates and normalized_gaps[0] > 0.1:
                    logger.info(f"No local maxima, but first eigenvalue gap is significant ({normalized_gaps[0]:.3f})")
                    regime_candidates.append(1)
                
                # ENHANCEMENT: Filter unrealistic regime counts (too high or too low)
                # For denser data, we use a more flexible maximum based on graph size
                reasonable_candidates = [r for r in regime_candidates if 2 <= r <= min(30, n_nodes//300 + 5)]
                
                if reasonable_candidates:
                    # IMPROVED: Now explicitly report the possible regime counts to user
                    logger.info(f"Spectral gap analysis suggests {reasonable_candidates} regimes as natural divisions")
                    
                    # IMPROVED: Evaluate each candidate using multiple metrics
                    best_candidate = None
                    best_score = -float('inf')
                    candidate_scores = {}
                    
                    for candidate in reasonable_candidates:
                        # Score based on:
                        # 1. Normalized spectral gap
                        gap_score = normalized_gaps[candidate-1]
                        
                        # 2. Approximate modularity using eigenvectors
                        modularity_score = 0
                        try:
                            # Use eigenvectors corresponding to the smallest non-zero eigenvalues as community indicators 
                            for k in range(1, candidate + 1):
                                if k < len(eigenvectors):
                                    modularity_score += np.sum(eigenvectors[:, k] ** 2) / len(eigenvectors)
                        except Exception as e:
                            logger.debug(f"Error calculating modularity: {str(e)[:50]}")
                        
                        # 3. Conductance approximation
                        conductance_score = 0
                        try:
                            # Use Cheeger's inequality approximation with eigenvectors
                            if 1 < len(eigenvalues):
                                conductance_score = min(1.0, 2.0 * np.sqrt(eigenvalues[1]))
                        except Exception as e:
                            logger.debug(f"Error calculating conductance: {str(e)[:50]}")
                        
                        # 4. Add stability heuristic - prefer 3-8 regimes for financial data
                        stability_score = 0.0
                        if 3 <= candidate <= 8:
                            stability_score = 1.0 - (abs(candidate - 5) / 5.0)  # Peak at 5 regimes
                            
                        # Combine scores with weights favoring larger spectral gaps and stability
                        combined_score = (0.5 * gap_score) + (0.2 * modularity_score) + \
                                        (0.1 * (1 - conductance_score)) + (0.2 * stability_score)
                        candidate_scores[candidate] = combined_score
                        
                        # Log detailed score for this candidate
                        logger.debug(f"Regime count {candidate} scores - Gap: {gap_score:.3f}, " +
                                  f"Modularity: {modularity_score:.3f}, Conductance: {conductance_score:.3f}, " +
                                  f"Stability: {stability_score:.3f}, Combined: {combined_score:.3f}")
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_candidate = candidate
                    
                    # If we found a best candidate, use it
                    if best_candidate is not None:
                        n_regimes = best_candidate
                        # Describe why this candidate was chosen
                        gap_value = normalized_gaps[best_candidate-1]
                        logger.info(f"Selected {n_regimes} regimes based on spectral analysis (score: {best_score:.3f}, gap: {gap_value:.3f})")
                        
                        # Enhanced description of why this number of regimes is appropriate
                        if self.eigenvalue_profile['type'] in ["rapid-growth", "linear-growth"]:
                            logger.info(f"This regime count represents a natural division in the graph structure, " +
                                      f"with a significant spectral gap indicating well-separated clusters.")
                        elif self.eigenvalue_profile['type'] == "slow-growth":
                            logger.info(f"This regime count balances granularity and stability for dense HFT data, " +
                                      f"capturing subtle but important market behavior shifts.")
                        else:  # flat profile
                            logger.info(f"Given the relative flatness of the eigenvalue spectrum, this number of regimes " +
                                      f"represents a pragmatic balance between detail and noise in highly interconnected data.")
                    else:
                        # Fall back to using the candidate with the largest gap
                        max_gap_idx = np.argmax([normalized_gaps[i-1] for i in reasonable_candidates])
                        n_regimes = reasonable_candidates[max_gap_idx]
                        logger.info(f"Selected {n_regimes} regimes based on largest spectral gap")
                else:
                    # Default to 3-10 regimes based on graph size
                    n_regimes = min(max(3, n_nodes // 300), 10)
                    logger.info(f"No clear spectral gaps found, defaulting to {n_regimes} regimes based on graph size")
                    logger.info(f"The eigenvalue spectrum doesn't show distinct separations, suggesting " +
                                f"the data may have a more continuous structure without clear regime boundaries.")
            else:
                # Not enough eigenvalues, default to size-based regime count
                n_regimes = min(max(3, n_nodes // 300), 10)
                logger.info(f"Defaulting to {n_regimes} regimes due to insufficient spectral information")
                
            # Limit maximum number of regimes 
            n_regimes = min(n_regimes, 15)
                
            logger.info(f"Selected {n_regimes} regimes based on spectral analysis")
                
            # Step 3: Heat Kernel Signatures
            # Compute multi-scale heat kernel signatures for nodes
            # t_values represents different time scales
            t_values = [0.1, 1.0, 10.0]  # Multiple time scales
            node_hks_features = {}
            
            for node_idx, node in enumerate(G.nodes()):
                node_hks = []
                for t in t_values:
                    # HKS_t(v) = sum(e^(-λ_i * t) * φ_i(v)^2) for all i
                    hks_t = np.sum(np.exp(-eigenvalues * t) * eigenvectors[node_idx, :]**2)
                    node_hks.append(hks_t)
                node_hks_features[node] = node_hks
            
            logger.info(f"Computed Heat Kernel Signatures at {len(t_values)} time scales")
                
            # Step 4: Persistent Homology Features
            # For temporal data, incorporate temporal order in the filtration
            # Check if we have temporal information in nodes
            has_temporal = False
            try:
                has_temporal = 'mid_idx' in G.nodes[list(G.nodes())[0]]
            except:
                has_temporal = False
            
            if has_temporal:
                # Sort nodes by temporal index
                nodes_by_time = sorted(G.nodes(), key=lambda n: G.nodes[n].get('mid_idx', 0))
                
                # Compute persistence features based on temporal filtration
                persistence_features = {}
                for i, node in enumerate(nodes_by_time):
                    # Create subgraph of nodes up to this temporal point
                    temporal_subgraph = G.subgraph(nodes_by_time[:i+1])
                    
                    # Count connected components in this subgraph
                    n_components = nx.number_connected_components(temporal_subgraph)
                    
                    # Store persistence feature
                    persistence_features[node] = n_components
                
                logger.info("Computed Persistent Homology Features based on temporal filtration")
                
                # Combine with HKS features
                for node in G.nodes():
                    if node in persistence_features:
                        # Normalize and add persistence feature
                        norm_persistence = persistence_features[node] / max(persistence_features.values())
                        node_hks_features[node].append(norm_persistence)
                    else:
                        # Default value if not in persistence features
                        node_hks_features[node].append(0.0)
                        
                # Add temporal sequence information explicitly to help with temporal coherence
                node_temporal_order = {node: i for i, node in enumerate(nodes_by_time)}
                for node in G.nodes():
                    if node in node_temporal_order:
                        norm_time = node_temporal_order[node] / len(nodes_by_time)
                        node_hks_features[node].append(norm_time)
                    else:
                        node_hks_features[node].append(0.5)  # Default middle value
            
            # Create feature matrix from node features
            node_list = list(G.nodes())
            feature_matrix = np.array([node_hks_features[n] for n in node_list])
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            
            # Step 5: ENHANCED - Use size-aware HDBSCAN clustering
            # This works better than standard spectral clustering for identifying natural regimes
            try:
                from hdbscan import HDBSCAN
                
                # Get node sizes for weighted clustering
                node_sizes = np.array([len(G.nodes[n].get('points', [])) if hasattr(G.nodes[n], 'get') else 1 for n in node_list])
                
                # Adjust weights to be proportional to node sizes
                # This gives larger nodes more influence in clustering
                weights = node_sizes / np.sum(node_sizes)
                
                # Set HDBSCAN parameters based on dataset characteristics
                if n_nodes > 1000:
                    # For large graphs, use more aggressive parameters
                    min_cluster_size = max(5, n_nodes // 100)
                    min_samples = max(2, min_cluster_size // 5)
                else:
                    # For smaller graphs, use more conservative parameters
                    min_cluster_size = max(5, n_nodes // 50)
                    min_samples = max(1, min_cluster_size // 8)
                
                # IMPROVEMENT: Try multiple HDBSCAN parameter combinations to target n_regimes
                best_labels = None
                best_n_clusters = 0
                best_noise_percentage = 100.0
                best_params = {}
                
                # Define parameter ranges to search
                epsilon_options = [0.1, 0.15, 0.2, 0.25, 0.3]
                min_samples_options = [1, 2, 3, 5]
                
                # Make smaller graphs use more aggressive parameters
                if n_nodes < 500:
                    epsilon_options = [0.15, 0.2, 0.25, 0.3, 0.35]
                
                # Try multiple parameter combinations to get close to the target regime count
                logger.info(f"Trying to optimize HDBSCAN parameters to target {n_regimes} regimes")
                
                for epsilon in epsilon_options:
                    for min_s in min_samples_options:
                        try:
                            # Initialize HDBSCAN with current parameters
                            hdbscan = HDBSCAN(
                                min_cluster_size=min_cluster_size,
                                min_samples=min_s,
                                cluster_selection_epsilon=epsilon,
                                cluster_selection_method='eom',  # Excess of Mass - more natural clusters
                                metric='euclidean'
                            )
                            
                            # Fit HDBSCAN with node weights
                            clusterer = hdbscan.fit(normalized_features)
                            labels = clusterer.labels_
                            
                            # Count clusters and noise
                            unique_labels = np.unique(labels)
                            current_n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
                            noise_count = np.sum(labels == -1)
                            noise_percentage = noise_count / len(labels) * 100
                            
                            # Score how close this parameter combination gets to our target
                            # We want to minimize noise while getting close to target number of regimes
                            distance_to_target = abs(current_n_clusters - n_regimes)
                            
                            # Only consider results that are reasonable (not too few or too many clusters, not too much noise)
                            is_reasonable = (current_n_clusters >= 2 and current_n_clusters <= n_regimes * 2 and noise_percentage < 40)
                            
                            # Update best if this is better
                            if is_reasonable:
                                # If we're closer to target regimes than previous best
                                if distance_to_target < abs(best_n_clusters - n_regimes):
                                    best_labels = labels
                                    best_n_clusters = current_n_clusters
                                    best_noise_percentage = noise_percentage
                                    best_params = {'min_samples': min_s, 'epsilon': epsilon}
                                # Or if equally close but with less noise
                                elif distance_to_target == abs(best_n_clusters - n_regimes) and noise_percentage < best_noise_percentage:
                                    best_labels = labels
                                    best_n_clusters = current_n_clusters
                                    best_noise_percentage = noise_percentage
                                    best_params = {'min_samples': min_s, 'epsilon': epsilon}
                        except Exception:
                            # Continue trying other parameters if this one fails
                            continue
                
                # If we found a reasonable configuration, use it
                if best_labels is not None:
                    labels = best_labels
                    n_clusters = best_n_clusters
                    noise_percentage = best_noise_percentage
                    
                    logger.info(f"HDBSCAN optimized to {n_clusters} regimes with {noise_percentage:.1f}% noise points")
                    logger.info(f"Best parameters: min_samples={best_params['min_samples']}, epsilon={best_params['epsilon']}")
                    
                    # ENHANCEMENT: Analyze regime contrastiveness and characteristics
                    # Collect stats about what makes each regime distinct
                    try:
                        # Map nodes to their regimes
                        node_to_regime = {}
                        for i, label in enumerate(labels):
                            if label >= 0:  # Skip noise points
                                node_to_regime[node_list[i]] = label
                                
                        # Analyze regime characteristics
                        regime_nodes = {i: [] for i in range(n_clusters)}
                        for node, regime in node_to_regime.items():
                            regime_nodes[regime].append(node)
                            
                        # Calculate regime sizes
                        regime_sizes = {r: len(nodes) for r, nodes in regime_nodes.items()}
                        
                        # Quantify the contrastiveness of regimes using various metrics
                        
                        # 1. Topological distinctiveness - measure graph-theoretic differences
                        #    - Average degree within regime vs outside
                        regime_degree_stats = {}
                        for regime, nodes in regime_nodes.items():
                            if nodes:
                                # Calculate average degree within regime
                                internal_degrees = [G.degree(n) for n in nodes]
                                internal_avg_degree = sum(internal_degrees) / len(internal_degrees) if internal_degrees else 0
                                
                                # Calculate average degree overall
                                all_degrees = [G.degree(n) for n in G.nodes()]
                                overall_avg_degree = sum(all_degrees) / len(all_degrees) if all_degrees else 0
                                
                                # Ratio of internal to overall
                                degree_ratio = internal_avg_degree / overall_avg_degree if overall_avg_degree > 0 else 1.0
                                
                                regime_degree_stats[regime] = {
                                    'internal_avg_degree': internal_avg_degree,
                                    'degree_ratio': degree_ratio
                                }
                        
                        # 2. Intra vs inter regime distances - how distinct are regimes from each other?
                        regime_distance_stats = {}
                        
                        # Access the distance matrix from self rather than global scope
                        if hasattr(self, 'distance_matrix') and self.distance_matrix is not None:
                            for regime, nodes in regime_nodes.items():
                                if len(nodes) > 1:
                                    # Get node indices
                                    regime_indices = [node_list.index(n) for n in nodes if n in node_list]
                                    
                                    if len(regime_indices) > 1:
                                        # Calculate mean intra-regime distance
                                        intra_distances = []
                                        for i, idx1 in enumerate(regime_indices[:-1]):
                                            for idx2 in regime_indices[i+1:]:
                                                if idx1 < len(self.distance_matrix) and idx2 < len(self.distance_matrix):
                                                    intra_distances.append(self.distance_matrix[idx1, idx2])
                                                
                                        intra_mean = np.mean(intra_distances) if intra_distances else 0
                                        
                                        # Calculate mean distance to other regimes
                                        inter_distances = []
                                        for other_regime, other_nodes in regime_nodes.items():
                                            if other_regime != regime:
                                                other_indices = [node_list.index(n) for n in other_nodes if n in node_list]
                                                for idx1 in regime_indices:
                                                    for idx2 in other_indices:
                                                        if idx1 < len(self.distance_matrix) and idx2 < len(self.distance_matrix):
                                                            inter_distances.append(self.distance_matrix[idx1, idx2])
                                        
                                        inter_mean = np.mean(inter_distances) if inter_distances else 0
                                        
                                        # Separation ratio - higher means more distinct
                                        separation_ratio = inter_mean / intra_mean if intra_mean > 0 else 0
                                        
                                        regime_distance_stats[regime] = {
                                            'intra_distance': intra_mean,
                                            'inter_distance': inter_mean,
                                            'separation_ratio': separation_ratio
                                        }
                        
                        # 3. Feature distinctiveness in the original space
                        # If we have the original feature matrix or node embeddings, we can calculate
                        # what features are most distinctive for each regime
                        
                        # Log contrastiveness statistics
                        logger.info(f"Regime contrastiveness analysis:")
                        
                        # Log topological contrastiveness
                        if regime_degree_stats:
                            # Find regime with highest degree ratio (most different connectivity pattern)
                            most_distinct = max(regime_degree_stats.items(), key=lambda x: abs(1 - x[1]['degree_ratio']))
                            logger.info(f"Most topologically distinctive: Regime {most_distinct[0]+1} " +
                                       f"(connectivity ratio: {most_distinct[1]['degree_ratio']:.2f}x)")
                        
                        # Log separation contrastiveness
                        if regime_distance_stats:
                            # Find regimes with highest separation ratio
                            most_separated = max(regime_distance_stats.items(), key=lambda x: x[1]['separation_ratio'])
                            logger.info(f"Most distinct regime: Regime {most_separated[0]+1} " +
                                       f"(separation ratio: {most_separated[1]['separation_ratio']:.2f}x)")
                            
                            # Calculate overall quality score for HDBSCAN regimes
                            avg_separation = np.mean([stats['separation_ratio'] for stats in regime_distance_stats.values()])
                            
                            # Log the distinctiveness score
                            logger.info(f"HDBSCAN found {n_clusters} sub-regimes with score {avg_separation:.3f}")
                            
                            # Store contrastiveness data for later analysis
                            self.regime_contrastiveness = {
                                'topology': regime_degree_stats,
                                'distances': regime_distance_stats
                            }
                            
                    except Exception as e:
                        logger.warning(f"Error during regime contrastiveness analysis: {str(e)[:100]}")
                else:
                    # If optimization failed, fall back to default parameters
                    logger.warning("HDBSCAN parameter optimization failed, using default parameters")
                    
                    # Initialize HDBSCAN with default parameters
                    hdbscan = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=0.25,  # More generous epsilon
                        cluster_selection_method='eom',
                        metric='euclidean'
                    )
                    
                    # Fit HDBSCAN
                    clusterer = hdbscan.fit(normalized_features)
                    labels = clusterer.labels_
                    
                    # Count clusters and noise
                    unique_labels = np.unique(labels)
                    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
                    noise_count = np.sum(labels == -1)
                    noise_percentage = noise_count / len(labels) * 100
                    
                    logger.info(f"HDBSCAN identified {n_clusters} natural regimes with {noise_percentage:.1f}% noise points")
                
                # If HDBSCAN identifies too few or too many clusters, or too much noise, fall back to spectral clustering
                if n_clusters < 2 or n_clusters > 20 or noise_percentage > 40:
                    logger.warning(f"HDBSCAN produced {n_clusters} regimes with {noise_percentage:.1f}% noise. Using spectral clustering instead.")
                    raise ValueError("Unsuitable HDBSCAN results")
                
                # Post-process HDBSCAN results to assign all noise points (-1 labels)
                if noise_count > 0:
                    # For each noise point, find its closest non-noise cluster
                    noise_indices = np.where(labels == -1)[0]
                    non_noise_indices = np.where(labels != -1)[0]
                    
                    if len(non_noise_indices) > 0:  # Only proceed if we have some non-noise clusters
                        # Get features for non-noise points
                        non_noise_features = normalized_features[non_noise_indices]
                        non_noise_labels = labels[non_noise_indices]
                        
                        # For each noise point
                        for idx in noise_indices:
                            # Get features for this noise point
                            point_features = normalized_features[idx:idx+1]
                            
                            # Calculate distances to all non-noise points
                            from scipy.spatial.distance import cdist
                            distances = cdist(point_features, non_noise_features, 'euclidean')[0]
                            
                            # Find k nearest neighbors
                            k = min(5, len(distances))
                            nearest_indices = np.argsort(distances)[:k]
                            
                            # Get labels of k nearest neighbors
                            neighbor_labels = non_noise_labels[nearest_indices]
                            
                            # Assign to most common neighbor label
                            from scipy.stats import mode
                            most_common = mode(neighbor_labels, keepdims=False)[0]
                            labels[idx] = most_common
                
                # Check again if we have a reasonable number of regimes
                # If HDBSCAN still found fewer than desired, use spectral clustering as fallback
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels)
                
                if n_clusters < 2 or abs(n_clusters - n_regimes) > n_regimes:
                    from sklearn.cluster import SpectralClustering
                    logger.warning(f"HDBSCAN results ({n_clusters} regimes) too different from spectral target ({n_regimes}), falling back to spectral clustering")
                    
                    # Fall back to spectral clustering
                    spectral = SpectralClustering(
                        n_clusters=n_regimes,  # Use the number from spectral gap analysis
                        assign_labels='discretize',
                        random_state=42
                    )
                    labels = spectral.fit_predict(normalized_features)
                    n_clusters = n_regimes  # We explicitly set this many clusters
                
            except Exception as e:
                # Fall back to spectral clustering if HDBSCAN fails
                logger.warning(f"HDBSCAN failed: {str(e)[:100]}. Using spectral clustering instead.")
                from sklearn.cluster import SpectralClustering
                
                # Apply spectral clustering with optimal number of regimes
                spectral = SpectralClustering(
                    n_clusters=n_regimes,
                    assign_labels='discretize',
                    random_state=42
                )
                
                labels = spectral.fit_predict(normalized_features)
            
            # Step 6: ENHANCEMENT - Merge regime nodes to create more balanced clusters
            # Convert to communities format - create mapping from label to community
            label_to_community = {}
            for i, label in enumerate(labels):
                if label not in label_to_community:
                    label_to_community[label] = []
                label_to_community[label].append(node_list[i])
            
            # Convert to list of communities
            communities = list(label_to_community.values())
            
            # Remove empty communities
            communities = [c for c in communities if c]
            
            logger.info(f"Topological approach identified {len(communities)} regimes")
            
            # ENHANCEMENT: Apply regime balancing to ensure more even distribution
            communities = self._balance_regime_sizes(G, communities)
            logger.info(f"After balancing: {len(communities)} regimes")
            
            # Apply temporal coherence enhancement
            if has_temporal:
                enhanced_communities = self._enhance_temporal_coherence(G, communities)
                logger.info(f"After temporal coherence enhancement: {len(enhanced_communities)} regimes")
                
                # Ensure we didn't collapse to a single regime
                if len(enhanced_communities) < 3 and len(communities) >= 3:
                    logger.warning("Temporal enhancement reduced communities below minimum 3 - reverting to original")
                    return communities
                
                return enhanced_communities
            
            return communities
            
        except Exception as e:
            logger.error(f"Error in advanced topological regime detection: {str(e)[:200]}... Falling back to basic method")
            
            # Fallback to basic spectral clustering
            try:
                # Create adjacency matrix
                adj_matrix = nx.to_numpy_array(G)
                
                # Force at least 3 clusters
                n_clusters = 3
                
                # Apply spectral clustering
                from sklearn.cluster import SpectralClustering
                spectral = SpectralClustering(
                    n_clusters=n_clusters, 
                    affinity='precomputed',
                    assign_labels='discretize',
                    random_state=42
                )
                
                # Add small value to diagonal to ensure matrix is positive semi-definite
                np.fill_diagonal(adj_matrix, 1.0)
                
                # Apply clustering
                labels = spectral.fit_predict(adj_matrix)
                    
                # Convert to communities format
                node_list = list(G.nodes())
                spectral_communities = {}
                for i, label in enumerate(labels):
                    if label not in spectral_communities:
                        spectral_communities[label] = []
                    spectral_communities[label].append(node_list[i])
                    
                communities = list(spectral_communities.values())
                logger.info(f"Fallback spectral clustering found {len(communities)} communities")
                return communities
                
            except Exception as inner_e:
                logger.error(f"Fallback method also failed: {str(inner_e)[:100]}. Using connected components.")
                # Last resort - use connected components
                return list(nx.connected_components(G))
                
    def _balance_regime_sizes(self, G: nx.Graph, communities: List[List[str]]) -> List[List[str]]:
        """
        Balance regime sizes by merging very small regimes and splitting very large ones.
        
        Args:
            G: NetworkX graph of mapper output
            communities: List of communities (regimes)
            
        Returns:
            Balanced list of communities
        """
        if len(communities) <= 1:
            return communities
            
        # First evaluate current quality
        original_quality = self._evaluate_regime_quality(G, communities)
        logger.info(f"Initial regime quality - Modularity: {original_quality['modularity']:.3f}, " +
                   f"Conductance: {original_quality['conductance']:.3f}, Score: {original_quality['combined_score']:.3f}")
        
        # Calculate the sizes of each community
        # Weight by node sizes if available
        community_sizes = []
        node_points = {}
        
        for node in G.nodes():
            if 'points' in G.nodes[node]:
                node_points[node] = len(G.nodes[node]['points'])
            elif 'size' in G.nodes[node]:
                node_points[node] = G.nodes[node]['size']
            else:
                node_points[node] = 1
        
        for i, community in enumerate(communities):
            # Sum node sizes or use node count
            size = sum(node_points.get(node, 1) for node in community)
            community_sizes.append((i, size, community))
        
        # Sort communities by size (smallest first)
        community_sizes.sort(key=lambda x: x[1])
        
        # Calculate minimum viable size (threshold for merging)
        total_size = sum(size for _, size, _ in community_sizes)
        min_viable_size = max(total_size * 0.05, 10)  # At least 5% of total or 10 points
        logger.info(f"Minimum viable regime size calculated: {min_viable_size:.1f} points")
        
        # Calculate median size for reference
        median_size = community_sizes[len(community_sizes) // 2][1]
        
        # Find small communities that need to be merged
        small_communities = [comm for _, size, comm in community_sizes if size < min_viable_size]
        
        # Find excessively large communities that could be split
        large_threshold = median_size * 5  # Communities 5x larger than median
        large_communities = [comm for _, size, comm in community_sizes if size > large_threshold]
        
        if not small_communities and not large_communities:
            # No balancing needed
            return communities
        
        # Process small communities
        if small_communities:
            # Ensure we don't reduce the regime count below 3 unless we have to
            # Only merge communities if we'll maintain at least 3 regimes or can't help it
            target_min_regimes = 3
            
            if len(communities) - len(small_communities) < target_min_regimes and len(communities) >= target_min_regimes:
                # Calculate how many small communities we can merge while maintaining target_min_regimes
                can_merge_count = len(communities) - target_min_regimes
                
                if can_merge_count > 0:
                    # Only merge the smallest ones
                    small_communities = [comm for _, _, comm in community_sizes[:can_merge_count]]
                    logger.info(f"Limiting merging to {can_merge_count} smallest regimes (of {len(community_sizes)} total) to maintain at least {target_min_regimes} regimes")
                else:
                    # Can't merge any without going below target
                    logger.info(f"Not merging any regimes to maintain minimum of {target_min_regimes} regimes")
                    small_communities = []
            
            if small_communities:
                logger.info(f"Merging {len(small_communities)} small regimes (threshold: {min_viable_size:.1f} points)")
                
                # Remove small communities from original list
                balanced_communities = [comm for comm in communities if comm not in small_communities]
                
                # For each small community, merge it with most similar larger community
                for small_comm in small_communities:
                    best_community = None
                    best_similarity = -1
                    
                    for comm in balanced_communities:
                        if comm == small_comm:
                            continue
                        
                        # Calculate connection strength between communities
                        connections = 0
                        for node1 in small_comm:
                            for node2 in comm:
                                if G.has_edge(node1, node2):
                                    connections += 1
                        
                        # Normalize by community sizes
                        similarity = connections / (len(small_comm) * len(comm)) if (len(small_comm) * len(comm)) > 0 else 0
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_community = comm
                    
                    # If no connections found, use the largest community
                    if best_community is None:
                        best_community = max(balanced_communities, key=len)
                    
                    # Merge small community into best community
                    best_community.extend(small_comm)
                
                communities = balanced_communities
        
        # Process large communities if needed
        if large_communities and len(communities) < 3:
            logger.info(f"Splitting {len(large_communities)} large regimes")
            
            # Remove large communities from list
            remaining_communities = [comm for comm in communities if comm not in large_communities]
            
            # Split each large community
            for large_comm in large_communities:
                # Create subgraph for this community
                subgraph = G.subgraph(large_comm)
                
                # Determine number of parts to split into (based on relative size)
                large_size = sum(node_points.get(node, 1) for node in large_comm)
                n_parts = max(2, min(5, int(large_size / (median_size + 1))))
                
                try:
                    # Try spectral clustering for division
                    from sklearn.cluster import SpectralClustering
                    
                    # Get adjacency matrix for subgraph
                    sub_adj = nx.to_numpy_array(subgraph)
                    np.fill_diagonal(sub_adj, 1.0)  # Ensure positive semi-definite
                    
                    # Apply spectral clustering
                    spectral = SpectralClustering(
                        n_clusters=n_parts, 
                        affinity='precomputed',
                        random_state=42
                    )
                    sub_labels = spectral.fit_predict(sub_adj)
                    
                    # Convert to subcommunities
                    subcommunities = []
                    for i in range(n_parts):
                        nodes = [list(subgraph.nodes())[j] for j in range(len(sub_labels)) if sub_labels[j] == i]
                        if nodes:  # Only add non-empty subcommunities
                            subcommunities.append(nodes)
                    
                    # Add subcommunities to result
                    remaining_communities.extend(subcommunities)
                    
                except Exception as e:
                    logger.warning(f"Failed to split large community: {str(e)[:100]}. Using simple division.")
                    # Fall back to simple partition
                    nodes = list(large_comm)
                    n_nodes = len(nodes)
                    
                    for i in range(n_parts):
                        start = i * n_nodes // n_parts
                        end = (i + 1) * n_nodes // n_parts
                        part = nodes[start:end]
                        if part:  # Only add non-empty parts
                            remaining_communities.append(part)
            
            communities = remaining_communities
        
        # Ensure we have reasonable results
        if not communities or len(communities) < 2:
            logger.warning("Balancing produced too few communities, reverting to original")
            return communities
        
        logger.info(f"After balancing: {len(communities)} communities with sizes: {[len(c) for c in communities]}")
        logger.info(f"After balancing: {len(communities)} regimes")
        return communities
    
    def _extract_topological_features(self, G: nx.Graph) -> Dict[str, List[float]]:
        """
        Extract topological features for each node to enhance community detection.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary mapping node IDs to feature vectors
        """
        # Initialize features
        features = {node: [] for node in G.nodes()}
        
        # Calculate basic centrality measures
        try:
            # 1. Eigenvector centrality - measures node importance
            eigen_centrality = nx.eigenvector_centrality_numpy(G)
            for node, value in eigen_centrality.items():
                features[node].append(value)
                
            # 2. Betweenness centrality - measures node's bridge role
            # Use approximate betweenness for larger graphs
            if len(G) > 100:
                between_centrality = nx.betweenness_centrality(G, k=min(50, len(G) // 2))
            else:
                between_centrality = nx.betweenness_centrality(G)
                
            for node, value in between_centrality.items():
                features[node].append(value)
            
            # 3. Degree centrality - number of connections
            for node in G.nodes():
                features[node].append(G.degree(node) / max(1, len(G) - 1))
                
            # 4. Clustering coefficient - local density
            clustering = nx.clustering(G)
            for node, value in clustering.items():
                features[node].append(value)
                
            # 5. Time-based features if available 
            for node in G.nodes():
                if 'mid_idx' in G.nodes[node]:
                    # Normalize using min-max scaling
                    all_mids = [G.nodes[n].get('mid_idx', 0) for n in G.nodes()]
                    min_mid, max_mid = min(all_mids), max(all_mids)
                    norm_mid = (G.nodes[node]['mid_idx'] - min_mid) / (max_mid - min_mid + 1e-6)
                    features[node].append(norm_mid)
                else:
                    # Use default value if mid_idx not available
                    features[node].append(0.5)
                    
            # 6. Size-based feature
            if 'size' in G.nodes[list(G.nodes())[0]]:
                all_sizes = [G.nodes[n].get('size', 1) for n in G.nodes()]
                max_size = max(all_sizes)
                for node in G.nodes():
                    size = G.nodes[node].get('size', 1)
                    features[node].append(size / max_size)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating topological features: {str(e)[:100]}... Using fallback.")
            # Return default features based on degree
            for node in G.nodes():
                features[node] = [G.degree(node) / max(1, len(G) - 1)]
            return features
    
    def _calculate_topological_complexity(self, G: nx.Graph) -> Dict[str, float]:
        """
        Calculate metrics that describe the topological complexity of the graph.
        Used to determine how many regimes would be appropriate.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary with complexity metrics
        """
        metrics = {}
        
        # Basic graph metrics
        n_nodes = len(G)
        n_edges = G.number_of_edges()
        
        try:
            # Calculate average clustering coefficient
            avg_clustering = nx.average_clustering(G)
            metrics['clustering'] = avg_clustering
            
            # Calculate node degree heterogeneity (standard deviation of degrees)
            degrees = [G.degree(n) for n in G.nodes()]
            degree_mean = np.mean(degrees)
            degree_std = np.std(degrees)
            heterogeneity = degree_std / degree_mean if degree_mean > 0 else 0
            metrics['heterogeneity'] = heterogeneity
            
            # Based on these metrics, estimate appropriate number of regimes
            # Complex graphs with high heterogeneity deserve more regimes
            base_regimes = 2 + int(2 * heterogeneity + 3 * (1 - avg_clustering))
            
            # Scale based on node count
            if n_nodes < 10:
                size_factor = 0.5
            elif n_nodes < 30:
                size_factor = 1.0
            elif n_nodes < 50:
                size_factor = 1.5
            else:
                size_factor = 2.0
                
            target_regimes = max(2, min(10, int(base_regimes * size_factor)))
            metrics['target_regimes'] = target_regimes
            
            # Component count can also inform regime count
            n_components = nx.number_connected_components(G)
            metrics['components'] = n_components
            
            # Adjust target regimes if we have many components
            if n_components > target_regimes:
                metrics['target_regimes'] = n_components
                
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating complexity metrics: {str(e)[:100]}...")
            # Fallback - estimate based on graph size
            target = max(2, min(8, int(np.sqrt(n_nodes) / 2)))
            return {
                'clustering': 0.5,
                'heterogeneity': 0.5,
                'components': 1,
                'target_regimes': target
            }
    
    def _enhance_temporal_coherence(self, G: nx.Graph, communities: List[List[str]]) -> List[List[str]]:
        """
        Enhance temporal coherence of regimes by considering temporal relationships.
        
        Args:
            G: NetworkX graph of mapper output
            communities: List of communities (regimes)
            
        Returns:
            Enhanced communities with improved temporal coherence
        """
        # Skip if we have fewer than 2 communities or no temporal information
        if len(communities) < 2:
            logger.info("Skipping temporal coherence enhancement: fewer than 2 regimes")
            return communities
            
        # Skip if we already have very few regimes to preserve diversity
        # Modified: Only skip if under 3 regimes (not under 4 as before)
        if len(communities) < 3:
            logger.info("Skipping temporal coherence enhancement to preserve regime diversity")
            return communities
            
        # Check if nodes have temporal information (window indices)
        has_temporal_info = False
        node_times = {}
        
        # Try to get temporal information from nodes
        for node in G.nodes():
            if 'mid_idx' in G.nodes[node]:
                has_temporal_info = True
                node_times[node] = G.nodes[node]['mid_idx']
            elif 'time' in G.nodes[node]:
                has_temporal_info = True
                node_times[node] = G.nodes[node]['time']
                
        if not has_temporal_info:
            logger.info("Skipping temporal coherence enhancement: no temporal information available")
            return communities
        
        try:
            # Compute temporal adjacency between communities
            temporal_adjacency = np.zeros((len(communities), len(communities)))
            
            # Calculate temporal transitions between regimes
            for i, comm_i in enumerate(communities):
                for j, comm_j in enumerate(communities):
                    if i == j:
                        continue
                        
                    # Get node indices as a proxy for time
                    comm_i_times = [node_times.get(node, 0) for node in comm_i 
                                   if node in node_times]
                    comm_j_times = [node_times.get(node, 0) for node in comm_j 
                                   if node in node_times]
                    
                    if not comm_i_times or not comm_j_times:
                        continue
                    
                    # Create time ranges for each community
                    comm_i_min, comm_i_max = min(comm_i_times), max(comm_i_times)
                    comm_j_min, comm_j_max = min(comm_j_times), max(comm_j_times)
                    
                    # Compute overlap or adjacency
                    if comm_i_max >= comm_j_min and comm_i_min <= comm_j_max:
                        # Communities overlap in time
                        overlap = min(comm_i_max, comm_j_max) - max(comm_i_min, comm_j_min)
                        temporal_adjacency[i, j] = overlap
                    else:
                        # Communities are adjacent in time (use negative distance as weight)
                        distance = min(abs(comm_i_max - comm_j_min), abs(comm_i_min - comm_j_max))
                        temporal_adjacency[i, j] = -distance
            
            # Normalize the adjacency matrix
            if np.max(temporal_adjacency) > np.min(temporal_adjacency):
                temp_max = np.max(temporal_adjacency)
                temp_min = np.min(temporal_adjacency)
                temporal_adjacency = (temporal_adjacency - temp_min) / (temp_max - temp_min)
            
            # Calculate maximum number of merges to keep at least 3 regimes
            target_min_regimes = 3
            max_merges = max(0, len(communities) - target_min_regimes)
            
            if max_merges == 0:
                logger.info(f"Skipping temporal coherence: already at minimum regime count of {target_min_regimes}")
                return communities
                
            # Set the threshold for merging (higher = more selective)
            # Adjust based on number of communities
            merge_threshold = 0.5 + 0.1 * len(communities)  # Increases with more communities
            merges_done = 0
            
            # Select pairs to merge based on temporal adjacency
            pairs_to_merge = []
            for i in range(len(communities)):
                for j in range(i+1, len(communities)):
                    if temporal_adjacency[i, j] > merge_threshold:
                        pairs_to_merge.append((i, j, temporal_adjacency[i, j]))
            
            # Sort by adjacency value (highest first)
            pairs_to_merge.sort(key=lambda x: x[2], reverse=True)
            
            # Create a mapping of merges
            merged_indices = list(range(len(communities)))
            
            # Track which communities have been merged
            already_merged = set()
            
            # Apply merges
            for i, j, score in pairs_to_merge:
                # Skip if we've reached our merger limit
                if merges_done >= max_merges:
                    break
                    
                # Skip if either community has already been merged
                if i in already_merged or j in already_merged:
                    continue
                
                # Map both to the lower index
                target = min(merged_indices[i], merged_indices[j])
                for k in range(len(merged_indices)):
                    if merged_indices[k] == merged_indices[i] or merged_indices[k] == merged_indices[j]:
                        merged_indices[k] = target
                        
                # Mark as merged
                already_merged.add(i)
                already_merged.add(j)
                merges_done += 1
                logger.debug(f"Merging regimes {i} and {j} with temporal adjacency score {score:.3f}")
            
            # Only proceed if we actually did any merges
            if merges_done == 0:
                logger.info("No communities merged for temporal coherence - all below threshold")
                return communities
                
            # Create new community structure
            new_communities = []
            for idx in sorted(set(merged_indices)):
                new_comm = []
                for i, old_idx in enumerate(merged_indices):
                    if old_idx == idx:
                        new_comm.extend(communities[i])
                new_communities.append(new_comm)
            
            logger.info(f"After temporal coherence enhancement: {len(new_communities)} regimes")
            return new_communities
            
        except Exception as e:
            logger.warning(f"Error during temporal coherence enhancement: {str(e)[:100]}")
            return communities
    
    def _identify_hierarchical_regimes(self, G: nx.Graph, 
                                     primary_communities: List[List[str]]) -> Dict[int, Tuple[int, int]]:
        """
        Identify hierarchical regime structure with parent-child relationships.
        
        Args:
            G: NetworkX graph of mapper output
            primary_communities: Primary regime communities
            
        Returns:
            Dictionary mapping window indices to tuples of (parent_regime, sub_regime)
        """
        logger.info("Identifying hierarchical market regime structure...")
        
        # Create mapping from nodes to primary regimes
        node_to_primary_regime = {}
        for regime_id, community in enumerate(primary_communities):
            for node_id in community:
                node_to_primary_regime[node_id] = regime_id + 1  # Add 1 to avoid 0 regime
        
        # Create mapping from window indices to primary regimes
        window_to_primary = {}
        for node_id, regime_id in node_to_primary_regime.items():
            if node_id in self.graph['nodes']:
                points = self.graph['nodes'][node_id]
                for idx in points:
                    window_to_primary[idx] = regime_id
        
        # Create a mapping of primary regime to windows
        primary_to_windows = {}
        for window_idx, regime_id in window_to_primary.items():
            if regime_id not in primary_to_windows:
                primary_to_windows[regime_id] = []
            primary_to_windows[regime_id].append(window_idx)
        
        # For each primary regime, identify sub-regimes
        window_to_hierarchy = {}
        
        for primary_id, windows in primary_to_windows.items():
            # Skip regimes with too few windows
            if len(windows) < max(10, self.config.min_cluster_size * 2):
                logger.info(f"Primary regime {primary_id} has only {len(windows)} windows, skipping sub-regime detection")
                # Assign (primary_id, 0) to all windows in this regime
                for window_idx in windows:
                    window_to_hierarchy[window_idx] = (primary_id, 0)
                continue
            
            # Extract lens values for windows in this regime
            if self.lens is not None:
                # Convert windows list to NumPy array for indexing
                windows_array = np.array(windows)
                regime_lens = self.lens[windows_array]
                
                # Determine appropriate number of sub-regimes based on primary regime size
                # Larger primary regimes can have more sub-regimes
                n_sub_regimes = min(5, max(2, int(np.sqrt(len(windows)) / 2)))
                
                logger.info(f"Identifying {n_sub_regimes} sub-regimes for primary regime {primary_id} with {len(windows)} windows")
                
                try:
                    # Use HDBSCAN for hierarchical sub-regime detection
                    if 'HDBSCAN' in globals():
                        # ENHANCEMENT: Parallel computation for large regimes
                        n_jobs = 4 if len(windows) > 1000 else 1
                        
                        # ENHANCEMENT: Optimized parameter selection for better clustering in HFT data
                        # Use more compute-intensive approach to find optimal parameters
                        best_score = -1
                        best_labels = None
                        best_n_clusters = 0
                        
                        # Enhanced grid search for HFT data - more fine-grained parameters
                        # For dense data, we need to try smaller cluster sizes and more epsilon variations
                        min_cluster_sizes = [
                            max(5, len(windows) // 25),  # Smaller clusters for dense data
                            max(5, len(windows) // 20), 
                            max(5, len(windows) // 15),
                            max(5, len(windows) // 10)
                        ]
                        min_samples_options = [1, 2, 3, 4] if len(windows) > 200 else [1, 2]
                        alpha_options = [0.8, 0.9, 1.0, 1.1] if len(windows) > 200 else [0.85, 1.0]
                        # More fine-grained epsilon values for detecting subtle regime differences
                        epsilon_options = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2] if len(windows) > 300 else [0.05, 0.1, 0.15]
                        
                        # For extremely dense data, try even more sensitive options
                        if hasattr(self, 'eigenvalue_profile') and self.eigenvalue_profile.get('type') == 'slow-growth':
                            epsilon_options = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15]
                            min_samples_options = [1] + min_samples_options
                        
                        # Calculate number of parameter combinations for logging
                        n_combinations = len(min_cluster_sizes) * len(min_samples_options) * len(alpha_options) * len(epsilon_options)
                        logger.debug(f"Trying {n_combinations} HDBSCAN parameter combinations for sub-regime detection")
                        
                        # Track all parameter combinations and their scores for analysis
                        param_scores = []
                        
                        # Grid search over different parameter combinations 
                        for min_cluster_size in min_cluster_sizes:
                            for min_samples in min_samples_options:
                                for alpha in alpha_options:
                                    for eps in epsilon_options:
                                        try:
                                            hdbscan = HDBSCAN(
                                                min_cluster_size=min_cluster_size,
                                                min_samples=min_samples,
                                                alpha=alpha,
                                                cluster_selection_epsilon=eps,
                                                cluster_selection_method='eom',  # Excess of mass usually works better for HFT data
                                                n_jobs=n_jobs
                                            )
                                            sub_labels = hdbscan.fit_predict(regime_lens)
                                            
                                            # Determine number of actual clusters (excluding noise)
                                            unique_labels = [label for label in np.unique(sub_labels) if label != -1]
                                            n_actual_clusters = len(unique_labels)
                                            
                                            # Only consider if we found multiple clusters
                                            if n_actual_clusters >= 2:
                                                # Calculate silhouette score if at least 2 clusters
                                                # Include only non-noise points
                                                non_noise_mask = sub_labels != -1
                                                non_noise_count = np.sum(non_noise_mask)
                                                
                                                if non_noise_count > n_actual_clusters:
                                                    from sklearn.metrics import silhouette_score
                                                    try:
                                                        if len(unique_labels) > 1:
                                                            # Calculate silhouette score
                                                            score = silhouette_score(
                                                                regime_lens[non_noise_mask], 
                                                                sub_labels[non_noise_mask]
                                                            )
                                                            
                                                            # Calculate noise ratio
                                                            noise_ratio = 1.0 - (non_noise_count / len(sub_labels))
                                                            
                                                            # Get probabilities if available (for confidence measurement)
                                                            cluster_confidence = 0.8  # Default
                                                            if hasattr(hdbscan, 'probabilities_'):
                                                                cluster_confidence = np.mean(hdbscan.probabilities_[non_noise_mask])
                                                            
                                                            # ENHANCEMENT FOR HFT DATA: Improved scoring
                                                            # Previous formula: score * (0.8 + 0.2 * n_actual_clusters / 5) * mapped_ratio
                                                            mapped_ratio = np.sum(non_noise_mask) / len(sub_labels)
                                                            
                                                            # More balanced formula for HFT data:
                                                            # 1. Reward silhouette score (cluster separation)
                                                            # 2. Reward higher number of clusters (but not too many)
                                                            # 3. Penalize too much noise (but accept some)
                                                            # 4. Reward higher cluster confidence
                                                            
                                                            # Penalize very high cluster counts (likely noise or overfitting)
                                                            cluster_penalty = 1.0
                                                            if n_actual_clusters > 10:
                                                                cluster_penalty = 0.7
                                                                
                                                            # Calculate optimal noise ratio (some noise is good to filter outliers)
                                                            optimal_noise = min(0.2, 50 / len(sub_labels))  # Adaptive based on size
                                                            noise_score = 1.0 - abs(noise_ratio - optimal_noise) * 3  # Penalize deviation from optimal
                                                            noise_score = max(0.5, min(1.0, noise_score))  # Bound between 0.5 and 1.0
                                                            
                                                            # Calculate final adjusted score
                                                            adjusted_score = (
                                                                score * 0.4 +                                       # Silhouette weight: 40%
                                                                min(1.0, n_actual_clusters / n_sub_regimes) * 0.3 + # Cluster count weight: 30%
                                                                noise_score * 0.15 +                               # Noise score weight: 15%
                                                                cluster_confidence * 0.15                          # Confidence weight: 15%
                                                            ) * cluster_penalty
                                                            
                                                            # Scale for target cluster count - give slight bonus for matching target
                                                            if abs(n_actual_clusters - n_sub_regimes) <= 1:
                                                                adjusted_score *= 1.1
                                                            
                                                            # Store parameters and score for later analysis
                                                            if best_score < 0 or best_labels is None:
                                                                # Log parameters for first successful run
                                                                logger.debug(f"First successful HDBSCAN parameters: min_cluster_size={min_cluster_size}, " +
                                                                          f"min_samples={min_samples}, alpha={alpha:.2f}, epsilon={eps}")
                                                            
                                                            if adjusted_score > best_score:
                                                                best_score = adjusted_score
                                                                best_labels = sub_labels
                                                                best_n_clusters = n_actual_clusters
                                                                best_params = {
                                                                    'min_cluster_size': min_cluster_size,
                                                                    'min_samples': min_samples,
                                                                    'alpha': alpha,
                                                                    'epsilon': eps
                                                                }
                                                                logger.debug(f"New best score: {adjusted_score:.3f} with {n_actual_clusters} clusters, " + 
                                                                          f"noise={noise_ratio:.2f}, silhouette={score:.3f}")
                                                    except:
                                                        # Silhouette score can fail in certain edge cases
                                                        continue
                                        except:
                                            # Skip this parameter combination if it fails
                                            continue
                        
                        # If we found a good clustering
                        if best_score > 0 and best_labels is not None:
                            logger.info(f"HDBSCAN found {best_n_clusters} sub-regimes for primary regime {primary_id} with score {best_score:.3f}")
                            
                            # ENHANCEMENT: Analyze contrastiveness between sub-regimes
                            try:
                                unique_clusters = np.unique(best_labels)
                                valid_clusters = [c for c in unique_clusters if c >= 0]
                                
                                # Create mapping from clusters to points
                                cluster_to_points = {}
                                for cluster in valid_clusters:
                                    cluster_to_points[cluster] = np.where(best_labels == cluster)[0]
                                
                                # Calculate characteristic features of each sub-regime
                                if self.distance_matrix is not None and len(windows) > 0:
                                    # 1. Calculate intra-cluster vs inter-cluster distances
                                    intra_cluster_distances = {}
                                    inter_cluster_distances = {}
                                    
                                    for cluster, points in cluster_to_points.items():
                                        if len(points) > 1:
                                            # Get actual window indices
                                            point_indices = [windows[i] for i in points]
                                            
                                            # Calculate intra-cluster distances (within cluster)
                                            intra_distances = []
                                            for i in range(len(point_indices)):
                                                for j in range(i+1, len(point_indices)):
                                                    idx1, idx2 = point_indices[i], point_indices[j]
                                                    if idx1 < len(self.distance_matrix) and idx2 < len(self.distance_matrix):
                                                        intra_distances.append(self.distance_matrix[idx1, idx2])
                                            
                                            if intra_distances:
                                                intra_cluster_distances[cluster] = {
                                                    'mean': float(np.mean(intra_distances)),
                                                    'median': float(np.median(intra_distances)),
                                                    'min': float(np.min(intra_distances)),
                                                    'max': float(np.max(intra_distances))
                                                }
                                                
                                            # Calculate inter-cluster distances (between clusters)
                                            for other_cluster, other_points in cluster_to_points.items():
                                                if cluster != other_cluster:
                                                    other_indices = [windows[i] for i in other_points]
                                                    
                                                    between_distances = []
                                                    for idx1 in point_indices:
                                                        for idx2 in other_indices:
                                                            if idx1 < len(self.distance_matrix) and idx2 < len(self.distance_matrix):
                                                                between_distances.append(self.distance_matrix[idx1, idx2])
                                                    
                                                    if between_distances:
                                                        key = (cluster, other_cluster)
                                                        inter_cluster_distances[key] = {
                                                            'mean': float(np.mean(between_distances)),
                                                            'median': float(np.median(between_distances)),
                                                            'min': float(np.min(between_distances)),
                                                            'max': float(np.max(between_distances))
                                                        }
                                    
                                    # Log cluster separation metrics
                                    if intra_cluster_distances and inter_cluster_distances:
                                        # Calculate separation ratios for each cluster
                                        separation_metrics = {}
                                        for cluster in valid_clusters:
                                            if cluster in intra_cluster_distances:
                                                intra_mean = intra_cluster_distances[cluster]['mean']
                                                # Get mean of all inter-cluster distances involving this cluster
                                                inter_distances = []
                                                for key, metrics in inter_cluster_distances.items():
                                                    if key[0] == cluster or key[1] == cluster:
                                                        inter_distances.append(metrics['mean'])
                                                
                                                if inter_distances and intra_mean > 0:
                                                    inter_mean = np.mean(inter_distances)
                                                    
                                                    # ENHANCED METRICS FOR HFT DATA SENSITIVITY
                                                    # 1. Traditional separation ratio (higher = more distinct)
                                                    separation_ratio = inter_mean / intra_mean
                                                    
                                                    # 2. Distribution overlap coefficient (lower = more distinct)
                                                    # Calculate lower overlap percentage between intra/inter distance distributions
                                                    intra_dist = np.array([d for key, values in intra_cluster_distances.items() 
                                                                      for d in [values['mean']] if key == cluster])
                                                    inter_dist = np.array([d for key, values in inter_cluster_distances.items() 
                                                                      for d in [values['mean']] if key[0] == cluster or key[1] == cluster])
                                                    
                                                    # Calculate overlap coefficient using histogram intersection method
                                                    try:
                                                        hist1, bin_edges = np.histogram(intra_dist, bins=10, density=True)
                                                        hist2, _ = np.histogram(inter_dist, bins=bin_edges, density=True)
                                                        # Histogram intersection
                                                        overlap_coef = np.sum(np.minimum(hist1, hist2)) * (bin_edges[1] - bin_edges[0])
                                                    except:
                                                        overlap_coef = 0.5  # Default if calculation fails
                                                    
                                                    # 3. Wasserstein distance (higher = more distinct)
                                                    try:
                                                        from scipy.stats import wasserstein_distance
                                                        wd = wasserstein_distance(intra_dist, inter_dist)
                                                    except:
                                                        wd = 0.0  # Default if calculation fails
                                                    
                                                    # 4. Statistical significance using t-test (higher = more significant)
                                                    try:
                                                        from scipy.stats import ttest_ind
                                                        t_stat, p_value = ttest_ind(intra_dist, inter_dist, equal_var=False)
                                                        # Check for NaN - this happens when distributions are identical or too small
                                                        if np.isnan(p_value):
                                                            significance = 0.5  # Neutral score for identical distributions
                                                        else:
                                                            significance = 1.0 - min(p_value, 0.99)  # Convert p-value to significance score
                                                    except Exception as e:
                                                        logger.debug(f"Error calculating significance: {str(e)[:50]}")
                                                        significance = 0.5  # Default if calculation fails
                                                    
                                                    # 5. Quantile-based separation (more robust to outliers)
                                                    try:
                                                        intra_q75 = np.percentile(intra_dist, 75)
                                                        inter_q25 = np.percentile(inter_dist, 25)
                                                        quantile_separation = (inter_q25 - intra_q75) / (intra_q75 + 1e-10)
                                                    except:
                                                        quantile_separation = 0.0
                                                    
                                                    # Store all metrics
                                                    separation_metrics[cluster] = {
                                                        'ratio': separation_ratio,
                                                        'overlap': 1.0 - overlap_coef,  # Invert so higher = more distinct
                                                        'wasserstein': wd,
                                                        'significance': significance,
                                                        'quantile_sep': quantile_separation,
                                                        # Combined score
                                                        'combined': (separation_ratio + (1.0 - overlap_coef) + 
                                                                    (wd * 10) + significance + 
                                                                    max(0, quantile_separation)) / 5.0
                                                    }
                                        
                                        # Find most and least distinct clusters using combined score
                                        if separation_metrics:
                                            # Check for NaN values in combined scores and filter them out
                                            valid_metrics = {k: v for k, v in separation_metrics.items() 
                                                           if not np.isnan(v['combined'])}
                                            
                                            if valid_metrics:
                                                # Find cluster with highest combined score
                                                most_distinct = max(valid_metrics.items(), 
                                                                  key=lambda x: x[1]['combined'])
                                                
                                                # Find cluster with lowest combined score
                                                least_distinct = min(valid_metrics.items(), 
                                                                   key=lambda x: x[1]['combined'])
                                            else:
                                                # If all are NaN, just use first cluster as a fallback
                                                first_cluster = next(iter(separation_metrics.items()))
                                                most_distinct = least_distinct = first_cluster
                                            
                                            # Calculate average scores across all metrics, handling NaN values
                                            avg_separation = np.nanmean([m['ratio'] for m in separation_metrics.values()])
                                            avg_overlap = np.nanmean([m['overlap'] for m in separation_metrics.values()])
                                            avg_wasserstein = np.nanmean([m['wasserstein'] for m in separation_metrics.values()])
                                            avg_significance = np.nanmean([m['significance'] for m in separation_metrics.values()])
                                            avg_combined = np.nanmean([m['combined'] for m in separation_metrics.values()])
                                            
                                            # Calculate microstructure distinctiveness for HFT data
                                            # Get window data for microstructure analysis
                                            try:
                                                microstructure_distinctiveness = []
                                                
                                                # Extract window data samples for each cluster
                                                cluster_windows = {}
                                                for c in valid_clusters:
                                                    point_indices = [windows[i] for i in cluster_to_points[c]]
                                                    if point_indices and len(point_indices) > 5:
                                                        cluster_windows[c] = point_indices
                                                
                                                # Calculate microstructure differences if we have the original windows
                                                if hasattr(self, 'windows') and len(self.windows) > 0:
                                                    for c1, indices1 in cluster_windows.items():
                                                        for c2, indices2 in cluster_windows.items():
                                                            if c1 < c2:  # Compare each pair once
                                                                # Calculate return distribution differences (HFT sensitive)
                                                                try:
                                                                    # Get sample windows for each cluster
                                                                    sample_size = min(20, len(indices1), len(indices2))
                                                                    c1_samples = np.random.choice(indices1, size=sample_size, replace=False)
                                                                    c2_samples = np.random.choice(indices2, size=sample_size, replace=False)
                                                                    
                                                                    # Extract actual window data
                                                                    windows1 = [self.windows[i] for i in c1_samples if i < len(self.windows)]
                                                                    windows2 = [self.windows[i] for i in c2_samples if i < len(self.windows)]
                                                                    
                                                                    if windows1 and windows2:
                                                                        # For each window, calculate return statistics
                                                                        def calc_micro_stats(win):
                                                                            if len(win.shape) > 1 and win.shape[1] > 0:
                                                                                # Use first dimension as price
                                                                                prices = win[:, 0]
                                                                                # Calculate returns
                                                                                returns = np.diff(prices) / (prices[:-1] + 1e-10)
                                                                                # Calculate microstructure statistics
                                                                                return {
                                                                                    'std': np.std(returns),
                                                                                    'skew': scipy.stats.skew(returns) if 'scipy.stats' in globals() else 0,
                                                                                    'kurtosis': scipy.stats.kurtosis(returns) if 'scipy.stats' in globals() else 0,
                                                                                    'abs_mean': np.mean(np.abs(returns)),
                                                                                    'jumps': np.sum(np.abs(returns) > np.std(returns) * 3) / len(returns)
                                                                                }
                                                                            return None
                                                                        
                                                                        # Calculate stats for each cluster
                                                                        stats1 = [calc_micro_stats(w) for w in windows1 if w is not None]
                                                                        stats2 = [calc_micro_stats(w) for w in windows2 if w is not None]
                                                                        
                                                                        # Remove None values
                                                                        stats1 = [s for s in stats1 if s is not None]
                                                                        stats2 = [s for s in stats2 if s is not None]
                                                                        
                                                                        if stats1 and stats2:
                                                                            # Calculate differences in distributions
                                                                            diffs = []
                                                                            for key in ['std', 'skew', 'kurtosis', 'abs_mean', 'jumps']:
                                                                                vals1 = np.array([s[key] for s in stats1])
                                                                                vals2 = np.array([s[key] for s in stats2])
                                                                                mean1, mean2 = np.mean(vals1), np.mean(vals2)
                                                                                # Normalize difference
                                                                                denom = max(abs(mean1), abs(mean2), 1e-10)
                                                                                diff = abs(mean1 - mean2) / denom
                                                                                diffs.append(diff)
                                                                            
                                                                            # Average difference across all metrics
                                                                            avg_diff = np.mean(diffs)
                                                                            microstructure_distinctiveness.append(avg_diff)
                                                                except Exception as e:
                                                                    logger.debug(f"Error in microstructure analysis: {str(e)[:100]}")
                                                
                                                # Summarize microstructure distinctiveness
                                                micro_distinct = np.mean(microstructure_distinctiveness) if microstructure_distinctiveness else 0.0
                                            except Exception as e:
                                                logger.debug(f"Error in microstructure distinctiveness calculation: {str(e)[:100]}")
                                                micro_distinct = 0.0
                                            
                                            # ENHANCEMENT: Add temporal analysis of sub-regimes
                                            # Check if sub-regimes are sequential or concurrent in time
                                            try:
                                                # For each cluster, get its temporal span
                                                cluster_time_spans = {}
                                                for cluster, points in cluster_to_points.items():
                                                    # Get window indices
                                                    window_indices = [windows[i] for i in points]
                                                    # Sort by time (window index is a proxy for time in sequential data)
                                                    sorted_indices = sorted(window_indices)
                                                    if sorted_indices:
                                                        cluster_time_spans[cluster] = (min(sorted_indices), max(sorted_indices))
                                                
                                                # Calculate overlap between clusters
                                                overlap_percentages = []
                                                for c1 in cluster_time_spans:
                                                    for c2 in cluster_time_spans:
                                                        if c1 < c2:  # Check each pair once
                                                            span1 = cluster_time_spans[c1]
                                                            span2 = cluster_time_spans[c2]
                                                            # Calculate overlap
                                                            overlap_start = max(span1[0], span2[0])
                                                            overlap_end = min(span1[1], span2[1])
                                                            if overlap_end >= overlap_start:
                                                                # There is overlap
                                                                overlap_length = overlap_end - overlap_start + 1
                                                                span1_length = span1[1] - span1[0] + 1
                                                                span2_length = span2[1] - span2[0] + 1
                                                                # Calculate percentage of overlap relative to shortest span
                                                                min_span = min(span1_length, span2_length)
                                                                overlap_pct = (overlap_length / min_span) * 100
                                                                overlap_percentages.append(overlap_pct)
                                                            else:
                                                                # No overlap
                                                                overlap_percentages.append(0)
                                                
                                                # Determine if sub-regimes are concurrent or sequential
                                                avg_overlap = np.mean(overlap_percentages) if overlap_percentages else 0
                                                
                                                # Store temporal analysis for later use
                                                self.sub_regime_temporal = {
                                                    primary_id: {
                                                        'time_spans': cluster_time_spans,
                                                        'avg_overlap': avg_overlap,
                                                        'sequential': avg_overlap < 30
                                                    }
                                                }
                                                
                                            except Exception as e:
                                                logger.warning(f"Error during temporal analysis: {str(e)[:100]}")
                                                avg_overlap = 0
                                            
                                            # Log detailed separation information
                                            logger.info(f"Sub-regime contrastiveness for primary regime {primary_id}:")
                                            logger.info(f"  - Average separation ratio: {avg_separation:.3f}")
                                            logger.info(f"  - Distributional distinctness: {avg_overlap*100:.3f}")
                                            logger.info(f"  - Statistical significance: {avg_significance:.3f}")
                                            logger.info(f"  - Microstructure distinctiveness: {micro_distinct:.3f}")
                                            
                                            # Log temporal analysis only if it's interesting (not always close to 100%)
                                            if avg_overlap < 95.0:
                                                logger.info(f"  - Sub-regime temporal analysis: {avg_overlap:.1f}% average overlap")
                                                
                                                # Only log regime type for non-concurrent regimes
                                                if avg_overlap < 70.0:
                                                    if avg_overlap > 30:
                                                        logger.info("  - Sub-regimes show mixed temporal behavior (partially overlapping)")
                                                    else:
                                                        logger.info("  - Sub-regimes appear to be sequential in time (regime shifts)")
                                            
                                            # Format scores to handle potential NaN values
                                            try:
                                                most_distinct_score = most_distinct[1]['combined']
                                                most_distinct_score_str = f"{most_distinct_score:.3f}" if not np.isnan(most_distinct_score) else "N/A"
                                            except:
                                                most_distinct_score_str = "N/A"
                                                
                                            try:
                                                least_distinct_score = least_distinct[1]['combined']
                                                least_distinct_score_str = f"{least_distinct_score:.3f}" if not np.isnan(least_distinct_score) else "N/A"
                                            except:
                                                least_distinct_score_str = "N/A"
                                            
                                            # Only log most/least distinct if there are actually multiple sub-regimes
                                            if len(valid_clusters) > 1:
                                                logger.info(f"  - Most distinct sub-regime: {most_distinct[0]+1} (score: {most_distinct_score_str})")
                                                logger.info(f"  - Least distinct sub-regime: {least_distinct[0]+1} (score: {least_distinct_score_str})")
                                            
                                            # Calculate overall regime meaningfulness score (0-1)
                                            regime_meaningful = min(1.0, (avg_separation * 0.3 + 
                                                                        avg_overlap * 0.2 + 
                                                                        avg_significance * 0.2 +
                                                                        avg_wasserstein * 10 * 0.1 + 
                                                                        micro_distinct * 0.2))
                                            
                                            # Handle potential NaN values in regime_meaningful
                                            if np.isnan(regime_meaningful):
                                                regime_meaningful = 0.5  # Default to medium distinctness if calculation fails
                                            
                                            # Only log quality message if it's meaningful (when there are multiple sub-regimes)
                                            if len(valid_clusters) > 1:
                                                if regime_meaningful < 0.3:
                                                    logger.warning(f"  - Sub-regimes may not be sufficiently distinct (score: {regime_meaningful:.3f})")
                                                elif regime_meaningful > 0.7:
                                                    logger.info(f"  - Sub-regimes are highly distinct (score: {regime_meaningful:.3f})")
                                                else:
                                                    logger.info(f"  - Sub-regimes are moderately distinct (score: {regime_meaningful:.3f})")
                            except Exception as e:
                                logger.warning(f"Error during contrastiveness analysis: {str(e)}")
                                    
                            # Assign windows to sub-regimes
                            for i, sub_label in enumerate(best_labels):
                                window_idx = windows[i]
                                if sub_label >= 0:  # Skip noise points
                                    # Store as (primary_regime, sub_regime)
                                    window_to_hierarchy[window_idx] = (primary_id, sub_label + 1)  # +1 to make 1-indexed
                                else:
                                    # Assign noise points to primary regime only
                                    window_to_hierarchy[window_idx] = (primary_id, 0)
                    
                    # Fall back to K-means for sub-regime clustering (faster and more stable)
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_sub_regimes, n_init=10, random_state=42)
                    sub_labels = kmeans.fit_predict(regime_lens)
                    
                    # Map windows to sub-regimes
                    for i, window_idx in enumerate(windows):
                        sub_regime = sub_labels[i] + 1  # Add 1 to avoid 0 sub-regime
                        window_to_hierarchy[window_idx] = (primary_id, sub_regime)
                    
                except Exception as e:
                    logger.error(f"Error identifying sub-regimes for primary regime {primary_id}: {e}")
                    # Default assignment
                    for window_idx in windows:
                        window_to_hierarchy[window_idx] = (primary_id, 0)
            else:
                # No lens data available, assign default hierarchy
                for window_idx in windows:
                    window_to_hierarchy[window_idx] = (primary_id, 0)
        
        # Log results
        sub_regime_counts = {}
        for (primary, sub) in window_to_hierarchy.values():
            key = f"{primary}_{sub}"
            if key not in sub_regime_counts:
                sub_regime_counts[key] = 0
            sub_regime_counts[key] += 1
        
        logger.info(f"Identified hierarchical regime structure with {len(sub_regime_counts)} distinct sub-regimes")
        
        return window_to_hierarchy
    
    def _apply_temporal_smoothing(self, regime_sequence, dates=None):
        """
        Apply temporal smoothing to regime sequence to reduce spurious switches.
        
        Args:
            regime_sequence: Detected regime sequence
            dates: Corresponding date sequence
            
        Returns:
            Smoothed regime sequence
        """
        if len(regime_sequence) <= 1:
            return regime_sequence

        # If regime sequence contains None values (as indicators for noise)
        # replace them with a specific "noise regime" marker for processing
        NOISE_MARKER = -999
        working_sequence = np.array([r if r is not None else NOISE_MARKER for r in regime_sequence])
        
        # Calculate run lengths
        runs = []
        current_regime = working_sequence[0]
        current_run_start = 0
        
        for i in range(1, len(working_sequence)):
            if working_sequence[i] != current_regime:
                run_length = i - current_run_start
                runs.append((current_regime, current_run_start, i-1, run_length))
                current_regime = working_sequence[i]
                current_run_start = i
        
        # Add the last run
        run_length = len(working_sequence) - current_run_start
        runs.append((current_regime, current_run_start, len(working_sequence)-1, run_length))
        
        # If we have fewer than 3 runs, no smoothing needed
        if len(runs) < 3:
            return regime_sequence
        
        # ENHANCED: Calculate adaptive thresholds based on data characteristics
        run_lengths = [run[3] for run in runs]
        median_run_length = np.median(run_lengths)
        # Reduce the threshold for considering a run "short" to preserve more regimes
        short_run_threshold = max(3, int(median_run_length * 0.3))  # Was 0.5, now 0.3
        
        # Log diagnostics about run lengths
        if hasattr(self, 'verbose') and self.verbose:
            logger.info(f"Run lengths: min={min(run_lengths)}, median={median_run_length}, max={max(run_lengths)}")
            logger.info(f"Short run threshold: {short_run_threshold}")
            
        # ENHANCED: Adaptive short run handling
        for i in range(1, len(runs) - 1):
            prev_regime, prev_start, prev_end, prev_length = runs[i-1]
            curr_regime, curr_start, curr_end, curr_length = runs[i]
            next_regime, next_start, next_end, next_length = runs[i+1]
            
            # Only smooth very short runs surrounded by the same regime
            if curr_length <= short_run_threshold:
                # ENHANCED: Be more conservative about smoothing
                # 1. Don't smooth if the run is at least 20% of the median length
                # (previously we would smooth all "short" runs)
                if curr_length >= 0.2 * median_run_length and curr_length > 2:
                    continue
                    
                # 2. Don't smooth NOISE_MARKER runs (these are important markers)
                if curr_regime == NOISE_MARKER:
                    continue
                    
                # 3. Only smooth if surrounded by the same regime
                if prev_regime == next_regime:
                    # Looks like a transient regime - replace with surrounding regime
                    working_sequence[curr_start:curr_end+1] = prev_regime
                
                # 4. NEW LOGIC: For A-B-A patterns where B is significant but short 
                # Consider the "intensity" of the B regime as a percentage of nearby points
                # to decide whether to keep it or smooth it
                elif prev_regime == next_regime and curr_length > 2:
                    # Calculate intensity by looking at window around the run
                    window_size = min(10, 2 * curr_length)
                    window_start = max(0, curr_start - window_size)
                    window_end = min(len(working_sequence), curr_end + window_size + 1)
                    
                    # Count how many B points exist in the window
                    b_count = np.sum(working_sequence[window_start:window_end] == curr_regime)
                    window_length = window_end - window_start
                    
                    # If B has significant presence, preserve it
                    if b_count / window_length > 0.25:
                        continue  # Skip smoothing to preserve this regime
                    else:
                        # This is likely a transient - smooth it out
                        working_sequence[curr_start:curr_end+1] = prev_regime
        
        # NEW FEATURE: Ensure we didn't over-smooth and collapse to a single regime
        unique_regimes_after = len(np.unique(working_sequence))
        unique_regimes_before = len(np.unique([r if r is not None else NOISE_MARKER for r in regime_sequence]))
        
        # If we've collapsed to a single regime, restore some diversity
        if unique_regimes_after == 1 and unique_regimes_before > 1:
            logger.warning("Temporal smoothing collapsed all regimes - restoring original sequence")
            # Revert to original sequence
            return regime_sequence
        
        # Convert back to original format with None for noise markers
        smoothed_sequence = [int(r) if r != NOISE_MARKER else None for r in working_sequence]
        
        # Log smoothing results
        if hasattr(self, 'verbose') and self.verbose:
            original_transitions = sum(1 for i in range(1, len(regime_sequence)) 
                                  if regime_sequence[i] != regime_sequence[i-1])
            smoothed_transitions = sum(1 for i in range(1, len(smoothed_sequence)) 
                                   if smoothed_sequence[i] != smoothed_sequence[i-1])
            
            logger.info(f"Temporal smoothing reduced transitions: {original_transitions} → {smoothed_transitions}")
        
        return smoothed_sequence
    
    def _fill_regime_gaps(self, regime_labels: np.ndarray) -> np.ndarray:
        """
        Fill gaps in regime labels using forward and backward filling.
        
        Args:
            regime_labels: Regime labels array with possible gaps
            
        Returns:
            Filled regime labels
        """
        # Count initial zeros before filling
        initial_zeros = np.sum(regime_labels == 0)
        if initial_zeros == 0:
            return regime_labels
            
        # Create a copy to avoid modifying the original
        filled_labels = regime_labels.copy()
        
        # Forward fill
        for i in range(1, len(filled_labels)):
            if filled_labels[i] == 0 and filled_labels[i-1] > 0:
                filled_labels[i] = filled_labels[i-1]
        
        # Backward fill for remaining zeros
        for i in range(len(filled_labels)-2, -1, -1):
            if filled_labels[i] == 0 and filled_labels[i+1] > 0:
                filled_labels[i] = filled_labels[i+1]
        
        # Count remaining zeros after filling
        remaining_zeros = np.sum(filled_labels == 0)
        
        # If there are still zeros, fill with most common regime
        if remaining_zeros > 0:
            # Find most common regime
            regimes, counts = np.unique(filled_labels[filled_labels > 0], return_counts=True)
            if len(regimes) > 0:
                most_common = regimes[np.argmax(counts)]
                filled_labels[filled_labels == 0] = most_common
            else:
                # If no non-zero regimes, use regime 1
                filled_labels[filled_labels == 0] = 1
        
        logger.info(f"Filled {initial_zeros - remaining_zeros} regime gaps with forward/backward fill")
        if remaining_zeros > 0:
            logger.info(f"Filled remaining {remaining_zeros} regime gaps with most common regime")
        
        return filled_labels

    def _create_cover(self, lens: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        Create a cover of the lens space using overlapping intervals.
        Optimized for financial data to create balanced interval sizes.
        
        Args:
            lens: 2D lens representation of data points
            
        Returns:
            Tuple of (hypercubes, bin_edges_0, bin_edges_1)
        """
        # Extract parameters from config
        n_intervals = self.config.n_intervals
        overlap_percentage = self.config.overlap_percentage
        
        # ENHANCEMENT: Detect if we're using sliding windows with stride=1
        # This is a common pattern in financial time series analysis
        is_sliding_window = False
        if hasattr(self, 'windows') and len(self.windows) > 2:
            # Check if windows are overlapping with stride 1
            window_size = self.config.window_size
            if self.windows[0].shape[0] == window_size and self.windows[1].shape[0] == window_size:
                # Check if the window indices are sequential
                # We can't directly use set operations with numpy arrays
                try:
                    # For time series data, compare the timestamps or indices
                    # If windows[i+1] contains same data as windows[i] shifted by 1, we have stride=1
                    if len(lens) > len(self.windows): 
                        # Use a safer method that works with numpy arrays
                        # Just check if the starting indices are consecutive
                        # This is a simplification but works well enough for most cases
                        is_sliding_window = True
                        logger.info("Detected sliding windows with stride=1, optimizing cover creation")
                except Exception as e:
                    logger.debug(f"Error detecting sliding windows: {str(e)}")
                    # Default to false on error
                    is_sliding_window = False
        
        # NEW: Enhanced cover strategy for topological feature extraction
        # For HFT/financial data, we want more balanced intervals and adaptive overlap
        
        # Calculate data density and distribution characteristics
        density = self._estimate_lens_density(lens)
        
        # Compute lens space statistics to guide cover parameters
        lens_stats = {}
        for dim in range(lens.shape[1]):
            dim_values = lens[:, dim]
            lens_stats[f'dim{dim}_mean'] = float(np.mean(dim_values))
            lens_stats[f'dim{dim}_std'] = float(np.std(dim_values))
            # Calculate skewness to identify asymmetric distributions
            lens_stats[f'dim{dim}_skew'] = float(np.mean(((dim_values - np.mean(dim_values)) / np.std(dim_values)) ** 3))
            
            # Identify outliers using IQR method
            q1, q3 = np.percentile(dim_values, [25, 75])
            iqr = q3 - q1
            lens_stats[f'dim{dim}_outliers'] = np.sum((dim_values < q1 - 1.5 * iqr) | (dim_values > q3 + 1.5 * iqr))
            
        logger.info(f"Lens space density: {density:.4f}, stats: {lens_stats}")
        
        # Adjust intervals and overlap based on lens space characteristics
        effective_intervals = n_intervals
        effective_overlap = overlap_percentage
        
        # For high-density lens spaces, increase intervals and overlap
        if density > 0.9:
            effective_intervals = min(30, n_intervals + 5)
            effective_overlap = min(0.8, overlap_percentage + 0.1)
            logger.info(f"High density lens space: increased intervals={effective_intervals}, overlap={effective_overlap:.2f}")
        
        # For sparse lens spaces, adjust intervals based on data distribution 
        elif density < 0.3:
            effective_intervals = max(10, n_intervals - 2)
            logger.info(f"Low density lens space: adjusted intervals={effective_intervals}")
            
        # For highly skewed data, use adaptive binning
        use_adaptive_binning = False
        for dim in range(lens.shape[1]):
            if abs(lens_stats[f'dim{dim}_skew']) > 1.0 or lens_stats[f'dim{dim}_outliers'] > len(lens) * 0.05:
                use_adaptive_binning = True
                logger.info(f"Detected skewed distribution, using adaptive binning")
                break
                
        # For stride=1 sliding windows, we need special treatment
        if is_sliding_window:
            # Use special cover strategy for sliding windows:
            # 1. Increase effective overlap to ensure points are captured in multiple intervals
            effective_overlap = max(0.75, overlap_percentage)
            # 2. Use more intervals to better capture the structure 
            effective_intervals = min(25, n_intervals + 5)
            
            logger.info(f"Using special cover for sliding windows: intervals={effective_intervals}, overlap={effective_overlap:.2f}")
            
            # Create more intervals but with higher overlap for sliding windows
            if use_adaptive_binning:
                hypercubes, bin_edges_0, bin_edges_1 = self._create_adaptive_cover(
                    lens, effective_intervals, effective_overlap
                )
            else:
                hypercubes, bin_edges_0, bin_edges_1 = self._create_balanced_cover(
                    lens, effective_intervals, effective_overlap
                )
        else:
            # Use standard cover creation for normal data with either balanced or adaptive approach
            if use_adaptive_binning:
                hypercubes, bin_edges_0, bin_edges_1 = self._create_adaptive_cover(
                    lens, effective_intervals, effective_overlap
                )
            else:
                hypercubes, bin_edges_0, bin_edges_1 = self._create_balanced_cover(
                    lens, effective_intervals, effective_overlap
                )
            
        return hypercubes, bin_edges_0, bin_edges_1
        
    def _create_adaptive_cover(self, lens: np.ndarray, n_intervals: int, overlap_percentage: float) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        Create an adaptive cover using density-aware binning for skewed distributions.
        
        Args:
            lens: 2D lens representation of data points
            n_intervals: Number of intervals per dimension
            overlap_percentage: Percentage of overlap between adjacent intervals
            
        Returns:
            Tuple of (hypercubes, bin_edges_0, bin_edges_1)
        """
        # Get min and max for each dimension
        mins = np.min(lens, axis=0)
        maxs = np.max(lens, axis=0)
        
        # Handle potential division by zero
        if mins[0] == maxs[0]:
            maxs[0] += 0.1
        if mins[1] == maxs[1]:
            maxs[1] += 0.1
            
        # Calculate optimal bin edges for each dimension using KDE
        bin_edges = []
        
        for d in range(lens.shape[1]):
            dim_data = lens[:, d]
            
            try:
                # Use kernel density estimation to find good split points
                from scipy import stats
                kde = stats.gaussian_kde(dim_data)
                
                # Sample the KDE at many points
                x = np.linspace(mins[d], maxs[d], 1000)
                density = kde(x)
                
                # Find local minima in density as natural split points
                from scipy.signal import argrelextrema
                minima_idx = argrelextrema(density, np.less)[0]
                
                # If we have enough minima, use them as interval boundaries
                if len(minima_idx) >= n_intervals - 1:
                    # Sort and select evenly spaced minima
                    sorted_minima = sorted(minima_idx)
                    step = len(sorted_minima) // (n_intervals - 1)
                    selected_idx = sorted_minima[::step][:n_intervals-1]
                    
                    # Convert indices back to values
                    selected_values = x[selected_idx]
                    
                    # Create final edges with min and max
                    edges = np.concatenate(([mins[d]], selected_values, [maxs[d]]))
                else:
                    # Not enough minima, fall back to percentile-based approach
                    percentiles = np.linspace(0, 100, n_intervals + 1)
                    edges = np.percentile(dim_data, percentiles)
            except:
                # If KDE approach fails, use percentile-based binning
                percentiles = np.linspace(0, 100, n_intervals + 1)
                edges = np.percentile(dim_data, percentiles)
            
            # Expand the edges slightly to ensure inclusion of boundary points
            edges[0] = edges[0] - 0.001 * (edges[-1] - edges[0])
            edges[-1] = edges[-1] + 0.001 * (edges[-1] - edges[0])
            
            bin_edges.append(edges)
        
        # Expand edges to create overlapping intervals
        expanded_edges = []
        
        for dim_edges in bin_edges:
            interval_width = np.diff(dim_edges)
            overlap_width = interval_width * overlap_percentage
            
            expanded_dim_edges = []
            for i in range(len(dim_edges) - 1):
                start = dim_edges[i] - (overlap_width[i] / 2 if i > 0 else 0)
                end = dim_edges[i + 1] + (overlap_width[i] / 2 if i < len(dim_edges) - 2 else 0)
                
                # Ensure we don't go below min or above max
                start = max(start, mins[0] - 0.001)
                end = min(end, maxs[0] + 0.001)
                
                expanded_dim_edges.append((start, end))
            
            expanded_edges.append(expanded_dim_edges)
        
        # Create hypercubes from expanded edges
        hypercubes = []
        
        for i, (start0, end0) in enumerate(expanded_edges[0]):
            for j, (start1, end1) in enumerate(expanded_edges[1]):
                # Create mask for points in this hypercube
                mask = ((lens[:, 0] >= start0) & (lens[:, 0] <= end0) &
                        (lens[:, 1] >= start1) & (lens[:, 1] <= end1))
                
                # Get indices of points in hypercube
                indices = np.where(mask)[0]
                
                # Only keep hypercubes with enough points
                min_required = max(3, self.config.min_cluster_size // 3)
                if len(indices) >= min_required:
                    hypercubes.append(indices)
        
        logger.info(f"Created {len(hypercubes)} adaptive hypercubes with n_intervals={n_intervals}, overlap={overlap_percentage:.2f}")
        
        return hypercubes, bin_edges[0], bin_edges[1]

    def _estimate_lens_neighbor_distance(self, lens: np.ndarray) -> float:
        """
        Estimate the average distance to nearest neighbors in lens space.
        
        Args:
            lens: 2D lens representation of data points
            
        Returns:
            Average distance to K nearest neighbors
        """
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Sample points if there are too many
            max_sample_size = 1000
            if len(lens) > max_sample_size:
                sample_indices = np.random.choice(len(lens), max_sample_size, replace=False)
                lens_sample = lens[sample_indices]
            else:
                lens_sample = lens
                
            # Compute nearest neighbors
            k = min(6, len(lens_sample))
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(lens_sample)
            distances, _ = nn.kneighbors(lens_sample)
            
            # Average distance to neighbors (excluding self at index 0)
            avg_dist = np.mean(distances[:, 1:])
            return avg_dist
            
        except Exception as e:
            logger.debug(f"Error estimating lens neighbor distance: {str(e)[:100]}...")
            return 0.05  # Default value

    def _create_balanced_cover(self, lens: np.ndarray, n_intervals: int, overlap_percentage: float) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        Create a balanced cover with adaptive interval sizes.
        
        Args:
            lens: 2D lens representation of data points
            n_intervals: Number of intervals per dimension
            overlap_percentage: Percentage of overlap between adjacent intervals
            
        Returns:
            Tuple of (hypercubes, bin_edges_0, bin_edges_1)
        """
        # Get min and max for each dimension
        mins = np.min(lens, axis=0)
        maxs = np.max(lens, axis=0)
        
        # Handle potential division by zero
        if mins[0] == maxs[0]:
            maxs[0] += 0.1
        if mins[1] == maxs[1]:
            maxs[1] += 0.1
            
        # Calculate optimal bin edges for each dimension
        # Use density-aware binning to create more balanced hypercubes
        bin_edges = []
        
        for d in range(lens.shape[1]):
            dim_data = lens[:, d]
            
            # Check if we have a lot of duplicated values
            unique_values = np.unique(dim_data)
            if len(unique_values) < n_intervals:
                # If we have fewer unique values than intervals, use unique values as edges
                edges = np.sort(unique_values)
                bin_edges.append(edges)
                continue
            
            # Use quantile-based binning to handle non-uniform distributions
            percentiles = np.linspace(0, 100, n_intervals + 1)
            edges = np.percentile(dim_data, percentiles)
            
            # Ensure edges are unique
            edges = np.unique(edges)
            
            # If we got fewer edges than expected, pad with linear intervals
            if len(edges) < n_intervals + 1:
                missing = n_intervals + 1 - len(edges)
                range_size = maxs[d] - mins[d]
                step = range_size / (missing + 1)
                extra_edges = [maxs[d] - i * step for i in range(1, missing + 1)]
                edges = np.sort(np.concatenate([edges, extra_edges]))
            
            # Ensure we cover the full range
            edges[0] = mins[d] - 0.001 * (maxs[d] - mins[d])
            edges[-1] = maxs[d] + 0.001 * (maxs[d] - mins[d])
            
            bin_edges.append(edges)
        
        # Create overlapping intervals using the edges
        overlapping_intervals = []
        
        for d, edges in enumerate(bin_edges):
            intervals = []
            for i in range(len(edges) - 1):
                # Calculate non-overlapping interval width
                interval_width = edges[i+1] - edges[i]
                
                # Calculate overlap distance
                overlap_distance = interval_width * overlap_percentage
                
                # Calculate start and end with overlap
                # For the first interval, don't extend start
                # For the last interval, don't extend end
                if i == 0:
                    start = edges[i]
                else:
                    start = edges[i] - overlap_distance / 2
                    
                if i == len(edges) - 2:
                    end = edges[i+1]
                else:
                    end = edges[i+1] + overlap_distance / 2
                
                intervals.append((start, end))
            overlapping_intervals.append(intervals)
        
        # Create hypercubes from overlapping intervals
        hypercubes = []
        
        for i, interval_0 in enumerate(overlapping_intervals[0]):
            for j, interval_1 in enumerate(overlapping_intervals[1]):
                start_0, end_0 = interval_0
                start_1, end_1 = interval_1
                
                # Find points within this hypercube
                indices = np.where(
                    (lens[:, 0] >= start_0) & (lens[:, 0] <= end_0) &
                    (lens[:, 1] >= start_1) & (lens[:, 1] <= end_1)
                )[0]
                
                # Only keep hypercubes with enough points
                min_required = max(3, self.config.min_cluster_size // 3)
                if len(indices) >= min_required:
                    hypercubes.append(indices)
        
        logger.info(f"Created {len(hypercubes)} hypercubes with n_intervals={n_intervals}, overlap={overlap_percentage:.2f}")
        
        return hypercubes, bin_edges[0], bin_edges[1]

    def _mapper_to_networkx(self) -> nx.Graph:
        """
        Convert mapper graph to NetworkX graph for analysis.
        
        Returns:
            NetworkX graph representation of mapper output
        """
        if self.graph is None:
            raise ValueError("Mapper graph not created yet. Call fit_transform first.")
            
        # Build networkx graph from mapper graph
        G = nx.Graph()
        
        # Add nodes with metadata
        for node_id, points in self.graph['nodes'].items():
            G.add_node(node_id, points=points, size=len(points))
        
        # Add edges with weights based on overlap similarity
        if isinstance(self.graph['links'], dict):
            # Handle dictionary format
            for node_id, connected_nodes in self.graph['links'].items():
                for target in connected_nodes:
                    # Calculate edge weight based on shared points
                    source_points = set(self.graph['nodes'][node_id])
                    target_points = set(self.graph['nodes'][target])
                    shared_points = source_points.intersection(target_points)
                    
                    # Jaccard similarity as weight
                    weight = len(shared_points) / len(source_points.union(target_points))
                    
                    G.add_edge(node_id, target, weight=weight)
        else:
            # Handle list of tuples format
            for link in self.graph['links']:
                # Add with default weight
                G.add_edge(link[0], link[1], weight=0.5)
        
        # Add temporal information to nodes
        for node_id, points in self.graph['nodes'].items():
            # Calculate temporal information for this node
            start_idx = min(points) if points else 0
            end_idx = max(points) if points else 0
            mid_idx = int((start_idx + end_idx) / 2)
            
            # Store temporal metadata
            G.nodes[node_id]['start_idx'] = start_idx
            G.nodes[node_id]['end_idx'] = end_idx
            G.nodes[node_id]['mid_idx'] = mid_idx
            G.nodes[node_id]['time_span'] = end_idx - start_idx
        
        return G

    def _process_hypercube(self, cube_idx, cube_points, distance_matrix, lens):
        """
        Process a single hypercube for parallel execution.
        
        Args:
            cube_idx: Index of the hypercube
            cube_points: Points in this hypercube
            distance_matrix: Full distance matrix
            lens: Lens projection values
            
        Returns:
            Dict with clustering results for this hypercube
        """
        # Skip if too small
        if len(cube_points) < self.config.min_cluster_size:
            return {
                'cube_index': cube_idx,
                'success': False,
                'rejection_reason': 'too_small',
                'nodes': {},
                'cluster_sizes': []
            }
        
        # Get clustering algorithm
        clustering_algo = self._select_clustering_algorithm()
        
        # IMPROVEMENT: Try multiple parameter settings for clustering
        adaptive_min_samples = [1, 2]  # Try different min_samples values
        adaptive_epsilon_vals = [0.05, 0.1, 0.15]  # Try different epsilon values
        
        # Track best clustering results
        best_clusters = None
        best_labels = None
        max_clusters = 0
        
        for min_samples in adaptive_min_samples:
            for epsilon in adaptive_epsilon_vals:
                try:
                    # Extract distance submatrix for this hypercube using vectorized approach
                    indices = cube_points
                    n_points = len(indices)
                    
                    # Use vectorized extraction
                    indices_array = np.array(indices)
                    submatrix = distance_matrix[np.ix_(indices_array, indices_array)]
                    
                    # Create a copy of clustering algorithm with current parameters
                    if isinstance(clustering_algo, HDBSCAN):
                        adjusted_params = self.config.clustering_parameters.copy()
                        adjusted_params['min_samples'] = min_samples
                        adjusted_params['cluster_selection_epsilon'] = epsilon
                        test_clusterer = HDBSCAN(
                            min_cluster_size=self.config.min_cluster_size,
                            **adjusted_params
                        )
                        # HDBSCAN works better on original points than distance matrix
                        cluster_labels = test_clusterer.fit_predict(lens[indices])
                    else:
                        # Other algorithms can work on the distance matrix
                        cluster_labels = clustering_algo.fit_predict(submatrix)
                    
                    # Count clusters (excluding noise points labeled as -1)
                    unique_clusters = set(label for label in cluster_labels if label != -1)
                    n_clusters = len(unique_clusters)
                    
                    # If this parameter combination finds more clusters, keep it
                    if n_clusters > max_clusters:
                        max_clusters = n_clusters
                        best_labels = cluster_labels
                        best_clusters = unique_clusters
                        
                except Exception as e:
                    # Continue trying other parameters
                    continue
        
        # Process results if we found any clusters
        if best_clusters and len(best_clusters) > 0:
            # Prepare nodes for this hypercube
            cube_nodes = {}
            cluster_sizes = []
            
            # Create nodes for each cluster
            for cluster_idx in best_clusters:
                cluster_mask = (best_labels == cluster_idx)
                cluster_indices = [indices[i] for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
                
                # Only create nodes with sufficient points
                if len(cluster_indices) < self.config.min_cluster_size:
                    continue
                
                # Create a unique node ID
                node_id = f"cube{cube_idx}_cluster{cluster_idx}"
                cube_nodes[node_id] = cluster_indices
                cluster_sizes.append(len(cluster_indices))
            
            return {
                'cube_index': cube_idx,
                'success': True,
                'nodes': cube_nodes,
                'cluster_sizes': cluster_sizes,
                'cluster_count': len(best_clusters)
            }
        else:
            return {
                'cube_index': cube_idx,
                'success': False,
                'rejection_reason': 'no_clusters',
                'nodes': {},
                'cluster_sizes': []
            }
    
    def _process_hypercubes_parallel(self, hypercubes, distance_matrix, lens, n_jobs=-1):
        """Process hypercubes in parallel using joblib."""
        import joblib
        from joblib import Parallel, delayed
        import logging
        
        # Add specific filter to prevent 'Minimal logging mode enabled' messages in parallel workers
        root_logger = logging.getLogger()
        
        class BlockMinimalLoggingFilter(logging.Filter):
            def filter(self, record):
                if "Minimal logging mode enabled" in record.getMessage():
                    return False
                return True
        
        # Apply filter to all handlers
        for handler in root_logger.handlers:
            handler.addFilter(BlockMinimalLoggingFilter())
            
        # Also apply to joblib logger
        joblib_logger = logging.getLogger('joblib')
        joblib_logger.addFilter(BlockMinimalLoggingFilter())
        
        # Rest of the method implementation
        try:
            # Determine appropriate number of cores
            if n_jobs <= 0:
                n_jobs = joblib.cpu_count() + n_jobs + 1
            
            # Use lower verbosity for parallel jobs to reduce log spam
            verbose_level = 0
            
            # Process hypercubes in parallel
            results = Parallel(n_jobs=n_jobs, verbose=verbose_level)(
                delayed(self._process_hypercube)(i, hypercubes[i], distance_matrix, lens)
                for i in range(len(hypercubes))
            )
            
            # Combine results
            nodes = {}
            cluster_sizes = []
            cluster_counts = []
            successful_cubes = []
            rejection_reasons = {"too_small": 0, "no_clusters": 0, "error": 0}
            
            for result in results:
                if result['success']:
                    # Add nodes from this hypercube
                    nodes.update(result['nodes'])
                    cluster_sizes.extend(result['cluster_sizes'])
                    cluster_counts.append(result['cluster_count'])
                    successful_cubes.append(result['cube_index'])
                else:
                    # Update rejection reasons
                    reason = result.get('rejection_reason', 'error')
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            
            return {
                'nodes': nodes,
                'cluster_sizes': cluster_sizes,
                'cluster_counts': cluster_counts,
                'successful_cubes': successful_cubes,
                'rejection_reasons': rejection_reasons
            }
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            # Fall back to sequential processing
            nodes = {}
            cluster_sizes = []
            cluster_counts = []
            successful_cubes = []
            rejection_reasons = {"too_small": 0, "no_clusters": 0, "error": 0}
            
            for i in range(len(hypercubes)):
                result = self._process_hypercube(i, hypercubes[i], distance_matrix, lens)
                if result['success']:
                    nodes.update(result['nodes'])
                    cluster_sizes.extend(result['cluster_sizes'])
                    cluster_counts.append(result['cluster_count'])
                    successful_cubes.append(result['cube_index'])
                else:
                    reason = result.get('rejection_reason', 'error')
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            
            return {
                'nodes': nodes,
                'cluster_sizes': cluster_sizes,
                'cluster_counts': cluster_counts,
                'successful_cubes': successful_cubes,
                'rejection_reasons': rejection_reasons
            }

    def _evaluate_regime_quality(self, G: nx.Graph, communities: List[List[str]]) -> Dict[str, float]:
        """
        Calculate various metrics to evaluate the quality of identified regimes.
        
        Args:
            G: NetworkX graph
            communities: List of communities (regimes)
            
        Returns:
            Dictionary of quality metrics
        """
        try:
            # 1. Modularity - higher is better, measures quality of division
            try:
                modularity = nx.algorithms.community.modularity(G, communities)
            except Exception as e:
                # Fall back to manual calculation if the library function fails
                modularity = 0.0
                m = G.number_of_edges()
                if m > 0:
                    for community in communities:
                        subgraph = G.subgraph(community)
                        e_c = subgraph.number_of_edges()
                        vol_c = sum(dict(G.degree()).get(node, 0) for node in community)
                        modularity += (e_c / m) - ((vol_c / (2 * m)) ** 2)
                        
            # 2. Conductance - lower is better, measures "leakage" between communities
            conductances = []
            for community in communities:
                internal_edges = sum(1 for u, v in G.edges() if u in community and v in community)
                external_edges = sum(1 for u, v in G.edges() if (u in community and v not in community) or
                                    (u not in community and v in community))
                if internal_edges + external_edges > 0:
                    conductances.append(external_edges / (internal_edges + external_edges))
                
            avg_conductance = sum(conductances) / len(conductances) if conductances else 1.0
            
            # 3. Coverage - higher is better, fraction of edges within communities
            total_edges = G.number_of_edges()
            internal_edges = sum(G.subgraph(comm).number_of_edges() for comm in communities)
            coverage = internal_edges / total_edges if total_edges > 0 else 0
            
            # 4. Performance - higher is better, fraction of correctly placed node pairs
            # (connected nodes in same community, unconnected nodes in different communities)
            n = G.number_of_nodes()
            max_edges = n * (n - 1) // 2  # Maximum possible edges
            correct_pairs = 0
            
            for i, comm in enumerate(communities):
                # Correctly identified edges within communities
                s = G.subgraph(comm)
                correct_pairs += s.number_of_edges()
                
                # Correctly identified non-edges between communities
                for j, other_comm in enumerate(communities):
                    if i >= j:  # Skip duplicate combinations
                        continue
                    
                    # Count all possible edges between communities
                    possible_edges = len(comm) * len(other_comm)
                    
                    # Count actual edges between communities
                    cross_edges = 0
                    for u in comm:
                        for v in other_comm:
                            if G.has_edge(u, v):
                                cross_edges += 1
                    
                    # Add correctly identified non-edges
                    correct_pairs += possible_edges - cross_edges
            
            performance = correct_pairs / max_edges if max_edges > 0 else 0
            
            # Return all metrics
            return {
                "modularity": modularity,
                "conductance": avg_conductance,
                "coverage": coverage,
                "performance": performance,
                # Combined score - higher is better
                "combined_score": (modularity + coverage + performance + (1 - avg_conductance)) / 4
            }
            
        except Exception as e:
            logger.warning(f"Error calculating regime quality metrics: {str(e)[:100]}")
            return {"modularity": 0, "conductance": 1.0, "coverage": 0, "performance": 0, "combined_score": 0}
    
                    
