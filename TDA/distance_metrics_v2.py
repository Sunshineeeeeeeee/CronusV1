"""
Financial distance metrics optimized for high-frequency trading (HFT) data.

This module provides specialized distance functions and efficient computation
methods for financial time series, with a focus on detecting market regimes
in high-frequency trading data.

Key features:
- Optimized distance metrics for financial time series (distribution-based, wavelet-based)
- Efficient distance matrix computation with parallelization
- Support for distributed computing with Dask

Dependencies:
- Required: numpy, scipy, pandas
- Optional: joblib (for parallel processing)
- Optional: dask (for distributed computing)
- Optional: pywt (for wavelet-based distances)
- Optional: ot (for optimized Wasserstein distance)

Example usage:
```python
from TDA.distance_metrics_v2 import compute_distance_matrix, create_financial_distance_function

# Create a distance function
distance_func = create_financial_distance_function(metric='distribution')

# Compute distance matrix
distance_matrix = compute_distance_matrix(windows, distance_func)
```
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Callable, Optional, Dict, Any
from scipy.stats import wasserstein_distance, energy_distance
import logging
from functools import partial
import warnings
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional libraries
try:
    import pywt  # PyWavelets for wavelet-based distances
    WAVELETS_AVAILABLE = True
except ImportError:
    WAVELETS_AVAILABLE = False
    logger.warning("PyWavelets not installed. Wavelet-based distances will not be available.")

# Try to import POT for optimized Wasserstein distance
try:
    import ot  # Python Optimal Transport library
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False
    logger.warning("POT not installed. Using scipy for Wasserstein distance.")

# Try to import Dask for distributed computing
try:
    import dask
    import dask.array as da
    from dask.distributed import Client, wait
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask not installed. Falling back to joblib for parallelization.")

class FinancialDistanceMetrics:
    """Distance metrics for financial time series in HFT."""
    
    @staticmethod
    def distribution_distance(x: np.ndarray, y: np.ndarray, method: str = 'wasserstein') -> float:
        """
        Optimized Wasserstein distance implementation for financial time series.
        Specifically tuned for detecting regime changes in HFT data when used with HDBSCAN.
        
        Args:
            x: First time series window
            y: Second time series window
            method: Distance method ('wasserstein' is recommended and optimized)
            
        Returns:
            Distribution distance value
        """
        # If method is not wasserstein, default to wasserstein
        if method != 'wasserstein':
            logger.info(f"Method {method} not optimal for regime detection, using wasserstein")
            method = 'wasserstein'
        
        # Flatten arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # For empty arrays, return a default high distance
        if len(x_flat) == 0 or len(y_flat) == 0:
            return 1.0
        
        # Handle very small arrays efficiently
        if len(x_flat) <= 3 or len(y_flat) <= 3:
            # For tiny arrays, use a simple euclidean distance
            len_diff = abs(len(x_flat) - len(y_flat))
            if len_diff > 0:
                # Pad the shorter array for comparison
                if len(x_flat) < len(y_flat):
                    x_pad = np.pad(x_flat, (0, len_diff), 'constant', constant_values=np.median(x_flat))
                    return np.sqrt(np.mean((x_pad - y_flat)**2))
                else:
                    y_pad = np.pad(y_flat, (0, len_diff), 'constant', constant_values=np.median(y_flat))
                    return np.sqrt(np.mean((x_flat - y_pad)**2))
            else:
                return np.sqrt(np.mean((x_flat - y_flat)**2))
        
        # Optimize calculation by downsampling very large arrays
        # This significantly speeds up distance calculation with minimal impact on results
        max_points = 1000  # Maximum number of points to consider
        if len(x_flat) > max_points:
            idx = np.linspace(0, len(x_flat)-1, max_points).astype(int)
            x_flat = x_flat[idx]
        if len(y_flat) > max_points:
            idx = np.linspace(0, len(y_flat)-1, max_points).astype(int)
            y_flat = y_flat[idx]
        
        # Use robust scaling to handle outliers in financial data
        x_q75, x_q25 = np.percentile(x_flat, [75, 25])
        y_q75, y_q25 = np.percentile(y_flat, [75, 25])
        x_iqr = x_q75 - x_q25
        y_iqr = y_q75 - y_q25
        x_median = np.median(x_flat)
        y_median = np.median(y_flat)
        
        # Avoid division by zero
        x_iqr = x_iqr if x_iqr > 0 else 1.0
        y_iqr = y_iqr if y_iqr > 0 else 1.0
        
        # Scale using IQR - more robust to outliers than standard deviation
        x_scaled = (x_flat - x_median) / x_iqr
        y_scaled = (y_flat - y_median) / y_iqr
        
        # Replace NaN values
        x_scaled = np.nan_to_num(x_scaled)
        y_scaled = np.nan_to_num(y_scaled)
        
        # Using wasserstein distance (Earth Mover's Distance)
        # Choose implementation based on data size and available libraries
        if POT_AVAILABLE and (len(x_scaled) > 100 or len(y_scaled) > 100):
            # Use POT for larger datasets (faster)
            try:
                # Create empirical distributions with uniform weights
                x_weights = np.ones(len(x_scaled)) / len(x_scaled)
                y_weights = np.ones(len(y_scaled)) / len(y_scaled)
                
                # Use POT's EMD implementation (faster than scipy's)
                return ot.emd2_1d(x_scaled, y_scaled, x_weights, y_weights)
            except Exception as e:
                logger.debug(f"POT implementation failed: {str(e)[:50]}... Falling back to scipy")
                # Fall back to scipy if POT fails
                return wasserstein_distance(x_scaled, y_scaled)
        else:
            # Use scipy's implementation for smaller datasets
            return wasserstein_distance(x_scaled, y_scaled)
    
    @staticmethod
    def wavelet_distance(x: np.ndarray, y: np.ndarray, 
                         wavelet: str = 'db4', 
                         level: int = 3) -> float:
        """
        Wavelet-based distance capturing multi-scale differences.
        Especially useful for high-frequency financial data with patterns at different timescales.
        
        Args:
            x: First time series
            y: Second time series
            wavelet: Wavelet function to use
            level: Decomposition level
            
        Returns:
            Wavelet distance value
        """
        if not WAVELETS_AVAILABLE:
            logger.warning("PyWavelets not available. Using euclidean distance as fallback.")
            return np.sqrt(np.mean((x.flatten() - y.flatten())**2))
        
        # Flatten arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Find power of 2 that's >= length
        # More efficient than calculating log2 directly
        max_len = max(len(x_flat), len(y_flat))
        pad_len = 1
        while pad_len < max_len:
            pad_len *= 2
            
        # Pad arrays with zeros
        x_padded = np.zeros(pad_len)
        y_padded = np.zeros(pad_len)
        x_padded[:len(x_flat)] = x_flat
        y_padded[:len(y_flat)] = y_flat
        
        # Perform wavelet decomposition
        try:
            # Use wavedec with mode="periodization" for better performance
            coeff_x = pywt.wavedec(x_padded, wavelet, level=level, mode="periodization")
            coeff_y = pywt.wavedec(y_padded, wavelet, level=level, mode="periodization")
            
            # Pre-calculate weights for all levels (vectorized)
            weights = np.zeros(level + 1)
            weights[0] = 0.4  # Approximation coefficients
            for i in range(1, level + 1):
                weights[i] = 0.6 * (2 ** -(i-1))
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Calculate distances for each level (vectorized)
            distances = np.zeros(level + 1)
            
            for i, (cx, cy) in enumerate(zip(coeff_x, coeff_y)):
                # Vectorized normalization
                cx_std = np.std(cx)
                cy_std = np.std(cy)
                
                # Avoid division by zero
                cx_norm = cx / cx_std if cx_std > 0 else cx
                cy_norm = cy / cy_std if cy_std > 0 else cy
                
                # Vectorized distance calculation
                distances[i] = np.sqrt(np.mean((cx_norm - cy_norm) ** 2))
            
            # Weighted sum of distances (vectorized)
            return np.sum(distances * weights)
        
        except Exception as e:
            logger.error(f"Error in wavelet distance: {e}")
            # Fall back to euclidean distance
            return np.sqrt(np.mean((x.flatten() - y.flatten())**2))


def create_financial_distance_function(metric: str = 'distribution',
                                     wavelet: str = 'db4',
                                     distribution_method: str = 'wasserstein',
                                     **kwargs) -> Callable:
    """
    Create a distance function for financial time series with HFT focus.
    
    Args:
        metric: Distance metric type ('distribution', 'wavelet')
        wavelet: Wavelet to use for wavelet-based distance
        distribution_method: Method for distribution distance
        **kwargs: Additional parameters for specific distance metrics
        
    Returns:
        Distance function that takes two time series and returns a distance value
    """
    logger.info(f"Creating financial distance function with metric: {metric}")
    
    if metric == 'distribution':
        return partial(FinancialDistanceMetrics.distribution_distance, method=distribution_method)
    
    elif metric == 'wavelet':
        if not WAVELETS_AVAILABLE:
            logger.warning("PyWavelets not available. Using euclidean distance instead.")
            return lambda x, y: np.sqrt(np.mean((x.flatten() - y.flatten())**2))
        return partial(FinancialDistanceMetrics.wavelet_distance, wavelet=wavelet, level=kwargs.get('level', 3))
    
    else:
        logger.warning(f"Unknown distance metric: {metric}, using distribution")
        return partial(FinancialDistanceMetrics.distribution_distance, method=distribution_method)


def compute_distance_matrix(windows: List[np.ndarray], 
                          distance_func: Callable,
                          running_locally: bool = True,
                          n_jobs: int = -1,  # Default to using all cores
                          block_size: int = None  # Will be calculated adaptively
                          ) -> np.ndarray:
    """
    Compute pairwise distance matrix for a list of time series windows.
    Optimized for HFT data with distributed processing support using block-wise computation.
    
    Args:
        windows: List of time series windows
        distance_func: Function to compute distance between two windows
        running_locally: If True, always use joblib with maximum cores, if False use Dask
        n_jobs: Number of jobs for parallel processing (-1 for all cores)
        block_size: Size of blocks for block-wise computation (None for adaptive)
        
    Returns:
        Distance matrix of shape (n_windows, n_windows)
    """
    n_windows = len(windows)
    logger.info(f"Computing distance matrix for {n_windows} windows")
    
    if n_windows == 0:
        return np.array([])
    
    # For very small datasets, just use direct computation
    if n_windows <= 10:
        dist_matrix = np.zeros((n_windows, n_windows))
        for i in range(n_windows):
            for j in range(i, n_windows):
                if i == j:
                    dist_matrix[i, j] = 0.0
                else:
                    dist = distance_func(windows[i], windows[j])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist  # Symmetric
        return dist_matrix
    
    # Calculate optimal chunk size for either implementation
    if n_jobs <= 0:
        n_jobs = max(os.cpu_count() - 1, 1)
    
    # Determine whether to use joblib (local) or Dask (distributed)
    if running_locally or not DASK_AVAILABLE:
        # Always use joblib when running locally, or if Dask is not available
        return _compute_with_joblib(windows, distance_func, n_jobs, n_windows, block_size)
    else:
        # Use Dask for distributed computing
        return _compute_with_dask(windows, distance_func, n_jobs, n_windows, block_size)

def _compute_with_joblib(windows, distance_func, n_jobs, n_windows, block_size=None):
    """
    Helper function for computing distance matrix using joblib.
    Optimized for local computation with improved chunking strategy.
    """
    try:
        from joblib import Parallel, delayed
        import numpy as np
        
        # Adaptive chunk sizing for optimal performance
        if block_size is None:
            # Calculate optimal chunk size based on problem size and cores
            # For larger datasets, create more chunks for better load balancing
            if n_windows > 10000:
                # Very large dataset: higher chunks per worker ratio
                chunks_per_worker = 8
            elif n_windows > 5000:
                # Large dataset
                chunks_per_worker = 6
            elif n_windows > 1000:
                # Medium dataset
                chunks_per_worker = 4
            else:
                # Small dataset: fewer chunks to reduce overhead
                chunks_per_worker = 2
                
            # Calculate chunk size to achieve the target chunks per worker
            chunk_size = max(10, n_windows // (n_jobs * chunks_per_worker))
        else:
            chunk_size = block_size
        
        logger.info(f"Computing with joblib using {n_jobs} workers, chunk size: {chunk_size}")
        
        # Create 2D chunking strategy for the upper triangle of the distance matrix
        chunks = []
        
        # Two chunking strategies:
        # 1. For smaller datasets: horizontal strips (better cache locality)
        # 2. For larger datasets: square tiles (better load balancing)
        if n_windows < 2000:
            # Horizontal strips for smaller datasets
            for i in range(0, n_windows, chunk_size):
                end_i = min(i + chunk_size, n_windows)
                chunks.append((i, end_i, i, n_windows))  # (start_i, end_i, start_j, end_j)
        else:
            # Square tiles for larger datasets (better load balancing)
            for i in range(0, n_windows, chunk_size):
                end_i = min(i + chunk_size, n_windows)
                for j in range(i, n_windows, chunk_size):
                    end_j = min(j + chunk_size, n_windows)
                    # Only compute the upper triangle
                    if j >= i:
                        chunks.append((i, end_i, j, end_j))
        
        # Function to compute a chunk of the distance matrix
        def compute_chunk(start_i, end_i, start_j, end_j):
            # Allocate memory only for the region we're computing
            chunk_height = end_i - start_i
            chunk_width = end_j - start_j
            chunk_result = np.zeros((chunk_height, chunk_width))
            
            # Only compute upper triangle within the chunk
            for i_local, i_global in enumerate(range(start_i, end_i)):
                for j_local, j_global in enumerate(range(max(i_global+1, start_j), end_j)):
                    dist = distance_func(windows[i_global], windows[j_global])
                    chunk_result[i_local, j_local + (start_j - max(i_global+1, start_j))] = dist
            
            return start_i, end_i, start_j, end_j, chunk_result
        
        # Compute chunks in parallel
        logger.info(f"Computing distance matrix with joblib using {n_jobs} workers and {len(chunks)} chunks")
        parallel = Parallel(n_jobs=n_jobs, verbose=1, prefer="processes")
        results = parallel(
            delayed(compute_chunk)(start_i, end_i, start_j, end_j) 
            for start_i, end_i, start_j, end_j in chunks
        )
        
        # Initialize distance matrix
        dist_matrix = np.zeros((n_windows, n_windows))
        
        # Fill matrix with computed distances
        for start_i, end_i, start_j, end_j, chunk_result in results:
            for i_local, i_global in enumerate(range(start_i, end_i)):
                for j_local, j_global in enumerate(range(max(i_global+1, start_j), end_j)):
                    j_adjusted = j_local + (start_j - max(i_global+1, start_j))
                    if j_adjusted < chunk_result.shape[1]:
                        val = chunk_result[i_local, j_adjusted]
                        if val > 0:  # Only set non-zero values
                            dist_matrix[i_global, j_global] = val
                            dist_matrix[j_global, i_global] = val  # Symmetric
        
        return dist_matrix
    
    except ImportError:
        logger.warning("joblib not available. Using sequential computation.")
        # Fall back to sequential computation
        dist_matrix = np.zeros((n_windows, n_windows))
        
        for i in range(n_windows):
            for j in range(i+1, n_windows):
                dist = distance_func(windows[i], windows[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # Symmetric
        
        return dist_matrix

def _compute_with_dask(windows, distance_func, n_jobs, n_windows, block_size=None):
    """
    Helper function for computing distance matrix using Dask.
    Optimized for distributed computation with improved chunking strategy.
    """
    try:
        # Check if Dask is available
        if not DASK_AVAILABLE:
            logger.warning("Dask not available. Falling back to joblib.")
            return _compute_with_joblib(windows, distance_func, n_jobs, n_windows, block_size)
        
        # Adaptive chunk sizing for Dask
        if block_size is None:
            # Dask works better with larger chunks to minimize scheduling overhead
            if n_windows > 20000:
                # Extra large dataset: aim for 1-2 chunks per worker
                chunks_per_worker = 2
            elif n_windows > 10000:
                # Very large dataset: aim for 2-3 chunks per worker
                chunks_per_worker = 3
            elif n_windows > 5000:
                # Large dataset: aim for 3-4 chunks per worker
                chunks_per_worker = 4
            else:
                # Medium dataset: aim for 4-6 chunks per worker
                chunks_per_worker = 6
                
            # Calculate block size based on desired chunks per worker
            block_size = max(100, n_windows // (n_jobs * chunks_per_worker))
        
        logger.info(f"Computing with Dask using {n_jobs} workers, block size: {block_size}")
        
        # Try to get existing client
        try:
            from dask.distributed import get_client
            client = get_client()
            logger.info("Using existing Dask client")
        except (ValueError, ImportError):
            logger.warning("No Dask client available, falling back to joblib")
            return _compute_with_joblib(windows, distance_func, n_jobs, n_windows, block_size)
        
        # Initialize distance matrix
        dist_matrix = np.zeros((n_windows, n_windows))
        
        # Advanced chunking strategy for Dask:
        # 1. Divide the matrix into square tiles for better data locality
        # 2. Process the upper triangle of the matrix only
        
        # Create chunks that maximize data locality and minimize communication
        n_chunks = max(1, min(n_jobs * chunks_per_worker, n_windows // block_size))
        chunk_size = max(block_size, n_windows // n_chunks)
        logger.info(f"Computing with {n_chunks} chunks of size ~{chunk_size}")
        
        futures = []
        
        # Define chunk computation function for square regions
        def compute_chunk(start_i, end_i, start_j, end_j, chunk_windows):
            chunk_result = {}
            # Only compute upper triangle
            for i_rel, i in enumerate(range(start_i, end_i)):
                for j in range(max(i+1, start_j), end_j):
                    dist = distance_func(chunk_windows[i_rel] if i >= start_i and i < end_i else windows[i], 
                                         windows[j])
                    chunk_result[(i, j)] = dist
            return chunk_result
        
        # Submit square tile chunks for computation
        for i in range(0, n_windows, chunk_size):
            end_i = min(i + chunk_size, n_windows)
            # Only send necessary windows to each worker to reduce data transfer
            chunk_windows = windows[i:end_i]
            
            # Process tiles in the upper triangle
            for j in range(i, n_windows, chunk_size):
                end_j = min(j + chunk_size, n_windows)
                futures.append(client.submit(
                    compute_chunk,
                    i, end_i, j, end_j, chunk_windows,
                    pure=False  # Ensure recomputation if needed
                ))
        
        # Gather results and fill matrix
        for future in client.as_completed(futures):
            try:
                chunk_result = future.result()
                # Fill both upper and lower triangles
                for (i, j), val in chunk_result.items():
                    dist_matrix[i, j] = val
                    dist_matrix[j, i] = val  # Symmetric
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
            
        return dist_matrix
        
    except Exception as e:
        logger.error(f"Error in Dask computation: {e}")
        logger.warning("Falling back to joblib")
        return _compute_with_joblib(windows, distance_func, n_jobs, n_windows, block_size)
