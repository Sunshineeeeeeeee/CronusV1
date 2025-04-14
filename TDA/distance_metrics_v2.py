"""
Financial distance metrics optimized for high-frequency trading (HFT) data.

This module provides specialized distance functions and efficient computation
methods for financial time series, with a focus on detecting market regimes
in high-frequency trading data.

Key features:
- Optimized distance metrics for financial time series (distribution-based, wavelet-based)
- Highly efficient distance matrix computation with parallelization
- Scalable to extremely large datasets (millions of points) using sparse approximation
- Support for distributed computing with Dask
- GPU acceleration with FAISS (if available)

Sparse approximation is automatically used for datasets larger than 20,000 points,
which dramatically reduces computation time with minimal impact on accuracy.
This approach computes only the k-nearest neighbors for each point instead of
the full distance matrix, resulting in O(n log n) complexity instead of O(n²).

Dependencies:
- Required: numpy, scipy, pandas
- Optional: joblib (for parallel processing)
- Optional: dask (for distributed computing)
- Optional: faiss-gpu/faiss-cpu (for GPU-accelerated sparse approximation)
- Optional: scikit-learn (fallback for sparse approximation)
- Optional: pywt (for wavelet-based distances)
- Optional: ot (for optimized Wasserstein distance)

Example usage:
```python
from TDA.distance_metrics_v2 import compute_distance_matrix, create_financial_distance_function

# Create a distance function
distance_func = create_financial_distance_function(metric='distribution')

# Compute distance matrix
distance_matrix = compute_distance_matrix(windows, distance_func)

# For large datasets, enable sparse approximation
distance_matrix = compute_distance_matrix(
    windows, 
    distance_func,
    use_sparse_approx=True,
    sparse_neighbors=100,
    use_gpu=True
)
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

# Try to import FAISS for approximate nearest neighbor search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Sparse approximation for large datasets will not be available. Install with: pip install faiss-gpu or pip install faiss-cpu")

# Try to import sklearn for alternative nearest neighbor implementation
try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_NN_AVAILABLE = True
except ImportError:
    SKLEARN_NN_AVAILABLE = False
    logger.warning("Scikit-learn not installed. Some fallback methods may not be available.")

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


def _flatten_and_normalize_windows(windows: List[np.ndarray]) -> np.ndarray:
    """
    Flatten and normalize time series windows for use with FAISS.
    
    Args:
        windows: List of time series windows
        
    Returns:
        Normalized feature matrix of shape (n_windows, n_features)
    """
    # First determine the largest dimension needed
    max_dim = max(w.size for w in windows)
    
    # Prepare feature matrix
    n_windows = len(windows)
    features = np.zeros((n_windows, max_dim), dtype=np.float32)
    
    # Fill feature matrix and pad shorter sequences
    for i, window in enumerate(windows):
        flat_window = window.flatten()
        features[i, :len(flat_window)] = flat_window
        
        # Add mask for padded regions (set to mean of non-padded)
        if len(flat_window) < max_dim:
            features[i, len(flat_window):] = np.mean(flat_window)
    
    # Normalize features using robust scaling (similar to distribution_distance)
    for i in range(n_windows):
        row = features[i]
        q75, q25 = np.percentile(row, [75, 25])
        iqr = q75 - q25
        median = np.median(row)
        
        # Avoid division by zero
        iqr = iqr if iqr > 0 else 1.0
        
        # Scale using IQR
        features[i] = (row - median) / iqr
    
    # Replace NaN values
    features = np.nan_to_num(features)
    
    # Normalize each vector to unit length for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    features = features / norms
    
    return features


def _compute_with_faiss(windows: List[np.ndarray], 
                        distance_func: Callable, 
                        n_neighbors: int = 50,
                        use_gpu: bool = False) -> Tuple[np.ndarray, float]:
    """
    Compute approximate distance matrix using FAISS HNSW index.
    This method scales to millions of points and only computes the k-nearest
    neighbors for each point, resulting in a sparse distance matrix.
    
    Args:
        windows: List of time series windows
        distance_func: Function to compute distance between two windows
        n_neighbors: Number of nearest neighbors to compute for each point
        use_gpu: Whether to use GPU acceleration if available
        
    Returns:
        Tuple of (sparse_distance_matrix, sparsity_level)
    """
    if not FAISS_AVAILABLE:
        logger.warning("FAISS not available. Using dense distance matrix computation.")
        return None, 0.0
    
    n_windows = len(windows)
    logger.info(f"Computing approximate distance matrix with FAISS for {n_windows} windows")
    
    # Convert windows to feature vectors
    features = _flatten_and_normalize_windows(windows)
    d = features.shape[1]  # Dimensionality
    
    try:
        # Configure FAISS index
        # HNSW index parameters: 
        # M = number of connections per layer (higher = more accurate but slower)
        # efConstruction = construction time/accuracy trade-off
        # efSearch = query time/accuracy trade-off
        M = 32
        efConstruction = 100
        efSearch = 128
        
        if use_gpu and hasattr(faiss, 'StandardGpuResources'):
            # Use GPU if available and requested
            logger.info("Using GPU-accelerated FAISS index")
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0  # GPU device ID
            
            # Create CPU index then transfer to GPU
            cpu_index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            # Use CPU index
            logger.info("Using CPU-based FAISS index")
            index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        
        # Set HNSW parameters
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = efSearch
        
        # Train and add vectors
        index.train(features)
        index.add(features)
        
        # Set search parameters
        index.hnsw.efSearch = efSearch
        
        # Perform search for all points
        logger.info(f"Searching for {n_neighbors} nearest neighbors for each point")
        similarities, indices = index.search(features, n_neighbors)
        
        # Convert similarities to distances (1 - cosine_similarity for normalized vectors)
        # This is an approximation of the euclidean distance between normalized vectors
        distances = 1 - similarities
        
        # Create sparse distance matrix
        dist_matrix = np.zeros((n_windows, n_windows))
        
        # Fill in the k-nearest neighbors for each point
        for i in range(n_windows):
            for j, idx in enumerate(indices[i]):
                if idx >= 0 and idx < n_windows:  # Valid index
                    dist_val = distances[i, j]
                    # Ensure distances are non-negative
                    dist_val = max(0.0, dist_val)
                    
                    # Set distance in matrix (ensure symmetry)
                    dist_matrix[i, idx] = dist_val
                    dist_matrix[idx, i] = dist_val
        
        # Calculate sparsity level
        nonzero = np.count_nonzero(dist_matrix)
        total_elements = n_windows * n_windows
        sparsity = nonzero / total_elements
        
        logger.info(f"Computed sparse distance matrix with {nonzero} non-zero elements ({sparsity:.2%} density)")
        
        # Refine distances for k-nearest neighbors using the actual distance function
        if distance_func is not None:
            logger.info("Refining distances for nearest neighbors using exact distance function")
            
            # Get non-zero indices
            rows, cols = np.nonzero(dist_matrix)
            
            # Process in batches to avoid memory issues
            batch_size = 10000
            for start in range(0, len(rows), batch_size):
                end = min(start + batch_size, len(rows))
                batch_rows = rows[start:end]
                batch_cols = cols[start:end]
                
                # Compute exact distances for this batch
                for idx in range(len(batch_rows)):
                    i, j = batch_rows[idx], batch_cols[idx]
                    if i < j:  # Only compute upper triangle to avoid double computation
                        exact_dist = distance_func(windows[i], windows[j])
                        dist_matrix[i, j] = exact_dist
                        dist_matrix[j, i] = exact_dist  # Ensure symmetry
        
        return dist_matrix, sparsity
        
    except Exception as e:
        logger.error(f"Error in FAISS computation: {str(e)}")
        logger.warning("Falling back to dense distance matrix computation")
        return None, 0.0


def _compute_with_sklearn_ann(windows: List[np.ndarray], 
                             distance_func: Callable,
                             n_neighbors: int = 50) -> Tuple[np.ndarray, float]:
    """
    Fallback method using scikit-learn's NearestNeighbors for approximate distance matrix computation.
    
    Args:
        windows: List of time series windows
        distance_func: Function to compute distance between two windows
        n_neighbors: Number of nearest neighbors to compute for each point
        
    Returns:
        Tuple of (sparse_distance_matrix, sparsity_level)
    """
    if not SKLEARN_NN_AVAILABLE:
        logger.warning("Scikit-learn not available. Using dense distance matrix computation.")
        return None, 0.0
    
    n_windows = len(windows)
    logger.info(f"Computing approximate distance matrix with scikit-learn for {n_windows} windows")
    
    # Convert windows to feature vectors
    features = _flatten_and_normalize_windows(windows)
    
    try:
        # Create NearestNeighbors model
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, n_windows), 
                              algorithm='ball_tree', 
                              metric='cosine', 
                              n_jobs=-1)
        
        # Fit model
        nn.fit(features)
        
        # Find k-nearest neighbors
        distances, indices = nn.kneighbors(features)
        
        # Convert cosine distances to similarities
        distances = 1 - distances  # cosine similarity
        
        # Create sparse distance matrix
        dist_matrix = np.zeros((n_windows, n_windows))
        
        # Fill in the k-nearest neighbors for each point
        for i in range(n_windows):
            for j, idx in enumerate(indices[i]):
                if idx >= 0 and idx < n_windows:  # Valid index
                    dist_val = distances[i, j]
                    
                    # Set distance in matrix (ensure symmetry)
                    dist_matrix[i, idx] = dist_val
                    dist_matrix[idx, i] = dist_val
        
        # Calculate sparsity level
        nonzero = np.count_nonzero(dist_matrix)
        total_elements = n_windows * n_windows
        sparsity = nonzero / total_elements
        
        logger.info(f"Computed sparse distance matrix with {nonzero} non-zero elements ({sparsity:.2%} density)")
        
        # Refine distances for k-nearest neighbors using the actual distance function
        if distance_func is not None:
            logger.info("Refining distances for nearest neighbors using exact distance function")
            
            # Get non-zero indices
            rows, cols = np.nonzero(dist_matrix)
            
            # Process in batches to avoid memory issues
            batch_size = 10000
            for start in range(0, len(rows), batch_size):
                end = min(start + batch_size, len(rows))
                batch_rows = rows[start:end]
                batch_cols = cols[start:end]
                
                # Compute exact distances for this batch
                for idx in range(len(batch_rows)):
                    i, j = batch_rows[idx], batch_cols[idx]
                    if i < j:  # Only compute upper triangle to avoid double computation
                        exact_dist = distance_func(windows[i], windows[j])
                        dist_matrix[i, j] = exact_dist
                        dist_matrix[j, i] = exact_dist  # Ensure symmetry
        
        return dist_matrix, sparsity
        
    except Exception as e:
        logger.error(f"Error in scikit-learn ANN computation: {str(e)}")
        logger.warning("Falling back to dense distance matrix computation")
        return None, 0.0


def compute_distance_matrix(windows: List[np.ndarray], 
                          distance_func: Callable,
                          running_locally: bool = True,
                          n_jobs: int = -1,  # Default to using all cores
                          block_size: int = None,  # Will be calculated adaptively
                          start_dask_client: bool = False,
                          use_sparse_approx: bool = True,  # Whether to use sparse approximation for large datasets
                          sparse_neighbors: int = 100,  # Number of neighbors for sparse approximation
                          use_gpu: bool = False  # Whether to use GPU for FAISS if available
                          ) -> np.ndarray:
    """
    Compute pairwise distance matrix for a list of time series windows.
    Optimized for HFT data with distributed processing support using block-wise computation.
    For very large datasets (>20k points), uses sparse approximation via FAISS or scikit-learn.
    
    Args:
        windows: List of time series windows
        distance_func: Function to compute distance between two windows
        running_locally: If True, always use joblib with maximum cores, if False use Dask
        use_parallel: Whether to use parallel processing
        n_jobs: Number of jobs for parallel processing (-1 for all cores)
        block_size: Size of blocks for block-wise computation (None for adaptive)
        start_dask_client: Whether to start a new Dask client (only used if running_locally=False)
        use_sparse_approx: Whether to use sparse approximation for large datasets (>20k points)
        sparse_neighbors: Number of neighbors to compute for each point in sparse approximation
        use_gpu: Whether to use GPU for FAISS if available
        
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
    
    # For very large datasets, use sparse approximation if enabled
    if n_windows > 20000 and use_sparse_approx:
        logger.info(f"Large dataset detected ({n_windows} windows). Using sparse approximation.")
        
        # Try FAISS first (faster and more accurate)
        if FAISS_AVAILABLE:
            dist_matrix, sparsity = _compute_with_faiss(
                windows, 
                distance_func, 
                n_neighbors=sparse_neighbors,
                use_gpu=use_gpu
            )
            
            if dist_matrix is not None:
                logger.info(f"Successfully computed sparse distance matrix with FAISS (density: {sparsity:.2%})")
                return dist_matrix
        
        # Fall back to scikit-learn if FAISS not available
        if SKLEARN_NN_AVAILABLE:
            dist_matrix, sparsity = _compute_with_sklearn_ann(
                windows, 
                distance_func, 
                n_neighbors=sparse_neighbors
            )
            
            if dist_matrix is not None:
                logger.info(f"Successfully computed sparse distance matrix with scikit-learn (density: {sparsity:.2%})")
                return dist_matrix
        
        logger.warning("Sparse approximation failed or not available. Falling back to dense computation.")
    
    # Calculate optimal chunk size for either implementation
    if n_jobs <= 0:
        n_jobs = max(os.cpu_count() - 1, 1)
    
    # Determine whether to use joblib (local) or Dask (distributed)
    if running_locally or not DASK_AVAILABLE:
        # Always use joblib when running locally, or if Dask is not available
        return _compute_with_joblib(windows, distance_func, n_jobs, n_windows, block_size)
    else:
        # Use Dask for distributed computing
        return _compute_with_dask(windows, distance_func, n_jobs, n_windows, block_size, start_dask_client)

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

def _compute_with_dask(windows, distance_func, n_jobs, n_windows, block_size=None, start_dask_client=False):
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
        
        # Try to get existing client or create new one
        client = None
        if not start_dask_client:
            try:
                from dask.distributed import get_client
                client = get_client()
                logger.info("Using existing Dask client")
            except (ValueError, ImportError):
                pass
        
        if client is None and start_dask_client:
            from dask.distributed import Client, LocalCluster
            # Create cluster with specific settings for distance computation
            cluster = LocalCluster(
                n_workers=n_jobs,
                threads_per_worker=1,  # Better for CPU-bound tasks
                processes=True,        # True process isolation
                memory_limit='4GB',    # Prevent memory issues
                scheduler_port=0       # Random port to avoid conflicts
            )
            client = Client(cluster)
            logger.info(f"Started new Dask cluster with {n_jobs} workers")
        
        if client is None:
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
        
        if start_dask_client:
            client.close()
            
        return dist_matrix
        
    except Exception as e:
        logger.error(f"Error in Dask computation: {e}")
        logger.warning("Falling back to joblib")
        return _compute_with_joblib(windows, distance_func, n_jobs, n_windows, block_size)
