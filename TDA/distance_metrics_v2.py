"""
Financial distance metrics optimized for high-frequency trading (HFT) data.

This module provides specialized distance functions and efficient computation
methods for financial time series, with a focus on detecting market regimes
in high-frequency trading data.

Key features:
- Optimized distance metrics for financial time series (distribution-based, wavelet-based)
- Efficient distance matrix computation with GPU acceleration
- Support for distributed computing with Dask and GPU clusters

Dependencies:
- Required: numpy, scipy, pandas
- Required for GPU: cupy, cusignal
- Optional: joblib (for parallel processing)
- Optional: dask (for distributed computing)
- Optional: dask_cuda (for GPU clusters)
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

# Configure optimal transport backend for POT
import os
os.environ["OT_BACKEND"] = "numpy"
# We no longer need to disable JAX
# os.environ["JAX_PLATFORMS"] = ""
# os.environ["JAX_DISABLE_JIT"] = "1"

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Callable, Optional, Dict, Any
from scipy.stats import wasserstein_distance, energy_distance
import logging
from functools import partial
import warnings
import time
import math

# Silence warnings about cusignal
warnings.filterwarnings("ignore", message="cusignal not installed")

# Configure logging to be less verbose
class LogFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.seen_messages = set()
        # Higher threshold for less filtering in development
        self.repeat_threshold = 1  # Set to 1 to be even more aggressive
        # Track when we've seen specific message patterns
        self.seen_patterns = {
            "gpu_info": False,
            "gpu_acceleration": False,
            "hdbscan_config": False,
            "memory_adjustment": False,
            "cusignal_warning": False,
            "ott_available": False,
            "ott_unavailable": False,
            "cuml_available": False,
            "mock_memory": False,
            "mapper_running": False
        }
        # Counters for parallelization messages
        self.worker_counters = {
            "adjusted_min_cluster_size": 0,
            "using_gpu_hdbscan": 0,
            "gpu_accelerated_nn": 0,
            "memory_requirement": 0
        }
        
    def filter(self, record):
        message = record.getMessage()
        
        # COMPLETELY SUPPRESS certain messages
        
        # 1. Suppress cusignal warnings entirely
        if "cusignal not installed" in message:
            return False
            
        # 2. Suppress GPU acceleration warnings entirely  
        if "GPU-accelerated optimal transport not available" in message:
            return False
            
        # 3. Suppress all GPU detection messages
        if "GPU acceleration enabled" in message or "NVIDIA H100" in message:
            return False
            
        # 4. Suppress all HDBSCAN configuration messages
        if "Using HDBSCAN with" in message:
            return False
            
        # 5. Suppress memory requirement adjustments
        if "Adjusting memory requirement for H100" in message:
            return False
            
        # 6. Suppress memory allocation messages  
        if "Added mock memory_allocated function" in message:
            return False
            
        # 7. Suppress cuML availability messages
        if "cuML" in message and "available for GPU-accelerated" in message:
            return False
            
        # 8. Suppress repetitive OTT/JAX messages
        if "JAX and OTT available" in message or "JAX/OTT not available" in message:
            return False
            
        # 9. Suppress high-performance GPU detection messages
        if "Detected high-performance GPU" in message:
            return False
            
        # 10. Suppress repetitive cluster size adjustments in parallel processing
        if "Adjusted min_cluster_size to" in message:
            return False
                
        # 11. Suppress repetitive GPU-accelerated HDBSCAN messages
        if "Using GPU-accelerated HDBSCAN from cuML" in message:
            return False
            
        # 12. Suppress batch size and computation messages
        if any(pattern in message for pattern in [
            "Using adaptive batches of size",
            "based on available GPU memory ratio",
            "Starting optimized batched Wasserstein distance computation",
            "Using adaptive batch sizes:",
            "Using JAX batch size:",
            "GPU memory: Free", 
            "Required memory:",
            "Using CuPy vectorized batch Wasserstein"
        ]):
            return False
            
        # 13. Replace detailed cluster startup with simplified message
        if ("Processing" in message and "hypercubes" in message) or \
           ("parallel processing" in message and "hypercubes" in message):
            if not self.seen_patterns.get("mapper_running", False):
                self.seen_patterns["mapper_running"] = True
                # Replace with simple message that MapperTDA is running
                record.msg = "Running MapperTDA on data..."
                record.args = ()
                return True
            return False
            
        # Allow any other messages to pass through
        return True

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addFilter(LogFilter())

# Add an aggressive filter for "Minimal logging mode enabled" messages
class BlockMinimalLoggingFilter(logging.Filter):
    def filter(self, record):
        if "Minimal logging mode enabled" in record.getMessage():
            return False
        return True

# Apply this filter to the root logger to block all "Minimal logging mode enabled" messages
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(BlockMinimalLoggingFilter())

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

# Cache for GPU memory checks to avoid redundant calls
_GPU_MEM_CACHE = {
    'last_check_time': 0,
    'available_memory': 0,
    'total_memory': 0,
    'cache_valid': False
}

# Function to check GPU memory with caching
def _get_gpu_memory(force_refresh=False):
    """Get GPU memory info with caching to reduce overhead."""
    global _GPU_MEM_CACHE
    
    current_time = time.time()
    # Refresh cache if it's invalid, forced, or more than 1 second old
    if (not _GPU_MEM_CACHE['cache_valid'] or 
        force_refresh or 
        current_time - _GPU_MEM_CACHE['last_check_time'] > 1.0):
        
        if cp_available:
            try:
                dev = Device()
                mem_free = dev.mem_info[0] / (1024**3)  # Free memory in GB
                mem_total = dev.mem_info[1] / (1024**3)  # Total memory in GB
                
                _GPU_MEM_CACHE.update({
                    'last_check_time': current_time,
                    'available_memory': mem_free,
                    'total_memory': mem_total,
                    'cache_valid': True
                })
            except Exception:
                # If there's an error, invalidate cache
                _GPU_MEM_CACHE['cache_valid'] = False
                return 0, 0
        else:
            _GPU_MEM_CACHE['cache_valid'] = False
            return 0, 0
    
    return _GPU_MEM_CACHE['available_memory'], _GPU_MEM_CACHE['total_memory']

# Function to ensure GPU has enough memory
def _ensure_gpu_memory(required_gb, safety_factor=1.5):
    """Check if GPU has enough memory with caching to reduce overhead."""
    if not cp_available:
        return False
        
    mem_free, _ = _get_gpu_memory()
    
    # Apply safety factor for memory requirements
    required_with_safety = required_gb * safety_factor
    
    if mem_free >= required_with_safety:
        return True
    
    # If memory check fails, force a refresh and try again
    mem_free, _ = _get_gpu_memory(force_refresh=True)
    
    # Attempt to free memory
    try:
        cp.get_default_memory_pool().free_all_blocks()
        mem_free, _ = _get_gpu_memory(force_refresh=True)
    except Exception:
        pass
        
    return mem_free >= required_with_safety

# Try to import JAX and OTT for GPU-accelerated Wasserstein distance
try:
    try:
        import jax
        import jax.numpy as jnp
        
        # Try to import OTT components
        try:
            from ott.geometry import pointcloud
            from ott.problems.linear import linear_problem
            from ott.solvers.linear import sinkhorn
            
            # Set JAX to use GPU
            jax.config.update('jax_platform_name', 'gpu')
            JAX_AVAILABLE = True
            OTT_AVAILABLE = True
            logger.info("JAX and OTT available for GPU-accelerated Wasserstein distance")
        except ImportError:
            JAX_AVAILABLE = True
            OTT_AVAILABLE = False
            logger.warning("JAX available but OTT not installed. Using alternative methods for Wasserstein distance.")
            
    except RuntimeError as e:
        # Handle version mismatch errors
        if "jaxlib" in str(e) and "version" in str(e):
            logger.warning(f"JAX version mismatch: {e}. Using alternative methods for Wasserstein distance.")
        else:
            logger.warning(f"JAX runtime error: {e}. Using alternative methods for Wasserstein distance.")
        JAX_AVAILABLE = False
        OTT_AVAILABLE = False
    
except ImportError:
    JAX_AVAILABLE = False
    OTT_AVAILABLE = False
    logger.warning("JAX/OTT not available. Using alternative methods for Wasserstein distance.")

# Try to import cusignal for GPU-accelerated signal processing
try:
    import cusignal
    CUSIGNAL_AVAILABLE = True and cp_available
except ImportError:
    CUSIGNAL_AVAILABLE = False
    logger.warning("cusignal not installed. GPU signal processing will not be available.")

# Try to import cuML for GPU-accelerated machine learning
try:
    import cuml
    CUML_AVAILABLE = True and cp_available
except ImportError:
    CUML_AVAILABLE = False
    logger.warning("cuML not installed. GPU-accelerated ML will not be available.")

# Try to import optimal transport libraries
try:
    # POT will use numpy backend due to the environment variable set at the top of the file
    import ot  # Python Optimal Transport library
    POT_AVAILABLE = True
    
    # Check if GPU version of OT is available
    try:
        import ot.gpu
        OT_GPU_AVAILABLE = True and cp_available
    except (ImportError, AttributeError):
        OT_GPU_AVAILABLE = False
        logger.warning("GPU-accelerated optimal transport not available.")
except ImportError as e:
    logger.warning(f"POT not installed or import error: {str(e)}. Using scipy for Wasserstein distance.")
    POT_AVAILABLE = False
    OT_GPU_AVAILABLE = False

# Try to import PyWavelets
try:
    import pywt  # PyWavelets for wavelet-based distances
    WAVELETS_AVAILABLE = True
except ImportError:
    WAVELETS_AVAILABLE = False
    logger.warning("PyWavelets not installed. Wavelet-based distances will not be available.")

# Try to import Dask for distributed computing
try:
    import dask
    import dask.array as da
    from dask.distributed import Client, wait
    DASK_AVAILABLE = True
    
    # Check for GPU-accelerated Dask
    try:
        from dask_cuda import LocalCUDACluster
        DASK_CUDA_AVAILABLE = True and cp_available
    except ImportError:
        DASK_CUDA_AVAILABLE = False
        logger.warning("dask_cuda not installed. Distributed GPU computation not available.")
except ImportError:
    DASK_AVAILABLE = False
    DASK_CUDA_AVAILABLE = False
    logger.warning("Dask not installed. Falling back to local computation.")

# GPU memory management utilities
def _gpu_mem_usage():
    """Get current GPU memory usage in GB."""
    if not cp_available:
        return 0.0
    
    try:
        # Try different approaches to get memory usage in case API changes
        try:
            # First try the memory_allocated function
            mem_used = cp.cuda.memory_allocated() / (1024 ** 3)  # GB
        except AttributeError:
            # If memory_allocated is not available, try get_current_stream().used_bytes()
            try:
                mem_used = cp.cuda.get_current_stream().used_bytes() / (1024 ** 3)  # GB
            except (AttributeError, TypeError):
                # If that fails too, get device memory info
                dev = cp.cuda.Device()
                mem_total = dev.mem_info[1] / (1024 ** 3)  # Total memory in GB
                mem_free = dev.mem_info[0] / (1024 ** 3)   # Free memory in GB
                mem_used = mem_total - mem_free
                
        return mem_used
    except Exception as e:
        logger.error(f"Error getting GPU memory usage: {e}")
        return 0.0

def to_gpu(arr, force_copy=False):
    """Safely transfer a numpy array to GPU memory."""
    if not cp_available:
        return arr
    
    try:
        # If it's already a cupy array, return it
        if isinstance(arr, cp.ndarray):
            return arr.copy() if force_copy else arr
        
        # Check if it's a numpy array
        if not isinstance(arr, np.ndarray):
            # Convert to numpy array first
            logger.debug(f"Converting {type(arr)} to numpy array before GPU transfer")
            arr = np.asarray(arr)
            
        # Ensure we have a contiguous array with the right dtype
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
            
        # Use float32 for better GPU performance
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        
        # Estimate memory requirements and check availability
        mem_needed = arr.nbytes / (1024 ** 3)  # GB
        if not _ensure_gpu_memory(mem_needed * 1.5):  # Add 50% safety margin
            logger.warning(f"Not enough GPU memory ({mem_needed*1.5:.2f} GB required)")
            return arr
        
        # Transfer to GPU
        logger.debug(f"Transferring array of shape {arr.shape} to GPU")
        return cp.asarray(arr)
    except Exception as e:
        logger.warning(f"Failed to transfer array to GPU: {str(e)}")
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

# Add this near the top of the file, after the cp_available check
# Create a mock memory_allocated function if it doesn't exist
if cp_available and not hasattr(cp.cuda, 'memory_allocated'):
    # Create a function that gets memory info from the device
    def memory_allocated():
        dev = cp.cuda.Device()
        mem_total = dev.mem_info[1]  # Total memory in bytes
        mem_free = dev.mem_info[0]   # Free memory in bytes
        return mem_total - mem_free
    
    # Add it to cp.cuda
    cp.cuda.memory_allocated = memory_allocated
    logger.info("Added mock memory_allocated function to CuPy")

class FinancialDistanceMetrics:
    """Distance metrics for financial time series in HFT with GPU acceleration."""
    
    @staticmethod
    def distribution_distance(x: np.ndarray, y: np.ndarray, method: str = 'wasserstein', use_gpu: bool = None) -> float:
        """
        Optimized Wasserstein distance implementation for financial time series.
        Specifically tuned for detecting regime changes in HFT data when used with HDBSCAN.
        
        Args:
            x: First time series window
            y: Second time series window
            method: Distance method ('wasserstein' is recommended and optimized)
            use_gpu: Whether to use GPU acceleration (None for auto-detection)
            
        Returns:
            Distribution distance value
        """
        # Check if GPU is available and requested
        if use_gpu is None:
            use_gpu = cp_available
        
        # Skip GPU if it's not available or not requested
        if not use_gpu or not cp_available:
            # Convert inputs to numpy arrays if they're gpu arrays
            if isinstance(x, cp.ndarray):
                x = cp.asnumpy(x)
            if isinstance(y, cp.ndarray):
                y = cp.asnumpy(y)
            return FinancialDistanceMetrics._cpu_distribution_distance(x, y, method)
        
        # Check if inputs are already GPU arrays, if not, transfer them
        if not isinstance(x, cp.ndarray):
            x_gpu = to_gpu(x)
        else:
            x_gpu = x
            
        if not isinstance(y, cp.ndarray):
            y_gpu = to_gpu(y)
        else:
            y_gpu = y
            
        try:
            # Call GPU implementation
            result = FinancialDistanceMetrics._gpu_distribution_distance(x_gpu, y_gpu, method)
            
            # Make sure we return a scalar Python float, not a device array
            if isinstance(result, cp.ndarray):
                result = float(cp.asnumpy(result))
            elif hasattr(result, 'item'):
                result = result.item()
                
            return float(result)
            
        except Exception as e:
            logger.warning(f"GPU distance calculation failed: {e}, falling back to CPU")
            
            # Convert to CPU if needed
            if isinstance(x, cp.ndarray):
                x = cp.asnumpy(x)
            if isinstance(y, cp.ndarray):
                y = cp.asnumpy(y)
                
            return FinancialDistanceMetrics._cpu_distribution_distance(x, y, method)
    
    @staticmethod
    def _cpu_distribution_distance(x: np.ndarray, y: np.ndarray, method: str = 'wasserstein') -> float:
        """CPU implementation of distribution distance."""
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
    def _gpu_distribution_distance(x, y, method='wasserstein') -> float:
        """
        Calculate the Wasserstein or Energy distance between two distributions using GPU acceleration.
        Optimized with vectorized GPU computation.
        
        Parameters:
        -----------
        x, y : array-like
            Values from the two distributions to compare
        method : str, default='wasserstein'
            Distance metric to use, either 'wasserstein' or 'energy'
        
        Returns:
        --------
        float : The computed distance
        """
        # Ensure inputs are flattened arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Quick return for empty arrays or identical distributions
        if len(x_flat) == 0 or len(y_flat) == 0:
            return 0.0
        if cp.array_equal(x_flat, y_flat):
            return 0.0
            
        # For very small arrays, use a simpler approach
        if len(x_flat) <= 3 or len(y_flat) <= 3:
            # For tiny arrays, use a simple euclidean distance
            len_diff = abs(len(x_flat) - len(y_flat))
            if len_diff > 0:
                # Pad the shorter array
                if len(x_flat) < len(y_flat):
                    x_pad = cp.pad(x_flat, (0, len_diff), 'constant', constant_values=cp.median(x_flat))
                    return float(cp.sqrt(cp.mean((x_pad - y_flat)**2)))
                else:
                    y_pad = cp.pad(y_flat, (0, len_diff), 'constant', constant_values=cp.median(y_flat))
                    return float(cp.sqrt(cp.mean((x_flat - y_pad)**2)))
            else:
                return float(cp.sqrt(cp.mean((x_flat - y_flat)**2)))
        
        # Optimize calculation by downsampling very large arrays
        max_points = 1000  # Maximum points to consider
        if len(x_flat) > max_points:
            idx = cp.linspace(0, len(x_flat)-1, max_points).astype(cp.int32)
            x_flat = x_flat[idx]
        if len(y_flat) > max_points:
            idx = cp.linspace(0, len(y_flat)-1, max_points).astype(cp.int32)
            y_flat = y_flat[idx]
            
        # Use robust scaling to handle outliers
        try:
            # Using percentiles for robust scaling
            x_median = cp.median(x_flat)
            y_median = cp.median(y_flat)
            
            # Sort values for percentile calculation
            x_sorted = cp.sort(x_flat)
            y_sorted = cp.sort(y_flat)
            
            # Calculate IQR
            x_q75 = x_sorted[int(0.75 * len(x_sorted))]
            x_q25 = x_sorted[int(0.25 * len(x_sorted))]
            y_q75 = y_sorted[int(0.75 * len(y_sorted))]
            y_q25 = y_sorted[int(0.25 * len(y_sorted))]
            
            x_iqr = x_q75 - x_q25
            y_iqr = y_q75 - y_q25
            
            # Avoid division by zero
            x_iqr = x_iqr if x_iqr > 0 else 1.0
            y_iqr = y_iqr if y_iqr > 0 else 1.0
            
            # Scale using IQR
            x_scaled = (x_flat - x_median) / x_iqr
            y_scaled = (y_flat - y_median) / y_iqr
            
            # Replace NaN values
            x_scaled = cp.nan_to_num(x_scaled)
            y_scaled = cp.nan_to_num(y_scaled)
            
            # Compute Wasserstein distance directly with GPU
            if method == 'wasserstein':
                # Sort values (required for 1D Wasserstein)
                x_sorted = cp.sort(x_scaled)
                y_sorted = cp.sort(y_scaled)
                
                # Get common size for comparison
                n = min(len(x_sorted), len(y_sorted))
                
                # Generate uniform points for both distributions
                if len(x_sorted) != len(y_sorted):
                    # Compute CDF points
                    cdf_points = cp.linspace(0, 1, n)
                    
                    # Interpolate to same number of points
                    x_quantiles = cp.interp(cdf_points, 
                                           cp.linspace(0, 1, len(x_sorted)), 
                                           x_sorted)
                    y_quantiles = cp.interp(cdf_points,
                                           cp.linspace(0, 1, len(y_sorted)),
                                           y_sorted)
                else:
                    # If same length, use directly
                    x_quantiles = x_sorted
                    y_quantiles = y_sorted
                
                # Calculate L2 Wasserstein distance
                w_dist = cp.sqrt(cp.mean((x_quantiles - y_quantiles)**2))
                
                # Convert to Python float
                return float(cp.asnumpy(w_dist))
            
            elif method == 'energy':
                # Energy distance on GPU
                # Formula: 2*E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]
                
                # First term: 2*E[|X-Y|]
                xy_dists = cp.abs(x_scaled.reshape(-1, 1) - y_scaled.reshape(1, -1))
                xy_term = 2 * cp.mean(xy_dists)
                
                # Second term: E[|X-X'|]
                xx_dists = cp.abs(x_scaled.reshape(-1, 1) - x_scaled.reshape(1, -1))
                xx_term = cp.mean(xx_dists)
                
                # Third term: E[|Y-Y'|]
                yy_dists = cp.abs(y_scaled.reshape(-1, 1) - y_scaled.reshape(1, -1))
                yy_term = cp.mean(yy_dists)
                
                # Compute energy distance
                e_dist = xy_term - xx_term - yy_term
                
                # Convert to Python float
                return float(cp.asnumpy(e_dist))
                
            else:
                # Default to Wasserstein if method not recognized
                logger.warning(f"Method {method} not recognized, using wasserstein")
                # Recursively call with wasserstein
                return FinancialDistanceMetrics._gpu_distribution_distance(x, y, 'wasserstein')
                
        except Exception as e:
            logger.warning(f"GPU distance calculation error: {e}")
            # Free GPU memory
            cp.get_default_memory_pool().free_all_blocks()
            
            # Fall back to CPU implementation
            x_cpu = cp.asnumpy(x) if isinstance(x, cp.ndarray) else x
            y_cpu = cp.asnumpy(y) if isinstance(y, cp.ndarray) else y
            return FinancialDistanceMetrics._cpu_distribution_distance(x_cpu, y_cpu, method)
    
    @staticmethod
    def wavelet_distance(x: np.ndarray, y: np.ndarray, 
                         wavelet: str = 'db4', 
                         level: int = 3,
                         use_gpu: bool = None) -> float:
        """
        Wavelet-based distance capturing multi-scale differences.
        Especially useful for high-frequency financial data with patterns at different timescales.
        
        Args:
            x: First time series
            y: Second time series
            wavelet: Wavelet function to use
            level: Decomposition level
            use_gpu: Whether to use GPU acceleration (None for auto-detection)
            
        Returns:
            Wavelet distance value
        """
        # Determine if we should use GPU
        if use_gpu is None:
            use_gpu = cp_available and CUSIGNAL_AVAILABLE
        
        # Skip GPU if not available, not requested, or wavelet processing not available
        if not use_gpu or not cp_available or not CUSIGNAL_AVAILABLE:
            return FinancialDistanceMetrics._cpu_wavelet_distance(x, y, wavelet, level)
        else:
            try:
                return FinancialDistanceMetrics._gpu_wavelet_distance(x, y, wavelet, level)
            except Exception as e:
                logger.warning(f"GPU wavelet distance calculation failed: {e}, falling back to CPU")
                return FinancialDistanceMetrics._cpu_wavelet_distance(x, y, wavelet, level)
    
    @staticmethod
    def _cpu_wavelet_distance(x: np.ndarray, y: np.ndarray, wavelet: str = 'db4', level: int = 3) -> float:
        """CPU implementation of wavelet distance."""
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
    
    @staticmethod
    def _gpu_wavelet_distance(x: np.ndarray, y: np.ndarray, wavelet: str = 'db4', level: int = 3) -> float:
        """GPU implementation of wavelet distance."""
        # Estimate memory requirements and ensure GPU has enough memory
        x_size = x.nbytes
        y_size = y.nbytes
        mem_needed = (x_size + y_size) * 4 / (1024**3)  # Approximate memory in GB with extra space for calculations
        
        if not _ensure_gpu_memory(mem_needed):
            # Fall back to CPU if not enough GPU memory
            return FinancialDistanceMetrics._cpu_wavelet_distance(x, y, wavelet, level)
        
        # Flatten arrays and transfer to GPU
        if isinstance(x, cp.ndarray):
            x_flat = x.flatten()
        else:
            x_flat = to_gpu(x.flatten())
            
        if isinstance(y, cp.ndarray):
            y_flat = y.flatten()
        else:
            y_flat = to_gpu(y.flatten())
        
        # Find power of 2 that's >= length
        max_len = max(x_flat.size, y_flat.size)
        pad_len = 1
        while pad_len < max_len:
            pad_len *= 2
            
        # Pad arrays with zeros
        x_padded = cp.zeros(pad_len, dtype=cp.float32)
        y_padded = cp.zeros(pad_len, dtype=cp.float32)
        x_padded[:x_flat.size] = x_flat
        y_padded[:y_flat.size] = y_flat
        
        try:
            # Use cusignal's wavelet decomposition if available
            # Note: cusignal's wavelet API might differ from PyWavelets
            # We're using a compatible subset of functionality
            # Might need to adjust based on actual cusignal version
            coeff_x = cusignal.cwt(x_padded, wavelet, level)
            coeff_y = cusignal.cwt(y_padded, wavelet, level)
            
            # Pre-calculate weights for all levels
            weights = cp.zeros(level + 1, dtype=cp.float32)
            weights[0] = 0.4  # Approximation coefficients
            for i in range(1, level + 1):
                weights[i] = 0.6 * (2 ** -(i-1))
            weights = weights / cp.sum(weights)  # Normalize weights
            
            # Calculate distances for each level
            distances = cp.zeros(level + 1, dtype=cp.float32)
            
            for i, (cx, cy) in enumerate(zip(coeff_x, coeff_y)):
                # Normalization
                cx_std = cp.std(cx)
                cy_std = cp.std(cy)
                
                # Avoid division by zero
                cx_norm = cx / cx_std if cx_std > 0 else cx
                cy_norm = cy / cy_std if cy_std > 0 else cy
                
                # Distance calculation
                distances[i] = cp.sqrt(cp.mean((cx_norm - cy_norm) ** 2))
            
            # Weighted sum of distances
            result = float(cp.sum(distances * weights).get())
            
            # Clean up GPU memory
            if not isinstance(x, cp.ndarray):
                del x_flat
            if not isinstance(y, cp.ndarray):
                del y_flat
            del x_padded, y_padded, coeff_x, coeff_y, weights, distances
            
            # Force garbage collection to release GPU memory
            cp.get_default_memory_pool().free_all_blocks()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in GPU wavelet distance: {e}")
            
            # Clean up GPU memory before falling back to CPU
            try:
                del x_padded, y_padded
                if not isinstance(x, cp.ndarray):
                    del x_flat
                if not isinstance(y, cp.ndarray):
                    del y_flat
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
                
            # Fall back to CPU implementation
            return FinancialDistanceMetrics._cpu_wavelet_distance(x, y, wavelet, level)


def create_financial_distance_function(metric: str = 'distribution',
                                     wavelet: str = 'db4',
                                     distribution_method: str = 'wasserstein',
                                     use_gpu: bool = None,
                                     **kwargs) -> Callable:
    """
    Create a distance function for financial time series with HFT focus.
    
    Args:
        metric: Distance metric type ('distribution', 'wavelet')
        wavelet: Wavelet to use for wavelet-based distance
        distribution_method: Method for distribution distance
        use_gpu: Whether to use GPU acceleration (None for auto-detection)
        **kwargs: Additional parameters for specific distance metrics
        
    Returns:
        Distance function that takes two time series and returns a distance value
    """
    # Determine if we should use GPU
    if use_gpu is None:
        use_gpu = cp_available
    
    if use_gpu and cp_available:
        gpu_str = "with GPU acceleration"
    else:
        gpu_str = "on CPU"
        
    logger.info(f"Creating financial distance function with metric: {metric} {gpu_str}")
    
    # Create a wrapper function to ensure float return value
    def ensure_float_wrapper(func):
        def wrapped(x, y):
            result = func(x, y)
            # Explicitly handle CuPy arrays
            if hasattr(cp, 'ndarray') and isinstance(result, cp.ndarray):
                result = float(cp.asnumpy(result))
            # Handle JAX arrays
            elif hasattr(result, 'item'):
                result = result.item()
            # Final conversion to Python float
            return float(result)
        return wrapped
    
    if metric == 'distribution':
        dist_func = partial(FinancialDistanceMetrics.distribution_distance, method=distribution_method, use_gpu=use_gpu)
        return ensure_float_wrapper(dist_func)
    
    elif metric == 'wavelet':
        if not WAVELETS_AVAILABLE and not (CUSIGNAL_AVAILABLE and use_gpu):
            logger.warning("Neither PyWavelets nor cusignal available. Using euclidean distance instead.")
            # Create a euclidean distance function that returns float
            def euclidean_dist(x, y):
                return float(np.sqrt(np.mean((x.flatten() - y.flatten())**2)))
            return euclidean_dist
        
        wavelet_func = partial(FinancialDistanceMetrics.wavelet_distance, wavelet=wavelet, level=kwargs.get('level', 3), use_gpu=use_gpu)
        return ensure_float_wrapper(wavelet_func)
    
    else:
        logger.warning(f"Unknown distance metric: {metric}, using distribution")
        dist_func = partial(FinancialDistanceMetrics.distribution_distance, method=distribution_method, use_gpu=use_gpu)
        return ensure_float_wrapper(dist_func)


def compute_distance_matrix(windows: List[np.ndarray], 
                           distance_func: Callable,
                           use_gpu: bool = None,
                           running_locally: bool = True,
                           n_jobs: int = -1,  # Default to using all cores
                           block_size: int = None,  # Will be calculated adaptively
                           memory_limit: float = None  # Memory limit in GB
                           ) -> np.ndarray:
    """
    Compute pairwise distance matrix for a list of time series windows.
    Optimized for HFT data with GPU acceleration and distributed processing support.
    
    Args:
        windows: List of time series windows
        distance_func: Function to compute distance between two windows
        use_gpu: Whether to use GPU acceleration (None for auto-detection)
        running_locally: If True, process on local machine; if False use Dask
        n_jobs: Number of jobs for parallel processing (-1 for all cores)
        block_size: Size of blocks for block-wise computation (None for adaptive)
        memory_limit: Memory limit in GB (None for automatic estimation)
        
    Returns:
        Distance matrix of shape (n_windows, n_windows)
    """
    n_windows = len(windows)
    
    # Early return for empty input
    if n_windows == 0:
        return np.array([])
    
    # Early return for single window
    if n_windows == 1:
        return np.array([[0.0]])
    
    # Determine if we should use GPU - auto-detect if not specified
    if use_gpu is None:
        use_gpu = cp_available
    
    # Disable GPU if requested or not available
    if not cp_available:
        use_gpu = False
    
    computation_type = "GPU" if use_gpu else "CPU"
    logger.info(f"Computing distance matrix for {n_windows} windows using {computation_type}")
    
    # Check if we have large arrays that need to be flattened
    window_size = None
    window_sizes_consistent = True
    
    # Check and standardize window shapes
    for i, win in enumerate(windows):
        # Skip None values
        if win is None:
            continue
            
        if hasattr(win, 'shape'):
            # Get size of first valid window
            if window_size is None:
                window_size = win.size
            # Check if all windows have the same size
            elif win.size != window_size:
                window_sizes_consistent = False
                logger.warning(f"Window {i} has size {win.size}, expected {window_size}. Will flatten windows.")
                break
    
    # Print memory usage for large computations
    if n_windows > 100:
        if use_gpu and cp_available:
            # Get GPU memory info
            device = cp.cuda.Device()
            total_mem = device.mem_info[1] / (1024**3)  # Total memory in GB
            free_mem = device.mem_info[0] / (1024**3)   # Free memory in GB
            logger.info(f"GPU memory: {free_mem:.2f}GB free / {total_mem:.2f}GB total")
            
            # Force memory cleanup
            cp.get_default_memory_pool().free_all_blocks()
            free_mem_after = device.mem_info[0] / (1024**3)
            logger.info(f"After cleanup: {free_mem_after:.2f}GB free")
    
    # For very small datasets, use direct computation
    if n_windows <= 10:
        logger.info(f"Small dataset ({n_windows} windows), using direct computation")
        # For tiny datasets, use simple implementation
        if use_gpu:
            return _compute_tiny_matrix_gpu(windows, distance_func, n_windows)
        else:
            return _compute_tiny_matrix_cpu(windows, distance_func, n_windows)
    
    # For medium datasets, use vectorized computation with batching
    elif n_windows <= 5000:
        logger.info(f"Medium dataset ({n_windows} windows), using optimized computation")
        if use_gpu:
            try:
                # Try fast vectorized computation for medium datasets
                return _compute_with_gpu(windows, distance_func, n_windows, block_size, memory_limit)
            except Exception as e:
                logger.warning(f"GPU computation failed: {e}, falling back to CPU")
                # Fall back to CPU
                return _compute_with_joblib(windows, distance_func, n_jobs, n_windows, block_size)
        else:
            return _compute_with_joblib(windows, distance_func, n_jobs, n_windows, block_size)
    
    # For large datasets, use distributed computation if possible
    else:
        logger.info(f"Large dataset ({n_windows} windows), using distributed computation")
        if use_gpu:
            if DASK_CUDA_AVAILABLE and not running_locally:
                # Use distributed GPU computation
                return _compute_with_dask_gpu(windows, distance_func, n_jobs, n_windows, block_size, memory_limit)
            else:
                # Use local GPU computation
                return _compute_with_gpu(windows, distance_func, n_windows, block_size, memory_limit)
        else:
            if DASK_AVAILABLE and not running_locally:
                # Use distributed CPU computation
                return _compute_with_dask(windows, distance_func, n_jobs, n_windows, block_size)
            else:
                # Use local CPU computation
                return _compute_with_joblib(windows, distance_func, n_jobs, n_windows, block_size)

def _compute_tiny_matrix_cpu(windows, distance_func, n_windows):
    """Helper function for tiny matrices on CPU."""
    dist_matrix = np.zeros((n_windows, n_windows))
    start_time = time.time()
    last_update_time = start_time
    total_computations = n_windows * (n_windows - 1) // 2
    computations_done = 0
    
    for i in range(n_windows):
        for j in range(i, n_windows):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                dist = distance_func(windows[i], windows[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # Symmetric
                computations_done += 1
                
                # Check if a minute has passed since the last update
                current_time = time.time()
                if current_time - last_update_time >= 60:  # 60 seconds = 1 minute
                    progress_percent = (computations_done / total_computations) * 100
                    logger.info(f"Distance matrix progress: {computations_done}/{total_computations} distances computed ({progress_percent:.1f}%)")
                    last_update_time = current_time
    
    return dist_matrix

def _compute_tiny_matrix_gpu(windows, distance_func, n_windows):
    """Helper function for tiny matrices on GPU."""
    try:
        logger.info("Starting _compute_tiny_matrix_gpu for %d windows", n_windows)
        
        # Allocate result matrix on GPU
        dist_matrix = cp.zeros((n_windows, n_windows), dtype=cp.float32)
        
        # Transfer windows to GPU if needed
        gpu_windows = []
        for w in windows:
            if not isinstance(w, cp.ndarray):
                gpu_windows.append(to_gpu(w))
            else:
                gpu_windows.append(w)
        
        logger.info("Windows transferred to GPU successfully")
        
        # Initialize timing variables for progress reporting
        start_time = time.time()
        last_update_time = start_time
        total_computations = n_windows * (n_windows - 1) // 2
        computations_done = 0
        
        # Compute distances
        for i in range(n_windows):
            for j in range(i, n_windows):
                try:
                    if i == j:
                        dist_matrix[i, j] = 0.0
                    else:
                        # Get the distance value
                        dist = distance_func(gpu_windows[i], gpu_windows[j])
                        
                        # Explicit type checking and conversion
                        if isinstance(dist, cp.ndarray):
                            logger.debug("Converting CuPy array to scalar")
                            dist = float(cp.asnumpy(dist))
                        elif hasattr(dist, 'item'):  # Handle JAX arrays
                            logger.debug("Converting JAX array to scalar")
                            dist = dist.item()
                        
                        # Ensure we have a plain Python float
                        dist = float(dist)
                        
                        # Assign to the matrix
                        dist_matrix[i, j] = dist
                        dist_matrix[j, i] = dist  # Symmetric
                        
                        # Count completed computation
                        computations_done += 1
                        
                        # Check if a minute has passed since the last update
                        current_time = time.time()
                        if current_time - last_update_time >= 60:  # 60 seconds = 1 minute
                            progress_percent = (computations_done / total_computations) * 100
                            logger.info(f"Distance matrix progress: {computations_done}/{total_computations} distances computed ({progress_percent:.1f}%)")
                            last_update_time = current_time
                            
                except Exception as e:
                    logger.error(f"Error computing distance for windows ({i},{j}): {e}")
                    raise
        
        logger.info("Distances computed successfully on GPU")
        
        # Transfer result back to CPU using .get() explicitly
        logger.debug("Transferring result matrix from GPU to CPU")
        result = cp.asnumpy(dist_matrix)
        
        # Clean up GPU memory
        del dist_matrix
        if not any(isinstance(w, cp.ndarray) for w in windows):
            del gpu_windows
        cp.get_default_memory_pool().free_all_blocks()
        logger.info("GPU memory cleaned up successfully")
        
        return result
        
    except Exception as e:
        logger.warning(f"GPU tiny matrix computation failed: {str(e)}, falling back to CPU")
        return _compute_tiny_matrix_cpu(windows, distance_func, n_windows)

def _compute_with_gpu(windows, distance_func, n_windows, block_size=None, memory_limit=None):
    """
    Helper function for computing distance matrix using GPU.
    Optimized for large-scale computation with intelligent batching.
    """
    try:
        if not cp_available:
            logger.warning("CuPy not available. Falling back to CPU computation.")
            return _compute_with_joblib(windows, distance_func, -1, n_windows, block_size)
        
        # Convert windows to numpy arrays for consistent handling
        windows_np = [np.asarray(w) if not isinstance(w, (np.ndarray, cp.ndarray)) else w for w in windows]
        
        # Estimate memory requirements
        window_size = windows_np[0].size * (4 if windows_np[0].dtype == np.float32 else 8)  # in bytes
        total_window_mem = window_size * n_windows / (1024**3)  # in GB
        matrix_mem = (n_windows**2 * 4) / (1024**3)  # in GB (assuming float32)
        
        # Get GPU memory info
        if memory_limit is None:
            device = cp.cuda.Device()
            total_mem = device.mem_info[1] / (1024**3)  # Total memory in GB
            free_mem = device.mem_info[0] / (1024**3)   # Free memory in GB
            memory_limit = free_mem * 0.8  # Use 80% of free memory
            
        logger.info(f"GPU memory: Free {free_mem:.2f}GB / Total {total_mem:.2f}GB, Limit: {memory_limit:.2f}GB")
        logger.info(f"Required memory: {matrix_mem:.2f}GB for matrix + {total_window_mem:.2f}GB for windows")
        
        # Force cleanup before starting computation
        cp.get_default_memory_pool().free_all_blocks()
        
        # Check if we have enough memory for vectorized computation
        vectorized_memory_needed = matrix_mem + (total_window_mem * 4)  # Extra space for computation
        direct_compute_memory_needed = matrix_mem + (window_size * 2 / (1024**3))  # Just for two windows
        
        # Prepare result matrix
        dist_matrix = np.zeros((n_windows, n_windows), dtype=np.float32)
        
        # Prefer JAX for very fast computation if available and memory permits
        if JAX_AVAILABLE and vectorized_memory_needed < memory_limit:
            logger.info("Using JAX vectorized batch Wasserstein for distance matrix computation")
            try:
                # Determine optimal batch size based on memory
                max_bytes = memory_limit * (1024**3) * 0.5  # Use half for safety
                batch_size = max(1, int(max_bytes // (n_windows * 4)))
                batch_size = min(batch_size, n_windows)
                logger.info(f"JAX batch size: {batch_size}")
                
                # Process in batches
                for i in range(0, n_windows, batch_size):
                    end_i = min(i + batch_size, n_windows)
                    X = np.stack([windows_np[k].flatten() for k in range(i, end_i)])
                    Y = np.stack([w.flatten() for w in windows_np])
                    
                    # Compute batch
                    batch_dist = np.array(batched_wasserstein_jax(X, Y))
                    dist_matrix[i:end_i, :] = batch_dist
                    
                    # Update progress
                    logger.info(f"Computed JAX batch {i}:{end_i} of {n_windows}")
                
                return dist_matrix
                
            except Exception as e:
                logger.warning(f"JAX computation failed: {str(e)}. Falling back to CuPy.")
                cp.get_default_memory_pool().free_all_blocks()
        
        # Use CuPy vectorized approach with adaptive batching if memory permits
        if cp_available and vectorized_memory_needed < memory_limit * 2:  # More relaxed memory check for CuPy
            logger.info("Using CuPy vectorized batch Wasserstein for distance matrix computation")
            try:
                # Get optimal batch size based on available memory
                device = cp.cuda.Device()
                free_mem_bytes = device.mem_info[0]
                
                # Use more aggressive batching strategy
                # Estimate memory per window pair and determine batch size
                mem_per_window_pair = 4 * windows_np[0].size * 5  # 5x overhead for processing
                total_pairs = n_windows * n_windows
                
                # Limit memory usage to 30% of available to leave room for processing
                max_mem_usage = free_mem_bytes * 0.3
                batch_elements = max_mem_usage / mem_per_window_pair
                batch_ratio = np.sqrt(batch_elements / total_pairs)
                
                # Calculate batch sizes with minimum of 1
                batch_size_x = max(1, int(n_windows * batch_ratio))
                batch_size_y = max(1, int(n_windows * batch_ratio))
                
                # Cap batch sizes for very large datasets
                batch_size_x = min(batch_size_x, 500)
                batch_size_y = min(batch_size_y, 500)
                
                logger.info(f"Using adaptive batch sizes: {batch_size_x}x{batch_size_y}")
                
                # Prepare data in optimal format
                flattened_windows = [w.flatten() for w in windows_np]
                
                # Process in batches using optimized batched_wasserstein_cupy
                for i in range(0, n_windows, batch_size_x):
                    end_i = min(i + batch_size_x, n_windows)
                    X = np.stack(flattened_windows[i:end_i])
                    
                    for j in range(0, n_windows, batch_size_y):
                        end_j = min(j + batch_size_y, n_windows)
                        Y = np.stack(flattened_windows[j:end_j])
                        
                        # Compute distances
                        try:
                            batch_result = batched_wasserstein_cupy(X, Y)
                            dist_matrix[i:end_i, j:end_j] = cp.asnumpy(batch_result)
                        except Exception as e:
                            logger.warning(f"Batch computation failed: {str(e)}. Computing pair by pair.")
                            # Fall back to pair-by-pair calculation for this batch
                            for ii, idx_i in enumerate(range(i, end_i)):
                                for jj, idx_j in enumerate(range(j, end_j)):
                                    try:
                                        dist = distance_func(windows_np[idx_i], windows_np[idx_j])
                                        dist_matrix[idx_i, idx_j] = dist
                                        # Copy to symmetric position if not on diagonal
                                        if idx_i != idx_j:
                                            dist_matrix[idx_j, idx_i] = dist
                                    except Exception as e2:
                                        logger.error(f"Individual pair calculation failed: {str(e2)}")
                        
                        # Clear GPU memory between batches
                        cp.get_default_memory_pool().free_all_blocks()
                        
                    # Log progress
                    logger.info(f"Processed {end_i}/{n_windows} rows")
                
                # Make sure the matrix is symmetric (in case we missed some entries)
                for i in range(n_windows):
                    for j in range(i+1, n_windows):
                        if dist_matrix[i, j] == 0 and dist_matrix[j, i] != 0:
                            dist_matrix[i, j] = dist_matrix[j, i]
                        elif dist_matrix[j, i] == 0 and dist_matrix[i, j] != 0:
                            dist_matrix[j, i] = dist_matrix[i, j]
                
                return dist_matrix
                
            except Exception as e:
                logger.warning(f"CuPy vectorized approach failed: {str(e)}. Falling back to direct computation.")
                cp.get_default_memory_pool().free_all_blocks()
        
        # If above methods fail or memory is insufficient, use direct window-to-window computation
        # This is slower but requires minimal memory
        logger.info("Using direct GPU computation for distance matrix")
        
        # Start timing for progress updates
        start_time = time.time()
        last_update_time = start_time
        total_computations = (n_windows * (n_windows - 1)) // 2
        computations_done = 0
        
        # Compute pairwise distances directly
        for i in range(n_windows):
            for j in range(i, n_windows):
                try:
                    if i == j:
                        dist_matrix[i, j] = 0.0
                    else:
                        # Directly use the distance function
                        dist = distance_func(windows_np[i], windows_np[j])
                        dist_matrix[i, j] = dist
                        dist_matrix[j, i] = dist  # Symmetric
                        
                        # Update computation count
                        computations_done += 1
                        
                    # Report progress every 60 seconds
                    current_time = time.time()
                    if current_time - last_update_time >= 60:  # 60 seconds = 1 minute
                        progress_percent = (computations_done / total_computations) * 100
                        logger.info(f"Distance matrix progress: {computations_done}/{total_computations} ({progress_percent:.1f}%)")
                        last_update_time = current_time
                        
                except Exception as e:
                    logger.error(f"Error computing distance for pair ({i},{j}): {str(e)}")
                    # Use a default value
                    dist_matrix[i, j] = 1.0
                    dist_matrix[j, i] = 1.0
        
        return dist_matrix
        
    except Exception as e:
        logger.error(f"GPU computation failed completely: {str(e)}")
        logger.warning("Falling back to CPU computation")
        cp.get_default_memory_pool().free_all_blocks()
        return _compute_with_joblib(windows, distance_func, -1, n_windows, block_size)

def _compute_with_dask_gpu(windows, distance_func, n_jobs, n_windows, block_size=None, memory_limit=None):
    """
    Helper function for computing distance matrix using Dask-CUDA.
    Optimized for distributed GPU computation.
    """
    try:
        # Verify requirements
        if not DASK_CUDA_AVAILABLE:
            logger.warning("Dask-CUDA not available. Falling back to local GPU computation.")
            return _compute_with_gpu(windows, distance_func, n_windows, block_size, memory_limit)
        
        # Set block size for optimal chunking
        if block_size is None:
            # Dask-CUDA works better with larger chunks
            if n_windows > 50000:
                chunks_per_worker = 2
            elif n_windows > 20000:
                chunks_per_worker = 3
            elif n_windows > 5000:
                chunks_per_worker = 4
            else:
                chunks_per_worker = 6
                
            # Calculate block size
            block_size = max(100, n_windows // (n_jobs * chunks_per_worker))
        
        logger.info(f"Computing with Dask-CUDA using {n_jobs} workers, block size: {block_size}")
        
        # Create GPU cluster if needed
        try:
            from dask.distributed import get_client
            client = get_client()
            logger.info("Using existing Dask client")
        except (ValueError, ImportError):
            # Create a new local CUDA cluster
            from dask_cuda import LocalCUDACluster
            from dask.distributed import Client
            
            # Start cluster with one worker per GPU
            cluster = LocalCUDACluster(n_workers=n_jobs)
            client = Client(cluster)
            logger.info(f"Created new Dask-CUDA cluster with {n_jobs} workers")
        
        # Initialize result matrix
        dist_matrix = np.zeros((n_windows, n_windows), dtype=np.float32)
        
        # Create chunks that maximize GPU utilization and minimize communication
        chunks = []
        for i in range(0, n_windows, block_size):
            end_i = min(i + block_size, n_windows)
            # Process upper triangle
            for j in range(i, n_windows, block_size):
                end_j = min(j + block_size, n_windows)
                chunks.append((i, end_i, j, end_j))
        
        # Set up progress tracking
        start_time = time.time()
        last_update_time = start_time
        total_computations = n_windows * (n_windows - 1) // 2
        computations_done = 0
        active_futures = set()
        completed_futures = set()
        
        # Define chunk computation function
        def compute_chunk_gpu(start_i, end_i, start_j, end_j, chunk_windows):
            import cupy as cp
            
            # Transfer windows to GPU if they aren't already
            gpu_windows = []
            for w in chunk_windows:
                if not isinstance(w, cp.ndarray):
                    gpu_windows.append(cp.asarray(w))
                else:
                    gpu_windows.append(w)
            
            result = {}
            computation_count = 0
            # Only compute upper triangle
            for i_rel, i in enumerate(range(start_i, end_i)):
                window_i = gpu_windows[i_rel]
                for j in range(max(i+1, start_j), end_j):
                    j_idx = j - start_j if j >= start_j and j < end_j else None
                    window_j = gpu_windows[j_idx] if j_idx is not None else cp.asarray(windows[j])
                    
                    # Calculate distance
                    dist = float(distance_func(window_i, window_j))
                    result[(i, j)] = dist
                    computation_count += 1
            
            # Clean up GPU memory
            del gpu_windows
            cp.get_default_memory_pool().free_all_blocks()
            
            return result, computation_count
        
        # Submit chunks to workers
        futures = []
        for i, end_i, j, end_j in chunks:
            # Select windows for this chunk
            chunk_windows = windows[i:end_i]
            
            # Submit to worker
            future = client.submit(
                compute_chunk_gpu,
                i, end_i, j, end_j, chunk_windows,
                pure=False
            )
            futures.append(future)
            active_futures.add(future)
        
        # Main computation loop
        while len(completed_futures) < len(futures):
            try:
                # Check for newly completed futures
                new_completed = set()
                for future in active_futures:
                    if future.done():
                        new_completed.add(future)
                        completed_futures.add(future)
                active_futures -= new_completed

                # Process newly completed futures
                for future in new_completed:
                    try:
                        chunk_result, chunk_computations = future.result()
                        # Fill both upper and lower triangles
                        for (i, j), val in chunk_result.items():
                            dist_matrix[i, j] = val
                            dist_matrix[j, i] = val  # Symmetric
                        
                        # Update computation count
                        computations_done += chunk_computations
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                        continue

                # Check if a minute has passed since the last update
                current_time = time.time()
                if current_time - last_update_time >= 60:  # 60 seconds = 1 minute
                    progress_percent = (len(completed_futures) / len(futures)) * 100
                    comp_percent = (computations_done / total_computations) * 100 if total_computations > 0 else 0
                    logger.info(f"Distance matrix progress: {len(completed_futures)}/{len(futures)} chunks processed ({progress_percent:.1f}%), approx. {computations_done}/{total_computations} distances computed ({comp_percent:.1f}%)")
                    last_update_time = current_time

                # Sleep briefly to avoid busy waiting
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in main computation loop: {e}")
                raise

        return dist_matrix
        
    except Exception as e:
        logger.error(f"Error in Dask-CUDA computation: {e}")
        logger.warning("Falling back to local GPU computation")
        return _compute_with_gpu(windows, distance_func, n_windows, block_size, memory_limit)

def _compute_with_joblib(windows, distance_func, n_jobs, n_windows, block_size=None):
    """
    Helper function for computing distance matrix using joblib.
    Optimized for local CPU computation with improved chunking strategy.
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
            try:
                # Allocate memory only for the region we're computing
                chunk_height = end_i - start_i
                chunk_width = end_j - start_j
                chunk_result = np.zeros((chunk_height, chunk_width))
                
                # Only compute upper triangle within the chunk
                for i_local, i_global in enumerate(range(start_i, end_i)):
                    for j_local, j_global in enumerate(range(max(i_global+1, start_j), end_j)):
                        try:
                            dist = distance_func(windows[i_global], windows[j_global])
                            j_adjusted = j_global - start_j
                            if 0 <= j_adjusted < chunk_width:
                                chunk_result[i_local, j_adjusted] = dist
                        except Exception as e:
                            logger.error(f"Error computing distance for pair ({i_global}, {j_global}): {e}")
                            # Use a default high distance value
                            j_adjusted = j_global - start_j
                            if 0 <= j_adjusted < chunk_width:
                                chunk_result[i_local, j_adjusted] = 1.0
                
                return start_i, end_i, start_j, end_j, chunk_result
            except Exception as e:
                logger.error(f"Error in chunk computation ({start_i}:{end_i}, {start_j}:{end_j}): {e}")
                # Return empty result with correct dimensions
                return start_i, end_i, start_j, end_j, np.zeros((end_i - start_i, end_j - start_j))
        
        # Compute chunks in parallel
        logger.info(f"Computing distance matrix with joblib using {n_jobs} workers and {len(chunks)} chunks")
        
        # Set up variables for progress reporting
        start_time = time.time()
        last_update_time = start_time
        
        # Initialize counters for progress reporting
        total_computations = n_windows * (n_windows - 1) // 2
        remaining_chunks = len(chunks)
        processed_chunks = 0
        estimated_computations = 0
        
        # Use joblib's verbose parameter to avoid too many progress messages
        parallel = Parallel(n_jobs=n_jobs, verbose=0, prefer="processes")
        
        # Process chunks in batches to allow progress monitoring
        batch_size = min(max(10, len(chunks) // 10), 100)  # Process at most 100 chunks at a time
        
        # Initialize distance matrix
        dist_matrix = np.zeros((n_windows, n_windows))
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            # Process the current batch
            batch_results = parallel(
                delayed(compute_chunk)(start_i, end_i, start_j, end_j) 
                for start_i, end_i, start_j, end_j in batch_chunks
            )
            
            # Update processed chunks count
            processed_chunks += len(batch_chunks)
            
            # Fill matrix with computed distances from this batch
            comp_in_batch = 0
            for start_i, end_i, start_j, end_j, chunk_result in batch_results:
                chunk_height = end_i - start_i
                chunk_width = end_j - start_j
                
                # Ensure chunk_result has correct dimensions
                if chunk_result.shape != (chunk_height, chunk_width):
                    logger.warning(f"Chunk result shape mismatch. Expected: ({chunk_height}, {chunk_width}), Got: {chunk_result.shape}")
                    # Create correctly shaped array
                    fixed_result = np.zeros((chunk_height, chunk_width))
                    # Copy as much data as possible
                    min_height = min(chunk_height, chunk_result.shape[0])
                    min_width = min(chunk_width, chunk_result.shape[1])
                    fixed_result[:min_height, :min_width] = chunk_result[:min_height, :min_width]
                    chunk_result = fixed_result
                
                # Fill the matrix
                for i_local in range(chunk_height):
                    i_global = start_i + i_local
                    for j_local in range(chunk_width):
                        j_global = start_j + j_local
                        if j_global > i_global:  # Only upper triangle
                            val = chunk_result[i_local, j_local]
                            if val > 0:  # Only set non-zero values
                                dist_matrix[i_global, j_global] = val
                                dist_matrix[j_global, i_global] = val  # Symmetric
                                comp_in_batch += 1
            
            # Update total estimated computations
            estimated_computations += comp_in_batch
            
            # Check if a minute has passed since the last update
            current_time = time.time()
            if current_time - last_update_time >= 60:  # 60 seconds = 1 minute
                progress_percent = (processed_chunks / len(chunks)) * 100
                estimated_percent = (estimated_computations / total_computations) * 100 if total_computations > 0 else 0
                logger.info(f"Distance matrix progress: {processed_chunks}/{len(chunks)} chunks processed ({progress_percent:.1f}%), approx. {estimated_computations}/{total_computations} distances computed ({estimated_percent:.1f}%)")
                last_update_time = current_time
        
        return dist_matrix
    
    except ImportError:
        logger.warning("joblib not available. Using sequential computation.")
        # Fall back to sequential computation
        dist_matrix = np.zeros((n_windows, n_windows))
        
        # Initialize timing variables for progress reporting
        start_time = time.time()
        last_update_time = start_time
        total_computations = n_windows * (n_windows - 1) // 2
        computations_done = 0
        
        for i in range(n_windows):
            for j in range(i+1, n_windows):
                try:
                    dist = distance_func(windows[i], windows[j])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist  # Symmetric
                except Exception as e:
                    logger.error(f"Error computing distance for pair ({i}, {j}): {e}")
                    # Use a default high distance value
                    dist_matrix[i, j] = 1.0
                    dist_matrix[j, i] = 1.0
        
                # Update progress
                computations_done += 1
                
                # Check if a minute has passed since the last update
                current_time = time.time()
                if current_time - last_update_time >= 60:  # 60 seconds = 1 minute
                    progress_percent = (computations_done / total_computations) * 100
                    logger.info(f"Distance matrix progress: {computations_done}/{total_computations} distances computed ({progress_percent:.1f}%)")
                    last_update_time = current_time
            
        return dist_matrix

# Add this utility function to help debug types
def _trace_array_types(arrays, prefix="", level=logging.DEBUG):
    """Helper function to trace array types for debugging."""
    for i, arr in enumerate(arrays):
        if level == logging.DEBUG:
            logger.debug(f"{prefix} Array {i}: Type={type(arr)}, Shape={getattr(arr, 'shape', 'unknown')}")
        else:
            logger.info(f"{prefix} Array {i}: Type={type(arr)}, Shape={getattr(arr, 'shape', 'unknown')}")

# Move the SilentFilter class before configure_tda_logging function
class SilentFilter(logging.Filter):
    def filter(self, record):
        # Block ALL messages from filter_functions_v2
        # Additionally block the "Minimal logging mode enabled" message to avoid repetition
        message = record.getMessage()
        if "Minimal logging mode enabled" in message:
            return False
            
        # Filter out specific batch size messages that the user wants to hide
        patterns_to_filter = [
            "Using adaptive batches of size",
            "based on available GPU memory ratio",
            "Starting optimized batched Wasserstein distance computation",
            "Using adaptive batch sizes:",
            "Using JAX batch size:",
            "GPU memory: Free", 
            "Required memory:",
            "Using CuPy vectorized batch Wasserstein"
        ]
        
        for pattern in patterns_to_filter:
            if pattern in message:
                return False
                
        return True  # Allow other messages

def configure_tda_logging(level=None):
    """
    Configure logging globally for the TDA module to reduce noise and improve readability.
    
    Args:
        level: Optional logging level (e.g., logging.INFO, logging.DEBUG)
              If None, uses the CRONUS_LOG_LEVEL environment variable or defaults to INFO
    
    Returns:
        The configured root logger
    """
    # Check environment variable for log level
    if level is None:
        level_name = os.environ.get("CRONUS_LOG_LEVEL", "INFO")
        level = getattr(logging, level_name, logging.INFO)
    
    # Check if verbose mode is enabled (default to minimal logs)
    verbose_mode = os.environ.get("CRONUS_VERBOSE", "0").lower() in ["1", "true", "yes"]
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Create console handler with formatted output
    console = logging.StreamHandler()
    console.setLevel(level)
    
    # Use a simpler format for minimal logging
    if verbose_mode:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        
    console.setFormatter(formatter)
    root_logger.addHandler(console)
    
    # Check if LogFilter class exists in this module
    if 'LogFilter' in globals():
        # Apply custom filter to TDA module loggers
        tda_filter = LogFilter()
        for logger_name in ['distance_metrics_v2', 'mapper_core_v2']:
            module_logger = logging.getLogger(logger_name)
            module_logger.addFilter(tda_filter)
    
    # Special ultra-silent treatment for filter_functions_v2
    filter_logger = logging.getLogger('filter_functions_v2')
    filter_logger.addFilter(SilentFilter())
    
    # Set the level to ERROR to further reduce noise
    filter_logger.setLevel(logging.ERROR)
    
    # Apply SilentFilter to root logger to filter batch size and computation messages
    silent_filter = SilentFilter()
    root_logger.addFilter(silent_filter)
    
    # Create a filter to suppress the minimal logging message in worker processes
    class SuppressMinimalLoggingFilter(logging.Filter):
        def filter(self, record):
            if "Minimal logging mode enabled" in record.getMessage():
                # Only allow this message from the main logger, not from worker processes
                if record.name in ['__main__', 'root']:
                    return True
                return False
            return True
    
    # Apply this filter to all loggers to prevent duplicate messages
    for handler in root_logger.handlers:
        handler.addFilter(SuppressMinimalLoggingFilter())
    
    # Set higher levels for noisy libraries
    for lib_logger in ['joblib', 'matplotlib', 'numba', 'cuml', 'tensorflow', 'pandas', 
                      'sklearn', 'distributed', 'dask']:
        logging.getLogger(lib_logger).setLevel(logging.WARNING)
    
    # Add file handler if CRONUS_LOG_FILE is set
    log_file = os.environ.get("CRONUS_LOG_FILE")
    if log_file:
        try:
            # Create logs directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Add file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            # Also apply the minimal logging filter to the file handler
            file_handler.addFilter(SuppressMinimalLoggingFilter())
        except Exception as e:
            root_logger.warning(f"Failed to set up log file: {str(e)}")
    
    # Silence ALL common warnings
    warnings.filterwarnings("ignore")
    
    # Add initial message about logging mode, but only from the main process
    minimal_logging_recorded = False
    if not hasattr(configure_tda_logging, 'minimal_logging_recorded'):
        configure_tda_logging.minimal_logging_recorded = False
        minimal_logging_recorded = True
        configure_tda_logging.minimal_logging_recorded = True
    
    if verbose_mode:
        if minimal_logging_recorded:
            root_logger.info("Verbose logging mode enabled")
    else:
        if minimal_logging_recorded:
            root_logger.info("Minimal logging mode enabled (set CRONUS_VERBOSE=1 for detailed logs)")
    
    return root_logger

# Auto-configure logging when imported
configure_tda_logging()

def batched_wasserstein_jax(X, Y):
    """
    Compute pairwise Wasserstein distances between batches X and Y using JAX vectorization.
    X: shape (N, window_size)
    Y: shape (M, window_size)
    Returns: shape (N, M) distance matrix
    """
    if not (JAX_AVAILABLE):
        raise RuntimeError("JAX is not available for batched Wasserstein computation.")
    import jax
    import jax.numpy as jnp
    
    def single_wasserstein(x, y):
        # Normalize
        x_min, x_max = jnp.min(x), jnp.max(x)
        y_min, y_max = jnp.min(y), jnp.max(y)
        global_min = jnp.minimum(x_min, y_min)
        global_max = jnp.maximum(x_max, y_max)
        def norm(arr):
            return jnp.where(global_max - global_min > 1e-10, (arr - global_min) / (global_max - global_min), arr)
        x_norm = norm(x)
        y_norm = norm(y)
        # Sort for 1D Wasserstein
        x_sorted = jnp.sort(x_norm)
        y_sorted = jnp.sort(y_norm)
        n = jnp.minimum(x_sorted.shape[0], y_sorted.shape[0])
        cdf_points = jnp.linspace(0, 1, n)
        xq = jnp.interp(cdf_points, jnp.linspace(0, 1, x_sorted.shape[0]), x_sorted)
        yq = jnp.interp(cdf_points, jnp.linspace(0, 1, y_sorted.shape[0]), y_sorted)
        w_dist = jnp.mean(jnp.abs(xq - yq) ** 2) ** 0.5
        # Scale back
        w_dist = jnp.where(global_max - global_min > 1e-10, w_dist * (global_max - global_min), w_dist)
        return w_dist
    # Vectorize over X and Y
    vmap_x = jax.vmap(lambda x: jax.vmap(lambda y: single_wasserstein(x, y))(Y))(X)
    return vmap_x

def batched_wasserstein_cupy(X, Y):
    """
    Compute pairwise Wasserstein distances between batches X and Y using CuPy vectorization.
    Optimized for maximum GPU utilization and performance.
    
    Args:
        X: shape (N, window_size) - flattened windows for first batch
        Y: shape (M, window_size) - flattened windows for second batch
        
    Returns:
        Distance matrix of shape (N, M)
    """
    if not cp_available:
        raise RuntimeError("CuPy is not available for batched Wasserstein computation.")
    
    # For progress tracking
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    # Log the start of computation
    logger.info(f"Starting optimized batched Wasserstein distance computation ({n_x}x{n_y} matrix)")
    
    try:
        # Transfer data to GPU and ensure they're float32 for better performance
        X_gpu = cp.asarray(X, dtype=cp.float32)
        Y_gpu = cp.asarray(Y, dtype=cp.float32)
        
        # Free memory before intensive computation
        cp.get_default_memory_pool().free_all_blocks()
        
        # Get available GPU memory and adjust batch size accordingly
        total_memory = cp.cuda.Device().mem_info[1]
        available_memory = cp.cuda.Device().mem_info[0]
        memory_ratio = available_memory / total_memory
        
        # Adaptive batch sizing based on available memory and problem size
        max_elements = int(0.3 * available_memory / 4)  # Assuming float32 (4 bytes)
        batch_x = min(n_x, max(1, int(np.sqrt(max_elements / X.shape[1]))))
        batch_y = min(n_y, max(1, int(np.sqrt(max_elements / Y.shape[1]))))
        
        logger.info(f"Using adaptive batches of size {batch_x}x{batch_y} based on available GPU memory ratio {memory_ratio:.2f}")
        
        # Initialize result matrix
        result = cp.zeros((n_x, n_y), dtype=cp.float32)
        
        # Process in optimized batches
        for i in range(0, n_x, batch_x):
            end_i = min(i + batch_x, n_x)
            X_batch = X_gpu[i:end_i]
            
            for j in range(0, n_y, batch_y):
                end_j = min(j + batch_y, n_y)
                Y_batch = Y_gpu[j:end_j]
                
                # Compute batch distances using vectorized operations
                
                # 1. Prepare for broadcasting
                X_reshaped = X_batch.reshape(end_i - i, 1, X.shape[1])
                Y_reshaped = Y_batch.reshape(1, end_j - j, Y.shape[1])
                
                # 2. Calculate global min/max for normalization
                x_min = cp.min(X_reshaped, axis=2, keepdims=True)
                x_max = cp.max(X_reshaped, axis=2, keepdims=True)
                y_min = cp.min(Y_reshaped, axis=2, keepdims=True)
                y_max = cp.max(Y_reshaped, axis=2, keepdims=True)
                
                global_min = cp.minimum(x_min, y_min)
                global_max = cp.maximum(x_max, y_max)
                range_factor = global_max - global_min
                
                # Add epsilon to avoid division by zero
                range_factor = cp.maximum(range_factor, 1e-10)
                
                # 3. Normalize both arrays
                X_norm = (X_reshaped - global_min) / range_factor
                Y_norm = (Y_reshaped - global_min) / range_factor
                
                # 4. Sort arrays along last dimension for Wasserstein distance
                X_sorted = cp.sort(X_norm, axis=2)
                Y_sorted = cp.sort(Y_norm, axis=2)
                
                # 5. Calculate distances using L2 norm of sorted arrays
                # This is a vectorized implementation of the 1D Wasserstein distance
                n = min(X.shape[1], Y.shape[1])
                
                # Ensure uniform CDF by taking equally spaced points
                if X.shape[1] != Y.shape[1]:
                    # Interpolate to same number of points
                    cdf_points = cp.linspace(0, 1, n, dtype=cp.float32)
                    
                    # Create temporary arrays for interpolation results
                    X_interp = cp.zeros((end_i - i, end_j - j, n), dtype=cp.float32)
                    Y_interp = cp.zeros((end_i - i, end_j - j, n), dtype=cp.float32)
                    
                    # Vectorized interpolation
                    x_indices = cp.linspace(0, 1, X.shape[1])
                    y_indices = cp.linspace(0, 1, Y.shape[1])
                    
                    # Use GPU-accelerated interpolation
                    for idx_i in range(end_i - i):
                        for idx_j in range(end_j - j):
                            X_interp[idx_i, idx_j] = cp.interp(cdf_points, x_indices, X_sorted[idx_i, idx_j])
                            Y_interp[idx_i, idx_j] = cp.interp(cdf_points, y_indices, Y_sorted[idx_i, idx_j])
                            
                    # Calculate L2 distance between interpolated points
                    diffs = (X_interp - Y_interp) ** 2
                    distances = cp.sqrt(cp.mean(diffs, axis=2))
                else:
                    # If arrays have the same length, we can compute directly
                    diffs = (X_sorted - Y_sorted) ** 2
                    distances = cp.sqrt(cp.mean(diffs, axis=2))
                
                # 6. Scale distances back by the range
                # Extract the scale factor (drop the last dimension)
                scales = cp.squeeze(range_factor, axis=2)
                distances = distances * scales
                
                # 7. Store in result matrix
                result[i:end_i, j:end_j] = distances
                
                # 8. Free memory between batches
                del X_norm, Y_norm, X_sorted, Y_sorted, diffs, distances
                if X.shape[1] != Y.shape[1]:
                    del X_interp, Y_interp
                cp.get_default_memory_pool().free_all_blocks()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in optimized batched Wasserstein computation: {str(e)}")
        # Try to free GPU memory in case of error
        cp.get_default_memory_pool().free_all_blocks()
        raise

def normalize_and_sort(arr, g_mins, g_maxs, reshape=False):
    """Helper function to normalize and sort arrays for Wasserstein distance"""
    # Normalize the array
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    arr_norm = (arr - g_mins) / (g_maxs - g_mins + epsilon)
    
    # Sort each row
    if reshape:
        # Reshape to 2D if needed
        arr_norm = arr_norm.reshape(arr_norm.shape[0], -1)
    
    # Sort along rows
    arr_sorted = cp.sort(arr_norm, axis=1)
    
    return arr_sorted

def _compute_with_dask(windows, distance_func, n_jobs, n_windows, block_size=None):
    """
    Helper function for computing distance matrix using Dask.
    Optimized for distributed CPU computation with improved chunking strategy.
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
            # Create a new client
            from dask.distributed import Client, LocalCluster
            cluster = LocalCluster(n_workers=n_jobs)
            client = Client(cluster)
            logger.info(f"Created new Dask cluster with {n_jobs} workers")
        
        # Initialize distance matrix
        dist_matrix = np.zeros((n_windows, n_windows))
        
        # Advanced chunking strategy for Dask:
        # 1. Divide the matrix into square tiles for better data locality
        # 2. Process the upper triangle of the matrix only
        
        # Create chunks that maximize data locality and minimize communication
        n_chunks = max(1, min(n_jobs * chunks_per_worker, n_windows // block_size))
        chunk_size = max(block_size, n_windows // n_chunks)
        logger.info(f"Computing with {n_chunks} chunks of size ~{chunk_size}")
        
        # Set up progress tracking
        start_time = time.time()
        last_update_time = start_time
        total_computations = n_windows * (n_windows - 1) // 2
        computations_done = 0
        active_futures = set()
        completed_futures = set()
        
        # Define chunk computation function for square regions
        def compute_chunk(start_i, end_i, start_j, end_j, chunk_windows):
            chunk_result = {}
            computations = 0
            # Only compute upper triangle
            for i_rel, i in enumerate(range(start_i, end_i)):
                for j in range(max(i+1, start_j), end_j):
                    dist = distance_func(chunk_windows[i_rel] if i >= start_i and i < end_i else windows[i], 
                                         windows[j])
                    chunk_result[(i, j)] = dist
                    computations += 1
            return chunk_result, computations
        
        # Submit square tile chunks for computation
        futures = []
        for i in range(0, n_windows, chunk_size):
            end_i = min(i + chunk_size, n_windows)
            # Only send necessary windows to each worker to reduce data transfer
            chunk_windows = windows[i:end_i]
            
            # Process tiles in the upper triangle
            for j in range(i, n_windows, chunk_size):
                end_j = min(j + chunk_size, n_windows)
                future = client.submit(
                    compute_chunk,
                    i, end_i, j, end_j, chunk_windows,
                    pure=False  # Ensure recomputation if needed
                )
                futures.append(future)
                active_futures.add(future)
        
        # Main computation loop
        while len(completed_futures) < len(futures):
            try:
                # Check for newly completed futures
                new_completed = set()
                for future in active_futures:
                    if future.done():
                        new_completed.add(future)
                        completed_futures.add(future)
                active_futures -= new_completed

                # Process newly completed futures
                for future in new_completed:
                    try:
                        chunk_result, chunk_computations = future.result()
                        # Fill both upper and lower triangles
                        for (i, j), val in chunk_result.items():
                            dist_matrix[i, j] = val
                            dist_matrix[j, i] = val  # Symmetric
                        
                        # Update computation count
                        computations_done += chunk_computations
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                        continue

                # Check if a minute has passed since the last update
                current_time = time.time()
                if current_time - last_update_time >= 60:  # 60 seconds = 1 minute
                    progress_percent = (len(completed_futures) / len(futures)) * 100
                    comp_percent = (computations_done / total_computations) * 100 if total_computations > 0 else 0
                    logger.info(f"Distance matrix progress: {len(completed_futures)}/{len(futures)} chunks processed ({progress_percent:.1f}%), approx. {computations_done}/{total_computations} distances computed ({comp_percent:.1f}%)")
                    last_update_time = current_time

                # Sleep briefly to avoid busy waiting
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in main computation loop: {e}")
                raise

        return dist_matrix
        
    except Exception as e:
        logger.error(f"Error in Dask computation: {e}")
        logger.warning("Falling back to joblib")
        return _compute_with_joblib(windows, distance_func, n_jobs, n_windows, block_size)

