import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Callable, Union, Optional, Any
from sklearn.cluster import DBSCAN, AgglomerativeClustering, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import logging
import os
import kmapper as km
import time
from datetime import datetime

# Import from local modules
try:
    from .filter_functions_v2 import FinancialLensFactory
    from .distance_metrics_v2 import create_financial_distance_function, compute_distance_matrix
except ImportError:
    # For standalone/development use
    from filter_functions_v2 import FinancialLensFactory
    from distance_metrics_v2 import create_financial_distance_function, compute_distance_matrix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
                 enable_adaptive_clustering: bool = True):
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
            'enable_adaptive_clustering': self.enable_adaptive_clustering
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
        Optimize parameters to improve direct mapping percentage.

        Args:
            n_windows: Number of windows in dataset
            lens_density: Optional lens space density estimate
        """
        logger.info(f"Optimizing mapper parameters for {n_windows} windows")

        # For large datasets, we need more intervals and higher overlap
        if n_windows > 5000:
            self.n_intervals = max(self.n_intervals, 15)
            self.overlap_percentage = min(
                0.7, max(self.overlap_percentage, 0.5))
            logger.info(
                f"Large dataset: increased intervals to {self.n_intervals}, overlap to {self.overlap_percentage:.2f}")

        elif n_windows > 1000:
            self.n_intervals = max(self.n_intervals, 12)
            self.overlap_percentage = min(
                0.65, max(self.overlap_percentage, 0.45))
            logger.info(
                f"Medium dataset: set intervals to {self.n_intervals}, overlap to {self.overlap_percentage:.2f}")

        # Adjust min_cluster_size based on dataset size
        original_min_cluster_size = self.min_cluster_size

        if n_windows > 10000:
            # For very large datasets, increase min_cluster_size to avoid too
            # many small clusters
            self.min_cluster_size = max(self.min_cluster_size, 8)
        elif n_windows > 2000:
            # For large datasets
            self.min_cluster_size = max(self.min_cluster_size, 5)
        elif n_windows < 100:
            # For very small datasets, reduce min_cluster_size
            self.min_cluster_size = max(2, min(self.min_cluster_size, 3))

        if self.min_cluster_size != original_min_cluster_size:
            logger.info(
                f"Adjusted min_cluster_size from {original_min_cluster_size} to {self.min_cluster_size}")

        # Adjust HDBSCAN parameters for better clustering
        if n_windows > 1000:
            # For larger datasets, slightly increase epsilon to capture more
            # structure
            self.clustering_parameters['cluster_selection_epsilon'] = min(0.1,
                self.clustering_parameters.get('cluster_selection_epsilon', 0.05) * 1.5)

        # If we know the lens density, use it to tune parameters
        if lens_density is not None:
            if lens_density < 0.1:  # Sparse lens space
                # Increase overlap to capture more connections
                self.overlap_percentage = min(
    0.75, self.overlap_percentage * 1.3)
                logger.info(
                    f"Sparse lens space: increased overlap to {self.overlap_percentage:.2f}")
            elif lens_density > 0.5:  # Dense lens space
                # Slightly reduce overlap to avoid too many connections
                self.overlap_percentage = max(
    0.4, self.overlap_percentage * 0.9)
                logger.info(
                    f"Dense lens space: adjusted overlap to {self.overlap_percentage:.2f}")

        return self


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
            stride = max(1, n_samples // 5000)
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
        
        if algo == 'hdbscan':
            # Configure HDBSCAN - better for financial data with varying
            # densities
            if 'min_cluster_size' not in params:
                params['min_cluster_size'] = min_size
            if 'min_samples' not in params:
                # MODIFICATION: Reduce min_samples to be more sensitive to
                # smaller clusters
                params['min_samples'] = max(
    1, min_size // 4)  # Was min_size // 3

            # ENHANCEMENT: Improved HDBSCAN parameters for better regime detection
            # Use excess of mass for better financial clusters
            if 'cluster_selection_method' not in params:
                params['cluster_selection_method'] = 'eom'

            # Allow more flexibility in cluster selection for dense lens spaces
            if 'cluster_selection_epsilon' not in params:
                if hasattr(
    self,
    'lens') and hasattr(
        self,
         '_estimate_lens_density'):
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
            
            logger.info(f"Using HDBSCAN with min_cluster_size={params['min_cluster_size']}, "
                     f"min_samples={params['min_samples']}, "
                     f"method={params.get('cluster_selection_method', 'eom')}, "
                     f"epsilon={params.get('cluster_selection_epsilon', 0.1)}")
            
            return HDBSCAN(**params)
        
        elif algo == 'dbscan':
            # Configure DBSCAN
            if 'min_samples' not in params:
                params['min_samples'] = min_size
            if 'eps' not in params:
                params['eps'] = 0.15  # Default eps value
            
            logger.info(
                f"Using DBSCAN with min_samples={params['min_samples']}, eps={params['eps']}")
            
            return DBSCAN(**params)
        
        elif algo == 'agglomerative':
            # Configure Agglomerative Clustering
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
            return HDBSCAN(
    min_cluster_size=min_size,
    min_samples=max(
        1,
         min_size // 4))

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
        for i in range(len(lens)):
            distances_i = distance_matrix[i]
            # Get 6 closest (including self)
            indices = np.argsort(distances_i)[:6]
            # Skip self (which is at index 0)
            all_nearest_neighbors[i] = [
                idx for idx in indices[1:6] if idx != i]

        # Perform clustering on each hypercube
        nodes = {}
        links = {}
        all_points = set(range(len(lens)))
        mapped_points = set()

        # Keep track of clustering info for debugging
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
                        submatrix = np.zeros((n_points, n_points))

                        for i in range(n_points):
                            for j in range(i + 1, n_points):
                                submatrix[i,
    j] = distance_matrix[indices[i],
     indices[j]]
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

        # Enhanced link creation: Find links between nodes (if they share points)
        # 1. First pass: create links based on shared points (standard mapper
        # approach)
        for node_id1 in nodes.keys():
            for node_id2 in nodes.keys():
                if node_id1 < node_id2:  # Avoid duplicates and self-links
                    # Find shared points
                    points1 = set(nodes[node_id1])
                    points2 = set(nodes[node_id2])
                    shared_points = points1.intersection(points2)
                    
                    # Only create link if enough points are shared
                    if len(shared_points) > 0:
                        # Calculate relative overlap for filtering
                        overlap_ratio = len(shared_points) / \
                                            min(len(points1), len(points2))
                        
                        # Lower threshold to capture more connections
                        if overlap_ratio > 0.05:  # At least 5% overlap
                            if node_id1 not in links:
                                links[node_id1] = []
                            links[node_id1].append(node_id2)
                            
                            # Add reverse link for consistency
                            if node_id2 not in links:
                                links[node_id2] = []
                            links[node_id2].append(node_id1)
        
        # 2. Second pass: add links based on distance between node centers in lens space
        # This helps connect disconnected components and improves the
        # topological representation
        if len(nodes) > 1:
            try:
                # Calculate node centers in lens space
                node_centers = {}
                for node_id, point_indices in nodes.items():
                    if point_indices:
                        node_centers[node_id] = np.mean(
                            lens[point_indices], axis=0)

                # Find nearest neighbors for each node
                for node_id1, center1 in node_centers.items():
                    # Only add links for nodes with few connections
                    if node_id1 in links and len(links[node_id1]) >= 3:
                        continue

                    # Calculate distances to all other centers
                    distances = {}
                    for node_id2, center2 in node_centers.items():
                        if node_id1 != node_id2:
                            distances[node_id2] = np.linalg.norm(
                                center1 - center2)

                    # Sort by distance
                    sorted_nodes = sorted(
    distances.keys(), key=lambda k: distances[k])

                    # Add links to closest nodes (if not already linked)
                    max_neighbors = 2  # Add at most 2 nearest neighbor links
                    added = 0

                    for node_id2 in sorted_nodes:
                        if added >= max_neighbors:
                            break

                        # Skip if already linked
                        if node_id1 in links and node_id2 in links[node_id1]:
                            continue
                        if node_id2 in links and node_id1 in links[node_id2]:
                            continue

                        # Add link based on lens space proximity
                        if node_id1 not in links:
                            links[node_id1] = []
                        links[node_id1].append(node_id2)

                        if node_id2 not in links:
                            links[node_id2] = []
                        links[node_id2].append(node_id1)

                        added += 1
                        logger.debug(
                            f"Added proximity link between {node_id1} and {node_id2}")
            except Exception as e:
                logger.warning(
                    f"Error creating proximity links: {str(e)[:100]}...")

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
        
        # If hierarchical regimes are enabled, identify sub-regimes
        if self.config.hierarchical_regimes:
            hierarchical_regimes = self._identify_hierarchical_regimes(
                G, communities)
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
        for regime_id, community in enumerate(communities):
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
        Detect regimes (communities) optimized for financial data with HDBSCAN clustering.
        Enhanced to identify distinct market regimes more reliably with focus on topological distinctions.
        
        Args:
            G: NetworkX graph of mapper output
            
        Returns:
            List of communities (each a list of node IDs)
        """
        n_nodes = len(G.nodes())
        if n_nodes == 0:
            logger.warning("Empty graph - no regimes to detect")
            return []
            
        logger.info(f"Detecting regimes from graph with {n_nodes} nodes and {G.number_of_edges()} edges")
        
        # Create backup copy of graph for alternative methods
        G_backup = G.copy()
        
        # Extract topological features to enhance community detection
        # These features help distinguish nodes based on their position in the topology
        node_topo_features = self._extract_topological_features(G)
        
        # Calculate the topological diversity target
        # If we have complex structure, we should aim for more regimes
        complexity_metrics = self._calculate_topological_complexity(G)
        target_regimes = complexity_metrics['target_regimes']
        
        logger.info(f"Graph complexity metrics - Target regimes: {target_regimes}, " 
                   f"Average clustering: {complexity_metrics['clustering']:.3f}, "
                   f"Heterogeneity: {complexity_metrics['heterogeneity']:.3f}")
        
        # ENHANCED: Force a minimum of 3 regimes unless the graph is trivially small
        if target_regimes < 3 and n_nodes > 10:
            target_regimes = 3
            logger.info(f"Adjusted target regimes to minimum of {target_regimes}")
        
        # Try multiple community detection algorithms and ensemble the results
        communities_candidates = []
        
        # 1. Try Louvain method first (best for financial data)
        try:
            import community as community_louvain
            
            # Use Louvain method for community detection
            partition = community_louvain.best_partition(G)
            
            # Convert partition dictionary to list of communities
            community_to_nodes = {}
            for node, community_id in partition.items():
                if community_id not in community_to_nodes:
                    community_to_nodes[community_id] = []
                community_to_nodes[community_id].append(node)
            
            louvain_communities = list(community_to_nodes.values())
            logger.info(f"Louvain method found {len(louvain_communities)} communities")
            communities_candidates.append(("louvain", louvain_communities))
            
        except ImportError:
            logger.warning("Louvain community detection not available")
        
        # 2. Try Leiden algorithm if available (often better than Louvain)
        try:
            import leidenalg
            import igraph as ig
            
            # Convert NetworkX graph to igraph
            edges = list(G.edges())
            g_ig = ig.Graph(n=n_nodes, edges=edges, directed=False)
            
            # Map node IDs to integers
            node_map = {node: i for i, node in enumerate(G.nodes())}
            inv_node_map = {i: node for node, i in node_map.items()}
            
            # Apply Leiden algorithm
            partition = leidenalg.find_partition(
                g_ig, 
                leidenalg.ModularityVertexPartition,
                n_iterations=10
            )
            
            # Convert partition to list of communities
            leiden_communities = []
            for cluster in partition:
                community = [inv_node_map[idx] for idx in cluster]
                leiden_communities.append(community)
            
            logger.info(f"Leiden algorithm found {len(leiden_communities)} communities")
            communities_candidates.append(("leiden", leiden_communities))
            
        except ImportError:
            logger.debug("Leiden algorithm not available")
        
        # 3. Try centrality-based clustering
        try:
            # Use eigenvector centrality to identify important nodes
            centrality = nx.eigenvector_centrality_numpy(G)
            
            # ENHANCED: Use more quantiles for finer-grained division
            # Create node subsets based on centrality quantiles 
            quantiles = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]  # More fine-grained
            centrality_values = sorted(centrality.values())
            thresholds = [centrality_values[int(q * len(centrality_values))] for q in quantiles]
            
            # Group nodes by centrality
            centrality_communities = [[] for _ in range(len(quantiles) + 1)]
            for node, value in centrality.items():
                # Find which quantile the node belongs to
                quantile_idx = 0
                while quantile_idx < len(thresholds) and value > thresholds[quantile_idx]:
                    quantile_idx += 1
                centrality_communities[quantile_idx].append(node)
            
            # Remove empty communities
            centrality_communities = [c for c in centrality_communities if c]
            logger.info(f"Centrality-based approach found {len(centrality_communities)} communities")
            communities_candidates.append(("centrality", centrality_communities))
        except Exception as e:
            logger.debug(f"Centrality-based clustering failed: {str(e)[:100]}...")
        
        # 4. Try NetworkX's community detection methods
        try:
            from networkx.algorithms import community
            
            # For large graphs, use faster greedy modularity algorithm
            if n_nodes > 100:
                nx_communities = list(community.greedy_modularity_communities(G))
                method = "greedy modularity"
            else:
                # For smaller graphs, use more accurate Girvan-Newman algorithm
                # but limit to 4 communities for performance reasons
                comp = community.girvan_newman(G)
                # Get the first few iterations of the algorithm
                limited_comp = []
                for i, c in enumerate(comp):
                    limited_comp = list(c)
                    if i >= 3:  # Stop after finding 4 levels of communities
                        break
                
                # Choose the best partition level based on modularity
                best_modularity = -1
                best_partition = None
                for partition in limited_comp:
                    mod = community.modularity(G, partition)
                    if mod > best_modularity:
                        best_modularity = mod
                        best_partition = partition
                
                nx_communities = list(best_partition) if best_partition else []
                method = f"Girvan-Newman (modularity: {best_modularity:.3f})"
            
            logger.info(f"{method} found {len(nx_communities)} communities")
            communities_candidates.append(("nx_community", nx_communities))
            
        except Exception as e:
            logger.debug(f"NetworkX community detection failed: {str(e)[:100]}...")
        
        # 5. Try spectral clustering 
        try:
            from sklearn.cluster import SpectralClustering
                
            # Create adjacency matrix
            adj_matrix = nx.to_numpy_array(G)
                
            # ENHANCED: Force a higher n_clusters value
            n_clusters = max(3, min(8, target_regimes))
                
            # Apply spectral clustering
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
                
            spectral_communities = list(spectral_communities.values())
            logger.info(f"Spectral clustering found {len(spectral_communities)} communities")
            communities_candidates.append(("spectral", spectral_communities))
        except Exception as e:
            logger.debug(f"Spectral clustering failed: {str(e)[:100]}...")
        
        # 6. Try topological clustering using extracted features
        if node_topo_features:
            try:
                from sklearn.cluster import KMeans
                
                # Extract feature matrix 
                feature_matrix = np.array([node_topo_features[n] for n in G.nodes()])
                
                # Normalize features
                feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-6)
                
                # Replace NaN values with 0
                feature_matrix = np.nan_to_num(feature_matrix)
                
                # ENHANCED: Force a higher n_clusters value
                n_clusters = max(3, min(10, target_regimes))
                
                # Apply KMeans clustering with multiple random starts
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                topo_labels = kmeans.fit_predict(feature_matrix)
                
                # Convert to communities format
                node_list = list(G.nodes())
                topo_communities = {}
                for i, label in enumerate(topo_labels):
                    if label not in topo_communities:
                        topo_communities[label] = []
                    topo_communities[label].append(node_list[i])
                
                topo_communities = list(topo_communities.values())
                logger.info(f"Topological clustering found {len(topo_communities)} communities")
                communities_candidates.append(("topological", topo_communities))
            except Exception as e:
                logger.debug(f"Topological clustering failed: {str(e)[:100]}...")
        
        # 7. NEW: Try temporal-based clustering
        try:
            # Check if we have temporal information
            has_temporal = 'mid_idx' in G.nodes[list(G.nodes())[0]]
            
            if has_temporal:
                # Sort nodes by temporal mid-point
                nodes_by_time = sorted(G.nodes(), key=lambda n: G.nodes[n].get('mid_idx', 0))
                
                # Divide into temporal segments (at least 3)
                n_segments = max(3, min(8, len(nodes_by_time) // 5))
                segment_size = len(nodes_by_time) // n_segments
                
                temporal_communities = []
                for i in range(n_segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(nodes_by_time)
                    temporal_communities.append(nodes_by_time[start_idx:end_idx])
                
                logger.info(f"Temporal-based approach found {len(temporal_communities)} communities")
                communities_candidates.append(("temporal", temporal_communities))
        except Exception as e:
            logger.debug(f"Temporal-based clustering failed: {str(e)[:100]}...")
            
        # 8. Use lens space clustering as additional method
        if hasattr(self, 'lens') and hasattr(self, 'graph') and 'nodes' in self.graph:
            try:
                from sklearn.cluster import KMeans
                
                # Extract points from each node
                all_mapped_points = set()
                node_to_points = {}
                for node_id, points in self.graph['nodes'].items():
                    node_to_points[node_id] = points
                    all_mapped_points.update(points)
                
                # Create mapping from points to nodes
                point_to_node = {}
                for node_id, points in node_to_points.items():
                    for point in points:
                        if point not in point_to_node:
                            point_to_node[point] = []
                        point_to_node[point].append(node_id)
                
                # Get lens values for unique points
                unique_points = sorted(list(all_mapped_points))
                lens_values = self.lens[unique_points]
                
                # ENHANCED: Force more lens clusters
                lens_target_regimes = max(4, min(8, int(0.5 / (0.01 + np.mean(np.std(lens_values, axis=0))))))
                
                # Apply KMeans to lens values
                kmeans = KMeans(n_clusters=lens_target_regimes, n_init=10, random_state=42)
                lens_labels = kmeans.fit_predict(lens_values)
                
                # Map lens clusters back to graph nodes
                lens_communities = [[] for _ in range(lens_target_regimes)]
                for i, point in enumerate(unique_points):
                    if point in point_to_node:
                        for node_id in point_to_node[point]:
                            lens_label = lens_labels[i]
                            lens_communities[lens_label].append(node_id)
                
                # Remove duplicates within communities
                lens_communities = [list(set(community)) for community in lens_communities]
                
                # Remove empty communities
                lens_communities = [c for c in lens_communities if c]
                
                logger.info(f"Lens-based clustering found {len(lens_communities)} communities")
                communities_candidates.append(("lens", lens_communities))
            except Exception as e:
                logger.debug(f"Lens-based clustering failed: {str(e)[:100]}...")
                
        # NEW: Add a method that ensures diversity by splitting largest communities
        try:
            # Find the most diverse (highest number) results from above methods
            most_diverse_method = None
            most_communities = 0
            
            for method, communities in communities_candidates:
                if len(communities) > most_communities:
                    most_communities = len(communities)
                    most_diverse_method = (method, communities)
            
            if most_diverse_method and most_communities >= 3:
                # Use the most diverse method as a starting point
                method_name, diverse_communities = most_diverse_method
                logger.info(f"Using {method_name} with {len(diverse_communities)} communities as diversity baseline")
                
                # Add this explicitly as a candidate
                communities_candidates.append(("diverse_" + method_name, diverse_communities))
        except Exception as e:
            logger.debug(f"Diversity enhancement failed: {str(e)[:100]}...")
        
        # 9. Special handling for cases where no good community structure is found
        if not communities_candidates or all(len(c[1]) <= 1 for c in communities_candidates):
            logger.warning("No clear community structure found - attempting direct clustering of nodes")
            
            try:
                # Extract features from nodes for clustering
                node_features = {}
                
                # Get size feature (number of points in node)
                for node_id in G.nodes():
                    if 'size' in G.nodes[node_id]:
                        node_features[node_id] = [G.nodes[node_id]['size']]
                    else:
                        # Default to size 1 if not specified
                        node_features[node_id] = [1]
                
                # Add temporal features if available
                for node_id in G.nodes():
                    features = node_features[node_id]
                    if 'mid_idx' in G.nodes[node_id]:
                        features.append(G.nodes[node_id]['mid_idx'])
                    if 'time_span' in G.nodes[node_id]:
                        features.append(G.nodes[node_id]['time_span'])
                
                # Normalize features
                feature_matrix = np.array(list(node_features.values()))
                if feature_matrix.shape[1] > 1:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    feature_matrix = scaler.fit_transform(feature_matrix)
                
                # Apply K-means clustering to node features
                from sklearn.cluster import KMeans
                
                # ENHANCED: Force a minimum number of clusters
                n_clusters = max(3, min(8, target_regimes))
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(feature_matrix)
                
                # Convert to communities format
                node_list = list(node_features.keys())
                kmeans_communities = {}
                for i, label in enumerate(labels):
                    if label not in kmeans_communities:
                        kmeans_communities[label] = []
                    kmeans_communities[label].append(node_list[i])
                
                kmeans_communities = list(kmeans_communities.values())
                logger.info(f"Node feature clustering found {len(kmeans_communities)} communities")
                communities_candidates.append(("node_features", kmeans_communities))
                
            except Exception as e:
                logger.error(f"Node feature clustering failed: {str(e)[:100]}...")
        
        # NEW: Set a minimum desired number of communities for the ensemble
        min_desired_communities = 3
        
        # Combine all candidate methods to create diverse regime detection
        # Instead of just picking one method, we'll create an ensemble approach
        if len(communities_candidates) > 1:
            # First, evaluate all candidates
            candidate_scores = []
            
            for method, communities in communities_candidates:
                # Skip if empty or just one community
                if not communities or len(communities) <= 1:
                    continue
                
                # Calculate basic quality metrics
                sizes = [len(c) for c in communities]
                coverage = sum(sizes) / n_nodes
                evenness = 1 - (np.std(sizes) / np.mean(sizes)) if np.mean(sizes) > 0 else 0
                
                # Calculate diversity score - higher when number of communities closer to target
                n_communities = len(communities)
                diversity_score = 1.0 - abs(n_communities - target_regimes) / (target_regimes + n_communities)
                
                # ENHANCED: Strongly reward methods that find at least the minimum number of communities
                if n_communities >= min_desired_communities:
                    diversity_score += 0.2
                
                # Evaluate community quality with modularity if possible
                try:
                    # Convert communities to sets for modularity calculation
                    community_sets = [set(c) for c in communities]
                    modularity = nx.community.modularity(G, community_sets)
                except Exception:
                    modularity = 0
                
                # Combined score - prioritize diversity and modularity
                score = (modularity * 0.3 + coverage * 0.2 + evenness * 0.2 + diversity_score * 0.3)
                
                candidate_scores.append((method, communities, score, n_communities, modularity))
                logger.info(f"Method: {method}, Communities: {n_communities}, Modularity: {modularity:.3f}, Score: {score:.3f}")
            
            # Sort by score (descending)
            candidate_scores.sort(key=lambda x: x[2], reverse=True)
            
            if candidate_scores:
                # Try to create an ensemble using the top candidates
                top_candidates = candidate_scores[:min(3, len(candidate_scores))]
                
                if len(top_candidates) > 1 and top_candidates[0][2] - top_candidates[-1][2] < 0.2:
                    # Close scores - use ensemble approach
                    logger.info("Using ensemble approach for community detection")
                    
                    # Create node-to-community mapping for each candidate
                    node_community_votes = {node: {} for node in G.nodes()}
                    
                    for method, communities, score, _, _ in top_candidates:
                        # Normalize score for voting weight
                        weight = score / sum(c[2] for c in top_candidates)
                        
                        # Map communities to node assignments
                        for comm_id, nodes in enumerate(communities):
                            for node in nodes:
                                if node not in node_community_votes:
                                    continue
                                
                                label = f"{method}_{comm_id}"
                                if label not in node_community_votes[node]:
                                    node_community_votes[node][label] = 0
                                node_community_votes[node][label] += weight
                    
                    # Cluster nodes based on their voting patterns
                    node_vectors = {}
                    all_labels = set()
                    for node, votes in node_community_votes.items():
                        all_labels.update(votes.keys())
                    
                    all_labels = list(all_labels)
                    label_to_idx = {label: i for i, label in enumerate(all_labels)}
                    
                    # Create feature vectors from voting patterns
                    for node, votes in node_community_votes.items():
                        vector = np.zeros(len(all_labels))
                        for label, weight in votes.items():
                            vector[label_to_idx[label]] = weight
                        node_vectors[node] = vector
                    
                    # Apply clustering to voting vectors
                    from sklearn.cluster import KMeans
                    
                    # Convert to matrix
                    node_list = list(node_vectors.keys())
                    feature_matrix = np.array([node_vectors[n] for n in node_list])
                    
                    # Determine number of clusters - weighted average of top candidates
                    # ENHANCED: Ensure a minimum number of clusters
                    n_clusters = max(min_desired_communities, int(sum(c[3] * c[2] for c in top_candidates) / sum(c[2] for c in top_candidates)))
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(feature_matrix)
                    
                    # Create communities from consensus clustering
                    ensemble_communities = [[] for _ in range(n_clusters)]
                    for i, node in enumerate(node_list):
                        ensemble_communities[labels[i]].append(node)
                    
                    ensemble_communities = [c for c in ensemble_communities if c]
                    logger.info(f"Ensemble method created {len(ensemble_communities)} communities")
                    
                    # Apply temporal coherence enhancement to ensemble communities
                    enhanced_communities = self._enhance_temporal_coherence(G, ensemble_communities)
                    
                    # Ensure we didn't collapse to a single community
                    if len(enhanced_communities) < len(ensemble_communities) and len(enhanced_communities) < min_desired_communities:
                        logger.warning(f"Temporal enhancement reduced communities below minimum {min_desired_communities} - reverting to original")
                        return ensemble_communities
                    
                    return enhanced_communities
                else:
                    # One clear winner - use its result
                    best_method, best_communities, best_score, _, _ = top_candidates[0]
                    logger.info(f"Selected {best_method} method with {len(best_communities)} communities (score: {best_score:.3f})")
                    
                    # Apply temporal coherence enhancement to best communities
                    enhanced_communities = self._enhance_temporal_coherence(G, best_communities)
                    
                    # Ensure we didn't collapse to a single community
                    if len(enhanced_communities) < len(best_communities) and len(enhanced_communities) < min_desired_communities:
                        logger.warning(f"Temporal enhancement reduced communities below minimum {min_desired_communities} - reverting to original")
                        return best_communities
                    
                    return enhanced_communities
        
        # If no good candidate found, use connected components as last resort
        logger.warning("No good community structure found - using manual divisive approach")
        
        # As a last resort, try temporal division
        try:
            # Divide nodes by temporal attributes if available
            temporal_nodes = []
            for node in G.nodes():
                if 'mid_idx' in G.nodes[node]:
                    mid_idx = G.nodes[node]['mid_idx']
                    temporal_nodes.append((node, mid_idx))
            
            if temporal_nodes:
                # Sort by mid index
                temporal_nodes.sort(key=lambda x: x[1])
                
                # Create at least 3 communities
                n_temporal = max(3, min(5, len(temporal_nodes) // 5))
                segment_size = len(temporal_nodes) // n_temporal
                
                temporal_communities = []
                for i in range(n_temporal):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < n_temporal - 1 else len(temporal_nodes)
                    community = [temporal_nodes[j][0] for j in range(start_idx, end_idx)]
                    temporal_communities.append(community)
                
                logger.info(f"Using temporal division into {len(temporal_communities)} regimes")
                return temporal_communities
        except Exception as e:
            logger.error(f"Temporal division failed: {str(e)[:100]}...")
        
        # Last resort - use connected components
            communities = list(nx.connected_components(G))
        
        # If not enough communities, subdivide largest
        if len(communities) < min_desired_communities:
            communities = list(communities)  # Convert to list for modification
            communities.sort(key=len, reverse=True)  # Sort by size
            
            while len(communities) < min_desired_communities:
                # Split largest community
                largest = communities[0]
                if len(largest) <= 2:
                    break  # Can't split further
                
                # Create subgraph and find internal division
                subgraph = G.subgraph(largest)
                
                # Try spectral bisection
                try:
                    # Create adjacency matrix
                    adj_matrix = nx.to_numpy_array(subgraph)
                    np.fill_diagonal(adj_matrix, 1.0)
                    
                    from sklearn.cluster import SpectralClustering
                    spectral = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
                    labels = spectral.fit_predict(adj_matrix)
                    
                    node_list = list(subgraph.nodes())
                    split1 = [node_list[i] for i in range(len(node_list)) if labels[i] == 0]
                    split2 = [node_list[i] for i in range(len(node_list)) if labels[i] == 1]
                    
                    # Replace largest community with two new ones
                    communities = communities[1:] + [split1, split2]
                    communities.sort(key=len, reverse=True)
                    
                except Exception:
                    # If spectral fails, just split in half by index
                    nodes = list(largest)
                    split_point = len(nodes) // 2
                    communities = communities[1:] + [nodes[:split_point], nodes[split_point:]]
                    communities.sort(key=len, reverse=True)
        
        logger.info(f"Using {len(communities)} communities as final regimes")
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
        Enhance temporal coherence by merging overlapping time periods.
        
        Args:
            G: NetworkX graph
            communities: Initial communities
            
        Returns:
            Temporally enhanced communities
        """
        # Don't apply temporal coherence if we already have few communities
        # to ensure we maintain regime diversity
        if len(communities) <= 3:
            logger.info("Skipping temporal coherence enhancement to preserve regime diversity")
            return communities
            
        # Calculate temporal range for each community
        community_time_ranges = []
        
        for i, community in enumerate(communities):
            all_start_idx = []
            all_end_idx = []
            
            for node_id in community:
                if 'start_idx' in G.nodes[node_id] and 'end_idx' in G.nodes[node_id]:
                    all_start_idx.append(G.nodes[node_id]['start_idx'])
                    all_end_idx.append(G.nodes[node_id]['end_idx'])
            
            if all_start_idx and all_end_idx:
                start = min(all_start_idx)
                end = max(all_end_idx)
                mid = (start + end) / 2
                size = len(community)  # Track community size
                community_time_ranges.append((i, start, end, mid, size))
        
        # Sort by mid-point for temporal order
        community_time_ranges.sort(key=lambda x: x[3])
        
        # Check for significant overlaps
        merged_indices = set()
        merged_communities = []
        
        for i in range(len(community_time_ranges)):
            if i in merged_indices:
                continue
                
            idx_i, start_i, end_i, mid_i, size_i = community_time_ranges[i]
            merged_group = [idx_i]
            
            # Look for communities that have significant temporal overlap
            for j in range(i+1, len(community_time_ranges)):
                if j in merged_indices:
                    continue
                    
                idx_j, start_j, end_j, mid_j, size_j = community_time_ranges[j]
                
                # Calculate temporal overlap
                overlap_start = max(start_i, start_j)
                overlap_end = min(end_i, end_j)
                overlap = max(0, overlap_end - overlap_start)
                
                # ENHANCED: Use stricter criteria for merging communities
                # 1. Calculate relative sizes of communities for merge decision
                size_ratio = min(size_i, size_j) / max(size_i, size_j) if max(size_i, size_j) > 0 else 0
                
                # 2. Calculate temporal distance between midpoints as a percentage of total range
                total_range = max(end_j, end_i) - min(start_i, start_j)
                mid_distance = abs(mid_i - mid_j) / (total_range + 1e-6) if total_range > 0 else 1.0
                
                # 3. Only merge if there's very significant overlap AND similar sizes
                # Require at least 65% overlap of the smaller range (increased from 50%)
                # AND at least 0.3 size ratio (to prevent large communities absorbing small ones)
                # AND midpoints are relatively close to each other
                smaller_range = min(end_i - start_i, end_j - start_j)
                if (smaller_range > 0 and 
                    overlap / smaller_range > 0.65 and      # Increased threshold
                    size_ratio > 0.3 and                   # Added size ratio constraint
                    mid_distance < 0.5):                   # Added midpoint distance constraint
                    
                    # Calculate topological similarity (if nodes share connections)
                    nodes_i = set(communities[idx_i])
                    nodes_j = set(communities[idx_j])
                    
                    # If communities have few nodes in common, don't merge
                    if len(nodes_i.intersection(nodes_j)) < 0.2 * min(len(nodes_i), len(nodes_j)):
                        # Don't merge communities with little node overlap
                        continue
                    
                    # Calculate regime characteristic similarity if we have lens data
                    # Only merge if community characteristics are similar
                    if hasattr(self, 'lens') and hasattr(self, 'graph') and 'nodes' in self.graph:
                        try:
                            # Get points for each community
                            points_i = set()
                            points_j = set()
                            
                            for node in nodes_i:
                                if node in self.graph['nodes']:
                                    points_i.update(self.graph['nodes'][node])
                            
                            for node in nodes_j:
                                if node in self.graph['nodes']:
                                    points_j.update(self.graph['nodes'][node])
                            
                            # If we have points in both communities, calculate similarity in lens space
                            if points_i and points_j:
                                # Convert to lists for indexing
                                points_i_list = list(points_i)
                                points_j_list = list(points_j)
                                
                                # Get lens values for both communities
                                lens_i = np.mean(self.lens[points_i_list], axis=0)
                                lens_j = np.mean(self.lens[points_j_list], axis=0)
                                
                                # Calculate Euclidean distance in lens space
                                lens_distance = np.linalg.norm(lens_i - lens_j)
                                
                                # Calculate average lens space distance as a reference
                                global_avg_distance = np.mean(np.std(self.lens, axis=0)) * 2
                                
                                # Only merge if communities are similar in lens space
                                if lens_distance > global_avg_distance * 0.8:
                                    # Too different in lens space, don't merge
                                    continue
                        except Exception as e:
                            # If this fails, continue with the merge
                            pass
                    
                    merged_group.append(idx_j)
                    merged_indices.add(j)
                    # Update range for future comparisons
                    start_i = min(start_i, start_j)
                    end_i = max(end_i, end_j)
                    # Update size for future comparisons
                    size_i = size_i + size_j
            
            # Create merged community
            merged_community = []
            for idx in merged_group:
                merged_community.extend(communities[idx])
            
            # Ensure we have no duplicates in the merged community
            merged_community = list(set(merged_community))
            merged_communities.append(merged_community)
        
        # CRITICAL: Prevent collapse to a single regime
        # If we have too few merged communities compared to the original, 
        # retain some of the original community structure
        if len(merged_communities) <= 2 and len(communities) > 4:
            logger.warning(f"Temporal coherence would collapse too many communities ({len(communities)} → {len(merged_communities)}) - preserving diversity")
            
            # Choose more finely-granulated partition based on community sizes
            if len(communities) >= 4:
                # Sort original communities by size (largest first)
                sorted_communities = sorted(communities, key=len, reverse=True)
                # Take at least half of the communities (minimum 3)
                top_communities_count = max(3, len(communities) // 2)
                merged_communities = sorted_communities[:top_communities_count]
                logger.info(f"Preserved {top_communities_count} largest original communities")
            else:
                # Just use the original communities
                merged_communities = communities
                logger.info("Preserved original communities to maintain regime diversity")
        
        # Also enforce a minimum number of communities
        if len(merged_communities) < 3 and len(communities) >= 3:
            logger.warning("Enforcing minimum of 3 communities for regime diversity")
            # Use original if it has enough communities
            return communities
        
        if len(merged_communities) < len(communities):
            logger.info(f"Enhanced temporal coherence: {len(communities)} → {len(merged_communities)} regimes")
            return merged_communities
        else:
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
                        
                        # ENHANCEMENT: Optimized parameter selection for better clustering
                        # Use more compute-intensive approach to find optimal parameters
                        best_score = -1
                        best_labels = None
                        best_n_clusters = 0
                        
                        # Grid search over different parameter combinations
                        for min_cluster_size in [max(5, len(windows) // 20), max(5, len(windows) // 15), max(5, len(windows) // 10)]:
                            for min_samples in [1, 2, 3]:
                                for alpha in [0.85, 1.0, 1.2]:
                                    for eps in [0.05, 0.1, 0.15]:
                                        try:
                                            hdbscan = HDBSCAN(
                                                min_cluster_size=min_cluster_size,
                                                min_samples=min_samples,
                                                alpha=alpha,
                                                cluster_selection_epsilon=eps,
                                                cluster_selection_method='eom',
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
                                                if np.sum(non_noise_mask) > n_actual_clusters:
                                                    from sklearn.metrics import silhouette_score
                                                    try:
                                                        if len(unique_labels) > 1:
                                                            score = silhouette_score(
                                                                regime_lens[non_noise_mask], 
                                                                sub_labels[non_noise_mask]
                                                            )
                                                            # Also reward higher number of meaningful clusters 
                                                            # and higher percentage of mapped points
                                                            mapped_ratio = np.sum(non_noise_mask) / len(sub_labels)
                                                            adjusted_score = score * (0.8 + 0.2 * n_actual_clusters / 5) * mapped_ratio
                                                            
                                                            if adjusted_score > best_score:
                                                                best_score = adjusted_score
                                                                best_labels = sub_labels
                                                                best_n_clusters = n_actual_clusters
                                                    except:
                                                        # Silhouette score can fail in certain edge cases
                                                        continue
                                        except:
                                            # Skip this parameter combination if it fails
                                            continue
                        
                        # If we found a good clustering
                        if best_score > 0 and best_labels is not None:
                            logger.info(f"HDBSCAN found {best_n_clusters} sub-regimes for primary regime {primary_id} with score {best_score:.3f}")
                            
                            # Map windows to sub-regimes
                            for i, window_idx in enumerate(windows):
                                # Add 1 to avoid 0 sub-regime, but keep -1 as 0 (noise)
                                sub_regime = best_labels[i] + 1 if best_labels[i] != -1 else 0
                                window_to_hierarchy[window_idx] = (primary_id, sub_regime)
                                
                            continue  # Skip fallback methods
                    
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

 