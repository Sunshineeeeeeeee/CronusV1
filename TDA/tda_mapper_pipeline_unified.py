import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import time
import os

from .tda_mapper_complex import MapperComplex
from .tda_mapper_filter import MapperFilters
from .tda_mapper_distance import MapperDistances
from .tda_mapper_homology import MapperHomology
from ...Diffusion.Volatility_regimes.tda_mapper_viz import MapperViz

class MapperPipeline:
    """
    Complete pipeline for Mapper analysis.
    
    This class combines filter functions, distance metrics, complex construction,
    homology calculation, and visualization into a unified pipeline.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the Mapper pipeline.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print verbose output
        """
        self.verbose = verbose
        
        # Initialize components
        self.mapper_complex = None
        self.mapper_homology = None
        self.mapper_viz = None
        
        # Data containers
        self.points = None
        self.filter_values = None
        self.timestamps = None
        self.features = None
        
        # Pipeline configuration
        self.filter_function = None
        self.distance_metric = None
        self.clusterer = None
        
        # Results
        self.persistence_diagram = None
        self.betti_numbers = None
        self.regime_labels = None
        
        if verbose:
            print("Mapper Pipeline initialized")
    
    def set_data(self, points, timestamps=None, feature_names=None):
        """
        Set input data for the pipeline.
        
        Parameters:
        -----------
        points : numpy.ndarray
            Point cloud data
        timestamps : numpy.ndarray or None
            Timestamps for each point
        feature_names : list or None
            Names of features
            
        Returns:
        --------
        self
        """
        self.points = points
        self.timestamps = timestamps
        
        # Create feature names if not provided
        if feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(points.shape[1])]
        else:
            self.feature_names = feature_names
            
        return self
    
    def set_filter(self, filter_type='pca', **kwargs):
        """
        Set the filter function for Mapper.
        
        Parameters:
        -----------
        filter_type : str
            Type of filter function ('pca', 'tsne', 'umap', 'eccentricity', etc.)
        **kwargs : dict
            Additional parameters for the filter function
            
        Returns:
        --------
        self
        """
        if filter_type == 'pca':
            n_components = kwargs.get('n_components', 2)
            self.filter_function = MapperFilters.pca_filter(n_components)
        elif filter_type == 'kernel_pca':
            n_components = kwargs.get('n_components', 2)
            kernel = kwargs.get('kernel', 'rbf')
            gamma = kwargs.get('gamma', None)
            self.filter_function = MapperFilters.kernel_pca_filter(n_components, kernel, gamma)
        elif filter_type == 'tsne':
            n_components = kwargs.get('n_components', 2)
            perplexity = kwargs.get('perplexity', 30.0)
            self.filter_function = MapperFilters.tsne_filter(n_components, perplexity)
        elif filter_type == 'umap':
            n_components = kwargs.get('n_components', 2)
            n_neighbors = kwargs.get('n_neighbors', 15)
            min_dist = kwargs.get('min_dist', 0.1)
            self.filter_function = MapperFilters.umap_filter(n_components, n_neighbors, min_dist)
        elif filter_type == 'eccentricity':
            metric = kwargs.get('metric', 'euclidean')
            p = kwargs.get('p', 2)
            self.filter_function = MapperFilters.eccentricity_filter(metric, p)
        elif filter_type == 'density':
            bandwidth = kwargs.get('bandwidth', 1.0)
            self.filter_function = MapperFilters.density_filter(bandwidth)
        elif filter_type == 'persistence':
            max_dimension = kwargs.get('max_dimension', 1)
            self.filter_function = MapperFilters.persistent_homology_filter(max_dimension)
        elif filter_type == 'entropy':
            bins = kwargs.get('bins', 10)
            self.filter_function = MapperFilters.entropy_filter(bins)
        elif callable(filter_type):
            # User-provided function
            self.filter_function = filter_type
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
            
        if self.verbose:
            print(f"Set filter function: {filter_type}")
            
        return self
    
    def set_distance_metric(self, metric_type='euclidean', **kwargs):
        """
        Set the distance metric for Mapper.
        
        Parameters:
        -----------
        metric_type : str
            Type of distance metric ('euclidean', 'manhattan', 'cosine', etc.)
        **kwargs : dict
            Additional parameters for the distance metric
            
        Returns:
        --------
        self
        """
        if metric_type == 'euclidean':
            self.distance_metric = MapperDistances.euclidean_distance
        elif metric_type == 'manhattan':
            self.distance_metric = MapperDistances.manhattan_distance
        elif metric_type == 'cosine':
            self.distance_metric = MapperDistances.cosine_distance
        elif metric_type == 'correlation':
            self.distance_metric = MapperDistances.correlation_distance
        elif metric_type == 'chebyshev':
            self.distance_metric = MapperDistances.chebyshev_distance
        elif metric_type == 'minkowski':
            p = kwargs.get('p', 2)
            self.distance_metric = lambda X: MapperDistances.minkowski_distance(X, p)
        elif metric_type == 'mahalanobis':
            self.distance_metric = MapperDistances.mahalanobis_distance
        elif metric_type == 'dtw':
            self.distance_metric = MapperDistances.dtw_distance
        elif metric_type == 'time_series':
            ts_metric = kwargs.get('ts_metric', 'dtw')
            self.distance_metric = lambda X: MapperDistances.time_series_distance(X, ts_metric)
        elif metric_type == 'temporal_weighted':
            if self.timestamps is None:
                raise ValueError("Timestamps must be provided for temporal_weighted distance")
            alpha = kwargs.get('alpha', 0.5)
            beta = kwargs.get('beta', 0.1)
            self.distance_metric = lambda X: MapperDistances.temporal_weighted_distance(
                X, self.timestamps, alpha, beta)
        elif callable(metric_type):
            # User-provided function
            self.distance_metric = metric_type
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
            
        if self.verbose:
            print(f"Set distance metric: {metric_type}")
            
        return self
    
    def set_clusterer(self, clusterer_type='dbscan', **kwargs):
        """
        Set the clustering algorithm for Mapper.
        
        Parameters:
        -----------
        clusterer_type : str or object
            Type of clusterer ('dbscan', 'agglomerative', etc.) or a custom clusterer
        **kwargs : dict
            Additional parameters for the clusterer
            
        Returns:
        --------
        self
        """
        if clusterer_type == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        elif clusterer_type == 'agglomerative':
            n_clusters = kwargs.get('n_clusters', None)
            linkage = kwargs.get('linkage', 'ward')
            distance_threshold = kwargs.get('distance_threshold', None)
            self.clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, 
                linkage=linkage,
                distance_threshold=distance_threshold
            )
        elif hasattr(clusterer_type, 'fit_predict'):
            # User-provided clusterer
            self.clusterer = clusterer_type
        else:
            raise ValueError(f"Unsupported clusterer type: {clusterer_type}")
            
        if self.verbose:
            print(f"Set clusterer: {clusterer_type}")
            
        return self
    
    def fit(self, n_intervals=10, overlap_frac=0.5, cover_type='uniform', 
           max_dimension=2, min_persistence=0):
        """
        Fit the Mapper pipeline to data.
        
        Parameters:
        -----------
        n_intervals : int
            Number of intervals in the cover
        overlap_frac : float
            Fraction of overlap between intervals
        cover_type : str
            Type of cover ('uniform' or 'balanced')
        max_dimension : int
            Maximum homology dimension
        min_persistence : float
            Minimum persistence value
            
        Returns:
        --------
        self
        """
        if self.points is None:
            raise ValueError("Data must be set first")
            
        start_time = time.time()
        
        # Create MapperComplex if not already created
        if self.mapper_complex is None:
            self.mapper_complex = MapperComplex(verbose=self.verbose)
            
        # Set default filter function if not set
        if self.filter_function is None:
            if self.verbose:
                print("No filter function set, using PCA by default")
            self.filter_function = MapperFilters.pca_filter(2)
            
        # Calculate distance matrix if needed
        if self.distance_metric is not None:
            distance_matrix = self.distance_metric(self.points)
        else:
            distance_matrix = None
            
        # Fit Mapper complex
        self.mapper_complex.fit(
            points=self.points,
            filter_function=self.filter_function,
            cover_type=cover_type,
            n_intervals=n_intervals,
            overlap_frac=overlap_frac,
            clusterer=self.clusterer
        )
        
        # Compute homology if requested
        if max_dimension > 0:
            if self.mapper_homology is None:
                self.mapper_homology = MapperHomology(self.mapper_complex)
            else:
                self.mapper_homology.set_mapper_complex(self.mapper_complex)
                
            self.persistence_diagram = self.mapper_homology.compute_homology(
                max_dimension=max_dimension,
                min_persistence=min_persistence
            )
            
            self.betti_numbers = self.mapper_homology.betti_numbers
            
            if self.verbose:
                print(f"Computed homology with Betti numbers: {self.betti_numbers}")
                
        # Initialize visualization
        self.mapper_viz = MapperViz(self.mapper_complex)
        
        if self.verbose:
            print(f"Mapper fit completed in {time.time() - start_time:.2f} seconds")
            
        return self
    
    def assign_regimes(self, n_regimes=None):
        """
        Assign regime labels to points based on Mapper clustering.
        
        Parameters:
        -----------
        n_regimes : int or None
            Number of regimes to identify (if None, uses the number of connected components)
            
        Returns:
        --------
        numpy.ndarray
            Regime labels
        """
        if self.mapper_complex is None or self.mapper_complex.clusters is None:
            raise ValueError("Mapper must be fitted first")
            
        # Get connected components if n_regimes is None
        if n_regimes is None:
            connected_components = list(nx.connected_components(self.mapper_complex.graph))
            n_regimes = len(connected_components)
            
            # Map node indices to regime labels
            node_to_regime = {}
            for i, component in enumerate(connected_components):
                for node in component:
                    node_to_regime[node] = i
        else:
            # Simple assignment based on node index modulo n_regimes
            node_to_regime = {node: node % n_regimes for node in self.mapper_complex.graph.nodes}
            
        # Initialize regime labels with -1 (unassigned)
        self.regime_labels = np.full(len(self.points), -1)
        
        # Assign each point to its cluster's regime
        for node, indices in enumerate(self.mapper_complex.clusters):
            regime = node_to_regime.get(node, -1)
            self.regime_labels[indices] = regime
            
        if self.verbose:
            print(f"Assigned {n_regimes} regimes to data points")
            
        return self.regime_labels
    
    def visualize_mapper(self, mode='2d', **kwargs):
        """
        Visualize the Mapper complex.
        
        Parameters:
        -----------
        mode : str
            Visualization mode ('2d', '3d', 'projection')
        **kwargs : dict
            Additional parameters for visualization
            
        Returns:
        --------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
            Figure object
        """
        if self.mapper_viz is None:
            raise ValueError("Mapper must be fitted first")
            
        if mode == '2d':
            return self.mapper_viz.visualize_2d(**kwargs)
        elif mode == '3d':
            return self.mapper_viz.visualize_3d_plotly(**kwargs)
        elif mode == 'projection':
            features = kwargs.get('features', self.points)
            return self.mapper_viz.visualize_point_cloud_projection(features=features, **kwargs)
        elif mode == 'heatmap':
            features = kwargs.get('features', None)
            if features is None:
                features = pd.DataFrame(self.points, columns=self.feature_names)
            return self.mapper_viz.create_cluster_heatmap(features, **kwargs)
        elif mode == 'time_evolution':
            if self.timestamps is None:
                raise ValueError("Timestamps must be provided for time_evolution visualization")
            return self.mapper_viz.visualize_time_evolution(self.timestamps, **kwargs)
        else:
            raise ValueError(f"Unsupported visualization mode: {mode}")
    
    def visualize_homology(self, title="Mapper Persistence Diagram", figsize=(12, 10)):
        """
        Visualize the homology of the Mapper complex.
        
        Parameters:
        -----------
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.mapper_homology is None:
            raise ValueError("Homology must be computed first")
            
        return self.mapper_homology.plot_mapper_with_betti_numbers(figsize=figsize)
    
    def save_results(self, output_dir='mapper_results'):
        """
        Save Mapper analysis results to files.
        
        Parameters:
        -----------
        output_dir : str
            Output directory
            
        Returns:
        --------
        dict
            Dictionary of saved file paths
        """
        if self.mapper_complex is None:
            raise ValueError("Mapper must be fitted first")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save filter values
        filter_values_path = os.path.join(output_dir, 'filter_values.csv')
        filter_values = self.mapper_complex.filter_values
        if len(filter_values.shape) == 1:
            filter_values = filter_values.reshape(-1, 1)
        pd.DataFrame(filter_values).to_csv(filter_values_path, index=False)
        saved_files['filter_values'] = filter_values_path
        
        # Save cluster assignments
        clusters_path = os.path.join(output_dir, 'clusters.csv')
        cluster_df = pd.DataFrame({'point_index': range(len(self.points)), 'cluster': -1})
        for i, indices in enumerate(self.mapper_complex.clusters):
            cluster_df.loc[indices, 'cluster'] = i
        cluster_df.to_csv(clusters_path, index=False)
        saved_files['clusters'] = clusters_path
        
        # Save regime labels if available
        if self.regime_labels is not None:
            regimes_path = os.path.join(output_dir, 'regimes.csv')
            pd.DataFrame({'point_index': range(len(self.points)), 'regime': self.regime_labels}).to_csv(
                regimes_path, index=False)
            saved_files['regimes'] = regimes_path
            
        # Save persistence diagram if available
        if self.persistence_diagram is not None:
            persistence_path = os.path.join(output_dir, 'persistence.csv')
            pd.DataFrame([(dim, birth, death) for dim, (birth, death) in self.persistence_diagram],
                       columns=['dimension', 'birth', 'death']).to_csv(persistence_path, index=False)
            saved_files['persistence'] = persistence_path
            
        # Save visualizations
        viz_2d_path = os.path.join(output_dir, 'mapper_2d.png')
        self.visualize_mapper(mode='2d').savefig(viz_2d_path)
        saved_files['viz_2d'] = viz_2d_path
        
        if self.persistence_diagram is not None:
            homology_path = os.path.join(output_dir, 'homology.png')
            self.visualize_homology().savefig(homology_path)
            saved_files['homology'] = homology_path
            
        if self.verbose:
            print(f"Saved Mapper results to {output_dir}")
            
        return saved_files 