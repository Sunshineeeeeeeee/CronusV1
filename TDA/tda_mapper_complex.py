import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class MapperComplex:
    """
    Implementation of the Mapper algorithm for topological data analysis.
    
    The Mapper algorithm creates a simplified representation of high-dimensional data
    by constructing a simplicial complex based on the data's topological structure.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the Mapper complex builder.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print verbose output
        """
        self.verbose = verbose
        
        # Data attributes
        self.points = None
        self.distance_matrix = None
        
        # Mapper attributes
        self.filter_values = None
        self.cover = None
        self.clusters = None
        self.complex = None
        self.graph = None
        
    def fit(self, points, filter_function=None, cover_type='uniform', 
            n_intervals=10, overlap_frac=0.5, clusterer=None, n_cubes=10):
        """
        Fit the Mapper complex to data.
        
        Parameters:
        -----------
        points : numpy.ndarray
            Input point cloud data
        filter_function : callable or None
            Function to map points to R^n
        cover_type : str
            Type of cover ('uniform' or 'balanced')
        n_intervals : int
            Number of intervals in the cover
        overlap_frac : float
            Fraction of overlap between adjacent intervals
        clusterer : object or None
            Clustering algorithm to use
        n_cubes : int
            Number of cubes in each dimension for balanced cover
            
        Returns:
        --------
        self
        """
        if self.verbose:
            print("Fitting Mapper complex...")
            
        # Store points
        self.points = points
        
        # Apply filter function
        if filter_function is None:
            # Default: use first principal component
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            self.filter_values = pca.fit_transform(points).flatten()
        else:
            self.filter_values = filter_function(points)
            
        # Construct cover
        self._construct_cover(cover_type, n_intervals, overlap_frac, n_cubes)
        
        # Cluster points in each cover element
        self._cluster_points_in_cover(clusterer)
        
        # Construct the complex
        self._construct_complex()
        
        if self.verbose:
            print(f"Constructed Mapper complex with {len(self.complex)} simplices")
            
        return self
    
    def _construct_cover(self, cover_type, n_intervals, overlap_frac, n_cubes):
        """
        Construct a cover of the filter range.
        
        Parameters:
        -----------
        cover_type : str
            Type of cover
        n_intervals : int
            Number of intervals
        overlap_frac : float
            Overlap fraction
        n_cubes : int
            Number of cubes per dimension
        """
        if cover_type == 'uniform':
            self._construct_uniform_cover(n_intervals, overlap_frac)
        elif cover_type == 'balanced':
            self._construct_balanced_cover(n_cubes, overlap_frac)
        else:
            raise ValueError(f"Unsupported cover type: {cover_type}")
    
    def _construct_uniform_cover(self, n_intervals, overlap_frac):
        """
        Construct a uniform cover of the filter range.
        
        Parameters:
        -----------
        n_intervals : int
            Number of intervals
        overlap_frac : float
            Overlap fraction
        """
        # Min and max of filter values
        min_val = np.min(self.filter_values)
        max_val = np.max(self.filter_values)
        
        # Compute interval length
        interval_length = (max_val - min_val) / (n_intervals - (n_intervals - 1) * overlap_frac)
        
        # Compute overlap length
        overlap_length = interval_length * overlap_frac
        
        # Construct intervals
        intervals = []
        start = min_val
        for i in range(n_intervals):
            end = start + interval_length
            intervals.append((start, end))
            start = end - overlap_length
            
        self.cover = intervals
    
    def _construct_balanced_cover(self, n_cubes, overlap_frac):
        """
        Construct a balanced cover of the filter range.
        
        Parameters:
        -----------
        n_cubes : int
            Number of cubes per dimension
        overlap_frac : float
            Overlap fraction
        """
        # For 1D filter values, this is equivalent to uniform cover
        # This method would be extended for multi-dimensional filter values
        self._construct_uniform_cover(n_cubes, overlap_frac)
    
    def _cluster_points_in_cover(self, clusterer):
        """
        Cluster points in each cover element.
        
        Parameters:
        -----------
        clusterer : object or None
            Clustering algorithm
        """
        # Default clusterer: DBSCAN
        if clusterer is None:
            from sklearn.cluster import DBSCAN
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            
        # Initialize clusters list
        self.clusters = []
        
        # For each cover element
        for interval in self.cover:
            # Find points in this interval
            in_interval = ((self.filter_values >= interval[0]) & 
                          (self.filter_values <= interval[1]))
            
            if np.sum(in_interval) > 0:
                # Get points in this interval
                points_in_interval = self.points[in_interval]
                
                # Cluster points
                if len(points_in_interval) > 1:
                    labels = clusterer.fit_predict(points_in_interval)
                    
                    # Extract clusters (ignoring noise points with label -1)
                    unique_labels = np.unique(labels)
                    for label in unique_labels:
                        if label != -1:  # Ignore noise
                            # Get indices of points in this cluster
                            cluster_indices = np.where(in_interval)[0][labels == label]
                            
                            # Add cluster to list
                            self.clusters.append(cluster_indices)
                else:
                    # Single point, add as a cluster
                    self.clusters.append(np.where(in_interval)[0])
    
    def _construct_complex(self):
        """
        Construct the Mapper simplicial complex.
        """
        # Initialize empty complex
        self.complex = []
        
        # Add 0-simplices (nodes)
        self.complex.extend([(i,) for i in range(len(self.clusters))])
        
        # Add 1-simplices (edges)
        for i in range(len(self.clusters)):
            for j in range(i+1, len(self.clusters)):
                # Check if clusters share points
                if len(np.intersect1d(self.clusters[i], self.clusters[j])) > 0:
                    self.complex.append((i, j))
        
        # Convert to NetworkX graph for visualization
        self.graph = nx.Graph()
        
        # Add nodes
        for i in range(len(self.clusters)):
            self.graph.add_node(i, size=len(self.clusters[i]))
            
        # Add edges
        for i, j in self.complex:
            if len((i,)) == 1 and len((j,)) == 1:  # Only add edges between nodes
                self.graph.add_edge(i, j)
    
    def visualize(self, node_color='size', node_size='size', ax=None, figsize=(10, 10)):
        """
        Visualize the Mapper complex as a graph.
        
        Parameters:
        -----------
        node_color : str or array-like
            Node color attribute
        node_size : str or array-like
            Node size attribute
        ax : matplotlib.axes.Axes or None
            Axes to plot on
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.graph is None:
            raise ValueError("Mapper complex must be constructed first")
            
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
            
        # Prepare node colors
        if node_color == 'size':
            node_colors = [self.graph.nodes[n]['size'] for n in self.graph.nodes]
        else:
            node_colors = node_color
            
        # Prepare node sizes
        if node_size == 'size':
            node_sizes = [50 + 10 * self.graph.nodes[n]['size'] for n in self.graph.nodes]
        else:
            node_sizes = node_size
            
        # Draw graph
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, ax=ax, with_labels=True, 
                node_color=node_colors, node_size=node_sizes, 
                cmap=plt.cm.viridis, edge_color='gray', alpha=0.8)
        
        ax.set_title("Mapper Complex")
        
        return fig 