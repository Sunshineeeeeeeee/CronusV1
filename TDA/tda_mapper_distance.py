import numpy as np
from scipy.spatial.distance import pdist, squareform, cosine, euclidean
from sklearn.metrics import pairwise_distances
import networkx as nx

class MapperDistances:
    """
    Distance functions for the Mapper algorithm.
    
    This class implements various distance functions that can be used
    with the Mapper algorithm to compute distances between points.
    """
    
    @staticmethod
    def euclidean_distance(X):
        """
        Compute Euclidean distance matrix.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        return squareform(pdist(X, metric='euclidean'))
    
    @staticmethod
    def manhattan_distance(X):
        """
        Compute Manhattan (L1) distance matrix.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        return squareform(pdist(X, metric='cityblock'))
    
    @staticmethod
    def cosine_distance(X):
        """
        Compute cosine distance matrix.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        return squareform(pdist(X, metric='cosine'))
    
    @staticmethod
    def correlation_distance(X):
        """
        Compute correlation distance matrix.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        return squareform(pdist(X, metric='correlation'))
    
    @staticmethod
    def chebyshev_distance(X):
        """
        Compute Chebyshev (L-infinity) distance matrix.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        return squareform(pdist(X, metric='chebyshev'))
    
    @staticmethod
    def minkowski_distance(X, p=2):
        """
        Compute Minkowski distance matrix.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        p : float
            Power parameter
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        return squareform(pdist(X, metric='minkowski', p=p))
    
    @staticmethod
    def mahalanobis_distance(X):
        """
        Compute Mahalanobis distance matrix.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        from sklearn.covariance import EmpiricalCovariance
        
        # Compute covariance matrix
        cov = EmpiricalCovariance().fit(X)
        VI = np.linalg.inv(cov.covariance_)
        
        # Compute Mahalanobis distances
        return pairwise_distances(X, metric='mahalanobis', VI=VI)
    
    @staticmethod
    def dtw_distance(X):
        """
        Compute Dynamic Time Warping distance matrix.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data (time series)
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        try:
            # Try to use dtaidistance if available
            from dtaidistance import dtw
            return dtw.distance_matrix(X)
        except ImportError:
            # Fallback to a simple implementation
            n = len(X)
            dist_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i+1, n):
                    dist = MapperDistances._simple_dtw(X[i], X[j])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
                    
            return dist_matrix
    
    @staticmethod
    def _simple_dtw(x, y):
        """
        Simple implementation of Dynamic Time Warping.
        
        Parameters:
        -----------
        x : numpy.ndarray
            First time series
        y : numpy.ndarray
            Second time series
            
        Returns:
        --------
        float
            DTW distance
        """
        # Ensure arrays are of the same shape
        if len(x) != len(y):
            # Interpolate to make them the same length
            if len(x) > len(y):
                y = np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, len(y)), y)
            else:
                x = np.interp(np.linspace(0, 1, len(y)), np.linspace(0, 1, len(x)), x)
                
        n, m = len(x), len(y)
        
        # Initialize cost matrix
        D = np.zeros((n + 1, m + 1))
        D[0, :] = np.inf
        D[:, 0] = np.inf
        D[0, 0] = 0
        
        # Fill cost matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(x[i-1] - y[j-1])
                D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
                
        return D[n, m]
    
    @staticmethod
    def graph_distance(G, algorithm='dijkstra'):
        """
        Compute graph distance matrix.
        
        Parameters:
        -----------
        G : networkx.Graph
            Input graph
        algorithm : str
            Algorithm for shortest path ('dijkstra', 'bellman-ford', 'floyd-warshall')
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        n = len(G.nodes())
        dist_matrix = np.zeros((n, n))
        
        if algorithm == 'dijkstra':
            # Compute shortest paths using Dijkstra's algorithm
            for i in G.nodes():
                paths = nx.single_source_dijkstra_path_length(G, i)
                for j, length in paths.items():
                    dist_matrix[i, j] = length
        elif algorithm == 'bellman-ford':
            # Compute shortest paths using Bellman-Ford algorithm
            for i in G.nodes():
                paths = nx.single_source_bellman_ford_path_length(G, i)
                for j, length in paths.items():
                    dist_matrix[i, j] = length
        elif algorithm == 'floyd-warshall':
            # Compute shortest paths using Floyd-Warshall algorithm
            path_lengths = dict(nx.floyd_warshall(G))
            for i in G.nodes():
                for j in G.nodes():
                    dist_matrix[i, j] = path_lengths[i][j]
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        return dist_matrix
    
    @staticmethod
    def time_series_distance(X, metric='dtw'):
        """
        Compute distance matrix for time series data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input time series data
        metric : str
            Distance metric ('dtw', 'euclidean', 'correlation')
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        if metric == 'dtw':
            return MapperDistances.dtw_distance(X)
        elif metric == 'euclidean':
            return MapperDistances.euclidean_distance(X)
        elif metric == 'correlation':
            return MapperDistances.correlation_distance(X)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    @staticmethod
    def temporal_weighted_distance(X, timestamps, alpha=0.5, beta=0.1):
        """
        Compute temporally-weighted distance matrix.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        timestamps : numpy.ndarray
            Timestamps for each point
        alpha : float
            Weight for temporal component
        beta : float
            Decay rate for temporal distance
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        # Compute Euclidean distance matrix
        base_dist = MapperDistances.euclidean_distance(X)
        
        # Convert timestamps to numerical values if needed
        if np.issubdtype(timestamps.dtype, np.datetime64):
            # Convert to nanoseconds, then to seconds
            t0 = timestamps[0]
            time_diffs = timestamps - t0
            time_indices = time_diffs.astype('timedelta64[ns]').astype(np.float64) / 1e9
        else:
            time_indices = np.array(timestamps)
        
        # Normalize time indices to [0,1] range
        if len(time_indices) > 1:
            time_indices = (time_indices - time_indices.min()) / (time_indices.max() - time_indices.min())
        
        # Compute temporal weight matrix
        n = len(X)
        temporal_weight = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Temporal weighting
                time_dist = abs(time_indices[i] - time_indices[j])
                weight = 1 + alpha * (1 - np.exp(-beta * time_dist))
                
                temporal_weight[i, j] = weight
                temporal_weight[j, i] = weight
        
        # Apply temporal weights to distance matrix
        dist_matrix = base_dist * temporal_weight
        
        return dist_matrix 