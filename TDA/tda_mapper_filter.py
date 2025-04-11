import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS
from scipy.stats import entropy
import gudhi
import umap

class MapperFilters:
    """
    Collection of filter functions for the Mapper algorithm.
    
    This class implements various filter functions that can be used
    with the Mapper algorithm to map high-dimensional data to R^n.
    """
    
    @staticmethod
    def pca_filter(n_components=2):
        """
        Principal Component Analysis filter.
        
        Parameters:
        -----------
        n_components : int
            Number of components
            
        Returns:
        --------
        callable
            Filter function
        """
        def filter_function(X):
            pca = PCA(n_components=n_components)
            return pca.fit_transform(X)
        
        return filter_function
    
    @staticmethod
    def kernel_pca_filter(n_components=2, kernel='rbf', gamma=None):
        """
        Kernel Principal Component Analysis filter.
        
        Parameters:
        -----------
        n_components : int
            Number of components
        kernel : str
            Kernel type ('rbf', 'poly', 'sigmoid', 'cosine')
        gamma : float or None
            Kernel coefficient
            
        Returns:
        --------
        callable
            Filter function
        """
        def filter_function(X):
            kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
            return kpca.fit_transform(X)
        
        return filter_function
    
    @staticmethod
    def tsne_filter(n_components=2, perplexity=30.0, early_exaggeration=12.0, 
                   learning_rate='auto', n_iter=1000):
        """
        t-SNE filter.
        
        Parameters:
        -----------
        n_components : int
            Number of components
        perplexity : float
            Perplexity parameter
        early_exaggeration : float
            Early exaggeration factor
        learning_rate : float or 'auto'
            Learning rate
        n_iter : int
            Number of iterations
            
        Returns:
        --------
        callable
            Filter function
        """
        def filter_function(X):
            tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                        early_exaggeration=early_exaggeration, 
                        learning_rate=learning_rate, n_iter=n_iter)
            return tsne.fit_transform(X)
        
        return filter_function
    
    @staticmethod
    def mds_filter(n_components=2, metric=True):
        """
        Multidimensional Scaling filter.
        
        Parameters:
        -----------
        n_components : int
            Number of components
        metric : bool
            Whether to use metric MDS
            
        Returns:
        --------
        callable
            Filter function
        """
        def filter_function(X):
            mds = MDS(n_components=n_components, metric=metric)
            return mds.fit_transform(X)
        
        return filter_function
    
    @staticmethod
    def umap_filter(n_components=2, n_neighbors=15, min_dist=0.1):
        """
        UMAP filter.
        
        Parameters:
        -----------
        n_components : int
            Number of components
        n_neighbors : int
            Number of neighbors
        min_dist : float
            Minimum distance
            
        Returns:
        --------
        callable
            Filter function
        """
        def filter_function(X):
            reducer = umap.UMAP(n_components=n_components, 
                               n_neighbors=n_neighbors, 
                               min_dist=min_dist)
            return reducer.fit_transform(X)
        
        return filter_function
    
    @staticmethod
    def eccentricity_filter(metric='euclidean', p=2):
        """
        Eccentricity filter.
        
        Parameters:
        -----------
        metric : str
            Distance metric
        p : int
            Power parameter
            
        Returns:
        --------
        callable
            Filter function
        """
        def filter_function(X):
            from scipy.spatial.distance import pdist, squareform
            
            # Compute distance matrix
            dist_matrix = squareform(pdist(X, metric=metric))
            
            # Compute eccentricity
            eccentricity = np.mean(dist_matrix ** p, axis=1) ** (1/p)
            
            return eccentricity.reshape(-1, 1)
        
        return filter_function
    
    @staticmethod
    def density_filter(bandwidth=1.0):
        """
        Kernel density filter.
        
        Parameters:
        -----------
        bandwidth : float
            Bandwidth parameter
            
        Returns:
        --------
        callable
            Filter function
        """
        def filter_function(X):
            from sklearn.neighbors import KernelDensity
            
            # Fit kernel density estimator
            kde = KernelDensity(bandwidth=bandwidth).fit(X)
            
            # Evaluate density at each point
            log_density = kde.score_samples(X)
            density = np.exp(log_density)
            
            return density.reshape(-1, 1)
        
        return filter_function
    
    @staticmethod
    def persistent_homology_filter(max_dimension=1, max_edge_length=np.inf):
        """
        Persistent homology filter.
        
        Parameters:
        -----------
        max_dimension : int
            Maximum homology dimension
        max_edge_length : float
            Maximum edge length
            
        Returns:
        --------
        callable
            Filter function
        """
        def filter_function(X):
            from scipy.spatial.distance import pdist, squareform
            
            # Compute distance matrix
            dist_matrix = squareform(pdist(X))
            
            # Compute Rips complex
            rips = gudhi.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_edge_length)
            simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension)
            
            # Compute persistent homology
            simplex_tree.compute_persistence()
            persistence = simplex_tree.persistence()
            
            # Calculate total persistence for each dimension
            result = np.zeros((len(X), max_dimension + 1))
            for dim, (birth, death) in persistence:
                if dim <= max_dimension:
                    # Use persistence for each point
                    # This is a simplification
                    result[:, dim] += (death - birth) / len(X)
            
            return result
        
        return filter_function
    
    @staticmethod
    def entropy_filter(bins=10, base=None):
        """
        Entropy filter.
        
        Parameters:
        -----------
        bins : int
            Number of bins for histogram
        base : float or None
            Base of the logarithm
            
        Returns:
        --------
        callable
            Filter function
        """
        def filter_function(X):
            # Compute entropy for each feature
            entropies = []
            for i in range(X.shape[1]):
                # Create histogram
                hist, _ = np.histogram(X[:, i], bins=bins, density=True)
                
                # Compute entropy
                entropies.append(entropy(hist, base=base))
                
            # Weight each dimension by its entropy
            weights = np.array(entropies) / np.sum(entropies)
            
            # Compute weighted average for each point
            result = np.zeros(len(X))
            for i in range(X.shape[1]):
                result += weights[i] * X[:, i]
                
            return result.reshape(-1, 1)
        
        return filter_function 