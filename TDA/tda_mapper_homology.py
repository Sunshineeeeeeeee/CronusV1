import numpy as np
import networkx as nx
import gudhi
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

class MapperHomology:
    """
    Compute homology for Mapper complexes.
    
    This class implements methods to compute persistent homology
    for Mapper complexes, including converting the graph representation
    to a simplicial complex.
    """
    
    def __init__(self, mapper_complex=None):
        """
        Initialize the Mapper homology calculator.
        
        Parameters:
        -----------
        mapper_complex : MapperComplex or None
            Mapper complex object
        """
        self.mapper_complex = mapper_complex
        self.persistence_diagram = None
        self.betti_numbers = None
        
    def set_mapper_complex(self, mapper_complex):
        """
        Set the Mapper complex.
        
        Parameters:
        -----------
        mapper_complex : MapperComplex
            Mapper complex object
            
        Returns:
        --------
        self
        """
        self.mapper_complex = mapper_complex
        return self
    
    def compute_homology(self, max_dimension=2, min_persistence=0):
        """
        Compute homology of the Mapper complex.
        
        Parameters:
        -----------
        max_dimension : int
            Maximum homology dimension
        min_persistence : float
            Minimum persistence value
            
        Returns:
        --------
        list
            Persistence diagram
        """
        if self.mapper_complex is None or self.mapper_complex.graph is None:
            raise ValueError("Mapper complex must be set and fitted first")
            
        # Convert NetworkX graph to simplicial complex
        st = self._graph_to_simplex_tree(self.mapper_complex.graph, max_dimension)
        
        # Compute persistence
        st.compute_persistence()
        
        # Get persistence diagram
        self.persistence_diagram = st.persistence_pairs()
        
        # Filter by minimum persistence
        self.persistence_diagram = [(dim, (birth, death)) for dim, (birth, death) in self.persistence_diagram
                                   if death - birth >= min_persistence]
        
        # Compute Betti numbers
        self.betti_numbers = self._compute_betti_numbers(max_dimension)
        
        return self.persistence_diagram
    
    def _graph_to_simplex_tree(self, G, max_dimension=2):
        """
        Convert a NetworkX graph to a Gudhi simplex tree.
        
        Parameters:
        -----------
        G : networkx.Graph
            Input graph
        max_dimension : int
            Maximum simplicial dimension
            
        Returns:
        --------
        gudhi.SimplexTree
            Simplex tree object
        """
        # Create simplex tree
        st = gudhi.SimplexTree()
        
        # Add vertices (0-simplices)
        for node in G.nodes():
            st.insert([node], filtration=0)
            
        # Add edges (1-simplices)
        for u, v in G.edges():
            st.insert([u, v], filtration=0)
            
        # Add higher-dimensional simplices if needed
        if max_dimension >= 2:
            # Find all cliques up to size max_dimension + 1
            cliques = nx.find_cliques(G)
            for clique in cliques:
                if len(clique) > 2 and len(clique) <= max_dimension + 1:
                    st.insert(clique, filtration=0)
        
        return st
    
    def _compute_betti_numbers(self, max_dimension):
        """
        Compute Betti numbers from persistence diagram.
        
        Parameters:
        -----------
        max_dimension : int
            Maximum homology dimension
            
        Returns:
        --------
        list
            Betti numbers
        """
        betti = [0] * (max_dimension + 1)
        
        if self.persistence_diagram is None:
            return betti
            
        for dim, (birth, death) in self.persistence_diagram:
            if dim <= max_dimension and death == float('inf'):
                betti[dim] += 1
                
        return betti
    
    def plot_persistence_diagram(self, title="Mapper Persistence Diagram", ax=None, figsize=(8, 8)):
        """
        Plot persistence diagram.
        
        Parameters:
        -----------
        title : str
            Plot title
        ax : matplotlib.axes.Axes or None
            Axes to plot on
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.persistence_diagram is None:
            raise ValueError("Homology must be computed first")
            
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
            
        # Extract birth-death pairs
        points = []
        for dim, (birth, death) in self.persistence_diagram:
            if death != float('inf'):  # Skip infinite death times
                points.append((dim, birth, death))
                
        if not points:
            ax.text(0.5, 0.5, "No finite persistence pairs", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
            
        # Sort by dimension
        points_by_dim = {}
        for dim, birth, death in points:
            if dim not in points_by_dim:
                points_by_dim[dim] = []
            points_by_dim[dim].append((birth, death))
            
        # Plot points by dimension
        colors = plt.cm.tab10.colors
        for dim, pairs in points_by_dim.items():
            pairs = np.array(pairs)
            ax.scatter(pairs[:, 0], pairs[:, 1], color=colors[dim % len(colors)], 
                      label=f"Dimension {dim}", alpha=0.8)
            
        # Plot diagonal
        ax_min = min([birth for _, birth, _ in points]) if points else 0
        ax_max = max([death for _, _, death in points]) if points else 1
        margin = (ax_max - ax_min) * 0.1
        diag_min = ax_min - margin
        diag_max = ax_max + margin
        ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', alpha=0.5)
        
        # Add labels and legend
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(title)
        if points_by_dim:
            ax.legend()
            
        return fig
    
    def plot_mapper_with_betti_numbers(self, figsize=(12, 10)):
        """
        Plot Mapper graph with Betti numbers.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.mapper_complex is None or self.mapper_complex.graph is None:
            raise ValueError("Mapper complex must be set and fitted first")
            
        if self.betti_numbers is None:
            raise ValueError("Homology must be computed first")
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot Mapper graph
        self.mapper_complex.visualize(ax=ax1)
        
        # Plot Betti numbers
        dimensions = range(len(self.betti_numbers))
        ax2.bar(dimensions, self.betti_numbers, color='skyblue')
        ax2.set_xlabel("Dimension")
        ax2.set_ylabel("Betti Number")
        ax2.set_title("Betti Numbers")
        ax2.set_xticks(dimensions)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add Betti number values as text
        for i, betti in enumerate(self.betti_numbers):
            ax2.text(i, betti + 0.1, str(betti), ha='center')
            
        plt.tight_layout()
        
        return fig 