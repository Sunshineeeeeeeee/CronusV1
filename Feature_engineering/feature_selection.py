import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold, SelectKBest
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Union
import os
import warnings
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

class FeatureSelector:
    """
    Feature selection class for regime features before Topological Data Analysis (TDA).
    
    This class provides various methods to select the most informative and diverse features:
    1. Signal-to-Noise Ratio - Estimates feature stability over time
    2. Variance-Based Selection - Identifies features with unique/orthogonal variance
    3. Feature Diversity - Ensures selected features capture different regimes
    4. Mutual Information Penalization - Reduces redundancy between features
    5. Information Gain - Selects features with maximum diversity using minimal set
    
    The goal is to reduce dimensionality while preserving topological structure for TDA.
    """
    
    def __init__(
        self, 
        feature_prefix: str = 'regime_feature_',
        timestamp_col: str = 'Timestamp',
        core_cols: List[str] = None,
        random_state: int = 42
    ):
        """
        Initialize the feature selector.
        
        Parameters:
        -----------
        feature_prefix : str
            Prefix for feature columns in the dataframe
        timestamp_col : str
            Timestamp column name
        core_cols : List[str]
            Core columns to keep (non-feature columns)
        random_state : int
            Random state for reproducibility
        """
        self.feature_prefix = feature_prefix
        self.timestamp_col = timestamp_col
        
        if core_cols is None:
            self.core_cols = [timestamp_col, 'Value', 'Volume', 'Volatility']
        else:
            self.core_cols = core_cols
            
        self.random_state = random_state
        self.feature_scores = {}
        self.selected_features = []
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        data_path : str
            Path to CSV file containing regime features
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        df = pd.read_csv(data_path)
        
        # Ensure timestamp is datetime type if exists
        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get feature column names from dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        List[str]
            List of feature column names
        """
        return [col for col in df.columns if col.startswith(self.feature_prefix)]
    
    def calculate_signal_to_noise(
        self, 
        df: pd.DataFrame,
        window_sizes: List[int] = [20, 50, 100],
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate enhanced signal-to-noise ratio for each feature across multiple time windows.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        window_sizes : List[int]
            List of window sizes for rolling signal-to-noise calculation
        normalize : bool
            Whether to normalize scores to [0, 1]
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their SNR scores
        """
        feature_cols = self.get_feature_columns(df)
        
        # Calculate signal-to-noise ratio across multiple windows
        snr_dict = {}
        
        for col in feature_cols:
            # Calculate SNR for each window size
            window_snrs = []
            
            for window in window_sizes:
                # Skip if window is larger than data
                if window >= len(df):
                    continue
                    
                # Calculate rolling mean and std
                rolling_mean = df[col].rolling(window=window).mean()
                rolling_std = df[col].rolling(window=window).std()
                
                # Calculate SNR as |mean| / std
                # Replace inf and NaN values with 0
                snr = np.abs(rolling_mean) / rolling_std
                snr = snr.fillna(0).replace([np.inf, -np.inf], 0)
                
                # Compute stability of SNR (lower std of SNR is better)
                snr_stability = 1.0 / (1.0 + snr.std())
                
                # Weight by window size (larger windows are more reliable)
                weighted_snr = snr.mean() * snr_stability * (window / max(window_sizes))
                
                window_snrs.append(weighted_snr)
            
            # Use average SNR across all windows
            if window_snrs:
                snr_dict[col] = np.mean(window_snrs)
            else:
                snr_dict[col] = 0
        
        # Normalize if requested
        if normalize and max(snr_dict.values()) > 0:
            max_val = max(snr_dict.values())
            snr_dict = {k: v / max_val for k, v in snr_dict.items()}
        
        # Store scores
        self.feature_scores['signal_to_noise'] = snr_dict
        
        return snr_dict

    def calculate_orthogonal_variance(
        self, 
        df: pd.DataFrame,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate variance contribution that is orthogonal to other features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        normalize : bool
            Whether to normalize scores to [0, 1]
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their orthogonal variance scores
        """
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].copy()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        # Calculate orthogonal variance for each feature
        orthogonal_var = {}
        
        for col in feature_cols:
            # Get all other features
            other_cols = [c for c in feature_cols if c != col]
            
            if not other_cols:  # If only one feature
                orthogonal_var[col] = X_scaled[col].var()
                continue
                
            # Get current feature and other features
            y = X_scaled[col]
            X_others = X_scaled[other_cols]
            
            # PCA on other features to capture their main components
            # This helps avoid collinearity issues
            pca = PCA(n_components=min(len(other_cols), 5, len(df)-1))
            X_others_pca = pca.fit_transform(X_others)
            
            # Calculate correlation between feature and each principal component
            feature_pc_corr = []
            for i in range(X_others_pca.shape[1]):
                pc = X_others_pca[:, i]
                corr = np.corrcoef(y, pc)[0, 1]
                feature_pc_corr.append(corr ** 2)  # Use R² to measure relationship
            
            # Total variance explained by other features
            if feature_pc_corr:
                explained_var = sum(feature_pc_corr)
            else:
                explained_var = 0
                
            # Orthogonal variance = total variance - explained variance
            # Ensure it's not negative due to numerical issues
            orthogonal_var[col] = max(1.0 - explained_var, 0)
        
        # Normalize if requested
        if normalize and max(orthogonal_var.values()) > 0:
            max_val = max(orthogonal_var.values())
            orthogonal_var = {k: v / max_val for k, v in orthogonal_var.items()}
        
        # Store scores
        self.feature_scores['orthogonal_variance'] = orthogonal_var
        
        return orthogonal_var
    
    def calculate_feature_diversity(
        self,
        df: pd.DataFrame,
        n_clusters: int = 5,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate how well each feature captures different regimes.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        n_clusters : int
            Number of regimes/clusters to consider
        normalize : bool
            Whether to normalize scores to [0, 1]
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their regime diversity scores
        """
        feature_cols = self.get_feature_columns(df)
        diversity_scores = {}
        
        # Normalize data for better clustering
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[feature_cols])
        
        # Perform clustering on all features to get regime labels
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        all_labels = kmeans.fit_predict(df_scaled)
        
        # Calculate entropy and silhouette for each feature
        for i, col in enumerate(feature_cols):
            # Extract feature as 2D array for KMeans
            X = df_scaled[:, i].reshape(-1, 1)
            
            # Cluster using only this feature
            feature_kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            feature_labels = feature_kmeans.fit_predict(X)
            
            # Calculate entropy of cluster sizes (higher entropy = more balanced clusters)
            cluster_counts = np.bincount(feature_labels, minlength=n_clusters)
            cluster_props = cluster_counts / len(feature_labels)
            entropy = -np.sum(cluster_props * np.log2(cluster_props + 1e-10))
            max_entropy = np.log2(n_clusters)
            normalized_entropy = entropy / max_entropy
            
            # Calculate mutual information between feature clusters and all-feature clusters
            # Lower MI is better - means this feature captures different regimes
            confusion = np.zeros((n_clusters, n_clusters))
            for k in range(len(feature_labels)):
                confusion[feature_labels[k], all_labels[k]] += 1
                
            row_sums = confusion.sum(axis=1, keepdims=True)
            col_sums = confusion.sum(axis=0, keepdims=True)
            expected = np.dot(row_sums, col_sums) / len(feature_labels)
            
            # Avoid division by zero
            expected = np.maximum(expected, 1e-10)
            
            # Calculate mutual information
            mutual_info = np.sum(confusion * np.log(confusion / expected + 1e-10))
            
            # Normalize MI by total entropy (results in range 0-1)
            max_mi = min(entropy, -np.sum(col_sums / len(feature_labels) * 
                                     np.log2(col_sums / len(feature_labels) + 1e-10)))
            
            if max_mi > 0:
                normalized_mi = mutual_info / max_mi
            else:
                normalized_mi = 0
            
            # Calculate feature diversity score:
            # Higher entropy (balanced clusters) is good
            # Lower mutual information (different from all-feature clustering) is good
            diversity_score = normalized_entropy * (1 - normalized_mi)
            
            # Ensure the score is positive
            diversity_scores[col] = abs(diversity_score)  # Fix: Use absolute value to ensure positive
        
        # Normalize if requested
        if normalize and max(diversity_scores.values()) > 0:
            max_val = max(diversity_scores.values())
            diversity_scores = {k: v / max_val for k, v in diversity_scores.items()}
        
        # Store scores
        self.feature_scores['diversity'] = diversity_scores
        
        return diversity_scores
        
    def calculate_mutual_information_penalty(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
        penalize_power: float = 2.0  # Higher value = stronger penalization
    ) -> Dict[str, float]:
        """
        Calculate mutual information between features and penalize redundancy.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        normalize : bool
            Whether to normalize scores to [0, 1]
        penalize_power : float
            Power to raise MI penalties to (higher = stronger penalization)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their uniqueness scores
        """
        feature_cols = self.get_feature_columns(df)
        n_features = len(feature_cols)
        
        if n_features <= 1:
            return {feature_cols[0]: 1.0} if n_features == 1 else {}
        
        # Calculate pairwise mutual information between features
        mi_matrix = np.zeros((n_features, n_features))
        
        for i, col_i in enumerate(feature_cols):
            X_i = df[col_i].values.reshape(-1, 1)
            
            for j, col_j in enumerate(feature_cols):
                if i == j:
                    # Self-information is just the entropy
                    mi_matrix[i, j] = 1.0
                elif j > i:
                    # Calculate mutual information
                    X_j = df[col_j].values.reshape(-1, 1)
                    mi = mutual_info_regression(X_i, df[col_j].values)[0]
                    
                    # Normalize by minimum entropy to get [0,1]
                    entropy_i = mutual_info_regression(X_i, X_i.ravel())[0]
                    entropy_j = mutual_info_regression(X_j, X_j.ravel())[0]
                    
                    if min(entropy_i, entropy_j) > 0:
                        norm_mi = mi / min(entropy_i, entropy_j)
                    else:
                        norm_mi = 0
                        
                    mi_matrix[i, j] = norm_mi
                    mi_matrix[j, i] = norm_mi
                    
        # Calculate uniqueness score (penalty for high MI with other features)
        uniqueness_scores = {}
        
        for i, col in enumerate(feature_cols):
            # Average mutual information with other features
            other_indices = [j for j in range(n_features) if j != i]
            if other_indices:
                # Calculate the average MI with other features
                avg_mi = np.mean(mi_matrix[i, other_indices])
                
                # Apply stronger penalization (higher power = stronger penalty)
                # Formula: uniqueness = 1 - avg_mi^power
                # This penalizes high MI more aggressively
                uniqueness_scores[col] = 1.0 - avg_mi ** penalize_power
            else:
                uniqueness_scores[col] = 1.0
        
        # Normalize if requested
        if normalize and max(uniqueness_scores.values()) > 0:
            max_val = max(uniqueness_scores.values())
            uniqueness_scores = {k: v / max_val for k, v in uniqueness_scores.items()}
        
        # Store scores
        self.feature_scores['uniqueness'] = uniqueness_scores
        
        return uniqueness_scores

    def get_feature_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix between features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Correlation matrix for features
        """
        feature_cols = self.get_feature_columns(df)
        return df[feature_cols].corr()
    
    def remove_correlated_features(
        self, 
        df: pd.DataFrame,
        threshold: float = 0.6,  # Lower threshold for stricter correlation removal
        reference_scores: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Remove highly correlated features with stricter threshold.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        threshold : float
            Correlation threshold for removal (0.0 to 1.0)
        reference_scores : Dict[str, float], optional
            Reference scores for deciding which correlated feature to keep
            
        Returns:
        --------
        List[str]
            List of features to keep after removing correlated features
        """
        feature_cols = self.get_feature_columns(df)
        
        # Calculate correlation matrix
        corr_matrix = self.get_feature_correlation_matrix(df)
        
        # If reference scores not provided, use orthogonal variance
        if reference_scores is None:
            reference_scores = self.calculate_orthogonal_variance(df, normalize=False)
            
        # Create list to store features to drop
        features_to_drop = []
        
        # Iterate through correlation matrix
        for i in range(len(feature_cols)):
            for j in range(i+1, len(feature_cols)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    # Get feature names
                    feature_i = feature_cols[i]
                    feature_j = feature_cols[j]
                    
                    # Decide which feature to drop based on reference scores
                    if reference_scores[feature_i] >= reference_scores[feature_j]:
                        if feature_j not in features_to_drop:
                            features_to_drop.append(feature_j)
                    else:
                        if feature_i not in features_to_drop:
                            features_to_drop.append(feature_i)
        
        # Return list of features to keep
        features_to_keep = [f for f in feature_cols if f not in features_to_drop]
        
        return features_to_keep
    
    def combine_feature_scores(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Combine scores from different methods with weights.
        
        Parameters:
        -----------
        weights : Dict[str, float], optional
            Weights for each scoring method
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their combined scores
        """
        if not self.feature_scores:
            raise ValueError("No feature scores calculated. Run scoring methods first.")
        
        # Default weights for TDA regime analysis with higher emphasis on uniqueness
        if weights is None:
            weights = {
                'signal_to_noise': 0.25,     # Stable features
                'orthogonal_variance': 0.3,  # Unique variance contribution
                'diversity': 0.2,            # Regime discrimination
                'uniqueness': 0.25           # Low mutual information with other features (increased)
            }
        
        # Adjust weights to use only available scores
        available_methods = set(self.feature_scores.keys())
        specified_methods = set(weights.keys())
        
        if not specified_methods.issubset(available_methods):
            warnings.warn(f"Some specified methods {specified_methods - available_methods} not available.")
            
        # Use only available methods with weights
        weights = {k: v for k, v in weights.items() if k in available_methods}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Get a list of all features across all methods
        all_features = set()
        for scores in self.feature_scores.values():
            all_features.update(scores.keys())
            
        # Ensure all scores are positive values before combining
        for method in self.feature_scores:
            self.feature_scores[method] = {k: abs(v) for k, v in self.feature_scores[method].items()}
        
        # Calculate combined scores
        combined_scores = {}
        for feature in all_features:
            score = 0.0
            for method, weight in weights.items():
                if method in self.feature_scores and feature in self.feature_scores[method]:
                    score += weight * self.feature_scores[method][feature]
            combined_scores[feature] = score
            
        # Ensure no negative values in combined scores (safeguard)
        combined_scores = {k: abs(v) for k, v in combined_scores.items()}
        
        return combined_scores
    
    def select_features(
        self, 
        df: pd.DataFrame,
        methods: List[str] = None,
        weights: Dict[str, float] = None,
        min_gain_threshold: float = 0.06,  # Slightly more permissive than 0.07
        max_features: int = 7,
        correlation_threshold: float = 0.65,  # Slightly more permissive than 0.6
        plot_gain_curve: bool = False,  # Changed to False since we're removing visualizations
        output_dir: str = None  # Made optional since we're not creating plots
    ) -> List[str]:
        """
        Select features using topology-first approach, prioritizing topological diversity.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        methods : List[str], optional
            List of methods to use for scoring features
        weights : Dict[str, float], optional
            Weights for each scoring method
        min_gain_threshold : float
            Minimum information gain threshold to include a feature (0.06 = 6%)
        max_features : int
            Maximum number of features to select (default=7)
        correlation_threshold : float
            Threshold for correlation removal (applied last)
        plot_gain_curve : bool
            Deprecated parameter kept for backward compatibility
        output_dir : str
            Deprecated parameter kept for backward compatibility
        
        Returns:
        --------
        List[str]
            List of selected feature names
        """
        feature_cols = self.get_feature_columns(df)
        
        # Print parameters
        print(f"Topology-First Feature Selection: min_gain={min_gain_threshold}, corr_threshold={correlation_threshold}, max_features={max_features}")
        
        # Step 1: Calculate metrics for each selection criterion
        # Default methods with topology priority
        if methods is None:
            methods = ['diversity', 'uniqueness', 'signal_to_noise', 'orthogonal_variance']
        
        print("Step 1: Computing feature metrics...")
        for method in methods:
            if method == 'diversity':
                # Topological diversity - how features capture different regimes
                print("  - Computing topological diversity scores...")
                self.calculate_feature_diversity(df)
            elif method == 'uniqueness':
                # Information uniqueness - penalize redundant information
                print("  - Computing information uniqueness scores...")
                self.calculate_mutual_information_penalty(df, penalize_power=2.0)
            elif method == 'signal_to_noise':
                # Signal stability
                print("  - Computing signal stability scores...")
                self.calculate_signal_to_noise(df)
            elif method == 'orthogonal_variance':
                # Variance contribution
                print("  - Computing orthogonal variance scores...")
                self.calculate_orthogonal_variance(df)
        
        # Step 2: Combine scores with custom weights to emphasize topology
        topology_weights = {
            'diversity': 0.35,       # Higher weight for topological diversity
            'uniqueness': 0.30,      # Strong weight for uniqueness/low redundancy
            'signal_to_noise': 0.20, # Less weight for stability
            'orthogonal_variance': 0.15  # Least weight for variance
        }
        
        # Use provided weights if given
        if weights:
            topology_weights = weights
        
        print(f"Step 2: Combining scores with weights: {topology_weights}")
        combined_scores = self.combine_feature_scores(topology_weights)
        
        # Step 3: Sort features by combined score
        sorted_features = sorted(
            [(f, combined_scores.get(f, 0)) for f in feature_cols], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Print top features by topology-focused score
        print("\nTop features by topology-focused score:")
        for i, (feature, score) in enumerate(sorted_features[:10]):
            print(f"  {i+1}. {feature}: {score:.4f}")
        
        # Step 4: Calculate information gain (before correlation filtering)
        print("\nStep 4: Calculating information gain (before correlation filtering)...")
        
        # Extract feature names only from sorted list
        ordered_features = [f for f, _ in sorted_features]
        
        # For information gain calculation
        selected_features = []
        info_gains = []
        cumulative_info = 0
        
        # Standardize data for PCA calculations
        X = df[feature_cols].copy()
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
        
        # Information gain calculation
        for feature in ordered_features:
            if not selected_features:
                # First feature - gain is just its variance
                feature_var = X_scaled[feature].var()
                total_var = X_scaled[feature_cols].var().sum()
                gain = feature_var / total_var if total_var > 0 else 0
                
                selected_features.append(feature)
                info_gains.append(gain)
                cumulative_info += gain
            else:
                # For subsequent features
                # Check if we've already reached max_features
                if max_features and len(selected_features) >= max_features:
                    print(f"Reached maximum number of features ({max_features})")
                    break
                    
                # Existing feature set
                X_selected = X_scaled[selected_features]
                
                # Calculate variance captured by existing feature set
                if len(selected_features) == 1:
                    existing_var = X_selected.var().values[0]
                else:
                    pca_existing = PCA()
                    pca_existing.fit(X_selected)
                    existing_var = sum(pca_existing.explained_variance_)
                
                # Add new feature and calculate variance
                X_new = X_scaled[selected_features + [feature]]
                
                if X_new.shape[1] == 1:
                    new_var = X_new.var().values[0]
                else:
                    pca_new = PCA()
                    pca_new.fit(X_new)
                    new_var = sum(pca_new.explained_variance_)
                
                # Calculate information gain
                if existing_var > 0:
                    gain = (new_var - existing_var) / existing_var
                else:
                    gain = new_var
                    
                # Ensure gain is positive
                gain = abs(gain)
                
                # Only add feature if gain is above threshold
                if gain >= min_gain_threshold:
                    selected_features.append(feature)
                    info_gains.append(gain)
                    cumulative_info += gain
                else:
                    print(f"Stopping at {len(selected_features)} features due to insufficient gain: {gain:.4f} < {min_gain_threshold}")
                    break
        
        print(f"After information gain: {len(selected_features)} features with cumulative gain: {cumulative_info:.4f}")
        
        # Step 5: NOW apply correlation filtering as the LAST step
        if len(selected_features) > 1:
            print(f"\nStep 5: Applying correlation filtering (threshold={correlation_threshold})...")
            
            # Get correlation matrix for selected features only
            selected_df = df[selected_features]
            corr_matrix = selected_df.corr()
            
            # Create a copy to track which features to keep
            features_to_keep = selected_features.copy()
            
            # For each pair of highly correlated features
            for i in range(len(selected_features)):
                for j in range(i+1, len(selected_features)):
                    if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                        # Get feature names and scores
                        feature_i = selected_features[i]
                        feature_j = selected_features[j]
                        score_i = combined_scores[feature_i]
                        score_j = combined_scores[feature_j]
                        
                        # Print correlation info
                        print(f"  High correlation ({corr_matrix.iloc[i, j]:.2f}) between {feature_i} and {feature_j}")
                        
                        # Decide which to drop based on combined score
                        if score_i >= score_j:
                            if feature_j in features_to_keep:
                                features_to_keep.remove(feature_j)
                                print(f"  → Removed {feature_j} (score: {score_j:.4f})")
                        else:
                            if feature_i in features_to_keep:
                                features_to_keep.remove(feature_i)
                                print(f"  → Removed {feature_i} (score: {score_i:.4f})")
            
            # Update selected features
            print(f"After correlation filtering: {len(features_to_keep)}/{len(selected_features)} features remain")
            selected_features = features_to_keep
        
        # Store feature info for logging
        self.info_gains = info_gains
        self.selected_features = selected_features
        self.combined_scores = combined_scores
        
        return selected_features
    
    def filter_dataframe(
        self, 
        df: pd.DataFrame,
        selected_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Filter dataframe to keep only selected features and core columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        selected_features : List[str], optional
            List of selected feature names. If None, use self.selected_features
            
        Returns:
        --------
        pd.DataFrame
            Filtered dataframe
        """
        if selected_features is None:
            if not self.selected_features:
                raise ValueError("No features selected. Run select_features first.")
            selected_features = self.selected_features
        
        # Ensure core columns exist in dataframe
        cols_to_keep = [col for col in self.core_cols if col in df.columns]
        cols_to_keep.extend(selected_features)
        
        return df[cols_to_keep]
    
    def save_selected_features(
        self, 
        df: pd.DataFrame,
        output_path: str,
        include_selection_info: bool = True
    ):
        """
        Save selected features to a CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original dataframe with all features
        output_path : str
            Path to save the filtered dataframe
        include_selection_info : bool
            Whether to include feature selection info in filename
            
        Returns:
        --------
        pd.DataFrame
            Filtered dataframe
        """
        if not self.selected_features:
            raise ValueError("No features selected. Run select_features first.")
        
        # Filter dataframe
        filtered_df = self.filter_dataframe(df)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Modify filename to include selection info if requested
        if include_selection_info:
            base_name, ext = os.path.splitext(output_path)
            output_path = f"{base_name}_top{len(self.selected_features)}{ext}"
        
        # Save to CSV
        filtered_df.to_csv(output_path, index=False)
        print(f"Selected features saved to {output_path}")
        
        return filtered_df


def run_feature_selection(
    input_path: str = None,
    df: pd.DataFrame = None,
    output_path: str = None,
    methods: List[str] = None,
    weights: Dict[str, float] = None,
    min_gain_threshold: float = 0.06,  # Topology-first default
    max_features: int = 7,
    correlation_threshold: float = 0.65,  # Topology-first default
    create_plots: bool = False,  # Changed to False since we're removing visualizations
    plots_dir: str = None,       # Made optional since we're not creating plots
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run topology-first feature selection pipeline for TDA.
    
    This approach prioritizes topological diversity first, then information
    uniqueness, then stability, with correlation filtering as the last step.
    
    Parameters:
    -----------
    input_path : str, optional
        Path to input CSV file with regime features
    df : pd.DataFrame, optional
        DataFrame containing features (alternative to input_path)
    output_path : str, optional
        Path to save selected features CSV
    methods : List[str], optional
        List of methods to use for selection
    weights : Dict[str, float], optional
        Weights for each method
    min_gain_threshold : float
        Minimum gain threshold (0.06 = 6% additional information)
    max_features : int
        Maximum number of features to select
    correlation_threshold : float
        Threshold for correlation removal (applied last)
    create_plots : bool
        Deprecated parameter kept for backward compatibility
    plots_dir : str
        Deprecated parameter kept for backward compatibility
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with selected features
    """
    if df is None and input_path is None:
        raise ValueError("Either df or input_path must be provided")
    
    # Initialize feature selector
    selector = FeatureSelector()
    
    # Load data if needed
    if df is None:
        if verbose:
            print(f"Loading data from {input_path}...")
        df = selector.load_data(input_path)
    
    if verbose:
        feature_cols = selector.get_feature_columns(df)
        print(f"Working with {len(df)} rows and {len(feature_cols)} features")
    
    # Default methods in order of priority for TDA
    if methods is None:
        methods = ['diversity', 'uniqueness', 'signal_to_noise', 'orthogonal_variance']
    
    # Default weights emphasizing topology
    if weights is None:
        weights = {
            'diversity': 0.35,       # Higher weight for topological diversity
            'uniqueness': 0.30,      # Strong weight for uniqueness/low redundancy
            'signal_to_noise': 0.20, # Less weight for stability
            'orthogonal_variance': 0.15  # Least weight for variance
        }
    
    # Select features using topology-first approach
    selected_features = selector.select_features(
        df,
        methods=methods,
        weights=weights,
        min_gain_threshold=min_gain_threshold,
        max_features=max_features,
        correlation_threshold=correlation_threshold,
        plot_gain_curve=False,
        output_dir=None
    )
    
    if verbose:
        print(f"\nFinal selection: {len(selected_features)} features")
        
        # Get a combined score for the selected features
        combined_scores = selector.combined_scores if hasattr(selector, 'combined_scores') else {}
        
        # Calculate PCA explained variance for the selected features
        # PCA is critical for TDA as it helps understand the reduced feature space's 
        # topological properties and how well it preserves the original data's structure
        if len(selected_features) > 1:
            X = df[selected_features].copy()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            n_components = min(3, len(selected_features))
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)
            explained_var = pca.explained_variance_ratio_
            
            print("\nSelected features - Key metrics:")
            print(f"  - Total variance captured: {sum(explained_var):.2%}")
            
            # Compute average correlation between selected features
            if len(selected_features) > 1:
                corr_matrix = df[selected_features].corr().abs()
                corr_values = []
                for i in range(len(selected_features)):
                    for j in range(i+1, len(selected_features)):
                        corr_values.append(corr_matrix.iloc[i, j])
                
                print(f"  - Average correlation: {np.mean(corr_values):.4f}")
                print(f"  - Maximum correlation: {np.max(corr_values):.4f}")
            
            # Principal components breakdown - important for understanding the
            # dimensionality reduction quality for TDA
            if n_components >= 2:
                print(f"  - PC1 explains: {explained_var[0]:.2%} of variance")
                print(f"  - PC2 explains: {explained_var[1]:.2%} of variance")
                if n_components >= 3:
                    print(f"  - PC3 explains: {explained_var[2]:.2%} of variance")
        
        # Print selected features
        print("\nSelected features (ranked by importance):")
        for i, feature in enumerate(selected_features):
            # Get score if available
            score = ""
            if feature in combined_scores:
                score = f" (score: {combined_scores[feature]:.4f})"
            print(f"  {i+1}. {feature}{score}")
    
    # Save selected features if output path provided
    if output_path:
        if verbose:
            print(f"\nSaving selected features to {output_path}...")
        filtered_df = selector.save_selected_features(df, output_path)
    else:
        filtered_df = selector.filter_dataframe(df)
    
    if verbose:
        print("\nFeature selection for TDA completed!")
    
    return filtered_df


if __name__ == "__main__":
    # Update the argument parser
    import argparse
    
    parser = argparse.ArgumentParser(description="Run topology-first feature selection optimized for TDA")
    parser.add_argument("--input", type=str, help="Path to input CSV file with regime features")
    parser.add_argument("--output", type=str, default=None, help="Path to save selected features CSV")
    parser.add_argument("--max_features", type=int, default=7, help="Maximum number of features to select (default=7)")
    parser.add_argument("--min_gain", type=float, default=0.06, help="Minimum information gain threshold (0.06 = 6%)")
    parser.add_argument("--corr_threshold", type=float, default=0.65, help="Correlation threshold (0.65)")
    
    args = parser.parse_args()
    
    # Ensure input is provided when running as a script
    if args.input is None:
        parser.error("--input is required when running as a script")
    
    # Run feature selection
    run_feature_selection(
        input_path=args.input,
        output_path=args.output,
        max_features=args.max_features,
        min_gain_threshold=args.min_gain,
        correlation_threshold=args.corr_threshold
    ) 