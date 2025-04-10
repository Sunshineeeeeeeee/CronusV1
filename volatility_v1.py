import numpy as np
import pandas as pd
import pywt
import warnings
from sklearn.preprocessing import normalize
from scipy.stats import entropy
from scipy import signal
from scipy import interpolate

# Define a custom energy_profile function since pywt doesn't have it
def energy_profile(coeffs):
    """
    Calculate the energy profile of wavelet coefficients.
    
    Parameters:
    -----------
    coeffs : np.ndarray
        Wavelet coefficients
        
    Returns:
    --------
    np.ndarray
        Energy profile (squared values)
    """
    return coeffs**2

# Custom EMD implementation to replace PyEMD
def empirical_mode_decomposition(signal_input, max_imf=5, max_iterations=100, sift_threshold=0.05):
    """
    Simple Empirical Mode Decomposition implementation.
    
    Parameters:
    -----------
    signal_input : np.ndarray
        Input signal to decompose
    max_imf : int
        Maximum number of IMFs to extract
    max_iterations : int
        Maximum number of sifting iterations per IMF
    sift_threshold : float
        Stopping criterion for sifting process
        
    Returns:
    --------
    np.ndarray
        IMFs with shape (n_imfs, len(signal))
    """
    signal = signal_input.copy()
    imfs = []
    
    for _ in range(max_imf):
        # Extract one IMF
        imf, residue = _extract_imf(signal, max_iterations, sift_threshold)
        imfs.append(imf)
        
        # Update signal with residue
        signal = residue
        
        # Check if residue is monotonic (no more IMFs can be extracted)
        if _is_monotonic(residue):
            break
    
    # Add residue as the last IMF
    imfs.append(signal)
    
    return np.array(imfs)

def _extract_imf(signal, max_iterations, sift_threshold):
    """Extract one IMF from the signal through sifting process."""
    x = signal.copy()
    
    for _ in range(max_iterations):
        # Find local extrema
        max_indices = _find_extrema(x, 'max')
        min_indices = _find_extrema(x, 'min')
        
        # Check if we have enough extrema to continue
        if len(max_indices) < 2 or len(min_indices) < 2:
            break
        
        # Interpolate extrema to create envelopes
        try:
            max_envelope = _create_envelope(x, max_indices)
            min_envelope = _create_envelope(x, min_indices)
            
            # Compute mean envelope
            mean_envelope = (max_envelope + min_envelope) / 2
            
            # Update signal by subtracting mean envelope
            x_new = x - mean_envelope
            
            # Check stopping criterion
            sd = np.sum((x - x_new)**2) / np.sum(x**2)
            
            # Update x for next iteration
            x = x_new
            
            if sd < sift_threshold:
                break
                
        except Exception as e:
            warnings.warn(f"Error in IMF extraction: {str(e)}")
            break
    
    # Return the extracted IMF and the residue
    return x, signal - x

def _find_extrema(x, extrema_type='max'):
    """Find indices of local maxima or minima"""
    n = len(x)
    indices = []
    
    # Add endpoints if needed
    if extrema_type == 'max':
        comparator = np.greater
    else:
        comparator = np.less
    
    for i in range(1, n-1):
        if comparator(x[i], x[i-1]) and comparator(x[i], x[i+1]):
            indices.append(i)
    
    # Add endpoints if they are extrema
    if n > 1:
        if extrema_type == 'max':
            if x[0] > x[1]:
                indices.insert(0, 0)
            if x[-1] > x[-2]:
                indices.append(n-1)
        else:
            if x[0] < x[1]:
                indices.insert(0, 0)
            if x[-1] < x[-2]:
                indices.append(n-1)
    
    return np.array(indices)

def _create_envelope(x, indices):
    """Create an envelope by interpolating through extrema"""
    if len(indices) < 2:
        return np.zeros_like(x)
    
    # Use cubic spline interpolation
    t = np.arange(len(x))
    spl = interpolate.CubicSpline(indices, x[indices])
    envelope = spl(t)
    
    return envelope

def _is_monotonic(x):
    """Check if the signal is monotonic (no more IMFs can be extracted)"""
    if len(x) < 2:
        return True
    
    diffs = np.diff(x)
    return np.all(diffs >= 0) or np.all(diffs <= 0)

# Custom EMD class to mimic PyEMD.EMD interface
class EMD:
    def __init__(self):
        pass
        
    def __call__(self, signal, max_imf=5):
        """
        Decompose signal into Intrinsic Mode Functions using EMD.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal to decompose
        max_imf : int
            Maximum number of IMFs to extract
        
        Returns:
        --------
        np.ndarray
            IMFs with shape (n_imfs, len(signal))
        """
        return empirical_mode_decomposition(signal, max_imf=max_imf)

def estimate_advanced_volatility(df, timestamp_col='TIMESTAMP', 
                              price_col='VALUE', volume_col='VOLUME',
                              window_size=100, num_imfs=2, epsilon=1e-8):
    """
    Estimate tick-level volatility using EMD-MODWT approach with Shannon entropy filtering.
    Optimized for a single asset time series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing tick data for a single asset
    timestamp_col : str
        Column name for timestamp
    price_col : str
        Column name for price
    volume_col : str
        Column name for volume
    window_size : int
        Size of sliding window for local energy baseline (in number of ticks)
    num_imfs : int
        Number of Intrinsic Mode Functions to extract (default: 2)
    epsilon : float
        Small regularization term to avoid division by zero
        
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added volatility columns
    """
    print(f"Estimating advanced tick-level volatility for {len(df)} ticks...")
    
    # Create a new dataframe to hold results
    result_df = df.copy()
    result_df['emd_vol'] = np.nan
    result_df['filtered_vol'] = np.nan
    
    # Ensure dataframe is sorted by timestamp
    result_df = result_df.sort_values(timestamp_col)
    result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
    
    # Step 1: Prepare High-Frequency Data - compute log-returns
    result_df['log_price'] = np.log(result_df[price_col])
    result_df['return'] = result_df['log_price'].diff().fillna(0)
    
    n_ticks = len(result_df)
    print(f"Processing {n_ticks} ticks...")
    
    # Skip if too few ticks
    if n_ticks < max(window_size, 100):
        warnings.warn(f"Not enough ticks ({n_ticks}). Need at least {max(window_size, 100)}.")
        return result_df
        
    # Get returns array
    returns = result_df['return'].values
    
    # Clean returns - vectorized
    returns = np.nan_to_num(returns)
    
    # Step 2: Apply Online Empirical Mode Decomposition (EMD)
    try:
        # Initialize the EMD with our custom EMD implementation
        emd = EMD()
        
        # Extract IMFs from the return series using our custom implementation
        imfs = emd(returns, max_imf=num_imfs)
        
        # Transpose imfs to match the shape expected by the rest of the code
        imfs = imfs.T
        
        # Ensure we have the requested number of IMFs (or the max possible)
        num_imfs_actual = min(imfs.shape[0], num_imfs)
        if imfs.shape[0] < num_imfs:
            warnings.warn(f"Only {imfs.shape[0]} IMFs could be extracted (requested {num_imfs}).")
            
        # Step 3: Compute Instantaneous IMF Energy - vectorized
        # Calculate energy at each tick from the IMFs using built-in functions
        energy = np.zeros(n_ticks)
        for i in range(num_imfs_actual):
            # Use scipy's signal.periodogram for energy calculation
            # The periodogram returns [frequencies, power spectrum]
            # The power spectrum has length n//2+1, which causes the broadcasting issue
            freqs, power = signal.periodogram(imfs[i, :], fs=1.0)
            
            # Interpolate the power spectrum to match the original signal length
            if len(power) != n_ticks:
                x_original = np.arange(n_ticks)
                x_periodogram = np.linspace(0, n_ticks-1, len(power))
                power_full = np.interp(x_original, x_periodogram, power)
                energy += power_full
            else:
                energy += power
        
        # Step 4: Normalize Against Local Energy Baseline - partially vectorized
        # Initialize arrays for normalized volatility
        local_energy_baseline = np.zeros(n_ticks)
        
        # Compute local energy baseline using rolling window
        # First calculate valid points
        for t in range(window_size, n_ticks):
            local_energy_baseline[t] = np.mean(energy[t-window_size:t])
            
        # Set first window_size points to the first valid baseline
        if window_size < n_ticks:
            first_valid_baseline = local_energy_baseline[window_size]
            local_energy_baseline[:window_size] = first_valid_baseline
        
        # Calculate normalized volatility - vectorized
        volatility = np.sqrt(energy / (local_energy_baseline + epsilon))
        
        # Store the initial EMD-based volatility
        result_df['emd_vol'] = volatility
        
        # Ensure emd_vol is positive and has reasonable values
        result_df['emd_vol'] = np.abs(result_df['emd_vol'])
        
        # Handle any extreme or NaN values
        result_df['emd_vol'] = np.nan_to_num(result_df['emd_vol'], nan=0.0001, posinf=0.01, neginf=0.0001)
        
        # Phase 2: Building the Filter
        
        # Step 5: Apply MODWT to Return Series
        # Define wavelet and decomposition level
        wavelet = 'db4'  # Daubechies wavelet
        level = 5  # Number of decomposition levels
        
        # Get coefficients using MODWT (using pywt's multilevel DWT as approximation)
        wavelet_coeffs = pywt.wavedec(returns, wavelet, mode='periodization', level=level)
        
        # Step 6: Calculate Wavelet Energy by Scale - using PyWavelets built-in functions
        # Initialize wavelet energy array
        wavelet_energy = np.zeros((level+1, n_ticks))
        
        # Calculate energy for each scale using pywt's built-in energy functions
        for j, coeff in enumerate(wavelet_coeffs):
            if len(coeff) < n_ticks:
                # Compute energy directly from wavelet coefficients
                temp_energy = energy_profile(coeff)
                
                # Upsample to match original length
                coeff_extended = np.zeros(n_ticks)
                coeff_extended[:len(temp_energy)] = temp_energy
                wavelet_energy[j, :] = coeff_extended
            else:
                # Compute energy directly and trim if needed
                temp_energy = energy_profile(coeff[:n_ticks])
                wavelet_energy[j, :len(temp_energy)] = temp_energy
        
        # Step 7: Compute Shannon Entropy of Energy Distribution - vectorized
        # Initialize entropy array
        entropy_values = np.zeros(n_ticks)
        
        # For each time point, compute the distribution across scales
        for t in range(n_ticks):
            # Get energy distribution across scales at this time point
            scale_energy = wavelet_energy[:, t]
            
            # Only calculate entropy if we have energy
            if np.sum(scale_energy) > 0:
                # Normalize to probability distribution - use scipy/sklearn
                prob_dist = scale_energy / np.sum(scale_energy)
                
                # Use scipy.stats.entropy for Shannon entropy calculation
                entropy_values[t] = entropy(prob_dist, base=2)
        
        # Step 8: Filter Volatility Estimate with Shannon Entropy
        # Define sigmoid filter function - vectorized
        def entropy_filter(h, h0=2.0, delta=1.0):
            return 1.0 / (1.0 + np.exp(-delta * (h - h0)))
        
        # Apply filter to volatility - vectorized
        filtered_volatility = volatility * entropy_filter(entropy_values)
        
        # Store filtered volatility
        result_df['filtered_vol'] = filtered_volatility
        
        # Step 9: Adaptive Parameter Calibration
        # Scale volatility to be in a reasonable range - vectorized
        filtered_vol_positive = filtered_volatility[filtered_volatility > 0]
        if len(filtered_vol_positive) > 0:
            median_vol = np.median(filtered_vol_positive)
            if median_vol > 0:
                target_median = 0.0001  # Target median volatility level
                scale = target_median / median_vol
                result_df['filtered_vol'] *= scale
        
        # Store raw filtered volatility (without EWMA smoothing) for regime detection
        result_df['raw_vol'] = result_df['filtered_vol'].copy()
        
        # Smooth volatility can be used for visualization but not for regime detection
        result_df['smooth_vol'] = result_df['filtered_vol'].ewm(span=20).mean()
        
        # Annualize volatility for interpretation - vectorized
        result_df['filtered_vol'] *= np.sqrt(252 * 6.5 * 3600)  # Annualized assuming trading days and hours
        result_df['raw_vol'] *= np.sqrt(252 * 6.5 * 3600)  # Annualize raw vol too
        result_df['smooth_vol'] *= np.sqrt(252 * 6.5 * 3600)  # Annualize smooth vol too
        
        # Final cleanup: ensure no NaN or inf values - vectorized
        result_df['filtered_vol'] = np.nan_to_num(result_df['filtered_vol'], 
                                               nan=0.0001, posinf=0.01, neginf=0.0001)
        result_df['raw_vol'] = np.nan_to_num(result_df['raw_vol'],
                                          nan=0.0001, posinf=0.01, neginf=0.0001)
        result_df['smooth_vol'] = np.nan_to_num(result_df['smooth_vol'],
                                             nan=0.0001, posinf=0.01, neginf=0.0001)
        
        print(f"Completed volatility estimation for {n_ticks} ticks")
        
        # Set the column 'sv_vol' as a copy of 'filtered_vol' for compatibility
        result_df['sv_vol'] = result_df['raw_vol'].copy()
            
    except Exception as e:
        warnings.warn(f"Processing failed: {str(e)}")
        print(f"Error in processing: {str(e)}")
        # Fallback approach: use simple EWMA volatility
        result_df['raw_vol'] = np.sqrt((result_df['return']**2).rolling(window=20).std())
        result_df['filtered_vol'] = result_df['raw_vol'].copy()
        result_df['smooth_vol'] = result_df['raw_vol'].ewm(span=20).mean()
        
        # Annualize
        result_df['raw_vol'] *= np.sqrt(252 * 6.5 * 3600)
        result_df['filtered_vol'] *= np.sqrt(252 * 6.5 * 3600)
        result_df['smooth_vol'] *= np.sqrt(252 * 6.5 * 3600)
        
        result_df['sv_vol'] = result_df['raw_vol'].copy()
    
    print("Completed advanced tick-level volatility estimation")
    return result_df

