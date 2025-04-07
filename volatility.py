import numpy as np
import pandas as pd
from scipy.signal import spectrogram

def estimate_tick_volatility(df, symbol_col='SYMBOL', timestamp_col='TIMESTAMP', 
                           price_col='VALUE', volume_col='VOLUME',
                           method='wavelet', window_size=1000, smooth_factor=0.5):
    """
    Estimate tick-level volatility using wavelet-based multi-scale volatility analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing tick data
    symbol_col : str
        Column name for symbol
    timestamp_col : str
        Column name for timestamp
    price_col : str
        Column name for price
    volume_col : str
        Column name for volume
    method : str
        Method to use: 'wavelet' (default)
    window_size : int
        Size of sliding window (in number of ticks)
    smooth_factor : float
        Smoothing factor for volatility estimates (0-1)
        
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added volatility column
    """
    print(f"Estimating advanced tick-level volatility for {len(df)} ticks...")
    
    result_df = df.copy()
    
    for symbol in df[symbol_col].unique():
        symbol_mask = result_df[symbol_col] == symbol
        symbol_data = result_df[symbol_mask].copy().sort_values(timestamp_col)
        
        symbol_data[timestamp_col] = pd.to_datetime(symbol_data[timestamp_col])
        
        symbol_data['log_price'] = np.log(symbol_data[price_col])
        symbol_data['return'] = symbol_data['log_price'].diff().fillna(0)
        
        n_ticks = len(symbol_data)
        
        print(f"Computing wavelet-based volatility for {symbol}...")
        
        # Use the spectrogram function to perform a windowed FFT (similar to wavelet analysis)
        # This gives us frequency components over time
        returns = symbol_data['return'].values
        returns = np.nan_to_num(returns)
        
        try:
            # Calculate spectrogram (windowed FFT)
            f, t, Sxx = spectrogram(
                returns,
                fs=1.0,  # Sampling frequency (normalized)
                nperseg=min(256, n_ticks//10),  # Window size
                noverlap=min(128, n_ticks//20),  # Overlap
                scaling='spectrum'  # Return spectrum
            )
            
            # Extract volatility at different frequency bands
            # Low frequencies (long-term volatility)
            low_freq_idx = slice(0, len(f)//10)
            # Medium frequencies
            med_freq_idx = slice(len(f)//10, len(f)//3)
            # High frequencies (short-term volatility)
            high_freq_idx = slice(len(f)//3, None)
            
            # Calculate power in each frequency band over time
            low_power = np.sqrt(np.sum(Sxx[low_freq_idx, :], axis=0))
            med_power = np.sqrt(np.sum(Sxx[med_freq_idx, :], axis=0))
            high_power = np.sqrt(np.sum(Sxx[high_freq_idx, :], axis=0))
            
            # Interpolate to get values for each tick
            wavelet_vol = np.zeros(n_ticks)
            tick_positions = np.linspace(0, 1, n_ticks)
            t_positions = np.linspace(0, 1, len(t))
            
            # Combine frequency bands with more weight to medium frequencies
            combined_power = (low_power * 0.3 + med_power * 0.5 + high_power * 0.2)
            
            # Interpolate to get values for each tick
            wavelet_vol = np.interp(tick_positions, t_positions, combined_power)
            
            # Scale to match typical volatility levels
            scale_factor = np.std(returns) / np.mean(wavelet_vol)
            wavelet_vol *= scale_factor
            
            # Smooth the results
            if smooth_factor > 0:
                window_smooth = int(window_size * smooth_factor)
                wavelet_vol = pd.Series(wavelet_vol).rolling(
                    window=window_smooth, min_periods=1, center=True
                ).mean().values
            
            symbol_data['wavelet_vol'] = wavelet_vol
            
        except Exception as e:
            print(f"Warning: Error in wavelet calculation: {e}")
            # Fallback to EWMA volatility
            symbol_data['wavelet_vol'] = np.sqrt(
               (symbol_data['return']**2).ewm(span=window_size//5, min_periods=10).mean()
            )
        
        # Copy results back to main dataframe    
        result_df.loc[symbol_mask, 'wavelet_vol'] = symbol_data['wavelet_vol']
        result_df.loc[symbol_mask, 'return'] = symbol_data['return']
    
    print("Completed advanced tick-level volatility estimation")
    return result_df