import pandas as pd
import numpy as np
from datetime import timedelta
import pywt 



def preprocess_tick_data(df, symbol_col='SYMBOL', timestamp_col='TIMESTAMP', 
                         price_col='VALUE', volume_col='VOLUME',
                         window=50, z_threshold=3.0, min_price_change_pct=0.05):
    """
    1. Filters to only include market hours (9:30 AM - 4:00 PM)
    2. Cleans outliers using Median Absolute Deviation (MAD)
    3. Removes anomalies
    4. Detection for isolated price spikes and reversals
    5. Isolated points detection with surrounding price analysis
    6. Wavelet transform for multi-scale analysis
        
    Returns:
    --------
    pd.DataFrame: Cleaned and preprocessed tick data
    pd.DataFrame: Diagnostic information with intermediate calculations and flags
    """
    print(f"Starting preprocessing with {len(df)} rows")
    
    outlier_counters = {
        'zscore': 0,
        'extreme_deviation': 0,
        'isolated_point': 0,
        'price_reversal': 0,
        'timestamp_group': 0,
        'price_velocity': 0,
        'suspicious_cluster': 0,
        'wavelet_outlier': 0
    }
    
    df_working = df.copy()
    
    # =============== STEP 1: FILTER TRADING HOURS ===============
    df_working[timestamp_col] = pd.to_datetime(df_working[timestamp_col], errors='coerce')
    df_working = df_working.dropna(subset=[timestamp_col])
    
    df_working = df_working[
        ((df_working[timestamp_col].dt.hour > 9) | 
         ((df_working[timestamp_col].dt.hour == 9) & (df_working[timestamp_col].dt.minute >= 30))) &
        (df_working[timestamp_col].dt.hour < 16)
    ]
    
    print(f"After filtering trading hours: {len(df_working)} rows")
    
    # =============== STEP 2: CLEAN OUTLIERS ===============
    if 'VOLATILITY' not in df_working.columns:
        df_working = df_working.sort_values([symbol_col, timestamp_col]).reset_index(drop=True)
        
        # Calculate volatility as the rolling 20-period mean of absolute percentage changes in price.
        df_working['VOLATILITY'] = df_working.groupby(symbol_col)[price_col].transform(
            lambda x: x.pct_change().abs().rolling(window=20, min_periods=1).mean() * 100
        )
    
    df_working = df_working.sort_values([symbol_col, timestamp_col]).reset_index(drop=True)
    
    df_clean_list = []
    diagnostics_list = []
    
    for symbol, symbol_data in df_working.groupby(symbol_col):
        symbol_data = symbol_data.copy()
        
        symbol_data['PRICE_CHANGE'] = symbol_data[price_col].diff()
        symbol_data['PRICE_CHANGE_PCT'] = (symbol_data['PRICE_CHANGE'] / symbol_data[price_col].shift(1)) * 100
        
        significant_change = abs(symbol_data['PRICE_CHANGE_PCT']) >= min_price_change_pct
        
        # Calculate rolling statistics
        rolling_median = symbol_data[price_col].rolling(window=window, center=True, min_periods=3).median()
        rolling_mad = (symbol_data[price_col] - rolling_median).abs().rolling(window=window, center=True, min_periods=3).median()
        rolling_std = symbol_data[price_col].rolling(window=20, center=True, min_periods=3).std()
        rolling_volume = symbol_data[volume_col].rolling(window=20, center=True, min_periods=3).mean()
        
        rolling_mad_adjusted = rolling_mad * 1.4826
        
        std_values = symbol_data[price_col].rolling(window=10, min_periods=1).std()
        rolling_mad_adjusted = rolling_mad_adjusted.mask(rolling_mad_adjusted == 0, std_values)

        # More robust measure of relative deviation compared to standard expected value procedure
        symbol_data['Z_SCORE'] = (symbol_data[price_col] - rolling_median) / rolling_mad_adjusted
        symbol_data['price_zscore'] = (symbol_data[price_col] - rolling_median) / rolling_std
        symbol_data['volume_ratio'] = symbol_data[volume_col] / rolling_volume
        symbol_data['MEDIAN_DIFF_PCT'] = abs((symbol_data[price_col] - rolling_median) / rolling_median * 100)
        
        volume_factor = 1 + (symbol_data[volume_col] / rolling_volume)
        volume_factor = volume_factor.clip(1, 2)
        symbol_data['Z_SCORE'] = symbol_data['Z_SCORE'] / volume_factor
        
        volatility_factor = 1 + (symbol_data['VOLATILITY'] / symbol_data['VOLATILITY'].rolling(window=window, center=True, min_periods=1).mean())
        volatility_factor = volatility_factor.clip(1, 2)
        symbol_data['Z_SCORE'] = symbol_data['Z_SCORE'] / volatility_factor
        
        # =============== WAVELET-BASED OUTLIER DETECTION ===============
        # Use wavelet transform for multi-scale analysis
        try:
            # Apply wavelet transform to price series
            price_array = symbol_data[price_col].dropna().values
            if len(price_array) > 100: 
                wavelet = 'db4'  # Daubechies wavelet
                coeffs = pywt.wavedec(price_array, wavelet, level=4)
                
                # Reconstruct signal without high-frequency components
                coeffs_filtered = list(coeffs)
                coeffs_filtered[1] = np.zeros_like(coeffs[1])
                
                # Reconstruct the filtered signal
                reconstructed = pywt.waverec(coeffs_filtered, wavelet)
                
                reconstructed = reconstructed[:len(price_array)]
                
                symbol_data['wavelet_diff'] = 0.0
                symbol_data.loc[symbol_data[price_col].notna(), 'wavelet_diff'] = price_array - reconstructed
                
                symbol_data['wavelet_diff_normalized'] = symbol_data['wavelet_diff'] / symbol_data[price_col]
                
                wavelet_threshold = 0.005  
                symbol_data['wavelet_outlier'] = abs(symbol_data['wavelet_diff_normalized']) > wavelet_threshold
                
                # Only flag as outlier if there's also other evidence of abnormality
                symbol_data['wavelet_outlier'] = symbol_data['wavelet_outlier'] & (
                    (abs(symbol_data['Z_SCORE']) > z_threshold * 0.7) |  
                    (abs(symbol_data['PRICE_CHANGE_PCT']) > 0.8)         
                )
            else:
                symbol_data['wavelet_diff'] = 0
                symbol_data['wavelet_diff_normalized'] = 0
                symbol_data['wavelet_outlier'] = False
        except Exception as e:
            print(f"Wavelet transform error for {symbol}: {e}")
            symbol_data['wavelet_diff'] = 0
            symbol_data['wavelet_diff_normalized'] = 0
            symbol_data['wavelet_outlier'] = False
        
        # Initialize outlier flags for each method
        symbol_data['zscore_outlier'] = False
        symbol_data['extreme_deviation_outlier'] = False
        symbol_data['isolated_point_outlier'] = False
        symbol_data['price_reversal_outlier'] = False
        
        # Initialize outlier detection
        zscore_outlier = (abs(symbol_data['Z_SCORE']) > z_threshold) & significant_change

        next_prices_similar = pd.DataFrame(index=symbol_data.index)
        for i in range(1, 4):
            next_prices_similar[f'shift_{i}'] = (
                abs(symbol_data[price_col].shift(-i) - symbol_data[price_col]) / 
                symbol_data[price_col] < 0.005
            )
        
        continuity_mask = next_prices_similar.any(axis=1)
        zscore_outlier = zscore_outlier & (~continuity_mask)
        
        symbol_data['zscore_outlier'] = zscore_outlier
        
        extreme_deviation = abs(symbol_data['PRICE_CHANGE_PCT']) > 1.5
        symbol_data['extreme_deviation_outlier'] = extreme_deviation
        
        # =============== ISOLATED POINTS DETECTION ===============
        isolated_price = (
            (symbol_data['MEDIAN_DIFF_PCT'] > 0.5) &  # Stricter significant deviation from median
            (abs(symbol_data['price_zscore']) > 2.0) &  # Stricter statistical deviation
            (symbol_data['volume_ratio'] < 1.0)  # More inclusive volume threshold
        )
        
        lookback_window = 5
        lookahead_window = 5
        min_stable_points = 3
        
        pct_changes = symbol_data[price_col].pct_change().abs()
        
        is_stable = pct_changes < 0.0015
        
        # For each potential isolated point, check surrounding prices
        for i in range(lookback_window, len(symbol_data) - lookahead_window):
            if not isolated_price.iloc[i]:
                current_price = symbol_data[price_col].iloc[i]
                
                prev_stable = is_stable.iloc[i-lookback_window:i]
                next_stable = is_stable.iloc[i+1:i+1+lookahead_window]
                
                prev_stable_count = prev_stable.rolling(min_stable_points, min_periods=1).sum().max()
                next_stable_count = next_stable.rolling(min_stable_points, min_periods=1).sum().max()
                
                if prev_stable_count >= min_stable_points-1 and next_stable_count >= min_stable_points-1:
                    prev_value = symbol_data[price_col].iloc[i-min_stable_points:i].median()
                    next_value = symbol_data[price_col].iloc[i+1:i+1+min_stable_points].median()
                    
                    # Check if current point significantly deviates from both before and after
                    prev_deviation = abs(current_price - prev_value) / prev_value
                    next_deviation = abs(current_price - next_value) / next_value
                    
                    if prev_deviation > 0.002 and next_deviation > 0.002:
                        isolated_price.iloc[i] = True
        
        symbol_data['isolated_point_outlier'] = isolated_price
        
        # =============== PRICE REVERSAL DETECTION ===============
        
        price_pcts = symbol_data[price_col].pct_change().fillna(0) * 100
        price_changes = np.zeros(len(symbol_data))
        
        price_reversal_mask = np.zeros(len(symbol_data), dtype=bool)
        
        lookback = 3
        lookahead = 3
        
        prices = symbol_data[price_col].values
        
        for i in range(lookback, len(prices) - lookahead):
            current_price = prices[i]
            
            next_prices = prices[i+1:i+1+lookahead]
            prev_prices = prices[i-lookback:i]
            
            forward_pct = (next_prices - current_price) / current_price * 100
            backward_pct = (current_price - prev_prices) / prev_prices * 100
            
            if abs(forward_pct[0]) > 0.5:  # Significant first move
                for j in range(1, len(forward_pct)):
                    if (forward_pct[j] * forward_pct[0] < 0 and  
                        abs(forward_pct[j]) > 0.3 * abs(forward_pct[0])):  
                        
                        price_reversal_mask[i] = True
                        price_reversal_mask[i+j+1] = True
                        
                        
                        price_reversal_mask[i+1:i+j+1] = True
                        break
        symbol_data['price_reversal_outlier'] = price_reversal_mask
        
        symbol_data['IS_OUTLIER'] = (
            symbol_data['zscore_outlier'] | 
            symbol_data['extreme_deviation_outlier'] | 
            symbol_data['isolated_point_outlier'] | 
            symbol_data['price_reversal_outlier'] |
            symbol_data['wavelet_outlier']
        )
        
        diagnostics_list.append(symbol_data.copy())
        
        outlier_counters['zscore'] += symbol_data['zscore_outlier'].sum()
        outlier_counters['extreme_deviation'] += symbol_data['extreme_deviation_outlier'].sum()
        outlier_counters['isolated_point'] += symbol_data['isolated_point_outlier'].sum()
        outlier_counters['price_reversal'] += symbol_data['price_reversal_outlier'].sum()
        outlier_counters['wavelet_outlier'] += symbol_data['wavelet_outlier'].sum()
        
        symbol_data = symbol_data[~symbol_data['IS_OUTLIER']].copy()
        df_clean_list.append(symbol_data)
    
    df_working = pd.concat(df_clean_list, ignore_index=True)
    df_diagnostics = pd.concat(diagnostics_list, ignore_index=True)
    
    print(f"After cleaning outliers: {len(df_working)} rows")
    
    # =============== STEP 3: TIME-BASED CLEANING & VELOCITY OUTLIERS ===============
    df_working = df_working.sort_values([symbol_col, timestamp_col]).reset_index(drop=True)
    
    df_working['date'] = df_working[timestamp_col].dt.date
    
    df_working['is_suspicious'] = False
    df_working['outlier_method'] = ''
    
    timestamp_groups = df_working.groupby([symbol_col, timestamp_col])
    
    for (symbol, timestamp), group in timestamp_groups:
        if len(group) > 1:
            price_range = group[price_col].max() - group[price_col].min()
            price_range_pct = price_range / group[price_col].min() * 100
            
            threshold = 0.5
            
            if price_range_pct > threshold:
                max_vol_idx = group[volume_col].idxmax()
                suspicious_idx = group.index.difference([max_vol_idx])
                
                if len(suspicious_idx) > 0:
                    df_working.loc[suspicious_idx, 'is_suspicious'] = True
                    df_working.loc[suspicious_idx, 'outlier_method'] = 'timestamp_group'
                    outlier_counters['timestamp_group'] += len(suspicious_idx)
    
    for symbol, symbol_group in df_working.groupby(symbol_col):
        symbol_mask = df_working[symbol_col] == symbol
        
        df_working.loc[symbol_mask, 'price_change'] = df_working.loc[symbol_mask, price_col].diff()
        df_working.loc[symbol_mask, 'price_change_pct'] = df_working.loc[symbol_mask, 'price_change'] / df_working.loc[symbol_mask, price_col].shift(1) * 100
        
        df_working.loc[symbol_mask, 'time_diff'] = df_working.loc[symbol_mask, timestamp_col].diff().dt.total_seconds()
        
        df_working.loc[symbol_mask, 'price_velocity'] = df_working.loc[symbol_mask, 'price_change_pct'] / df_working.loc[symbol_mask, 'time_diff']
        
        mask_zero_time = (df_working['time_diff'] == 0) & symbol_mask
        df_working.loc[mask_zero_time, 'price_velocity'] = 0
        
        df_working.loc[symbol_mask, 'rolling_vol_10min'] = df_working.loc[symbol_mask, 'VOLATILITY'].rolling(window=20, min_periods=5).mean()
        
        df_working.loc[symbol_mask, 'vol_adj_factor'] = np.clip(np.sqrt(df_working.loc[symbol_mask, 'volume_ratio']), 0.5, 2.0)
        
        df_working.loc[symbol_mask, 'velocity_threshold'] = 30.0 * df_working.loc[symbol_mask, 'vol_adj_factor']
        
        velocity_outliers = symbol_mask & (abs(df_working['price_velocity']) > df_working['velocity_threshold'])
        # Additional confirmation requirements
        velocity_outliers = velocity_outliers & (
            (abs(df_working['price_change_pct']) > 0.3) |
            (df_working['volume_ratio'] < 0.3)
        )
        df_working.loc[velocity_outliers, 'is_suspicious'] = True
        df_working.loc[velocity_outliers, 'outlier_method'] = 'price_velocity'
        outlier_counters['price_velocity'] += velocity_outliers.sum()
    
    # Handle suspicious clusters
    df_working['suspicious_group'] = (df_working['is_suspicious'] != df_working['is_suspicious'].shift()).cumsum()
    
    suspicious_groups = df_working[df_working['is_suspicious']].groupby(['suspicious_group', symbol_col])
    
    for (group_id, symbol), group_data in suspicious_groups:
        if len(group_data) > 5:
            time_range = (group_data[timestamp_col].max() - group_data[timestamp_col].min()).total_seconds()
            
            if time_range < 1.0:
                vwap = (group_data[price_col] * group_data[volume_col]).sum() / group_data[volume_col].sum()
                
                keep_idx = group_data.index[0]
                df_working.loc[keep_idx, price_col] = vwap
                df_working.loc[keep_idx, 'is_suspicious'] = False
                df_working.loc[keep_idx, 'outlier_method'] = ''
                
                remove_idx = group_data.index[1:]
                df_working.loc[remove_idx, 'is_suspicious'] = True
                df_working.loc[remove_idx, 'outlier_method'] = 'suspicious_cluster'
                outlier_counters['suspicious_cluster'] += len(remove_idx)
    
    df_full_diagnostics = df_working.copy()
    
    df_clean = df_working[~df_working['is_suspicious']].copy()
    
    diagnostic_cols = ['seconds_from_open', 'price_change', 'price_change_pct', 'time_diff', 
                      'price_velocity', 'is_suspicious', 'suspicious_group', 'Z_SCORE', 
                      'IS_OUTLIER', 'PRICE_CHANGE', 'PRICE_CHANGE_PCT', 'MEDIAN_DIFF_PCT',
                      'price_zscore', 'volume_ratio', 'outlier_method', 'zscore_outlier',
                      'extreme_deviation_outlier', 'isolated_point_outlier', 'price_reversal_outlier',
                      'rolling_vol_10min', 'velocity_threshold', 'vol_adj_factor',
                      'wavelet_diff', 'wavelet_diff_normalized', 'wavelet_outlier', 'date']
    
    df_clean = df_clean.drop(columns=[col for col in diagnostic_cols if col in df_clean.columns])
    
    print(f"Final clean dataset: {len(df_clean)} rows")
    
    print("\nOutlier counts by detection method:")
    for method, count in outlier_counters.items():
        print(f"  {method}: {count}")
    
    return df_clean, df_full_diagnostics, outlier_counters