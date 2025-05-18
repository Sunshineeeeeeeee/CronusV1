import numpy as np
import pandas as pd
from scipy import signal
import pywt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Union, Optional
import warnings
import time

# Import GPU libraries if available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    from cupyx.scipy import signal as cusignal
    HAS_CUSIGNAL = True
except ImportError:
    HAS_CUSIGNAL = False

# Import PyEMD if available, otherwise define a placeholder
try:
    # Try EMD for Empirical Mode Decomposition (Hilbert-Huang Transform)
    from PyEMD import EMD, EEMD
    HAS_EMD = True
except ImportError:
    try:
        # Alternative naming convention
        from emd import EMD, EEMD
        HAS_EMD = True
    except ImportError:
        try:
            # We have pyemd, but it's for Earth Mover's Distance, not Empirical Mode Decomposition
            import pyemd
            HAS_EMD = False
        except ImportError:
            HAS_EMD = False


class SignalProcessor:
    """
    Advanced signal processing methods for financial time series.
    Implements various methods to extract trend information from price data:
    - Wavelet analysis (CWT and DWT)
    - Hilbert-Huang Transform (if PyEMD is available)
    - Statistical methods
    - Adaptive filtering
    """
    
    def __init__(self, window_sizes: List[int] = [8, 16, 32, 64], 
                 denoising_threshold: float = 0.3,
                 wavelet: str = 'db4',
                 adaptive_weights: bool = True,
                 use_gpu: bool = True):
        """
        Initialize SignalProcessor with specified parameters.
        
        Args:
            window_sizes: List of window sizes for multi-scale analysis
            denoising_threshold: Threshold for wavelet denoising
            wavelet: Wavelet family to use
            adaptive_weights: Whether to use adaptive weights for timeframes
            use_gpu: Whether to use GPU acceleration when available
        """
        self.window_sizes = window_sizes
        self.denoising_threshold = denoising_threshold
        self.wavelet = wavelet
        self.adaptive_weights = adaptive_weights
        self.timeframe_weights = {w: 1.0 / len(window_sizes) for w in window_sizes}
        self.timeframe_correlations = {w: 0.0 for w in window_sizes}
        self.ensemble_methods = ['wavelet', 'kalman', 'changepoint']
        self.ensemble_weights = {m: 1.0 / len(self.ensemble_methods) for m in self.ensemble_methods}
        
        # Check GPU availability
        self.use_gpu = use_gpu
        self.has_torch_gpu = HAS_TORCH and torch.cuda.is_available()
        self.has_cupy = HAS_CUPY
        
        if self.use_gpu and (self.has_torch_gpu or self.has_cupy):
            print(f"GPU acceleration enabled: {'PyTorch CUDA' if self.has_torch_gpu else 'CuPy'}")
        else:
            print("Using CPU processing (GPU not available or disabled)")
            self.use_gpu = False
    
    def wavelet_denoising(self, data: np.ndarray, level: int = None) -> np.ndarray:
        """
        Apply wavelet denoising to remove noise from the signal.
        GPU-accelerated if PyTorch or CuPy is available.
        
        Args:
            data: Input signal
            level: Decomposition level (if None, will be calculated based on data length)
            
        Returns:
            Denoised signal
        """
        # Pad the signal if needed for wavelet transform
        original_len = len(data)
        power_of_2 = 2**np.ceil(np.log2(original_len))
        padding = int(power_of_2 - original_len)
        
        # Move data to GPU if available
        if self.has_cupy:
            # Use CuPy if available
            try:
                gpu_data = cp.asarray(data)
                return gpu_data, lambda x: cp.asnumpy(x)
            except:
                # Fallback to numpy if CuPy transfer fails
                warnings.warn("CuPy transfer failed, falling back to CPU")
                return data, lambda x: x
        elif self.has_torch_gpu:
            # Use PyTorch if available but CuPy isn't
            try:
                gpu_data = torch.from_numpy(data)
                if torch.cuda.is_available():
                    gpu_data = gpu_data.cuda()
                return gpu_data, lambda x: x.cpu().numpy() if torch.is_tensor(x) else x
            except:
                # Fallback to numpy if PyTorch transfer fails
                warnings.warn("PyTorch transfer failed, falling back to CPU")
                return data, lambda x: x
        else:
            # No GPU library available
            return data, lambda x: x
    
    def wavelet_transform(self, data: np.ndarray, method: str = 'dwt', 
                           scales: Optional[List[int]] = None,
                           adapt_for_hft: bool = True) -> Dict[str, np.ndarray]:
        """
        Apply wavelet transform to the data.
        
        Args:
            data: Input data as numpy array
            method: Transform method ('dwt' for discrete, 'cwt' for continuous)
            scales: Scales for the continuous wavelet transform
            adapt_for_hft: Whether to adapt parameters for HFT data
            
        Returns:
            Dictionary with transform results
        """
        result = {}
        
        # Filter out NaN values that may appear in HFT data
        data_clean = np.nan_to_num(data, nan=0.0)
        
        # For HFT data, smooth extreme outliers
        if adapt_for_hft:
            # Use median absolute deviation for robustness
            median = np.median(data_clean)
            mad = np.median(np.abs(data_clean - median))
            # Scale factor approximately equates to 3 standard deviations in normal distribution
            scale_factor = 4.0
            # Identify outliers
            outlier_mask = np.abs(data_clean - median) > scale_factor * mad
            # Replace outliers with capped values
            data_clean[outlier_mask] = np.sign(data_clean[outlier_mask] - median) * scale_factor * mad + median
        
        if method.lower() == 'dwt':
            try:
                import pywt
                
                # Determine appropriate wavelet level based on data length
                # For HFT, use fewer levels to avoid over-decomposition
                if adapt_for_hft:
                    n = len(data_clean)
                    # Adaptive level selection based on data length
                    max_level = min(pywt.dwt_max_level(n, self.wavelet), 4)  # Limit for HFT
                    # For very short sequences, use just 1 or 2 levels
                    level = max(1, min(max_level, int(np.log2(n) / 2)))
                else:
                    level = 3  # Default level
                
                # Select wavelet family appropriate for HFT (less complex)
                wavelet = 'haar' if adapt_for_hft else self.wavelet
                
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(data_clean, wavelet, level=level)
                
                # Store approximation and detail coefficients
                approx = coeffs[0]
                details = coeffs[1:]
                
                # For HFT, apply extra smoothing to approximation
                if adapt_for_hft and len(approx) > 3:
                    # Use simple moving average for final smoothing
                    window_size = min(len(approx) // 4, 3)
                    if window_size > 1:
                        approx = np.convolve(approx, np.ones(window_size)/window_size, mode='same')
                
                # Extract trend from approximation coefficients
                # Upsample to match original length
                if len(approx) < len(data_clean):
                    # Use linear interpolation for upsampling
                    x_orig = np.linspace(0, 1, len(approx))
                    x_new = np.linspace(0, 1, len(data_clean))
                    approx_upsampled = np.interp(x_new, x_orig, approx)
                else:
                    approx_upsampled = approx
                
                # Store results
                result['approx'] = approx_upsampled
                result['details'] = details
                result['trend_dwt'] = approx_upsampled
                
            except ImportError:
                print("Warning: PyWavelets not available. Install with: pip install PyWavelets")
                result['approx'] = data_clean
                result['trend_dwt'] = data_clean
                
        elif method.lower() == 'cwt':
            try:
                import pywt
                
                # For HFT, use a smaller range of scales focused on short-term patterns
                if scales is None:
                    if adapt_for_hft:
                        # For HFT, focus on smaller scales (higher frequencies)
                        max_scale = min(32, len(data_clean) // 4)
                        scales = np.arange(1, max_scale + 1)
                    else:
                        scales = np.arange(1, 64)
                
                # Use simpler wavelet for HFT
                wavelet = 'mexh' if adapt_for_hft else 'morl'
                
                # Perform CWT
                coef, freqs = pywt.cwt(data_clean, scales, wavelet)
                
                # Get scalogram (power of coefficients)
                scalogram = np.abs(coef)**2
                
                # Extract trend by selecting appropriate scales
                if adapt_for_hft:
                    # For HFT, focus on mid-range scales that capture short-term trends
                    # but filter out very high-frequency noise
                    mid_idx = len(scales) // 2
                    trend_idx = max(0, min(mid_idx, len(scales) // 4))
                else:
                    # For regular data, focus on lower frequencies for trends
                    trend_idx = len(scales) // 3
                
                trend_cwt = coef[trend_idx]
                
                # Store results
                result['coef'] = coef
                result['scales'] = scales
                result['scalogram'] = scalogram
                result['trend_cwt'] = trend_cwt
                
            except ImportError:
                print("Warning: PyWavelets not available. Install with: pip install PyWavelets")
                result['trend_cwt'] = data_clean
                
        return result
    
    def hilbert_huang_transform(self, data: np.ndarray, num_imfs: int = 5) -> Dict[str, np.ndarray]:
        """
        Apply Hilbert-Huang Transform for adaptive time-frequency analysis.
        GPU-accelerated for Hilbert transform if CuPy and CuSignal are available.
        
        Args:
            data: Input signal
            num_imfs: Number of Intrinsic Mode Functions to extract
            
        Returns:
            Dictionary with IMFs, instantaneous frequencies and amplitudes
        """
        result = {}
        
        if not HAS_EMD:
            # If PyEMD is not available, return empty results
            result['trend_hht'] = np.zeros_like(data)
            return result
        
        try:
            # Empirical Mode Decomposition (EMD is a CPU operation as it's iterative)
            emd = EMD()
            imfs = emd(data, max_imf=num_imfs)
            
            # Store IMFs
            result['imfs'] = imfs
            
            # Apply Hilbert transform - this can be GPU accelerated
            if self.has_cupy and HAS_CUSIGNAL:
                # Move IMFs to GPU
                imfs_gpu = cp.asarray(imfs)
                
                # Apply Hilbert transform on GPU
                analytic_signal = cusignal.hilbert(imfs_gpu)
                
                # Calculate amplitude and phase on GPU
                amplitude_envelope = cp.abs(analytic_signal)
                instantaneous_phase = cp.unwrap(cp.angle(analytic_signal))
                
                # Calculate frequency (difference of unwrapped phase)
                instantaneous_frequency = cp.diff(instantaneous_phase) / (2.0 * cp.pi)
                
                # Pad frequency to match original shape
                instantaneous_frequency = cp.vstack((
                    instantaneous_frequency, 
                    instantaneous_frequency[:, -1][:, cp.newaxis]
                ))
                
                # Move results back to CPU
                result['amplitude'] = cp.asnumpy(amplitude_envelope)
                result['frequency'] = cp.asnumpy(instantaneous_frequency)
                
                # Calculate trend using residual (last IMF) which represents the trend
                result['trend_hht'] = imfs[-1]
                
                # Normalize trend to [-1, 1] range
                max_abs = cp.max(cp.abs(cp.asarray(result['trend_hht'])))
                if max_abs > 0:
                    result['trend_hht'] = cp.asnumpy(cp.asarray(result['trend_hht']) / max_abs)
            
            elif self.has_torch_gpu and torch.cuda.is_available():
                # Alternative implementation using PyTorch
                # Move IMFs to GPU
                imfs_tensor = torch.tensor(imfs, dtype=torch.float32, device='cuda')
                
                # Apply Hilbert transform
                # PyTorch doesn't have a direct Hilbert transform, so we use FFT approach
                n = imfs.shape[1]
                # Create Hermitian-symmetric version for the Hilbert transform
                X = torch.fft.fft(imfs_tensor, dim=1)
                h = torch.zeros(n, device='cuda')
                if n % 2 == 0:
                    h[0] = h[n//2] = 1
                    h[1:n//2] = 2
                else:
                    h[0] = 1
                    h[1:(n+1)//2] = 2
                
                # Apply filter to create analytic signal
                analytic_signal = torch.zeros_like(X, dtype=torch.complex64)
                for i in range(imfs.shape[0]):
                    analytic_signal[i] = torch.fft.ifft(X[i] * h)
                
                # Calculate amplitude and phase
                amplitude_envelope = torch.abs(analytic_signal)
                instantaneous_phase = torch.angle(analytic_signal)
                instantaneous_phase = torch.unwrap(instantaneous_phase, dim=1)
                
                # Calculate frequency
                instantaneous_frequency = torch.diff(instantaneous_phase, dim=1) / (2.0 * torch.pi)
                # Pad frequency to match original shape
                instantaneous_frequency = torch.cat([
                    instantaneous_frequency, 
                    instantaneous_frequency[:, -1:]], dim=1)
                
                # Move results back to CPU
                result['amplitude'] = amplitude_envelope.cpu().numpy()
                result['frequency'] = instantaneous_frequency.cpu().numpy()
                
                # Calculate trend using residual (last IMF)
                result['trend_hht'] = imfs[-1]
                
                # Normalize trend to [-1, 1] range
                max_abs = torch.max(torch.abs(torch.tensor(result['trend_hht'], device='cuda')))
                if max_abs > 0:
                    result['trend_hht'] = (torch.tensor(result['trend_hht'], device='cuda') / max_abs).cpu().numpy()
            
            else:
                # Original CPU implementation
                # Apply Hilbert transform to get instantaneous amplitude and frequency
                analytic_signal = signal.hilbert(imfs)
                amplitude_envelope = np.abs(analytic_signal)
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
                
                result['amplitude'] = amplitude_envelope
                result['frequency'] = np.vstack((instantaneous_frequency, instantaneous_frequency[:, -1][:, np.newaxis]))
                
                # Calculate trend using residual (last IMF) which represents the trend
                result['trend_hht'] = imfs[-1]
                
                # Normalize trend to [-1, 1] range
                if np.max(np.abs(result['trend_hht'])) > 0:
                    result['trend_hht'] = result['trend_hht'] / np.max(np.abs(result['trend_hht']))
            
        except Exception as e:
            print(f"Error in Hilbert-Huang Transform: {e}")
            result['trend_hht'] = np.zeros_like(data)
            
        return result
    
    def adaptive_kalman_filter(self, data: np.ndarray, process_variance: float = 1e-4,
                               measurement_variance: float = 1e-2,
                               is_hft: bool = True) -> Dict[str, np.ndarray]:
        """
        Apply adaptive Kalman filter to estimate hidden state of time series.
        Optimized for HFT data with adaptive noise estimation and robustness to outliers.
        GPU-accelerated if PyTorch or CuPy is available.
        
        Args:
            data: Input time series as numpy array
            process_variance: Initial process noise variance
            measurement_variance: Initial measurement noise variance
            is_hft: Whether the data is from high-frequency trading
            
        Returns:
            Dictionary with filtered state and variance
        """
        # Clean data - replace NaNs and Infs
        data_clean = np.nan_to_num(data, nan=0.0)
        n = len(data_clean)
        
        # Check if GPU acceleration is available
        if self.has_torch_gpu and torch.cuda.is_available():
            # Move data to GPU
            data_tensor = torch.tensor(data_clean, device='cuda', dtype=torch.float32)
            
            # Initialize state estimates and variances (on GPU)
            state_mean = torch.zeros(n, device='cuda', dtype=torch.float32)
            state_var = torch.zeros(n, device='cuda', dtype=torch.float32)
            
            # Initialize robust statistics variables
            robust_std = torch.tensor(1.0, device='cuda', dtype=torch.float32)
            median_innovation = torch.tensor(0.0, device='cuda', dtype=torch.float32)
            
            # Initialize adaptive parameters for HFT
            if is_hft:
                measurement_var = torch.tensor(measurement_variance * 10, device='cuda', dtype=torch.float32)
                process_var = torch.tensor(process_variance * 5, device='cuda', dtype=torch.float32)
                adaptation_rate = 0.3
                
                # Create buffer for recent innovations
                buffer_size = min(20, max(5, n // 100))
                innovation_buffer = torch.zeros(buffer_size, device='cuda', dtype=torch.float32)
                
                # Forgetting factor parameters
                min_forgetting = torch.tensor(0.7, device='cuda', dtype=torch.float32)
                max_forgetting = torch.tensor(0.99, device='cuda', dtype=torch.float32)
                forgetting_factor = torch.ones(n, device='cuda', dtype=torch.float32) * 0.95
            else:
                measurement_var = torch.tensor(measurement_variance, device='cuda', dtype=torch.float32)
                process_var = torch.tensor(process_variance, device='cuda', dtype=torch.float32)
                adaptation_rate = 0.1
                buffer_size = 10
                innovation_buffer = torch.zeros(buffer_size, device='cuda', dtype=torch.float32)
            
            # Initialize with first observation
            if n > 0:
                state_mean[0] = data_tensor[0]
                state_var[0] = 1.0
            
            # Kalman filter recursion
            for t in range(1, n):
                # Predict step (prior)
                prior_mean = state_mean[t-1]
                prior_var = state_var[t-1] + process_var
                
                # Innovation (measurement residual)
                innovation = data_tensor[t] - prior_mean
                
                # Innovation variance
                innovation_var = prior_var + measurement_var
                
                # For HFT, detect and handle outliers
                if is_hft:
                    # Update innovation buffer (circular buffer)
                    innovation_buffer[t % buffer_size] = innovation
                    
                    # Compute robust statistics from buffer
                    if t >= buffer_size:
                        valid_innovations = innovation_buffer[innovation_buffer != 0]
                        if len(valid_innovations) > 0:
                            # Use median absolute deviation for robustness
                            median_innovation = np.median(valid_innovations)
                            mad = np.median(np.abs(valid_innovations - median_innovation))
                            
                            # Scaled MAD approximates standard deviation
                            robust_std = 1.4826 * mad
                            
                            # Update measurement variance based on robust statistics
                            # Higher weight to most recent observation for fast adaptation
                            measurement_var = (1 - adaptation_rate) * measurement_var + \
                                             adaptation_rate * (robust_std ** 2)
                    
                    # Check if current innovation is an outlier (using robust statistics)
                    # Use the robust_std value that was either calculated above or initialized earlier
                    local_measurement_var = measurement_var
                    if t >= buffer_size and robust_std > 0:
                        # If innovation is far from typical, reduce its impact
                        z_score = abs(innovation) / robust_std
                        if z_score > 3.0:  # More than 3 "robust standard deviations"
                            # Adjust measurement variance to reduce influence of outlier
                            local_measurement_var = measurement_var * (z_score / 3.0) ** 2
                        
                    # Adaptive forgetting factor based on innovation magnitude
                    # Smaller forgetting factor when large innovations (rapid changes)
                    if t > 1:
                        normalized_innovation = abs(innovation) / np.sqrt(innovation_var)
                        forgetting_factor[t] = max_forgetting - (max_forgetting - min_forgetting) * \
                                              min(1.0, normalized_innovation / 3.0)
                        
                        # Adjust process variance based on forgetting factor
                        # Lower forgetting factor (rapid changes) -> higher process variance
                        process_var = process_variance * (2.0 - forgetting_factor[t])
                else:
                    local_measurement_var = measurement_var
                
                # Kalman gain
                kalman_gain = prior_var / (prior_var + local_measurement_var)
                
                # Update step (posterior)
                state_mean[t] = prior_mean + kalman_gain * innovation
                state_var[t] = (1 - kalman_gain) * prior_var
            
            # Move results back to CPU
            result = {
                'state': state_mean.cpu().numpy(),
                'variance': state_var.cpu().numpy()
            }
            
            if is_hft:
                result['forgetting_factor'] = forgetting_factor.cpu().numpy()
            
            return result
            
        elif HAS_CUPY:
            # CuPy implementation - similar structure to PyTorch version
            # Move data to GPU
            data_gpu = cp.asarray(data_clean)
            
            # Initialize state estimates and variances
            state_mean = cp.zeros(n)
            state_var = cp.zeros(n)
            
            # Initialize robust statistics variables
            robust_std = cp.array(1.0)
            median_innovation = cp.array(0.0)
            
            # Initialize adaptive parameters for HFT
            if is_hft:
                measurement_var = cp.array(measurement_variance * 10)
                process_var = cp.array(process_variance * 5)
                adaptation_rate = 0.3
                
                buffer_size = min(20, max(5, n // 100))
                innovation_buffer = cp.zeros(buffer_size)
                
                min_forgetting = cp.array(0.7)
                max_forgetting = cp.array(0.99)
                forgetting_factor = cp.ones(n) * 0.95
            else:
                measurement_var = cp.array(measurement_variance)
                process_var = cp.array(process_variance)
                adaptation_rate = 0.1
                buffer_size = 10
                innovation_buffer = cp.zeros(buffer_size)
            
            # Initialize with first observation
            if n > 0:
                state_mean[0] = data_gpu[0]
                state_var[0] = 1.0
            
            # Kalman filter recursion
            for t in range(1, n):
                # Predict step (prior)
                prior_mean = state_mean[t-1]
                prior_var = state_var[t-1] + process_var
                
                # Innovation (measurement residual)
                innovation = data_gpu[t] - prior_mean
                
                # Innovation variance
                innovation_var = prior_var + measurement_var
                
                # For HFT, detect and handle outliers
                if is_hft:
                    # Update innovation buffer
                    innovation_buffer[t % buffer_size] = innovation
                    
                    # Compute robust statistics from buffer
                    if t >= buffer_size:
                        valid_mask = innovation_buffer != 0
                        if cp.any(valid_mask):
                            valid_innovations = innovation_buffer[valid_mask]
                            
                            # Calculate median and MAD
                            median_innovation = cp.median(valid_innovations)
                            mad = cp.median(cp.abs(valid_innovations - median_innovation))
                            
                            # Convert MAD to standard deviation
                            robust_std = 1.4826 * mad
                            
                            # Update measurement variance
                            measurement_var = (1 - adaptation_rate) * measurement_var + adaptation_rate * (robust_std ** 2)
                    
                    # Check for outliers
                    local_measurement_var = measurement_var
                    if t >= buffer_size and robust_std > 0:
                        z_score = cp.abs(innovation) / robust_std
                        if z_score > 3.0:
                            local_measurement_var = measurement_var * (z_score / 3.0) ** 2
                    
                    # Adaptive forgetting factor
                    if t > 1:
                        normalized_innovation = cp.abs(innovation) / cp.sqrt(innovation_var)
                        forgetting_factor[t] = max_forgetting - (max_forgetting - min_forgetting) * \
                                          cp.minimum(1.0, normalized_innovation / 3.0)
                        
                        # Adjust process variance
                        process_var = process_variance * (2.0 - forgetting_factor[t])
                else:
                    local_measurement_var = measurement_var
                
                # Kalman gain
                kalman_gain = prior_var / (prior_var + local_measurement_var)
                
                # Update step (posterior)
                state_mean[t] = prior_mean + kalman_gain * innovation
                state_var[t] = (1 - kalman_gain) * prior_var
            
            # Move results back to CPU
            result = {
                'state': cp.asnumpy(state_mean),
                'variance': cp.asnumpy(state_var)
            }
            
            if is_hft:
                result['forgetting_factor'] = cp.asnumpy(forgetting_factor)
            
            return result
            
        else:
            # Original CPU implementation
            # Initialize state estimates and variances
            state_mean = np.zeros(n)        # Filtered state mean
            state_var = np.zeros(n)         # Filtered state variance
            
            # Initialize robust statistics variables
            robust_std = 1.0  # Initialize with a default value
            median_innovation = 0.0
            
            # Initialize adaptive parameters specifically for HFT
            if is_hft:
                # Start with higher measurement noise for HFT due to microstructure noise
                measurement_var = measurement_variance * 10
                
                # Process variance for HFT should adapt quickly to rapid changes
                process_var = process_variance * 5
                
                # Adaptation rate - how quickly to adapt to new data
                # Higher for HFT to respond to rapid changes
                adaptation_rate = 0.3  
                
                # For HFT, use robust statistics for noise estimation
                # Create a buffer for recent innovations (prediction errors)
                buffer_size = min(20, max(5, n // 100))
                innovation_buffer = np.zeros(buffer_size)
                
                # Use adaptive forgetting factor for nonstationary data (common in HFT)
                min_forgetting = 0.7
                max_forgetting = 0.99
                forgetting_factor = np.ones(n) * 0.95  # Initial value
            else:
                # Standard parameters for regular data
                measurement_var = measurement_variance
                process_var = process_variance
                adaptation_rate = 0.1
                buffer_size = 10
                innovation_buffer = np.zeros(buffer_size)
            
            # Initialize with first observation
            if n > 0:
                state_mean[0] = data_clean[0]
                state_var[0] = 1.0
            
            # Kalman filter recursion
            for t in range(1, n):
                # Predict step (prior)
                prior_mean = state_mean[t-1]
                prior_var = state_var[t-1] + process_var
                
                # Innovation (measurement residual)
                innovation = data_clean[t] - prior_mean
                
                # Innovation (or residual) variance
                innovation_var = prior_var + measurement_var
                
                # For HFT, detect and handle outliers
                if is_hft:
                    # Update innovation buffer (circular buffer)
                    innovation_buffer[t % buffer_size] = innovation
                    
                    # Compute robust statistics from buffer
                    if t >= buffer_size:
                        valid_innovations = innovation_buffer[innovation_buffer != 0]
                        if len(valid_innovations) > 0:
                            # Use median absolute deviation for robustness
                            median_innovation = np.median(valid_innovations)
                            mad = np.median(np.abs(valid_innovations - median_innovation))
                            
                            # Scaled MAD approximates standard deviation
                            robust_std = 1.4826 * mad
                            
                            # Update measurement variance based on robust statistics
                            # Higher weight to most recent observation for fast adaptation
                            measurement_var = (1 - adaptation_rate) * measurement_var + \
                                             adaptation_rate * (robust_std ** 2)
                    
                    # Check if current innovation is an outlier (using robust statistics)
                    # Use the robust_std value that was either calculated above or initialized earlier
                    local_measurement_var = measurement_var
                    if t >= buffer_size and robust_std > 0:
                        # If innovation is far from typical, reduce its impact
                        z_score = abs(innovation) / robust_std
                        if z_score > 3.0:  # More than 3 "robust standard deviations"
                            # Adjust measurement variance to reduce influence of outlier
                            local_measurement_var = measurement_var * (z_score / 3.0) ** 2
                        
                    # Adaptive forgetting factor based on innovation magnitude
                    # Smaller forgetting factor when large innovations (rapid changes)
                    if t > 1:
                        normalized_innovation = abs(innovation) / np.sqrt(innovation_var)
                        forgetting_factor[t] = max_forgetting - (max_forgetting - min_forgetting) * \
                                              min(1.0, normalized_innovation / 3.0)
                        
                        # Adjust process variance based on forgetting factor
                        # Lower forgetting factor (rapid changes) -> higher process variance
                        process_var = process_variance * (2.0 - forgetting_factor[t])
                else:
                    local_measurement_var = measurement_var
                
                # Kalman gain
                kalman_gain = prior_var / (prior_var + local_measurement_var)
                
                # Update step (posterior)
                state_mean[t] = prior_mean + kalman_gain * innovation
                state_var[t] = (1 - kalman_gain) * prior_var
            
            result = {
                'state': state_mean,
                'variance': state_var
            }
            
            if is_hft:
                result['forgetting_factor'] = forgetting_factor
            
            return result
    
    def changepoint_detection(self, data: np.ndarray, window_size: int = 20) -> Dict[str, np.ndarray]:
        """
        Detect change points in the time series using robust statistical methods.
        
        Args:
            data: Input signal
            window_size: Size of the sliding window for detection
            
        Returns:
            Dictionary with change points and trend estimate
        """
        result = {}
        n = len(data)
        
        # Initialize arrays for storing results
        change_points = np.zeros(n, dtype=bool)
        trend = np.zeros(n)
        
        if n < 2 * window_size:
            result['change_points'] = change_points
            result['trend_cp'] = trend
            return result
        
        # Compute rolling statistics
        for i in range(window_size, n - window_size):
            window_before = data[i-window_size:i]
            window_after = data[i:i+window_size]
            
            # Mann-Whitney U test (non-parametric test for distribution shift)
            u_stat, p_value = stats.mannwhitneyu(window_before, window_after, alternative='two-sided')
            
            # Calculate trend strength based on means
            mean_before = np.mean(window_before)
            mean_after = np.mean(window_after)
            
            # Detect significant change points
            if p_value < 0.05:  # 5% significance level
                change_points[i] = True
                # Calculate trend based on direction of change
                trend[i] = mean_after - mean_before
            else:
                # Smaller trend signal for non-significant changes
                trend[i] = 0.2 * (mean_after - mean_before)
        
        # Smooth the trend
        trend = np.convolve(trend, np.ones(min(16, n//4))/min(16, n//4), mode='same')
        
        # Normalize trend to [-1, 1] range
        if np.max(np.abs(trend)) > 0:
            trend = trend / np.max(np.abs(trend))
        
        result['change_points'] = change_points
        result['trend_cp'] = trend
        
        return result
    
    def path_signatures(self, data: np.ndarray, depth: int = 2) -> Dict[str, np.ndarray]:
        """
        Compute path signatures to capture geometric features of the time series.
        GPU-accelerated if PyTorch or CuPy is available.
        
        Args:
            data: Input signal
            depth: Depth of the signature computation
            
        Returns:
            Dictionary with signature terms and trend estimate
        """
        result = {}
        
        # Using signatory with PyTorch CUDA acceleration if available
        if HAS_TORCH and torch.cuda.is_available():
            try:
                import signatory
                
                # Convert to torch tensor and move to GPU
                path = torch.tensor(data.reshape(-1, 1), dtype=torch.float32, device='cuda')
                
                # Compute signature on GPU
                sig = signatory.signature(path, depth)
                
                # Store signature terms (move back to CPU)
                result['signature'] = sig.cpu().numpy()
                
                # Use first order terms for trend estimation
                if depth >= 2 and len(sig) >= 2:
                    result['trend_sig'] = sig[1].cpu().numpy()
                else:
                    result['trend_sig'] = np.zeros_like(data)
                
                # Normalize trend to [-1, 1] range
                if np.max(np.abs(result['trend_sig'])) > 0:
                    result['trend_sig'] = result['trend_sig'] / np.max(np.abs(result['trend_sig']))
                
                return result
                
            except ImportError:
                # Signatory not available, fallback to CPU methods
                pass
        
        # Try using original signatory implementation on CPU
        try:
            # Try to import signature module if available
            import signatory
            import torch
            
            # Convert to torch tensor
            path = torch.tensor(data.reshape(-1, 1), dtype=torch.float32)
            
            # Compute signature
            sig = signatory.signature(path, depth)
            
            # Store signature terms
            result['signature'] = sig.numpy()
            
            # Use first order terms for trend estimation
            if depth >= 2 and len(sig) >= 2:
                result['trend_sig'] = sig[1].numpy()
            else:
                result['trend_sig'] = np.zeros_like(data)
                
        except ImportError:
            try:
                # Try iisignature as an alternative
                import iisignature
                
                # Using CuPy for pre/post-processing if available
                if HAS_CUPY:
                    # Move to GPU
                    data_gpu = cp.asarray(data)
                    
                    # Reshape data for iisignature (iisignature still works on CPU)
                    path = cp.asnumpy(data_gpu.reshape(-1, 1))
                    
                    # Compute signature (on CPU)
                    sig = iisignature.sig(path, depth)
                    
                    # Post-processing on GPU
                    sig_gpu = cp.asarray(sig)
                    
                    # Store signature terms
                    result['signature'] = cp.asnumpy(sig_gpu)
                    
                    # Use first order terms for trend estimation
                    if depth >= 2 and len(sig) >= 1:
                        trend_values = sig_gpu[0] if len(sig_gpu.shape) > 1 else sig_gpu
                        
                        # Expand to match data length using GPU interpolation
                        if len(trend_values) < len(data):
                            orig_indices = cp.arange(len(trend_values))
                            new_indices = cp.linspace(0, len(trend_values)-1, len(data))
                            
                            # Linear interpolation on GPU
                            result['trend_sig'] = cp.interp(new_indices, orig_indices, trend_values)
                            result['trend_sig'] = cp.asnumpy(result['trend_sig'])
                        else:
                            result['trend_sig'] = cp.asnumpy(trend_values[:len(data)])
                    else:
                        result['trend_sig'] = np.zeros_like(data)
                    
                else:
                    # Original CPU implementation
                    # Reshape data for iisignature
                    path = data.reshape(-1, 1)
                    
                    # Compute signature
                    sig = iisignature.sig(path, depth)
                    
                    # Store signature terms
                    result['signature'] = sig
                    
                    # Use first order terms for trend estimation
                    if depth >= 2 and len(sig) >= 1:
                        trend_values = sig[0] if len(sig.shape) > 1 else sig
                        # Expand to match data length
                        if len(trend_values) < len(data):
                            result['trend_sig'] = np.interp(
                                np.arange(len(data)),
                                np.linspace(0, len(data)-1, len(trend_values)),
                                trend_values
                            )
                        else:
                            result['trend_sig'] = trend_values[:len(data)]
                    else:
                        result['trend_sig'] = np.zeros_like(data)
                
            except ImportError:
                # If signatory and iisignature are not available, use an enhanced approximation with GPU if possible
                if HAS_CUPY:
                    # Move data to GPU
                    data_gpu = cp.asarray(data)
                    
                    # Step 1: Compute increments (first-order features)
                    increments = cp.diff(data_gpu, prepend=data_gpu[0])
                    
                    # Step 2: Compute moving averages of different window sizes
                    window_sizes = [5, 10, 20]
                    moving_avgs = []
                    for w in window_sizes:
                        if len(data_gpu) > w:
                            # Convolve on GPU
                            kernel = cp.ones(w, dtype=cp.float32) / w
                            ma = cp.convolve(data_gpu, kernel, mode='valid')
                            
                            # Pad to match original length
                            padding = len(data_gpu) - len(ma)
                            if padding > 0:
                                ma = cp.pad(ma, (padding, 0), mode='edge')
                            moving_avgs.append(ma)
                    
                    # Step 3: Compute local volatility (second-order features)
                    volatility = []
                    for w in window_sizes:
                        if len(data_gpu) > w:
                            # Calculate rolling standard deviation on GPU
                            vol = cp.zeros_like(data_gpu)
                            for i in range(len(data_gpu)):
                                start_idx = max(0, i-w)
                                vol[i] = cp.std(data_gpu[start_idx:i+1])
                            volatility.append(vol)
                    
                    # Step 4: Compute cumulative sums
                    cumulative_sum = cp.cumsum(increments)
                    cumulative_sq = cp.cumsum(increments**2)
                    cumulative_prod = cp.cumsum(increments * cp.roll(increments, 1))
                    
                    # Combine all features
                    features = [cumulative_sum, cumulative_sq, cumulative_prod]
                    if moving_avgs:
                        features.extend(moving_avgs)
                    if volatility:
                        features.extend(volatility)
                    
                    # Stack all features
                    result['signature_approx'] = cp.vstack(features).T
                    
                    # Create trend indicator from a combination of features
                    if len(moving_avgs) >= 2:
                        trend_indicator = moving_avgs[0] - moving_avgs[-1]
                    else:
                        # Fallback to simple smoothed increments
                        window_size_smooth = min(16, len(increments)//4)
                        kernel = cp.ones(window_size_smooth, dtype=cp.float32) / window_size_smooth
                        trend_indicator = cp.convolve(increments, kernel, mode='same')
                    
                    # Move results back to CPU
                    result['signature_approx'] = cp.asnumpy(result['signature_approx'])
                    result['trend_sig'] = cp.asnumpy(trend_indicator)
                
                else:
                    # Original CPU implementation
                    # Create a more sophisticated approximation of path signatures
                    # Step 1: Compute increments (first-order features)
                    increments = np.diff(data, prepend=data[0])
                    
                    # Step 2: Compute moving averages of different window sizes (capturing trends)
                    window_sizes = [5, 10, 20]
                    moving_avgs = []
                    for w in window_sizes:
                        if len(data) > w:
                            ma = np.convolve(data, np.ones(w)/w, mode='valid')
                            # Pad to match original length
                            padding = len(data) - len(ma)
                            if padding > 0:
                                ma = np.pad(ma, (padding, 0), mode='edge')
                            moving_avgs.append(ma)
            
                    # Step 3: Compute local volatility (second-order features)
                    volatility = []
                    for w in window_sizes:
                        if len(data) > w:
                            vol = np.array([np.std(data[max(0, i-w):i+1]) for i in range(len(data))])
                            volatility.append(vol)
                    
                    # Step 4: Compute cumulative sums (approximating iterated integrals)
                    cumulative_sum = np.cumsum(increments)
                    cumulative_sq = np.cumsum(increments**2)
                    cumulative_prod = np.cumsum(increments * np.roll(increments, 1))
                    
                    # Combine all features
                    features = [cumulative_sum, cumulative_sq, cumulative_prod]
                    if moving_avgs:
                        features.extend(moving_avgs)
                    if volatility:
                        features.extend(volatility)
                    
                    # Stack all features as our signature approximation
                    result['signature_approx'] = np.vstack(features).T
                    
                    # Create trend indicator from a combination of features
                    # Use the difference between short and long-term moving averages
                    if len(moving_avgs) >= 2:
                        trend_indicator = moving_avgs[0] - moving_avgs[-1]
                    else:
                        # Fallback to simple smoothed increments
                        trend_indicator = np.convolve(increments, np.ones(min(16, len(increments)//4))/min(16, len(increments)//4), mode='same')
                    
                    result['trend_sig'] = trend_indicator
            
            # Normalize trend to [-1, 1] range
            if np.max(np.abs(result['trend_sig'])) > 0:
                result['trend_sig'] = result['trend_sig'] / np.max(np.abs(result['trend_sig']))
        
        return result
    
    def update_timeframe_weights(self, df: pd.DataFrame, price_col: str, future_return_col: str = 'future_return'):
        """
        Update weights of different timeframes based on their correlation with future returns.
        
        Args:
            df: DataFrame containing price data and signals
            price_col: Name of the price column
            future_return_col: Name of the future return column
        """
        if not self.adaptive_weights:
            return
        
        if future_return_col not in df.columns:
            # Cannot update weights without future returns
            print("Warning: Cannot update timeframe weights without future returns column")
            return
            
        # Calculate correlations between trend signals and future returns
        correlations = {}
        for window in self.window_sizes:
            col = f'trend_{window}'
            if col in df.columns:
                # Calculate correlation and use absolute value (direction matters, not sign)
                corr = df[col].corr(df[future_return_col])
                if not np.isnan(corr):
                    correlations[window] = abs(corr)
                    self.timeframe_correlations[window] = corr  # Store original correlation for sign
        
        if not correlations:
            return
            
        # Calculate weights based on relative correlation strength
        total_corr = sum(correlations.values())
        if total_corr > 0:
            for window in self.window_sizes:
                if window in correlations:
                    # Set weight proportional to correlation strength
                    self.timeframe_weights[window] = correlations[window] / total_corr
                else:
                    # If no correlation was calculated, give a small weight
                    self.timeframe_weights[window] = 0.1 / len(self.window_sizes)
            
            # Ensure weights sum to 1
            total_weight = sum(self.timeframe_weights.values())
            for window in self.timeframe_weights:
                self.timeframe_weights[window] /= total_weight
                
            print(f"Updated timeframe weights: {self.timeframe_weights}")
    
    def update_ensemble_weights(self, df: pd.DataFrame, future_return_col: str = 'future_return'):
        """
        Update weights of different ensemble methods based on their performance.
        
        Args:
            df: DataFrame containing ensemble predictions
            future_return_col: Name of the future return column
        """
        if future_return_col not in df.columns:
            return
            
        # Calculate correlations between ensemble method signals and future returns
        method_correlations = {}
        for method in self.ensemble_methods:
            col = f'trend_{method}'
            if col in df.columns:
                corr = df[col].corr(df[future_return_col])
                if not np.isnan(corr):
                    method_correlations[method] = abs(corr)
        
        if not method_correlations:
            return
            
        # Calculate weights based on relative correlation strength
        total_corr = sum(method_correlations.values())
        if total_corr > 0:
            for method in self.ensemble_methods:
                if method in method_correlations:
                    self.ensemble_weights[method] = method_correlations[method] / total_corr
                else:
                    self.ensemble_weights[method] = 0.1 / len(self.ensemble_methods)
            
            # Ensure weights sum to 1
            total_weight = sum(self.ensemble_weights.values())
            for method in self.ensemble_weights:
                self.ensemble_weights[method] /= total_weight
                
            print(f"Updated ensemble weights: {self.ensemble_weights}")
            
    def calculate_multi_timeframe_trend(self, df: pd.DataFrame, price_col: str, 
                                       return_all_features: bool = False,
                                       disable_direction_correction: bool = False) -> pd.DataFrame:
        """
        Calculate trend strength across multiple timeframes.
        
        Args:
            df: DataFrame containing price data
            price_col: Name of the price column
            return_all_features: Whether to return all computed features
            disable_direction_correction: If True, will not flip trend direction based on correlation
            
        Returns:
            DataFrame with trend strength features added
        """
        # Create a copy of the DataFrame
        data = df.copy()
        
        # Calculate future price direction to ensure proper alignment
        # Add a 5-period future return to align with trend direction
        future_periods = 5
        if 'future_return' not in data.columns:
            data['future_return'] = data[price_col].pct_change(future_periods).shift(-future_periods)
        
        # Initialize columns for different timeframes
        for window in self.window_sizes:
            data[f'trend_{window}'] = np.nan
        
        # Add columns for different ensemble methods
        for method in self.ensemble_methods:
            data[f'trend_{method}'] = np.nan
            
        # Add columns for combined trend
        data['trend_strength'] = np.nan
        data['trend_agreement'] = np.nan
        data['ensemble_trend'] = np.nan
        
        # Calculate trend for each window size
        for window in self.window_sizes:
            print(f"Processing window size {window}...")
            
            # Create rolling windows
            for i in range(window, len(data)):
                prices = data[price_col].values[i-window:i]
                
                # Skip if window is too small
                if len(prices) < window/2:
                    continue
                
                # Process the price series
                results = self.process_price_series(prices)
                
                # Store the weighted trend for this window
                data.loc[i, f'trend_{window}'] = results['weighted_trend'][-1]
                
                # Store ensemble method trends
                for method in self.ensemble_methods:
                    if f'{method}_trend' in results:
                        data.loc[i, f'trend_{method}'] = results[f'{method}_trend'][-1]
        
        # Update timeframe weights based on correlations with future returns
        self.update_timeframe_weights(data, price_col)
        
        # Update ensemble weights
        self.update_ensemble_weights(data)
        
        # Combine trends from different timeframes with adaptive weights
        timeframe_trends = []
        for window in self.window_sizes:
            if self.adaptive_weights:
                # Apply weight to this timeframe
                timeframe_trends.append(data[f'trend_{window}'].fillna(0) * self.timeframe_weights[window])
            else:
                # Equal weights
                timeframe_trends.append(data[f'trend_{window}'].fillna(0))
        
        # Calculate combined trend strength (weighted average)
        if self.adaptive_weights:
            data['trend_strength'] = sum(timeframe_trends)
        else:
            data['trend_strength'] = np.mean(timeframe_trends, axis=0)
        
        # Calculate trend agreement
        trend_signs = [np.sign(data[f'trend_{window}'].fillna(0)) for window in self.window_sizes]
        # Agreement = |sum of signs| / number of timeframes
        data['trend_agreement'] = np.abs(np.sum(trend_signs, axis=0)) / len(self.window_sizes)
        
        # Weight the combined trend by agreement
        data['weighted_trend_strength'] = data['trend_strength'] * data['trend_agreement']
        
        # Combine ensemble method trends with adaptive weights
        ensemble_trends = []
        for method in self.ensemble_methods:
            method_col = f'trend_{method}'
            if method_col in data.columns:
                if self.adaptive_weights:
                    ensemble_trends.append(data[method_col].fillna(0) * self.ensemble_weights[method])
                else:
                    ensemble_trends.append(data[method_col].fillna(0))
                
        if ensemble_trends:
            if self.adaptive_weights:
                data['ensemble_trend'] = sum(ensemble_trends)
            else:
                data['ensemble_trend'] = np.mean(ensemble_trends, axis=0)
                
            # Calculate final trend as weighted average of timeframe and ensemble trends
            data['final_trend'] = 0.6 * data['weighted_trend_strength'] + 0.4 * data['ensemble_trend']
        else:
            # If no ensemble trends, use weighted trend strength
            data['final_trend'] = data['weighted_trend_strength']
        
        # Ensure trend directionality aligns with actual future price movements
        # Only if not disabled
        if not disable_direction_correction:
            # Calculate correlation between trend and future returns
            valid_mask = ~(data['final_trend'].isna() | data['future_return'].isna())
            if valid_mask.sum() > 10:  # Need enough samples for meaningful correlation
                trend_direction = np.corrcoef(
                    data.loc[valid_mask, 'final_trend'], 
                    data.loc[valid_mask, 'future_return']
                )[0, 1]
                
                # If correlation is negative, flip the sign of all trend features
                if trend_direction < 0:
                    print(f"Correcting trend direction (correlation: {trend_direction:.4f})")
                    for window in self.window_sizes:
                        data[f'trend_{window}'] = -data[f'trend_{window}']
                    
                    for method in self.ensemble_methods:
                        method_col = f'trend_{method}'
                        if method_col in data.columns:
                            data[method_col] = -data[method_col]
                    
                    data['trend_strength'] = -data['trend_strength']
                    data['weighted_trend_strength'] = -data['weighted_trend_strength']
                    data['ensemble_trend'] = -data['ensemble_trend'] if 'ensemble_trend' in data.columns else np.nan
                    data['final_trend'] = -data['final_trend']
                    
                    # Flip the correlations sign for future updates
                    for window in self.timeframe_correlations:
                        self.timeframe_correlations[window] = -self.timeframe_correlations[window]
        
        # Use the final trend as the weighted trend strength for backward compatibility
        data['weighted_trend_strength'] = data['final_trend']
        
        # Fill NaN values with 0
        data.fillna(0, inplace=True)
        
        # Return only the trend columns if not returning all features
        if not return_all_features:
            trend_cols = [
                'trend_strength', 'trend_agreement', 'weighted_trend_strength', 
                'ensemble_trend', 'final_trend'
            ] + [f'trend_{window}' for window in self.window_sizes] + [f'trend_{method}' for method in self.ensemble_methods]
            return data[trend_cols]
        
        return data
    
    def process_price_series(self, prices: np.ndarray, include_methods: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Process a price series with multiple signal extraction methods and return results.
        Uses GPU-accelerated implementations when available.
        
        Args:
            prices: Price series as numpy array
            include_methods: List of methods to include (None = all)
            
        Returns:
            Dictionary containing extracted signals and trends
        """
        result = {}
        
        # Normalize prices to percentage changes
        # Handle potential issue with zero prices in HFT data
        changes = np.zeros_like(prices)
        valid_indices = np.where(prices[:-1] > 0)[0]
        if len(valid_indices) > 0:
            changes[valid_indices + 1] = (prices[valid_indices + 1] - prices[valid_indices]) / prices[valid_indices]
        
        # Check if we should use GPU acceleration
        use_gpu = (HAS_TORCH and torch.cuda.is_available()) or HAS_CUPY
        
        # Apply various signal processing methods
        methods_to_use = include_methods or ['wavelet', 'kalman', 'changepoint']
        
        # Get signals from each method
        method_results = {}
        
        if 'wavelet' in methods_to_use:
            # Wavelet denoising - GPU accelerated if available
            denoised = self.wavelet_denoising(changes)
            wavelet_result = self.wavelet_transform(denoised)
            # Extract trend component (approximation coefficients)
            wavelet_trend = wavelet_result.get('approx', denoised)
            method_results['wavelet'] = {'trend': wavelet_trend}
            
        if 'kalman' in methods_to_use:
            # Kalman filtering - GPU accelerated if available
            kalman_result = self.adaptive_kalman_filter(changes)
            kalman_trend = kalman_result.get('state', changes)
            method_results['kalman'] = {'trend': kalman_trend}
            
        if 'changepoint' in methods_to_use:
            # Changepoint detection - adjust with smaller window for HFT
            changepoint_result = self.changepoint_detection(changes, window_size=min(len(changes)//8, 10))
            changepoint_trend = changepoint_result.get('trend_cp', changes)
            method_results['changepoint'] = {'trend': changepoint_trend}
            
        if 'hht' in methods_to_use:
            # Hilbert-Huang Transform - GPU accelerated for Hilbert transform part
            try:
                hht_result = self.hilbert_huang_transform(changes)
                # Use the sum of the first 2 IMFs as the trend
                hht_trend = np.zeros_like(changes)
                if 'imfs' in hht_result and len(hht_result['imfs']) >= 2:
                    hht_trend = hht_result['imfs'][0] + hht_result['imfs'][1]
                method_results['hht'] = {'trend': hht_trend}
            except:
                # HHT can fail for some price series
                pass
                
        if 'emd' in methods_to_use:
            # Earth Mover's Distance analysis - handle with smaller window for HFT
            try:
                emd_result = self.analyze_with_earth_movers_distance(prices, window_size=min(len(prices)//10, 10))
                emd_trend = emd_result.get('trend_emd', np.zeros_like(changes))
                method_results['emd'] = {'trend': emd_trend}
            except:
                pass
        
        # Combine results from all methods (ensemble approach)
        # GPU-accelerated ensembling if available
        if use_gpu and HAS_TORCH and torch.cuda.is_available():
            # Using PyTorch for GPU computation
            # 1. Extract and combine trends from different methods
            all_trends = [res['trend'] for res in method_results.values() if 'trend' in res]
            
            if all_trends:
                # Move trends to GPU
                gpu_trends = []
                for trend in all_trends:
                    # Handle outliers by capping extreme values
                    trend_tensor = torch.tensor(trend, dtype=torch.float32, device='cuda')
                    # Compute percentiles on GPU
                    sorted_vals, _ = torch.sort(trend_tensor)
                    n = sorted_vals.size(0)
                    p01_idx = max(0, int(0.01 * n))
                    p99_idx = min(n-1, int(0.99 * n))
                    p01, p99 = sorted_vals[p01_idx], sorted_vals[p99_idx]
                    # Cap the values
                    capped = torch.clamp(trend_tensor, p01, p99)
                    gpu_trends.append(capped)
                
                if gpu_trends:
                    # Stack trends for median computation
                    stacked_trends = torch.stack(gpu_trends)
                    # Compute ensemble trend as median (more robust than mean)
                    # PyTorch median along dim 0
                    ensemble_trend = torch.median(stacked_trends, dim=0).values
                else:
                    ensemble_trend = torch.zeros_like(torch.tensor(changes, device='cuda'))
            else:
                # Fallback to simple moving average
                window_size = min(len(changes)//10, 5)
                if window_size > 1:
                    changes_tensor = torch.tensor(changes, device='cuda')
                    # Use 1D convolution for moving average
                    kernel = torch.ones(1, 1, window_size, device='cuda') / window_size
                    padded = torch.nn.functional.pad(changes_tensor.view(1, 1, -1), (window_size//2, window_size//2), mode='replicate')
                    ensemble_trend = torch.nn.functional.conv1d(padded, kernel).view(-1)
                else:
                    ensemble_trend = torch.tensor(changes, device='cuda')
            
            # Apply exponential smoothing for continuity
            alpha = 0.3  # Smoothing factor
            smoothed_trend = torch.zeros_like(ensemble_trend)
            smoothed_trend[0] = ensemble_trend[0]
            
            # Compute standard deviation for outlier detection
            trend_std = torch.std(ensemble_trend)
            
            # Apply smoothing with outlier handling
            for i in range(1, len(ensemble_trend)):
                # Skip extreme jumps that might be data errors in HFT
                if torch.abs(ensemble_trend[i] - smoothed_trend[i-1]) > 5 * trend_std:
                    smoothed_trend[i] = smoothed_trend[i-1]
                else:
                    smoothed_trend[i] = alpha * ensemble_trend[i] + (1 - alpha) * smoothed_trend[i-1]
            
            # Normalize strength to [-1, 1]
            max_abs = torch.max(torch.abs(smoothed_trend))
            if max_abs > 0:
                strength = smoothed_trend / max_abs
            else:
                strength = smoothed_trend
            
            # Move results back to CPU
            result['ensemble_trend'] = ensemble_trend.cpu().numpy()
            result['weighted_trend'] = strength.cpu().numpy()
            
            # Store individual method trends
            for method, res in method_results.items():
                if 'trend' in res:
                    result[f'{method}_trend'] = res['trend']
        
        elif use_gpu and HAS_CUPY:
            # Using CuPy for GPU computation
            # Similar approach as PyTorch implementation
            all_trends = [res['trend'] for res in method_results.values() if 'trend' in res]
            
            if all_trends:
                # Move trends to GPU and handle outliers
                gpu_trends = []
                for trend in all_trends:
                    trend_gpu = cp.asarray(trend)
                    # Compute percentiles and clamp values
                    p01, p99 = cp.percentile(trend_gpu, [1, 99])
                    capped = cp.clip(trend_gpu, p01, p99)
                    gpu_trends.append(capped)
                
                if gpu_trends:
                    # Compute median across trends
                    stacked_trends = cp.stack(gpu_trends)
                    ensemble_trend = cp.median(stacked_trends, axis=0)
                else:
                    ensemble_trend = cp.zeros_like(cp.asarray(changes))
            else:
                # Fallback to simple moving average
                window_size = min(len(changes)//10, 5)
                if window_size > 1:
                    changes_gpu = cp.asarray(changes)
                    kernel = cp.ones(window_size, dtype=cp.float32) / window_size
                    ensemble_trend = cp.convolve(changes_gpu, kernel, mode='same')
                else:
                    ensemble_trend = cp.asarray(changes)
            
            # Apply exponential smoothing
            alpha = 0.3
            smoothed_trend = cp.zeros_like(ensemble_trend)
            smoothed_trend[0] = ensemble_trend[0]
            
            # Compute std for outlier detection
            trend_std = cp.std(ensemble_trend)
            
            # Apply smoothing with outlier handling
            for i in range(1, len(ensemble_trend)):
                if cp.abs(ensemble_trend[i] - smoothed_trend[i-1]) > 5 * trend_std:
                    smoothed_trend[i] = smoothed_trend[i-1]
                else:
                    smoothed_trend[i] = alpha * ensemble_trend[i] + (1 - alpha) * smoothed_trend[i-1]
            
            # Normalize strength
            max_abs = cp.max(cp.abs(smoothed_trend))
            if max_abs > 0:
                strength = smoothed_trend / max_abs
            else:
                strength = smoothed_trend
            
            # Move results back to CPU
            result['ensemble_trend'] = cp.asnumpy(ensemble_trend)
            result['weighted_trend'] = cp.asnumpy(strength)
            
            # Store individual method trends
            for method, res in method_results.items():
                if 'trend' in res:
                    result[f'{method}_trend'] = res['trend']
            
        else:
            # Original CPU implementation
            # 1. Average the trends from different methods
            all_trends = [res['trend'] for res in method_results.values() if 'trend' in res]
            if all_trends:
                # Handle outliers in HFT data by winsorizing extreme values
                capped_trends = []
                for trend in all_trends:
                    if len(trend) > 0:
                        # Cap extreme values at 99th percentile
                        p01, p99 = np.nanpercentile(trend, [1, 99])
                        capped = np.clip(trend, p01, p99)
                        capped_trends.append(capped)
                    else:
                        capped_trends.append(trend)
                
                if capped_trends:
                    # Compute ensemble trend as median instead of mean for robustness
                    ensemble_trend = np.nanmedian(capped_trends, axis=0)
                else:
                    ensemble_trend = np.zeros_like(changes)
            else:
                # Fallback to simple moving average for robustness
                window_size = min(len(changes)//10, 5)
                if window_size > 1:
                    ensemble_trend = np.convolve(changes, np.ones(window_size)/window_size, mode='same')
                else:
                    ensemble_trend = changes.copy()
            
            # Store individual method trends
            for method, res in method_results.items():
                if 'trend' in res:
                    result[f'{method}_trend'] = res['trend']
            
            # Calculate weighted trend
            # Apply robust exponential smoothing for continuity
            alpha = 0.3  # Smoothing factor
            smoothed_trend = np.zeros_like(ensemble_trend)
            smoothed_trend[0] = ensemble_trend[0]
            for i in range(1, len(ensemble_trend)):
                # Skip extreme jumps that might be data errors in HFT
                if np.abs(ensemble_trend[i] - smoothed_trend[i-1]) > 5 * np.nanstd(ensemble_trend):
                    smoothed_trend[i] = smoothed_trend[i-1]
                else:
                    smoothed_trend[i] = alpha * ensemble_trend[i] + (1 - alpha) * smoothed_trend[i-1]
            
            # Calculate trend strength (normalized)
            strength = smoothed_trend
            if len(strength) > 0 and np.nanmax(np.abs(strength)) > 0:
                strength = strength / np.nanmax(np.abs(strength))
            
            # Store trends without volatility estimate
            result['ensemble_trend'] = ensemble_trend
            result['weighted_trend'] = strength
        
        return result
    
    def analyze_with_earth_movers_distance(self, prices, window_size=20, min_window_size=5):
        """
        Analyze time series using Earth Mover's Distance (pyemd package).
        Optimized for HFT data with adaptable window size and robust error handling.
        
        Args:
            prices: Input price series
            window_size: Maximum size of sliding window for comparison (will adapt for HFT)
            min_window_size: Minimum window size to use
            
        Returns:
            Dictionary with trend indicators based on EMD
        """
        try:
            import pyemd
            from pyemd import emd
        except ImportError:
            print("Warning: pyemd not available. Install with: pip install PyEMD")
            return {'trend_emd': np.zeros_like(prices)}
            
        result = {}
        n = len(prices)
        
        # Adapt window size for HFT (smaller windows)
        # For very short time series, use smaller windows
        adaptive_window = min(window_size, max(min_window_size, n // 20))
        
        # Need at least 2 windows
        if n < 2 * adaptive_window:
            result['trend_emd'] = np.zeros_like(prices)
            return result
        
        # Initialize divergence array
        divergence = np.zeros(n)
        
        # Calculate for each point that has enough history
        for i in range(adaptive_window, n - adaptive_window + 1):
            # Get current and previous windows
            current_window = prices[i:i+adaptive_window]
            previous_window = prices[i-adaptive_window:i]
            
            try:
                # Check if windows have enough variability for meaningful histograms
                # HFT data often has minimal price movement, so use a relative threshold
                current_range = np.ptp(current_window)
                previous_range = np.ptp(previous_window)
                min_range_threshold = 1e-8 * np.mean(prices)
                
                if current_range < min_range_threshold or previous_range < min_range_threshold:
                    # Not enough variability, skip this window
                    continue
                
                # Filter out potential bad data points in HFT
                # Replace extreme outliers with median value
                current_median = np.median(current_window)
                previous_median = np.median(previous_window)
                current_mad = np.median(np.abs(current_window - current_median))
                previous_mad = np.median(np.abs(previous_window - previous_median))
                
                # Use median absolute deviation for robustness
                current_outlier_mask = np.abs(current_window - current_median) > 10 * current_mad
                previous_outlier_mask = np.abs(previous_window - previous_median) > 10 * previous_mad
                
                # Create clean windows
                clean_current = current_window.copy()
                clean_previous = previous_window.copy()
                
                if np.any(current_outlier_mask):
                    clean_current[current_outlier_mask] = current_median
                if np.any(previous_outlier_mask):
                    clean_previous[previous_outlier_mask] = previous_median
                
                # Add small noise to avoid identical values (common in HFT tick data)
                # Scale noise to price level to avoid introducing false patterns
                noise_scale = min_range_threshold / 10
                clean_current = clean_current + np.random.normal(0, noise_scale, size=len(clean_current))
                clean_previous = clean_previous + np.random.normal(0, noise_scale, size=len(clean_previous))
                
                # Calculate histograms with equal bin sizes across both windows
                min_val = min(np.min(clean_current), np.min(clean_previous))
                max_val = max(np.max(clean_current), np.max(clean_previous))
                
                # Ensure bins have a reasonable range
                if max_val - min_val < min_range_threshold:
                    continue
                
                # Use fewer bins for HFT data to avoid sparsity
                bin_count = min(8, adaptive_window // 2)
                bins = np.linspace(min_val, max_val, bin_count + 1)  
                
                # Calculate histograms
                current_hist, _ = np.histogram(clean_current, bins=bins)
                previous_hist, _ = np.histogram(clean_previous, bins=bins)
                
                # Ensure there's at least one count in each histogram to avoid div by zero
                if np.sum(current_hist) == 0:
                    current_hist[bin_count // 2] = 1
                if np.sum(previous_hist) == 0:
                    previous_hist[bin_count // 2] = 1
                    
                # Normalize histograms
                current_hist = current_hist.astype(float) / np.sum(current_hist)
                previous_hist = previous_hist.astype(float) / np.sum(previous_hist)
                
                # Create distance matrix - scale to price level
                bin_centers = (bins[:-1] + bins[1:]) / 2
                n_bins = len(bin_centers)
                distance_matrix = np.zeros((n_bins, n_bins))
                
                price_scale = max(1e-8, np.median(prices))
                for j in range(n_bins):
                    for k in range(n_bins):
                        # Normalize distance by price scale for HFT
                        distance_matrix[j, k] = abs(bin_centers[j] - bin_centers[k]) / price_scale
                
                # Calculate EMD
                emd_value = emd(previous_hist, current_hist, distance_matrix)
                divergence[i] = emd_value
            except Exception as e:
                # Skip calculation for this window if there's an error
                continue
        
        # Calculate trend direction using robust linear regression on windows
        trend_direction = np.zeros(n)
        for i in range(adaptive_window, n):
            window = prices[i-adaptive_window:i]
            x = np.arange(len(window))
            try:
                # Use Theil-Sen estimator for robustness against outliers in HFT
                try:
                    from scipy import stats
                    # Theil-Sen estimator is robust to outliers
                    slope = stats.theilslopes(window, x)[0]
                except (ImportError, ValueError):
                    # Fallback to regular polyfit if scipy not available or data issues
                    slope = np.polyfit(x, window, 1)[0]
                
                # Scale slope by median price for better comparability in HFT
                scale_factor = max(1e-8, np.median(window))
                trend_direction[i] = slope / scale_factor
            except Exception:
                pass
        
        # Normalize trend direction to [-1, 1] with robust scaling
        abs_trend = np.abs(trend_direction)
        robust_max = np.percentile(abs_trend[abs_trend > 0], 95) if np.any(abs_trend > 0) else 1
        
        if robust_max > 0:
            trend_direction = np.clip(trend_direction / robust_max, -1, 1)
        
        # Combine EMD divergence with trend direction
        # High EMD in the direction of trend = strong trend
        trend_emd = divergence * np.abs(trend_direction) * np.sign(trend_direction)
        
        # Smooth the result with robust smoothing for HFT
        window_size_smooth = min(adaptive_window//2, 3)
        if window_size_smooth > 1:
            # Use median filter instead of mean for robustness against outliers
            try:
                from scipy import signal
                trend_emd = signal.medfilt(trend_emd, window_size_smooth)
            except ImportError:
                # Fallback to simple moving average if scipy not available
                trend_emd = np.convolve(trend_emd, np.ones(window_size_smooth)/window_size_smooth, mode='same')
        
        # Normalize to [-1, 1] using robust scaling
        abs_trend_emd = np.abs(trend_emd)
        robust_max_emd = np.percentile(abs_trend_emd[abs_trend_emd > 0], 95) if np.any(abs_trend_emd > 0) else 1
        
        if robust_max_emd > 0:
            trend_emd = np.clip(trend_emd / robust_max_emd, -1, 1)
        
        # Store results, excluding intermediate calculations for HFT efficiency
        result['trend_emd'] = trend_emd
        
        return result

def extract_advanced_features(df: pd.DataFrame, price_col: str, 
                              window_sizes: List[int] = [8, 16, 32, 64]) -> pd.DataFrame:
    """
    Extract advanced features from price data using multiple signal processing methods.
    
    Args:
        df: DataFrame containing price data
        price_col: Name of the price column
        window_sizes: List of window sizes for multi-scale analysis
        
    Returns:
        DataFrame with added trend features
    """
    # Create SignalProcessor instance
    processor = SignalProcessor(window_sizes=window_sizes)
    
    # Calculate trend strength across multiple timeframes
    trend_features = processor.calculate_multi_timeframe_trend(df, price_col)
    
    # Add trend features to original DataFrame
    result = pd.concat([df, trend_features], axis=1)
    
    return result

def benchmark_gpu_vs_cpu(data_size=10000, iterations=10, methods=None):
    """
    Benchmark GPU vs CPU performance for signal processing tasks.
    
    Args:
        data_size: Size of the test data
        iterations: Number of iterations for reliable timing
        methods: List of methods to benchmark (default: all)
        
    Returns:
        DataFrame with benchmark results
    """
    import time
    import pandas as pd
    
    if methods is None:
        methods = ['wavelet_denoising', 'wavelet_transform', 'adaptive_kalman_filter', 'path_signatures']
    
    # Generate synthetic data
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, data_size))
    
    results = []
    
    # Create processors for CPU and GPU
    processor_cpu = SignalProcessor(use_gpu=False)
    
    if HAS_TORCH and torch.cuda.is_available():
        processor_gpu = SignalProcessor(use_gpu=True)
        gpu_type = "PyTorch CUDA"
    elif HAS_CUPY:
        processor_gpu = SignalProcessor(use_gpu=True)
        gpu_type = "CuPy"
    else:
        processor_gpu = None
        gpu_type = "Not Available"
    
    for method in methods:
        if method == 'wavelet_denoising':
            # Benchmark wavelet denoising
            if processor_gpu is not None:
                # Warm-up
                processor_gpu.wavelet_denoising(prices)
                
                # Measure GPU time
                start = time.time()
                for _ in range(iterations):
                    processor_gpu.wavelet_denoising(prices)
                gpu_time = (time.time() - start) / iterations
            else:
                gpu_time = float('nan')
            
            # Measure CPU time
            start = time.time()
            for _ in range(iterations):
                processor_cpu.wavelet_denoising(prices)
            cpu_time = (time.time() - start) / iterations
            
        elif method == 'wavelet_transform':
            # Benchmark wavelet transform
            if processor_gpu is not None:
                # Warm-up
                processor_gpu.wavelet_transform(prices)
                
                # Measure GPU time
                start = time.time()
                for _ in range(iterations):
                    processor_gpu.wavelet_transform(prices)
                gpu_time = (time.time() - start) / iterations
            else:
                gpu_time = float('nan')
            
            # Measure CPU time
            start = time.time()
            for _ in range(iterations):
                processor_cpu.wavelet_transform(prices)
            cpu_time = (time.time() - start) / iterations
            
        elif method == 'adaptive_kalman_filter':
            # Benchmark Kalman filter
            if processor_gpu is not None:
                # Warm-up
                processor_gpu.adaptive_kalman_filter(prices)
                
                # Measure GPU time
                start = time.time()
                for _ in range(iterations):
                    processor_gpu.adaptive_kalman_filter(prices)
                gpu_time = (time.time() - start) / iterations
            else:
                gpu_time = float('nan')
            
            # Measure CPU time
            start = time.time()
            for _ in range(iterations):
                processor_cpu.adaptive_kalman_filter(prices)
            cpu_time = (time.time() - start) / iterations
            
        elif method == 'path_signatures':
            # Benchmark path signatures
            if processor_gpu is not None:
                # Warm-up
                processor_gpu.path_signatures(prices)
                
                # Measure GPU time
                start = time.time()
                for _ in range(iterations):
                    processor_gpu.path_signatures(prices)
                gpu_time = (time.time() - start) / iterations
            else:
                gpu_time = float('nan')
            
            # Measure CPU time
            start = time.time()
            for _ in range(iterations):
                processor_cpu.path_signatures(prices)
            cpu_time = (time.time() - start) / iterations
            
        # Calculate speedup
        if not np.isnan(gpu_time) and gpu_time > 0:
            speedup = cpu_time / gpu_time
        else:
            speedup = float('nan')
            
        results.append({
            'Method': method,
            'CPU Time (s)': cpu_time,
            'GPU Time (s)': gpu_time,
            'GPU Type': gpu_type,
            'Speedup': speedup,
            'Data Size': data_size
        })
    
    return pd.DataFrame(results)

def batch_process_with_gpu(df, price_col, processor=None, batch_size=1000, overlap=100):
    """
    Process a large DataFrame in batches using GPU acceleration.
    This allows processing larger datasets than would fit in GPU memory.
    
    Args:
        df: DataFrame to process
        price_col: Name of the price column
        processor: SignalProcessor instance (or None to create a new one)
        batch_size: Size of each batch
        overlap: Overlap between consecutive batches for continuity
        
    Returns:
        DataFrame with added trend features
    """
    # Create processor if not provided
    if processor is None:
        processor = SignalProcessor(use_gpu=True)
    
    # Create a copy of the DataFrame to hold results
    result_df = df.copy()
    
    # Initialize trend columns
    trend_cols = [
        'trend_strength', 'trend_agreement', 'weighted_trend_strength', 
        'ensemble_trend', 'final_trend'
    ] + [f'trend_{window}' for window in processor.window_sizes]
    
    for col in trend_cols:
        result_df[col] = np.nan
    
    # Process in batches
    total_rows = len(df)
    for start_idx in range(0, total_rows, batch_size - overlap):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        print(f"Processing batch {start_idx}:{end_idx} ({end_idx-start_idx} rows)")
        
        # Process this batch
        batch_results = processor.calculate_multi_timeframe_trend(batch_df, price_col)
        
        # If this is not the first batch, exclude the overlapping part at the beginning
        if start_idx > 0:
            # Skip the overlapping rows we've already processed
            valid_start = overlap
            # Update only the new rows
            result_df.iloc[start_idx+valid_start:end_idx, result_df.columns.get_indexer(trend_cols)] = \
                batch_results.iloc[valid_start:, batch_results.columns.get_indexer(trend_cols)].values
        else:
            # For the first batch, use all results
            result_df.iloc[start_idx:end_idx, result_df.columns.get_indexer(trend_cols)] = \
                batch_results.iloc[:, batch_results.columns.get_indexer(trend_cols)].values
        
        # Free memory
        del batch_df
        del batch_results
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif HAS_CUPY:
            cp._default_memory_pool.free_all_blocks()
    
    return result_df

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Advanced Signal Processing")
    parser.add_argument("--data_path", type=str, help="Path to the CSV data file")
    parser.add_argument("--test_only", action="store_true", help="Only test preprocessing without model training")
    parser.add_argument("--disable_direction_correction", action="store_true", 
                        help="Disable automatic correction of trend direction")
    parser.add_argument("--benchmark", action="store_true", help="Run GPU vs CPU benchmark")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing large datasets")
    parser.add_argument("--disable_gpu", action="store_true", help="Disable GPU acceleration")
    args = parser.parse_args()
    
    if args.benchmark:
        # Run benchmark
        print("Running GPU vs CPU benchmark...")
        results = benchmark_gpu_vs_cpu(data_size=5000, iterations=5)
        print(results)
        
        # Plot benchmark results
        plt.figure(figsize=(10, 6))
        plt.bar(results['Method'] + ' (CPU)', results['CPU Time (s)'], alpha=0.7, label='CPU')
        plt.bar(results['Method'] + ' (GPU)', results['GPU Time (s)'], alpha=0.7, label='GPU')
        plt.ylabel('Time (seconds)')
        plt.title('GPU vs CPU Performance Comparison')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig("benchmark_results.png")
        plt.close()
        
        print(f"Benchmark results saved to benchmark_results.png")
    
    if args.data_path:
        try:
            # Load data
            df = pd.read_csv(args.data_path)
            
            # Get the price column (typically 'close' or similar)
            price_cols = [col for col in df.columns if col.lower() in ['close', 'price', 'last', 'value']]
            if not price_cols:
                raise ValueError("No price column found in data. Expected 'close', 'price', 'last', or 'value'")
            price_col = price_cols[0]
            
            print(f"Processing {len(df)} samples from {args.data_path}")
            print(f"Using '{price_col}' as price column")
            print(f"Direction correction: {'disabled' if args.disable_direction_correction else 'enabled'}")
            print(f"GPU acceleration: {'disabled' if args.disable_gpu else 'enabled'}")
            
            use_gpu = not args.disable_gpu and ((HAS_TORCH and torch.cuda.is_available()) or HAS_CUPY)
            
            # Create signal processor
            processor = SignalProcessor(use_gpu=use_gpu)
            
            if len(df) > args.batch_size and use_gpu:
                # Use batch processing for large datasets
                print(f"Using batch processing with size {args.batch_size}")
                result = batch_process_with_gpu(
                    df, price_col, processor=processor, 
                    batch_size=args.batch_size, overlap=100
                )
            else:
                # Extract features
                result = processor.calculate_multi_timeframe_trend(
                    df, price_col, 
                    disable_direction_correction=args.disable_direction_correction
                )
            
            print(f"Processed {len(result)} samples successfully")
            print("Trend columns:", [col for col in result.columns if 'trend' in col.lower()])
            
            if not args.test_only:
                # Plot results
                plt.figure(figsize=(15, 10))
                
                # Plot 1: Price and trend strength
                plt.subplot(2, 1, 1)
                plt.plot(df[price_col], 'b-', label='Price')
                plt.title('Price Data')
                plt.legend()
                
                # Plot 2: Trend strength
                plt.subplot(2, 1, 2)
                plt.plot(result['trend_strength'], 'g-', label='Trend Strength')
                plt.plot(result['weighted_trend_strength'], 'r-', label='Weighted Trend Strength')
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                plt.title('Trend Strength Indicators')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig("trend_analysis_results.png")
                plt.close()
                
                print(f"Features extracted and saved to trend_analysis_results.png")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc() 