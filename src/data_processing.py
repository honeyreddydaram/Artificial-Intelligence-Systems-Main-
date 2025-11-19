"""
ECG Data Processing Module
Comprehensive signal processing functions for ECG analysis including noise removal,
filtering, baseline correction, and signal quality assessment.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, medfilt, savgol_filter
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SIGNAL PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_ecg_signal(ecg_signal, fs=360, lowcut=0.5, highcut=40):
    """
    Complete ECG signal preprocessing pipeline.
    
    Parameters:
    -----------
    ecg_signal : array_like
        Raw ECG signal
    fs : int
        Sampling frequency in Hz
    lowcut : float
        Low cutoff frequency for bandpass filter
    highcut : float
        High cutoff frequency for bandpass filter
        
    Returns:
    --------
    processed_signal : numpy.ndarray
        Preprocessed ECG signal
    """
    
    try:
        # Step 1: Convert to numpy array and handle NaN values
        signal_array = np.array(ecg_signal, dtype=np.float64)
        
        # Remove NaN and infinite values
        if np.any(np.isnan(signal_array)) or np.any(np.isinf(signal_array)):
            signal_array = interpolate_missing_values(signal_array)
        
        # Step 2: Remove DC offset
        signal_array = signal_array - np.mean(signal_array)
        
        # Step 3: Apply bandpass filter
        filtered_signal = bandpass_filter(signal_array, lowcut, highcut, fs)
        
        # Step 4: Baseline correction
        corrected_signal = baseline_correction(filtered_signal, fs)
        
        # Step 5: Normalize signal
        normalized_signal = normalize_signal(corrected_signal)
        
        return normalized_signal
        
    except Exception as e:
        print(f"Error in signal preprocessing: {e}")
        return ecg_signal

def remove_noise_artifacts(ecg_signal, fs=360, powerline_freq=50):
    """
    Remove various types of noise and artifacts from ECG signal.
    
    Parameters:
    -----------
    ecg_signal : array_like
        ECG signal to clean
    fs : int
        Sampling frequency in Hz
    powerline_freq : float
        Powerline interference frequency (50 or 60 Hz)
        
    Returns:
    --------
    clean_signal : numpy.ndarray
        Cleaned ECG signal
    """
    
    try:
        signal_array = np.array(ecg_signal, dtype=np.float64)
        
        # Step 1: Remove powerline interference
        signal_array = notch_filter(signal_array, powerline_freq, fs)
        
        # Step 2: Remove baseline wander
        signal_array = remove_baseline_wander(signal_array, fs)
        
        # Step 3: Remove muscle artifacts (EMG noise)
        signal_array = remove_muscle_artifacts(signal_array, fs)
        
        # Step 4: Remove electrode motion artifacts
        signal_array = remove_motion_artifacts(signal_array, fs)
        
        # Step 5: Median filter for impulse noise
        signal_array = medfilt(signal_array, kernel_size=3)
        
        return signal_array
        
    except Exception as e:
        print(f"Error in noise removal: {e}")
        return ecg_signal

def bandpass_filter(ecg_signal, lowcut=0.5, highcut=40, fs=360, order=4):
    """
    Apply bandpass filter to ECG signal.
    
    Parameters:
    -----------
    ecg_signal : array_like
        Input ECG signal
    lowcut : float
        Low cutoff frequency
    highcut : float
        High cutoff frequency
    fs : int
        Sampling frequency
    order : int
        Filter order
        
    Returns:
    --------
    filtered_signal : numpy.ndarray
        Bandpass filtered signal
    """
    
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='band')
        
        # Apply zero-phase filtering
        filtered_signal = filtfilt(b, a, ecg_signal)
        
        return filtered_signal
        
    except Exception as e:
        print(f"Error in bandpass filtering: {e}")
        return ecg_signal

def notch_filter(ecg_signal, freq=50, fs=360, quality=30):
    """
    Apply notch filter to remove powerline interference.
    
    Parameters:
    -----------
    ecg_signal : array_like
        Input ECG signal
    freq : float
        Frequency to notch out (50 or 60 Hz)
    fs : int
        Sampling frequency
    quality : float
        Quality factor
        
    Returns:
    --------
    filtered_signal : numpy.ndarray
        Notch filtered signal
    """
    
    try:
        nyquist = fs / 2
        freq_norm = freq / nyquist
        
        # Design notch filter
        b, a = iirnotch(freq_norm, quality)
        
        # Apply filter
        filtered_signal = filtfilt(b, a, ecg_signal)
        
        return filtered_signal
        
    except Exception as e:
        print(f"Error in notch filtering: {e}")
        return ecg_signal

def baseline_correction(ecg_signal, fs=360, cutoff=0.8):
    """
    Remove baseline wander using high-pass filtering.
    
    Parameters:
    -----------
    ecg_signal : array_like
        Input ECG signal
    fs : int
        Sampling frequency
    cutoff : float
        High-pass cutoff frequency
        
    Returns:
    --------
    corrected_signal : numpy.ndarray
        Baseline corrected signal
    """
    
    try:
        nyquist = 0.5 * fs
        high = cutoff / nyquist
        
        # Design high-pass filter
        b, a = butter(3, high, btype='high')
        
        # Apply filter
        corrected_signal = filtfilt(b, a, ecg_signal)
        
        return corrected_signal
        
    except Exception as e:
        print(f"Error in baseline correction: {e}")
        return ecg_signal

# ============================================================================
# SPECIALIZED NOISE REMOVAL FUNCTIONS
# ============================================================================

def remove_baseline_wander(ecg_signal, fs=360):
    """
    Remove baseline wander using median filtering approach.
    
    Parameters:
    -----------
    ecg_signal : array_like
        Input ECG signal
    fs : int
        Sampling frequency
        
    Returns:
    --------
    corrected_signal : numpy.ndarray
        Signal with baseline wander removed
    """
    
    try:
        # Use median filter with window size of ~600ms
        window_size = int(0.6 * fs)
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        
        # Estimate baseline using median filter
        baseline = medfilt(ecg_signal, kernel_size=window_size)
        
        # Remove baseline
        corrected_signal = ecg_signal - baseline
        
        return corrected_signal
        
    except Exception as e:
        print(f"Error in baseline wander removal: {e}")
        return ecg_signal

def remove_muscle_artifacts(ecg_signal, fs=360):
    """
    Remove muscle artifacts (EMG noise) using Savitzky-Golay filter.
    
    Parameters:
    -----------
    ecg_signal : array_like
        Input ECG signal
    fs : int
        Sampling frequency
        
    Returns:
    --------
    clean_signal : numpy.ndarray
        Signal with muscle artifacts reduced
    """
    
    try:
        # Apply Savitzky-Golay filter to smooth high-frequency noise
        window_length = int(0.04 * fs)  # 40ms window
        if window_length % 2 == 0:
            window_length += 1
        
        # Ensure minimum window length
        window_length = max(window_length, 5)
        polyorder = min(3, window_length - 1)
        
        clean_signal = savgol_filter(ecg_signal, window_length, polyorder)
        
        return clean_signal
        
    except Exception as e:
        print(f"Error in muscle artifact removal: {e}")
        return ecg_signal

def remove_motion_artifacts(ecg_signal, fs=360):
    """
    Remove motion artifacts using adaptive filtering.
    
    Parameters:
    -----------
    ecg_signal : array_like
        Input ECG signal
    fs : int
        Sampling frequency
        
    Returns:
    --------
    clean_signal : numpy.ndarray
        Signal with motion artifacts reduced
    """
    
    try:
        # Use moving average to detect slow variations
        window_size = int(1.0 * fs)  # 1 second window
        
        # Calculate moving average
        moving_avg = uniform_filter1d(ecg_signal, size=window_size, mode='nearest')
        
        # Remove slow variations
        clean_signal = ecg_signal - moving_avg
        
        return clean_signal
        
    except Exception as e:
        print(f"Error in motion artifact removal: {e}")
        return ecg_signal

# ============================================================================
# SIGNAL QUALITY ASSESSMENT
# ============================================================================

def signal_quality_assessment(ecg_signal, fs=360):
    """
    Assess ECG signal quality using multiple metrics.
    
    Parameters:
    -----------
    ecg_signal : array_like
        ECG signal to assess
    fs : int
        Sampling frequency
        
    Returns:
    --------
    quality_metrics : dict
        Dictionary containing quality metrics
    """
    
    try:
        quality_metrics = {}
        
        # 1. Signal-to-noise ratio estimation
        signal_power = np.mean(ecg_signal**2)
        
        # Estimate noise using high-frequency components
        highpass_signal = bandpass_filter(ecg_signal, 20, fs/2-1, fs)
        noise_power = np.mean(highpass_signal**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            quality_metrics['snr_db'] = snr
        else:
            quality_metrics['snr_db'] = float('inf')
        
        # 2. Baseline stability
        baseline_variation = np.std(np.diff(ecg_signal))
        quality_metrics['baseline_stability'] = baseline_variation
        
        # 3. Signal saturation check
        max_value = np.max(np.abs(ecg_signal))
        saturation_percentage = np.sum(np.abs(ecg_signal) > 0.95 * max_value) / len(ecg_signal) * 100
        quality_metrics['saturation_percentage'] = saturation_percentage
        
        # 4. Flat line detection
        flat_threshold = 0.01 * np.std(ecg_signal)
        flat_samples = np.sum(np.abs(np.diff(ecg_signal)) < flat_threshold)
        flat_percentage = flat_samples / len(ecg_signal) * 100
        quality_metrics['flat_line_percentage'] = flat_percentage
        
        # 5. Frequency domain analysis
        freqs, psd = signal.welch(ecg_signal, fs, nperseg=min(1024, len(ecg_signal)//4))
        
        # Power in QRS frequency band (5-15 Hz)
        qrs_band_mask = (freqs >= 5) & (freqs <= 15)
        qrs_power = np.sum(psd[qrs_band_mask])
        
        # Total power
        total_power = np.sum(psd)
        
        # QRS power ratio
        if total_power > 0:
            qrs_power_ratio = qrs_power / total_power
            quality_metrics['qrs_power_ratio'] = qrs_power_ratio
        else:
            quality_metrics['qrs_power_ratio'] = 0
        
        # 6. Overall quality score (0-100)
        quality_score = calculate_overall_quality_score(quality_metrics)
        quality_metrics['overall_quality_score'] = quality_score
        
        return quality_metrics
        
    except Exception as e:
        print(f"Error in signal quality assessment: {e}")
        return {}

def calculate_overall_quality_score(quality_metrics):
    """
    Calculate overall signal quality score from individual metrics.
    
    Parameters:
    -----------
    quality_metrics : dict
        Individual quality metrics
        
    Returns:
    --------
    quality_score : float
        Overall quality score (0-100)
    """
    
    try:
        score = 100  # Start with perfect score
        
        # Penalize based on SNR
        snr = quality_metrics.get('snr_db', 20)
        if snr < 10:
            score -= 30
        elif snr < 15:
            score -= 15
        elif snr < 20:
            score -= 5
        
        # Penalize based on saturation
        saturation = quality_metrics.get('saturation_percentage', 0)
        if saturation > 5:
            score -= 40
        elif saturation > 1:
            score -= 20
        
        # Penalize based on flat lines
        flat_line = quality_metrics.get('flat_line_percentage', 0)
        if flat_line > 10:
            score -= 30
        elif flat_line > 5:
            score -= 15
        
        # Penalize based on QRS power ratio
        qrs_ratio = quality_metrics.get('qrs_power_ratio', 0.3)
        if qrs_ratio < 0.1:
            score -= 25
        elif qrs_ratio < 0.2:
            score -= 10
        
        # Ensure score is between 0 and 100
        quality_score = max(0, min(100, score))
        
        return quality_score
        
    except Exception as e:
        print(f"Error in quality score calculation: {e}")
        return 50  # Return neutral score on error

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def interpolate_missing_values(signal_array):
    """
    Interpolate missing values (NaN, inf) in signal.
    
    Parameters:
    -----------
    signal_array : numpy.ndarray
        Signal with potential missing values
        
    Returns:
    --------
    interpolated_signal : numpy.ndarray
        Signal with interpolated values
    """
    
    try:
        # Find invalid values
        invalid_mask = np.isnan(signal_array) | np.isinf(signal_array)
        
        if np.any(invalid_mask):
            # Get valid indices
            valid_indices = np.where(~invalid_mask)[0]
            invalid_indices = np.where(invalid_mask)[0]
            
            if len(valid_indices) > 1:
                # Interpolate missing values
                interpolated_values = np.interp(invalid_indices, valid_indices, signal_array[valid_indices])
                signal_array[invalid_mask] = interpolated_values
            else:
                # If too few valid values, replace with zeros
                signal_array[invalid_mask] = 0
        
        return signal_array
        
    except Exception as e:
        print(f"Error in value interpolation: {e}")
        return signal_array

def normalize_signal(ecg_signal, method='zscore'):
    """
    Normalize ECG signal using specified method.
    
    Parameters:
    -----------
    ecg_signal : array_like
        Input ECG signal
    method : str
        Normalization method ('zscore', 'minmax', 'robust')
        
    Returns:
    --------
    normalized_signal : numpy.ndarray
        Normalized signal
    """
    
    try:
        signal_array = np.array(ecg_signal)
        
        if method == 'zscore':
            # Z-score normalization
            mean_val = np.mean(signal_array)
            std_val = np.std(signal_array)
            if std_val > 0:
                normalized_signal = (signal_array - mean_val) / std_val
            else:
                normalized_signal = signal_array - mean_val
                
        elif method == 'minmax':
            # Min-max normalization
            min_val = np.min(signal_array)
            max_val = np.max(signal_array)
            if max_val > min_val:
                normalized_signal = (signal_array - min_val) / (max_val - min_val)
            else:
                normalized_signal = signal_array - min_val
                
        elif method == 'robust':
            # Robust normalization using median and MAD
            median_val = np.median(signal_array)
            mad_val = np.median(np.abs(signal_array - median_val))
            if mad_val > 0:
                normalized_signal = (signal_array - median_val) / mad_val
            else:
                normalized_signal = signal_array - median_val
                
        else:
            # Default: return original signal
            normalized_signal = signal_array
        
        return normalized_signal
        
    except Exception as e:
        print(f"Error in signal normalization: {e}")
        return ecg_signal

def detect_signal_artifacts(ecg_signal, fs=360):
    """
    Detect various types of artifacts in ECG signal.
    
    Parameters:
    -----------
    ecg_signal : array_like
        ECG signal to analyze
    fs : int
        Sampling frequency
        
    Returns:
    --------
    artifacts : dict
        Dictionary containing detected artifacts
    """
    
    try:
        artifacts = {}
        
        # 1. Detect sudden amplitude changes (electrode pop)
        diff_signal = np.diff(ecg_signal)
        threshold = 5 * np.std(diff_signal)
        sudden_changes = np.where(np.abs(diff_signal) > threshold)[0]
        artifacts['electrode_pops'] = len(sudden_changes)
        
        # 2. Detect flat segments
        flat_threshold = 0.01 * np.std(ecg_signal)
        flat_segments = []
        current_flat = 0
        
        for i in range(1, len(ecg_signal)):
            if abs(ecg_signal[i] - ecg_signal[i-1]) < flat_threshold:
                current_flat += 1
            else:
                if current_flat > fs * 0.1:  # Flat for more than 100ms
                    flat_segments.append(current_flat)
                current_flat = 0
        
        artifacts['flat_segments'] = len(flat_segments)
        
        # 3. Detect saturation
        max_amplitude = np.max(np.abs(ecg_signal))
        saturation_threshold = 0.95 * max_amplitude
        saturated_samples = np.sum(np.abs(ecg_signal) > saturation_threshold)
        artifacts['saturated_samples'] = saturated_samples
        
        # 4. Detect high frequency noise
        highfreq_signal = bandpass_filter(ecg_signal, 30, fs/2-1, fs)
        noise_power = np.mean(highfreq_signal**2)
        signal_power = np.mean(ecg_signal**2)
        
        if signal_power > 0:
            noise_ratio = noise_power / signal_power
            artifacts['high_freq_noise_ratio'] = noise_ratio
        else:
            artifacts['high_freq_noise_ratio'] = 0
        
        return artifacts
        
    except Exception as e:
        print(f"Error in artifact detection: {e}")
        return {}

# ============================================================================
# ADVANCED PROCESSING FUNCTIONS
# ============================================================================

def adaptive_filtering(ecg_signal, fs=360, adaptation_rate=0.01):
    """
    Apply adaptive filtering to ECG signal.
    
    Parameters:
    -----------
    ecg_signal : array_like
        Input ECG signal
    fs : int
        Sampling frequency
    adaptation_rate : float
        Adaptation rate for the filter
        
    Returns:
    --------
    filtered_signal : numpy.ndarray
        Adaptively filtered signal
    """
    
    try:
        # Simple LMS adaptive filter implementation
        signal_array = np.array(ecg_signal)
        filtered_signal = np.zeros_like(signal_array)
        
        # Filter parameters
        filter_length = int(0.1 * fs)  # 100ms filter
        weights = np.zeros(filter_length)
        
        for i in range(filter_length, len(signal_array)):
            # Get input vector
            x = signal_array[i-filter_length:i]
            
            # Calculate output
            y = np.dot(weights, x)
            filtered_signal[i] = y
            
            # Calculate error
            error = signal_array[i] - y
            
            # Update weights
            weights += adaptation_rate * error * x
        
        return filtered_signal
        
    except Exception as e:
        print(f"Error in adaptive filtering: {e}")
        return ecg_signal

def wavelet_denoising(ecg_signal, wavelet='db4', threshold_mode='soft'):
    """
    Apply wavelet denoising to ECG signal.
    
    Parameters:
    -----------
    ecg_signal : array_like
        Input ECG signal
    wavelet : str
        Wavelet type
    threshold_mode : str
        Thresholding mode ('soft' or 'hard')
        
    Returns:
    --------
    denoised_signal : numpy.ndarray
        Wavelet denoised signal
    """
    
    try:
        import pywt
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(ecg_signal, wavelet, level=6)
        
        # Calculate threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(ecg_signal)))
        
        # Apply thresholding
        coeffs_thresh = list(coeffs)
        coeffs_thresh[1:] = [pywt.threshold(detail, threshold, threshold_mode) 
                            for detail in coeffs_thresh[1:]]
        
        # Reconstruct signal
        denoised_signal = pywt.waverec(coeffs_thresh, wavelet)
        
        # Ensure same length as input
        if len(denoised_signal) != len(ecg_signal):
            denoised_signal = denoised_signal[:len(ecg_signal)]
        
        return denoised_signal
        
    except ImportError:
        print("PyWavelets not available, using alternative denoising")
        return bandpass_filter(ecg_signal, 0.5, 40)
    except Exception as e:
        print(f"Error in wavelet denoising: {e}")
        return ecg_signal

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def complete_signal_processing_pipeline(ecg_signal, fs=360, **kwargs):
    """
    Complete signal processing pipeline combining all methods.
    
    Parameters:
    -----------
    ecg_signal : array_like
        Raw ECG signal
    fs : int
        Sampling frequency
    **kwargs : dict
        Additional parameters for processing methods
        
    Returns:
    --------
    result : dict
        Dictionary containing processed signal and quality metrics
    """
    
    try:
        # Step 1: Initial preprocessing
        preprocessed = preprocess_ecg_signal(ecg_signal, fs)
        
        # Step 2: Noise removal
        clean_signal = remove_noise_artifacts(preprocessed, fs)
        
        # Step 3: Quality assessment
        quality_metrics = signal_quality_assessment(clean_signal, fs)
        
        # Step 4: Artifact detection
        artifacts = detect_signal_artifacts(clean_signal, fs)
        
        # Step 5: Additional processing based on quality
        if quality_metrics.get('overall_quality_score', 50) < 70:
            # Apply more aggressive processing for poor quality signals
            if 'use_wavelet' in kwargs and kwargs['use_wavelet']:
                clean_signal = wavelet_denoising(clean_signal)
            
            if 'use_adaptive' in kwargs and kwargs['use_adaptive']:
                clean_signal = adaptive_filtering(clean_signal, fs)
        
        # Compile results
        result = {
            'processed_signal': clean_signal,
            'quality_metrics': quality_metrics,
            'artifacts_detected': artifacts,
            'processing_parameters': {
                'sampling_frequency': fs,
                'methods_applied': ['preprocessing', 'noise_removal', 'quality_assessment']
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error in complete processing pipeline: {e}")
        return {
            'processed_signal': ecg_signal,
            'quality_metrics': {},
            'artifacts_detected': {},
            'processing_parameters': {}
        }

if __name__ == "__main__":
    # Example usage and testing
    print("ECG Data Processing Module")
    print("Available functions:")
    print("- preprocess_ecg_signal()")
    print("- remove_noise_artifacts()")
    print("- bandpass_filter()")
    print("- notch_filter()")
    print("- baseline_correction()")
    print("- signal_quality_assessment()")
    print("- complete_signal_processing_pipeline()")