"""
ECG Feature Extraction Module

This module provides functions for extracting features from ECG signals,
including statistical, morphological, and wavelet-based features.
"""

import numpy as np
from scipy import stats
import pandas as pd

# Check if advanced libraries are available
try:
    from scipy.fft import fft
    fft_available = True
except ImportError:
    fft_available = False

try:
    import pywt
    wavelet_available = True
except ImportError:
    wavelet_available = False

try:
    import tsfel
    tsfel_available = True
except ImportError:
    tsfel_available = False

def extract_statistical_features(signal):
    """
    Extract basic statistical features from an ECG signal segment.
    
    Parameters:
        signal (array): The ECG signal segment
        
    Returns:
        dict: Dictionary of statistical features
    """
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'range': np.max(signal) - np.min(signal),
        'median': np.median(signal),
        'rms': np.sqrt(np.mean(np.square(signal))),
        'skewness': stats.skew(signal),
        'kurtosis': stats.kurtosis(signal),
        'p5': np.percentile(signal, 5),
        'p25': np.percentile(signal, 25),
        'p75': np.percentile(signal, 75),
        'p95': np.percentile(signal, 95),
        'iqr': np.percentile(signal, 75) - np.percentile(signal, 25),
        'energy': np.sum(np.square(signal)),
    }
    
    # Add zero crossing rate
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(signal)
    
    # Add slope features
    slopes = np.diff(signal)
    features['mean_slope'] = np.mean(np.abs(slopes))
    features['max_slope'] = np.max(np.abs(slopes))
    
    return features

def extract_frequency_features(signal, fs=250):
    """
    Extract frequency domain features from an ECG signal segment.
    
    Parameters:
        signal (array): The ECG signal segment
        fs (float): Sampling frequency (Hz)
        
    Returns:
        dict: Dictionary of frequency domain features
    """
    features = {}
    
    if not fft_available:
        return features
    
    try:
        # Calculate FFT and frequency values
        n = len(signal)
        signal_fft = fft(signal)
        magnitude = np.abs(signal_fft[:n//2]) / n
        
        # Define frequency bands
        frequencies = np.fft.fftfreq(n, 1/fs)[:n//2]
        
        # Common ECG frequency bands
        bands = {
            'vlow': (0, 1),    # Very low frequency
            'low': (1, 5),     # Low frequency
            'mid': (5, 15),    # Mid frequency (contains most of the QRS complex)
            'high': (15, 50)   # High frequency
        }
        
        # Calculate power in each band
        for band_name, (low, high) in bands.items():
            # Find indices corresponding to the frequency band
            indices = np.logical_and(frequencies >= low, frequencies <= high)
            # Calculate the power in this band
            band_power = np.sum(magnitude[indices] ** 2)
            features[f'power_{band_name}'] = band_power
        
        # Calculate total power
        total_power = np.sum(magnitude ** 2)
        features['total_power'] = total_power
        
        # Calculate normalized power in each band
        for band_name in bands.keys():
            features[f'norm_power_{band_name}'] = features[f'power_{band_name}'] / total_power
        
        # Calculate spectral metrics
        if len(magnitude) > 0:
            features['spectral_mean'] = np.average(frequencies, weights=magnitude)
            features['spectral_std'] = np.sqrt(np.average((frequencies - features['spectral_mean'])**2, weights=magnitude))
            
            # Calculate the peak frequency
            peak_idx = np.argmax(magnitude)
            features['peak_frequency'] = frequencies[peak_idx]
            
            # Calculate spectral entropy
            normalized_magnitude = magnitude / np.sum(magnitude)
            nonzero_mask = normalized_magnitude > 0
            spectral_entropy = -np.sum(normalized_magnitude[nonzero_mask] * np.log2(normalized_magnitude[nonzero_mask]))
            features['spectral_entropy'] = spectral_entropy
        
        return features
    except Exception as e:
        print(f"Error extracting frequency features: {e}")
        return {}

def extract_wavelet_features(signal, wavelet='db4', levels=4):
    """
    Extract wavelet-based features from an ECG signal segment.
    
    Parameters:
        signal (array): The ECG signal segment
        wavelet (str): Wavelet type for decomposition
        levels (int): Number of decomposition levels
        
    Returns:
        dict: Dictionary of wavelet-based features
    """
    features = {}
    
    if not wavelet_available:
        return features
    
    try:
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=min(levels, pywt.dwt_max_level(len(signal), wavelet)))
        
        # Extract features from each coefficient level
        for i, coef in enumerate(coeffs):
            level_name = "approx" if i == 0 else f"detail_{i}"
            features[f"{level_name}_mean"] = np.mean(coef)
            features[f"{level_name}_std"] = np.std(coef)
            features[f"{level_name}_energy"] = np.sum(coef**2)
            features[f"{level_name}_max"] = np.max(coef)
            features[f"{level_name}_min"] = np.min(coef)
        
        return features
    except Exception as e:
        print(f"Error extracting wavelet features: {e}")
        return {}

def extract_morphological_features(signal, fs=250):
    """
    Extract morphological features from an ECG heartbeat segment.
    Assumes the heartbeat is aligned with R peak at a specific position.
    
    Parameters:
        signal (array): ECG heartbeat segment
        fs (float): Sampling frequency (Hz)
        
    Returns:
        dict: Dictionary of morphological features
    """
    features = {}
    beat_length = len(signal)
    
    # Estimate R peak location (assuming it's approximately in the middle of the signal)
    # The exact position depends on how segments were extracted
    r_idx = beat_length // 2
    
    try:
        # Find potential PQRST points in the heartbeat
        # P-wave is typically before the R peak
        p_search_start = max(0, r_idx - int(0.2 * fs))
        p_search_end = r_idx - int(0.05 * fs)
        if p_search_start < p_search_end:
            p_idx = p_search_start + np.argmax(signal[p_search_start:p_search_end])
        else:
            p_idx = max(0, r_idx - int(0.15 * fs))
        
        # Q-wave is the minimum just before R
        q_search_start = max(0, r_idx - int(0.05 * fs))
        q_search_end = r_idx
        if q_search_start < q_search_end:
            q_idx = q_search_start + np.argmin(signal[q_search_start:q_search_end])
        else:
            q_idx = max(0, r_idx - int(0.025 * fs))
        
        # S-wave is the minimum just after R
        s_search_start = r_idx
        s_search_end = min(beat_length, r_idx + int(0.05 * fs))
        if s_search_start < s_search_end:
            s_idx = s_search_start + np.argmin(signal[s_search_start:s_search_end])
        else:
            s_idx = min(beat_length - 1, r_idx + int(0.025 * fs))
        
        # T-wave is the maximum after S
        t_search_start = s_idx
        t_search_end = min(beat_length, s_idx + int(0.3 * fs))
        if t_search_start < t_search_end:
            t_idx = t_search_start + np.argmax(signal[t_search_start:t_search_end])
        else:
            t_idx = min(beat_length - 1, s_idx + int(0.15 * fs))
        
        # Calculate morphological features
        # Wave amplitudes
        features['p_amp'] = signal[p_idx]
        features['q_amp'] = signal[q_idx]
        features['r_amp'] = signal[r_idx]
        features['s_amp'] = signal[s_idx]
        features['t_amp'] = signal[t_idx]
        
        # Wave-to-wave amplitudes
        features['pq_amp'] = features['p_amp'] - features['q_amp']
        features['rs_amp'] = features['r_amp'] - features['s_amp']
        features['rt_amp'] = features['r_amp'] - features['t_amp']
        features['st_amp'] = features['s_amp'] - features['t_amp']
        
        # Time intervals (in seconds)
        features['pq_interval'] = (q_idx - p_idx) / fs
        features['qrs_interval'] = (s_idx - q_idx) / fs
        features['qt_interval'] = (t_idx - q_idx) / fs
        features['st_interval'] = (t_idx - s_idx) / fs
        
        # Slopes
        if q_idx > p_idx:
            features['p_q_slope'] = (signal[q_idx] - signal[p_idx]) / ((q_idx - p_idx) / fs)
        else:
            features['p_q_slope'] = 0
            
        if r_idx > q_idx:
            features['q_r_slope'] = (signal[r_idx] - signal[q_idx]) / ((r_idx - q_idx) / fs)
        else:
            features['q_r_slope'] = 0
            
        if s_idx > r_idx:
            features['r_s_slope'] = (signal[s_idx] - signal[r_idx]) / ((s_idx - r_idx) / fs)
        else:
            features['r_s_slope'] = 0
            
        if t_idx > s_idx:
            features['s_t_slope'] = (signal[t_idx] - signal[s_idx]) / ((t_idx - s_idx) / fs)
        else:
            features['s_t_slope'] = 0
        
        # Overall wave prominence
        features['r_prominence'] = features['r_amp'] - min(features['q_amp'], features['s_amp'])
        features['p_prominence'] = features['p_amp'] - min(signal[max(0, p_idx-int(0.1*fs)):p_idx+1]) if p_idx > 0 else 0
        features['t_prominence'] = features['t_amp'] - min(signal[s_idx:t_idx+1]) if t_idx > s_idx else 0
        
        return features
    
    except Exception as e:
        print(f"Error extracting morphological features: {e}")
        return {}

def extract_tsfel_features(signal, fs=250):
    """
    Extract features using TSFEL library.
    
    Parameters:
        signal (array): The ECG signal segment
        fs (float): Sampling frequency (Hz)
        
    Returns:
        dict: Dictionary of TSFEL features
    """
    if not tsfel_available:
        return {}
    
    try:
        # Get TSFEL configuration for time domain features
        cfg = tsfel.get_features_by_domain(domain="temporal")
        
        # Extract TSFEL features
        features_df = tsfel.time_series_features_extractor(cfg, signal.reshape(1, -1), fs=fs, verbose=0)
        
        # Convert to dictionary
        features = features_df.iloc[0].to_dict()
        
        return features
    except Exception as e:
        print(f"Error extracting TSFEL features: {e}")
        return {}

def extract_heartbeat_features(heartbeat, fs=250, include_advanced=True):
    """
    Extract comprehensive features from a single heartbeat.
    
    Parameters:
        heartbeat (array): Single heartbeat ECG segment
        fs (float): Sampling frequency (Hz)
        include_advanced (bool): Whether to include advanced features (wavelets, TSFEL)
        
    Returns:
        dict: Combined heartbeat features
    """
    # Basic statistical features
    features = extract_statistical_features(heartbeat)
    
    # Add frequency domain features
    freq_features = extract_frequency_features(heartbeat, fs)
    features.update(freq_features)
    
    # Add morphological features
    morph_features = extract_morphological_features(heartbeat, fs)
    features.update(morph_features)
    
    # Add advanced features if requested
    if include_advanced:
        # Add wavelet features if available
        if wavelet_available:
            wavelet_features = extract_wavelet_features(heartbeat)
            features.update(wavelet_features)
        
        # Add TSFEL features if available
        if tsfel_available:
            tsfel_features = extract_tsfel_features(heartbeat, fs)
            features.update(tsfel_features)
    
    return features

def extract_features_from_heartbeats(heartbeats, fs=250, include_advanced=True):
    """
    Extract features from multiple heartbeats.
    
    Parameters:
        heartbeats (array): Array of heartbeat segments
        fs (float): Sampling frequency (Hz)
        include_advanced (bool): Whether to include advanced features
        
    Returns:
        DataFrame: Features for each heartbeat
    """
    features_list = []
    
    for i, beat in enumerate(heartbeats):
        try:
            # Extract features for this heartbeat
            beat_features = extract_heartbeat_features(beat, fs, include_advanced)
            
            # Add beat index
            beat_features['beat_idx'] = i
            
            # Add to list
            features_list.append(beat_features)
            
            # Print progress
            if (i+1) % 50 == 0:
                print(f"Processed {i+1}/{len(heartbeats)} heartbeats...")
                
        except Exception as e:
            print(f"Error processing heartbeat {i}: {e}")
    
    # Convert to DataFrame
    if features_list:
        features_df = pd.DataFrame(features_list)
        
        # Handle any NaN values
        features_df = features_df.fillna(0)
        
        return features_df
    else:
        return pd.DataFrame()

def select_top_features(features_df, target=None, n_features=20, method='variance'):
    """
    Select top features based on various criteria.
    
    Parameters:
        features_df (DataFrame): DataFrame containing features
        target (Series, optional): Target variable for correlation-based selection
        n_features (int): Number of features to select
        method (str): Selection method ('variance', 'correlation')
        
    Returns:
        DataFrame: DataFrame with selected features
    """
    # Remove non-feature columns
    feature_cols = [col for col in features_df.columns 
                   if col not in ['beat_idx', 'target', 'class', 'label']]
    
    if method == 'variance':
        # Calculate variance of each feature
        variances = features_df[feature_cols].var()
        
        # Select top features by variance
        top_features = variances.sort_values(ascending=False).index[:n_features].tolist()
        
    elif method == 'correlation' and target is not None:
        # Calculate correlation with target
        correlations = features_df[feature_cols].corrwith(target).abs()
        
        # Select top features by correlation
        top_features = correlations.sort_values(ascending=False).index[:n_features].tolist()
        
    else:
        # Default to all features if method is not recognized
        top_features = feature_cols[:n_features]
    
    # Include non-feature columns
    selected_cols = [col for col in features_df.columns if col not in feature_cols or col in top_features]
    
    return features_df[selected_cols]

# Add this function to feature_extraction.py
def extract_features_in_batches(heartbeats, fs=250, include_advanced=True, batch_size=50):
    """
    Extract features in batches to reduce memory usage.
    """
    all_features = []
    total_batches = (len(heartbeats) + batch_size - 1) // batch_size
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(heartbeats))
        batch = heartbeats[start_idx:end_idx]
        
        # Process batch
        batch_features = [extract_heartbeat_features(beat, fs, include_advanced) for beat in batch]
        all_features.extend(batch_features)
        
    return pd.DataFrame(all_features)