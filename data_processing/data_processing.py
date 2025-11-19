"""
Data processing functions for ECG analysis.
"""

import numpy as np
from scipy.signal import butter, filtfilt

def preprocess_ecg_signal(signal, fs):
    """Preprocess ECG signal with bandpass filtering."""
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = 40.0 / nyquist
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)

def remove_noise_artifacts(signal, fs):
    """Remove noise and artifacts from ECG signal."""
    # Remove powerline interference
    t = np.arange(len(signal)) / fs
    powerline = np.sin(2 * np.pi * 50 * t)  # 50 Hz powerline
    powerline_coef = np.corrcoef(signal, powerline)[0, 1]
    signal = signal - powerline_coef * powerline
    
    # Remove baseline wander
    nyquist = fs / 2
    low = 0.5 / nyquist
    b, a = butter(3, low, btype='high')
    return filtfilt(b, a, signal)

def signal_quality_assessment(signal, fs):
    """Assess ECG signal quality."""
    metrics = {}
    
    # Signal-to-noise ratio
    signal_power = np.mean(signal ** 2)
    noise = signal - np.convolve(signal, np.ones(100)/100, mode='same')
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
    metrics['snr_db'] = snr
    
    # Signal amplitude
    metrics['amplitude_mean'] = np.mean(np.abs(signal))
    metrics['amplitude_std'] = np.std(signal)
    
    # Signal quality score (0-100)
    quality_score = min(100, max(0, 50 + snr))
    metrics['quality_score'] = quality_score
    
    return metrics 