"""
PQRST peak detection functions for ECG analysis.
"""

import numpy as np
from scipy.signal import find_peaks
from wfdb.processing import XQRS

def detect_all_peaks(signal, fs):
    """Detect all PQRST peaks in ECG signal."""
    try:
        # R-peak detection using XQRS
        xqrs = XQRS(sig=signal, fs=fs)
        xqrs.detect()
        r_peaks = xqrs.qrs_inds
        
        if len(r_peaks) == 0:
            return None
        
        # Initialize peak arrays
        p_peaks, q_peaks, s_peaks, t_peaks = [], [], [], []
        
        for r in r_peaks:
            # Q wave: minimum before R
            q_start = max(0, r - int(0.05 * fs))
            if q_start < r:
                q_idx = np.argmin(signal[q_start:r]) + q_start
                q_peaks.append(q_idx)
            
            # S wave: minimum after R
            s_end = min(len(signal), r + int(0.08 * fs))
            if r < s_end:
                s_idx = np.argmin(signal[r:s_end]) + r
                s_peaks.append(s_idx)
            
            # P wave: maximum before Q
            if q_peaks:
                p_start = max(0, q_peaks[-1] - int(0.2 * fs))
                if p_start < q_peaks[-1]:
                    p_idx = np.argmax(signal[p_start:q_peaks[-1]]) + p_start
                    p_peaks.append(p_idx)
            
            # T wave: maximum after S
            if s_peaks:
                t_start = s_peaks[-1]
                t_end = min(len(signal), s_peaks[-1] + int(0.3 * fs))
                if t_start < t_end:
                    t_idx = np.argmax(signal[t_start:t_end]) + t_start
                    t_peaks.append(t_idx)
        
        return {
            'p_peaks': np.array(p_peaks),
            'q_peaks': np.array(q_peaks),
            'r_peaks': np.array(r_peaks),
            's_peaks': np.array(s_peaks),
            't_peaks': np.array(t_peaks)
        }
        
    except Exception as e:
        print(f"Error in PQRST detection: {str(e)}")
        return None

def enhanced_peak_detection(signal, fs):
    """Enhanced PQRST peak detection with additional validation."""
    try:
        # First pass: detect all peaks
        peaks = detect_all_peaks(signal, fs)
        if peaks is None:
            return None
        
        # Validate R-R intervals
        r_peaks = peaks['r_peaks']
        rr_intervals = np.diff(r_peaks) / fs
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        
        # Remove outliers
        valid_rr = np.abs(rr_intervals - mean_rr) < 2 * std_rr
        valid_r_peaks = r_peaks[:-1][valid_rr]
        
        if len(valid_r_peaks) < 2:
            return None
        
        # Re-detect other peaks with validated R peaks
        p_peaks, q_peaks, s_peaks, t_peaks = [], [], [], []
        
        for r in valid_r_peaks:
            # Q wave
            q_start = max(0, r - int(0.05 * fs))
            if q_start < r:
                q_idx = np.argmin(signal[q_start:r]) + q_start
                q_peaks.append(q_idx)
            
            # S wave
            s_end = min(len(signal), r + int(0.08 * fs))
            if r < s_end:
                s_idx = np.argmin(signal[r:s_end]) + r
                s_peaks.append(s_idx)
            
            # P wave
            if q_peaks:
                p_start = max(0, q_peaks[-1] - int(0.2 * fs))
                if p_start < q_peaks[-1]:
                    p_idx = np.argmax(signal[p_start:q_peaks[-1]]) + p_start
                    p_peaks.append(p_idx)
            
            # T wave
            if s_peaks:
                t_start = s_peaks[-1]
                t_end = min(len(signal), s_peaks[-1] + int(0.3 * fs))
                if t_start < t_end:
                    t_idx = np.argmax(signal[t_start:t_end]) + t_start
                    t_peaks.append(t_idx)
        
        return {
            'p_peaks': np.array(p_peaks),
            'q_peaks': np.array(q_peaks),
            'r_peaks': np.array(valid_r_peaks),
            's_peaks': np.array(s_peaks),
            't_peaks': np.array(t_peaks)
        }
        
    except Exception as e:
        print(f"Error in enhanced peak detection: {str(e)}")
        return None 