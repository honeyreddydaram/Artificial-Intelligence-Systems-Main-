"""
Complete PQRST Detection Module
Advanced algorithms for detecting P, Q, R, S, T peaks in ECG signals.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from wfdb.processing import XQRS
import warnings
warnings.filterwarnings('ignore')

def detect_all_peaks(ecg_signal, fs=360, method='xqrs'):
    """
    Main function to detect all PQRST peaks.
    
    Parameters:
    -----------
    ecg_signal : array_like
        ECG signal data
    fs : int
        Sampling frequency
    method : str
        Detection method ('xqrs', 'pan_tompkins')
        
    Returns:
    --------
    peaks_dict : dict
        Dictionary containing all detected peaks
    """
    
    try:
        # Step 1: R-peak detection
        r_peaks = detect_r_peaks(ecg_signal, fs, method)
        
        if len(r_peaks) == 0:
            return None
        
        # Step 2: Detect other PQRST peaks
        p_peaks = detect_p_peaks(ecg_signal, r_peaks, fs)
        q_peaks = detect_q_peaks(ecg_signal, r_peaks, fs)
        s_peaks = detect_s_peaks(ecg_signal, r_peaks, fs)
        t_peaks = detect_t_peaks(ecg_signal, r_peaks, fs)
        
        # Step 3: Calculate heart rate
        heart_rate = calculate_heart_rate(r_peaks, fs)
        
        peaks_dict = {
            'r_peaks': r_peaks,
            'p_peaks': p_peaks,
            'q_peaks': q_peaks,
            's_peaks': s_peaks,
            't_peaks': t_peaks,
            'heart_rate': heart_rate,
            'fs': fs
        }
        
        return peaks_dict
        
    except Exception as e:
        print(f"Error in PQRST detection: {e}")
        return None

def detect_r_peaks(ecg_signal, fs=360, method='xqrs'):
    """
    Detect R-peaks using specified method.
    """
    
    try:
        if method == 'xqrs':
            # XQRS method - robust and reliable
            xqrs = XQRS(sig=ecg_signal, fs=fs)
            xqrs.detect()
            return xqrs.qrs_inds
            
        elif method == 'pan_tompkins':
            # Pan-Tompkins algorithm
            return pan_tompkins_detector(ecg_signal, fs)
            
        else:
            # Default to XQRS
            xqrs = XQRS(sig=ecg_signal, fs=fs)
            xqrs.detect()
            return xqrs.qrs_inds
            
    except Exception as e:
        print(f"Error in R-peak detection: {e}")
        return np.array([])

def pan_tompkins_detector(ecg_signal, fs=360):
    """
    Pan-Tompkins R-peak detection algorithm.
    """
    
    try:
        # Step 1: Bandpass filter (5-15 Hz)
        nyquist = 0.5 * fs
        low = 5 / nyquist
        high = 15 / nyquist
        b, a = butter(1, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)
        
        # Step 2: Differentiation
        diff_signal = np.diff(filtered_signal)
        
        # Step 3: Squaring
        squared_signal = diff_signal ** 2
        
        # Step 4: Moving window integration
        window_size = int(0.150 * fs)  # 150ms window
        integrated_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')
        
        # Step 5: Peak detection
        min_distance = int(0.6 * fs)  # Minimum 600ms between R-peaks
        height_threshold = np.max(integrated_signal) * 0.3
        
        peaks, _ = find_peaks(integrated_signal, distance=min_distance, height=height_threshold)
        
        return peaks
        
    except Exception as e:
        print(f"Error in Pan-Tompkins detection: {e}")
        return np.array([])

def detect_p_peaks(ecg_signal, r_peaks, fs=360):
    """
    Detect P-wave peaks.
    """
    
    p_peaks = []
    
    try:
        for r_idx in r_peaks:
            # Search window: 80-300ms before R-peak
            search_start = max(0, r_idx - int(0.30 * fs))
            search_end = max(0, r_idx - int(0.08 * fs))
            
            if search_start < search_end and search_end < len(ecg_signal):
                # Find local maxima in search window
                search_segment = ecg_signal[search_start:search_end]
                
                # Use peak finding with minimum prominence
                local_peaks, properties = find_peaks(
                    search_segment, 
                    prominence=0.1 * np.std(search_segment),
                    distance=int(0.05 * fs)
                )
                
                if len(local_peaks) > 0:
                    # Take the most prominent peak
                    prominences = properties['prominences']
                    best_peak_idx = local_peaks[np.argmax(prominences)]
                    p_peak = search_start + best_peak_idx
                    p_peaks.append(p_peak)
        
        return np.array(p_peaks)
        
    except Exception as e:
        print(f"Error in P-peak detection: {e}")
        return np.array([])

def detect_q_peaks(ecg_signal, r_peaks, fs=360):
    """
    Detect Q-wave peaks (negative deflection before R).
    """
    
    q_peaks = []
    
    try:
        for r_idx in r_peaks:
            # Search window: 50ms before R-peak
            search_start = max(0, r_idx - int(0.05 * fs))
            search_end = r_idx
            
            if search_start < search_end:
                # Find minimum in search window
                search_segment = ecg_signal[search_start:search_end]
                q_idx = np.argmin(search_segment)
                q_peak = search_start + q_idx
                q_peaks.append(q_peak)
        
        return np.array(q_peaks)
        
    except Exception as e:
        print(f"Error in Q-peak detection: {e}")
        return np.array([])

def detect_s_peaks(ecg_signal, r_peaks, fs=360):
    """
    Detect S-wave peaks (negative deflection after R).
    """
    
    s_peaks = []
    
    try:
        for r_idx in r_peaks:
            # Search window: 80ms after R-peak
            search_start = r_idx
            search_end = min(len(ecg_signal), r_idx + int(0.08 * fs))
            
            if search_start < search_end:
                # Find minimum in search window
                search_segment = ecg_signal[search_start:search_end]
                s_idx = np.argmin(search_segment)
                s_peak = search_start + s_idx
                s_peaks.append(s_peak)
        
        return np.array(s_peaks)
        
    except Exception as e:
        print(f"Error in S-peak detection: {e}")
        return np.array([])

def detect_t_peaks(ecg_signal, r_peaks, fs=360):
    """
    Detect T-wave peaks.
    """
    
    t_peaks = []
    
    try:
        for r_idx in r_peaks:
            # Search window: 150-400ms after R-peak
            search_start = min(len(ecg_signal), r_idx + int(0.15 * fs))
            search_end = min(len(ecg_signal), r_idx + int(0.40 * fs))
            
            if search_start < search_end:
                # Smooth the signal for T-wave detection
                search_segment = ecg_signal[search_start:search_end]
                
                # Apply smoothing
                if len(search_segment) > 10:
                    window_length = min(len(search_segment) // 3 * 2 - 1, int(0.08 * fs))
                    if window_length >= 5:
                        search_segment = savgol_filter(search_segment, window_length, 3)
                
                # Find local maxima
                local_peaks, properties = find_peaks(
                    search_segment,
                    prominence=0.05 * np.std(search_segment),
                    distance=int(0.05 * fs)
                )
                
                if len(local_peaks) > 0:
                    # Take the most prominent peak
                    prominences = properties['prominences']
                    best_peak_idx = local_peaks[np.argmax(prominences)]
                    t_peak = search_start + best_peak_idx
                    t_peaks.append(t_peak)
        
        return np.array(t_peaks)
        
    except Exception as e:
        print(f"Error in T-peak detection: {e}")
        return np.array([])

def calculate_heart_rate(r_peaks, fs=360):
    """
    Calculate heart rate from R-peaks.
    """
    
    try:
        if len(r_peaks) < 2:
            return None
        
        # Calculate RR intervals in seconds
        rr_intervals = np.diff(r_peaks) / fs
        
        # Calculate heart rate in BPM
        heart_rate = 60 / np.mean(rr_intervals)
        
        return heart_rate
        
    except Exception as e:
        print(f"Error in heart rate calculation: {e}")
        return None

def enhanced_peak_detection(ecg_signal, fs=360, use_ensemble=True):
    """
    Enhanced peak detection using ensemble methods.
    """
    
    try:
        if use_ensemble:
            # Use multiple methods and combine results
            xqrs_peaks = detect_r_peaks(ecg_signal, fs, 'xqrs')
            pt_peaks = detect_r_peaks(ecg_signal, fs, 'pan_tompkins')
            
            # Combine and validate peaks
            all_peaks = np.concatenate([xqrs_peaks, pt_peaks])
            
            # Remove duplicates (peaks within 100ms of each other)
            if len(all_peaks) > 0:
                all_peaks = np.sort(all_peaks)
                final_peaks = [all_peaks[0]]
                
                for peak in all_peaks[1:]:
                    if peak - final_peaks[-1] > int(0.1 * fs):  # 100ms minimum
                        final_peaks.append(peak)
                
                r_peaks = np.array(final_peaks)
            else:
                r_peaks = np.array([])
        else:
            # Use single best method
            r_peaks = detect_r_peaks(ecg_signal, fs, 'xqrs')
        
        # Detect other peaks
        if len(r_peaks) > 0:
            p_peaks = detect_p_peaks(ecg_signal, r_peaks, fs)
            q_peaks = detect_q_peaks(ecg_signal, r_peaks, fs)
            s_peaks = detect_s_peaks(ecg_signal, r_peaks, fs)
            t_peaks = detect_t_peaks(ecg_signal, r_peaks, fs)
            heart_rate = calculate_heart_rate(r_peaks, fs)
            
            return {
                'r_peaks': r_peaks,
                'p_peaks': p_peaks,
                'q_peaks': q_peaks,
                's_peaks': s_peaks,
                't_peaks': t_peaks,
                'heart_rate': heart_rate,
                'fs': fs
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error in enhanced peak detection: {e}")
        return None

def validate_peak_sequence(r_peaks, p_peaks, q_peaks, s_peaks, t_peaks, fs=360):
    """
    Validate the detected peak sequence for physiological correctness.
    """
    
    try:
        validated_peaks = {
            'r_peaks': r_peaks,
            'p_peaks': [],
            'q_peaks': [],
            's_peaks': [],
            't_peaks': []
        }
        
        for i, r_peak in enumerate(r_peaks):
            # Validate P-peaks
            valid_p = [p for p in p_peaks if abs(p - r_peak) < int(0.3 * fs) and p < r_peak]
            if valid_p:
                validated_peaks['p_peaks'].append(min(valid_p, key=lambda x: abs(x - r_peak)))
            
            # Validate Q-peaks
            valid_q = [q for q in q_peaks if abs(q - r_peak) < int(0.05 * fs) and q < r_peak]
            if valid_q:
                validated_peaks['q_peaks'].append(min(valid_q, key=lambda x: abs(x - r_peak)))
            
            # Validate S-peaks
            valid_s = [s for s in s_peaks if abs(s - r_peak) < int(0.08 * fs) and s > r_peak]
            if valid_s:
                validated_peaks['s_peaks'].append(min(valid_s, key=lambda x: abs(x - r_peak)))
            
            # Validate T-peaks
            valid_t = [t for t in t_peaks if int(0.15 * fs) < t - r_peak < int(0.4 * fs)]
            if valid_t:
                validated_peaks['t_peaks'].append(min(valid_t, key=lambda x: abs(x - r_peak)))
        
        # Convert to numpy arrays
        for key in validated_peaks:
            validated_peaks[key] = np.array(validated_peaks[key])
        
        return validated_peaks
        
    except Exception as e:
        print(f"Error in peak validation: {e}")
        return {
            'r_peaks': r_peaks,
            'p_peaks': np.array([]),
            'q_peaks': np.array([]),
            's_peaks': np.array([]),
            't_peaks': np.array([])
        }

def calculate_ecg_intervals(peaks_dict, fs=360):
    """
    Calculate ECG intervals from detected peaks.
    """
    
    try:
        intervals = {}
        
        r_peaks = peaks_dict.get('r_peaks', [])
        p_peaks = peaks_dict.get('p_peaks', [])
        q_peaks = peaks_dict.get('q_peaks', [])
        s_peaks = peaks_dict.get('s_peaks', [])
        t_peaks = peaks_dict.get('t_peaks', [])
        
        # RR intervals
        if len(r_peaks) > 1:
            rr_intervals = np.diff(r_peaks) / fs * 1000  # ms
            intervals['rr_mean'] = np.mean(rr_intervals)
            intervals['rr_std'] = np.std(rr_intervals)
        
        # PR intervals
        min_len = min(len(p_peaks), len(r_peaks))
        if min_len > 0:
            pr_intervals = (r_peaks[:min_len] - p_peaks[:min_len]) / fs * 1000
            pr_intervals = pr_intervals[pr_intervals > 0]
            if len(pr_intervals) > 0:
                intervals['pr_mean'] = np.mean(pr_intervals)
                intervals['pr_std'] = np.std(pr_intervals)
        
        # QRS duration
        min_len = min(len(q_peaks), len(s_peaks))
        if min_len > 0:
            qrs_intervals = (s_peaks[:min_len] - q_peaks[:min_len]) / fs * 1000
            qrs_intervals = qrs_intervals[qrs_intervals > 0]
            if len(qrs_intervals) > 0:
                intervals['qrs_mean'] = np.mean(qrs_intervals)
                intervals['qrs_std'] = np.std(qrs_intervals)
        
        # QT intervals
        min_len = min(len(q_peaks), len(t_peaks))
        if min_len > 0:
            qt_intervals = (t_peaks[:min_len] - q_peaks[:min_len]) / fs * 1000
            qt_intervals = qt_intervals[qt_intervals > 0]
            if len(qt_intervals) > 0:
                intervals['qt_mean'] = np.mean(qt_intervals)
                intervals['qt_std'] = np.std(qt_intervals)
        
        return intervals
        
    except Exception as e:
        print(f"Error in interval calculation: {e}")
        return {}

# Main detection function for compatibility
def detect_pqrst_peaks(ecg_signal, fs=360, method='xqrs'):
    """
    Main function for PQRST detection (compatibility wrapper).
    """
    return detect_all_peaks(ecg_signal, fs, method)

if __name__ == "__main__":
    print("PQRST Detection Module")
    print("Available functions:")
    print("- detect_all_peaks()")
    print("- enhanced_peak_detection()")
    print("- detect_r_peaks()")
    print("- detect_p_peaks()")
    print("- detect_q_peaks()")
    print("- detect_s_peaks()")
    print("- detect_t_peaks()")