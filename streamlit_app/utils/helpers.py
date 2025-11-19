"""
Complete Helper Functions Module - UPDATED VERSION
Comprehensive utility functions for ECG analysis, visualization, and data handling.
This version includes all fixes for WFDB file processing and additional improvements.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os
from datetime import datetime
import warnings
import re
import struct
warnings.filterwarnings('ignore')

# ============================================================================
# SYNTHETIC ECG GENERATION
# ============================================================================

def generate_synthetic_ecg(duration=10, fs=250, hr=60, noise_level=0.1):
    """
    Generate synthetic ECG signal with realistic morphology.
    
    Parameters:
    -----------
    duration : float
        Signal duration in seconds
    fs : int
        Sampling frequency in Hz
    hr : int
        Heart rate in beats per minute
    noise_level : float
        Level of noise to add (0-1)
        
    Returns:
    --------
    tuple : (time, ecg_signal)
        Time array and synthetic ECG signal
    """
    try:
        # Time array
        t = np.arange(0, duration, 1/fs)
        
        # Calculate RR interval
        rr_interval = 60/hr  # seconds
        
        # Generate R-peak locations
        r_peaks = np.arange(0, duration, rr_interval)
        
        # Initialize signal
        ecg = np.zeros_like(t)
        
        # Add each heartbeat
        for r_peak in r_peaks:
            # Time relative to R-peak
            t_rel = t - r_peak
            
            # P wave (Gaussian)
            p_wave = 0.25 * np.exp(-(t_rel - 0.2)**2 / (2 * 0.02**2))
            
            # QRS complex (combination of Gaussians)
            q_wave = -0.1 * np.exp(-(t_rel - 0.05)**2 / (2 * 0.01**2))
            r_wave = 1.0 * np.exp(-(t_rel - 0.0)**2 / (2 * 0.02**2))
            s_wave = -0.3 * np.exp(-(t_rel + 0.05)**2 / (2 * 0.02**2))
            
            # T wave (Gaussian)
            t_wave = 0.35 * np.exp(-(t_rel + 0.3)**2 / (2 * 0.05**2))
            
            # Add all components
            beat = p_wave + q_wave + r_wave + s_wave + t_wave
            
            # Add to signal
            ecg += beat
        
        # Add noise
        noise = np.random.normal(0, noise_level, len(t))
        ecg += noise
        
        # Normalize
        ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg))
        
        return t, ecg
        
    except Exception as e:
        print(f"Error generating synthetic ECG: {e}")
        return None, None

# ============================================================================
# IMPROVED WFDB FILE PROCESSING
# ============================================================================

def validate_wfdb_files(uploaded_files):
    """
    Validate uploaded WFDB files before processing.
    
    Parameters:
    -----------
    uploaded_files : list
        List of uploaded files
        
    Returns:
    --------
    tuple : (is_valid, error_message)
    """
    try:
        if not uploaded_files:
            return False, "No files uploaded"
        
        # Check file types
        dat_files = [f for f in uploaded_files if f.name.lower().endswith('.dat')]
        hea_files = [f for f in uploaded_files if f.name.lower().endswith('.hea')]
        
        if not dat_files:
            return False, "No .dat files found. Please upload WFDB data files."
        
        if not hea_files:
            return False, "No .hea files found. Please upload WFDB header files."
        
        # Check for matching pairs
        dat_bases = [os.path.splitext(f.name)[0] for f in dat_files]
        hea_bases = [os.path.splitext(f.name)[0] for f in hea_files]
        
        matching_pairs = set(dat_bases) & set(hea_bases)
        
        if not matching_pairs:
            return False, f"No matching .dat/.hea pairs found.\nDAT files: {dat_bases}\nHEA files: {hea_bases}"
        
        return True, f"Found {len(matching_pairs)} matching file pair(s): {list(matching_pairs)}"
        
    except Exception as e:
        return False, f"Error validating files: {str(e)}"

def parse_header_file(header_path):
    """
    Parse WFDB header file to extract metadata.
    
    Parameters:
    -----------
    header_path : str
        Path to the .hea file
        
    Returns:
    --------
    tuple : (fs, num_channels, gain, baseline, units, data_format)
    """
    try:
        with open(header_path, 'r') as f:
            lines = f.readlines()
        
        # First line contains basic info
        first_line = lines[0].strip().split()
        record_name = first_line[0]
        num_channels = int(first_line[1])
        fs = float(first_line[2]) if len(first_line) > 2 else 250.0
        
        print(f"Header info - Record: {record_name}, Channels: {num_channels}, FS: {fs}")
        
        # Initialize arrays for channel-specific info
        gain = []
        baseline = []
        units = []
        data_formats = []
        
        # Parse channel-specific lines
        for i in range(1, min(num_channels + 1, len(lines))):
            line = lines[i].strip().split()
            if len(line) >= 3:
                # WFDB format: filename format gain(baseline)/units
                filename = line[0] if len(line) > 0 else ""
                data_format = line[1] if len(line) > 1 else "16"
                gain_baseline_units = line[2] if len(line) > 2 else "200(0)/mV"
                
                # Store data format for each channel
                data_formats.append(data_format)
                
                # Parse gain(baseline)/units
                if '(' in gain_baseline_units and ')' in gain_baseline_units:
                    gain_part = gain_baseline_units.split('(')[0]
                    baseline_units = gain_baseline_units.split('(')[1]
                    baseline_part = baseline_units.split(')')[0]
                    units_part = baseline_units.split('/')[-1] if '/' in baseline_units else "mV"
                else:
                    # Try alternative parsing
                    parts = gain_baseline_units.split('/')
                    if len(parts) >= 2:
                        gain_baseline = parts[0]
                        units_part = parts[1]
                        if '(' in gain_baseline and ')' in gain_baseline:
                            gain_part = gain_baseline.split('(')[0]
                            baseline_part = gain_baseline.split('(')[1].split(')')[0]
                        else:
                            gain_part = gain_baseline
                            baseline_part = "0"
                    else:
                        gain_part = "200"
                        baseline_part = "0"
                        units_part = "mV"
                
                try:
                    gain.append(float(gain_part))
                    baseline.append(float(baseline_part))
                    units.append(units_part)
                except ValueError:
                    gain.append(200.0)  # Default gain
                    baseline.append(0.0)  # Default baseline
                    units.append("mV")   # Default units
                    data_formats.append("16")  # Default format
            else:
                # Default values if line is malformed
                gain.append(200.0)
                baseline.append(0.0)
                units.append("mV")
                data_formats.append("16")
        
        # Fill remaining channels with defaults if needed
        while len(gain) < num_channels:
            gain.append(200.0)
            baseline.append(0.0)
            units.append("mV")
            data_formats.append("16")
        
        print(f"Data formats detected: {data_formats[:5]}...")  # Show first 5
        
        # Determine the most common data format
        primary_format = data_formats[0] if data_formats else "16"
        
        return fs, num_channels, gain, baseline, units, primary_format
        
    except Exception as e:
        print(f"Error parsing header file: {str(e)}")
        # Return default values
        return 250.0, 1, [200.0], [0.0], ["mV"], "16"

def read_dat_file(dat_path, num_channels, gain, baseline):
    """
    Read WFDB .dat file and convert to physical units.
    
    Parameters:
    -----------
    dat_path : str
        Path to the .dat file
    num_channels : int
        Number of channels in the data
    gain : list
        Gain values for each channel
    baseline : list
        Baseline values for each channel
        
    Returns:
    --------
    np.ndarray : Signal data in physical units
    """
    try:
        # Read binary file
        with open(dat_path, 'rb') as f:
            data = f.read()
        
        print(f"Read {len(data)} bytes from DAT file")
        
        # Try different data formats
        signal_data = None
        
        # Format 1: 16-bit signed integers (most common)
        try:
            raw_data = np.frombuffer(data, dtype=np.int16)
            print(f"Raw 16-bit data length: {len(raw_data)} samples")
            print(f"Expected samples per channel: {len(raw_data) / num_channels}")
            
            if len(raw_data) % num_channels == 0:
                signal_data = raw_data.reshape(-1, num_channels)
                print(f"Successfully parsed as 16-bit signed integers: {signal_data.shape}")
            else:
                # Check if it's close to being divisible - might be padding or header
                remainder = len(raw_data) % num_channels
                print(f"16-bit format: length mismatch - remainder: {remainder}")
                
                # Try removing some samples from the end (common with padding)
                if remainder < num_channels:
                    truncated_length = len(raw_data) - remainder
                    truncated_data = raw_data[:truncated_length]
                    if len(truncated_data) % num_channels == 0:
                        signal_data = truncated_data.reshape(-1, num_channels)
                        print(f"Successfully parsed after truncating {remainder} samples: {signal_data.shape}")
        except Exception as e:
            print(f"Failed to parse as 16-bit signed: {e}")
        
        # Format 2: Try 12-bit packed data format (3 bytes for 2 samples)
        if signal_data is None:
            try:
                print("Trying 12-bit packed format...")
                # 12-bit data is often packed as 3 bytes for every 2 samples
                if len(data) % 3 == 0:
                    num_sample_pairs = len(data) // 3
                    raw_12bit = []
                    
                    for i in range(0, len(data), 3):
                        if i + 2 < len(data):
                            # Read 3 bytes
                            b1, b2, b3 = data[i], data[i+1], data[i+2]
                            
                            # Extract two 12-bit values
                            sample1 = b1 | ((b2 & 0x0F) << 8)
                            sample2 = ((b2 & 0xF0) >> 4) | (b3 << 4)
                            
                            # Convert to signed 12-bit
                            if sample1 > 2047:
                                sample1 -= 4096
                            if sample2 > 2047:
                                sample2 -= 4096
                                
                            raw_12bit.extend([sample1, sample2])
                    
                    raw_12bit = np.array(raw_12bit)
                    print(f"12-bit unpacked data length: {len(raw_12bit)} samples")
                    
                    if len(raw_12bit) % num_channels == 0:
                        signal_data = raw_12bit.reshape(-1, num_channels)
                        print(f"Successfully parsed as 12-bit packed: {signal_data.shape}")
                    else:
                        # Try truncating for 12-bit as well
                        remainder = len(raw_12bit) % num_channels
                        if remainder < num_channels:
                            truncated_length = len(raw_12bit) - remainder
                            truncated_data = raw_12bit[:truncated_length]
                            if len(truncated_data) % num_channels == 0:
                                signal_data = truncated_data.reshape(-1, num_channels)
                                print(f"Successfully parsed 12-bit after truncating {remainder} samples: {signal_data.shape}")
            except Exception as e:
                print(f"Failed to parse as 12-bit packed: {e}")
        
        # Format 3: Try reading with byte swapping (big-endian vs little-endian)
        if signal_data is None:
            try:
                print("Trying byte-swapped 16-bit format...")
                raw_data = np.frombuffer(data, dtype='>i2')  # Big-endian 16-bit
                print(f"Byte-swapped 16-bit data length: {len(raw_data)} samples")
                
                if len(raw_data) % num_channels == 0:
                    signal_data = raw_data.reshape(-1, num_channels)
                    print(f"Successfully parsed as byte-swapped 16-bit: {signal_data.shape}")
                else:
                    remainder = len(raw_data) % num_channels
                    if remainder < num_channels:
                        truncated_length = len(raw_data) - remainder
                        truncated_data = raw_data[:truncated_length]
                        if len(truncated_data) % num_channels == 0:
                            signal_data = truncated_data.reshape(-1, num_channels)
                            print(f"Successfully parsed byte-swapped after truncating {remainder} samples: {signal_data.shape}")
            except Exception as e:
                print(f"Failed to parse as byte-swapped 16-bit: {e}")
        
        # Format 4: Try as interleaved bytes (for some custom formats)
        if signal_data is None:
            try:
                print("Trying custom byte arrangement...")
                raw_data = np.frombuffer(data, dtype=np.uint8)
                
                # Try reading as interleaved high/low bytes
                if len(raw_data) % (num_channels * 2) == 0:
                    # Combine pairs of bytes in different order
                    raw_16bit = []
                    for i in range(0, len(raw_data), 2):
                        if i + 1 < len(raw_data):
                            # Try different byte orders
                            val = (raw_data[i+1] << 8) | raw_data[i]  # Little-endian
                            # Convert to signed 16-bit
                            if val > 32767:
                                val -= 65536
                            raw_16bit.append(val)
                    
                    raw_16bit = np.array(raw_16bit)
                    if len(raw_16bit) % num_channels == 0:
                        signal_data = raw_16bit.reshape(-1, num_channels)
                        print(f"Successfully parsed as custom byte arrangement: {signal_data.shape}")
            except Exception as e:
                print(f"Failed to parse as custom byte arrangement: {e}")
        
        if signal_data is None:
            # Provide more detailed error information
            error_msg = f"Could not parse DAT file. Debug info:\n"
            error_msg += f"- File size: {len(data)} bytes\n"
            error_msg += f"- Expected channels: {num_channels}\n"
            error_msg += f"- 16-bit samples: {len(data) // 2}\n"
            error_msg += f"- Samples per channel (16-bit): {(len(data) // 2) / num_channels:.2f}\n"
            error_msg += f"- 12-bit samples: {(len(data) // 3) * 2}\n"
            error_msg += f"- Samples per channel (12-bit): {((len(data) // 3) * 2) / num_channels:.2f}\n"
            raise ValueError(error_msg)
        
        # Convert to physical units
        physical_data = np.zeros_like(signal_data, dtype=np.float64)
        for ch in range(min(num_channels, signal_data.shape[1])):
            if ch < len(gain) and ch < len(baseline):
                physical_data[:, ch] = (signal_data[:, ch] - baseline[ch]) / gain[ch]
            else:
                physical_data[:, ch] = signal_data[:, ch] / 200.0  # Default gain
        
        print(f"Converted to physical units: range [{np.min(physical_data):.3f}, {np.max(physical_data):.3f}]")
        
        return physical_data if num_channels > 1 else physical_data.flatten()
        
    except Exception as e:
        print(f"Error reading DAT file: {str(e)}")
        raise

def normalize_base_name(name):
    """
    Normalize base name for file matching.
    """
    # Remove extension, lower case, remove non-alphanumeric chars
    base = os.path.splitext(name)[0]
    base = re.sub(r'[^a-zA-Z0-9]', '', base).lower()
    return base

def process_wfdb_to_csv(uploaded_files, temp_dir):
    """
    Process uploaded WFDB files and convert to CSV format.
    UPDATED VERSION with improved error handling and format support.
    
    Parameters:
    -----------
    uploaded_files : list
        List of uploaded files (.dat and .hea)
    temp_dir : str
        Temporary directory to save files
        
    Returns:
    --------
    tuple : (time, ecg_signal, fs, csv_data, csv_filename)
        Time array, ECG signal, sampling frequency, CSV data, and filename
    """
    try:
        # First validate files
        is_valid, validation_message = validate_wfdb_files(uploaded_files)
        if not is_valid:
            raise ValueError(f"File validation failed: {validation_message}")
        
        print(f"Validation passed: {validation_message}")
        
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded files to temp directory
        file_map = {}
        
        print("Processing uploaded files...")
        for file in uploaded_files:
            if hasattr(file, 'name') and hasattr(file, 'getvalue'):
                # Get the base name without extension
                base_name = os.path.splitext(file.name)[0]
                
                # Save both .dat and .hea files with the same base name
                if file.name.lower().endswith('.dat'):
                    file_path = os.path.join(temp_dir, f"{base_name}.dat")
                    with open(file_path, 'wb') as f:
                        f.write(file.getvalue())
                    if base_name not in file_map:
                        file_map[base_name] = {}
                    file_map[base_name]['dat'] = file_path
                    print(f"Saved DAT file: {file_path}")
                    
                elif file.name.lower().endswith('.hea'):
                    file_path = os.path.join(temp_dir, f"{base_name}.hea")
                    with open(file_path, 'wb') as f:
                        f.write(file.getvalue())
                    if base_name not in file_map:
                        file_map[base_name] = {}
                    file_map[base_name]['hea'] = file_path
                    print(f"Saved HEA file: {file_path}")
                else:
                    print(f"Skipping unsupported file: {file.name}")
        
        # Find a pair with both .dat and .hea
        for base_name, files in file_map.items():
            if 'dat' in files and 'hea' in files:
                print(f"\nProcessing record: {base_name}")
                print(f"Data file: {files['dat']}")
                print(f"Header file: {files['hea']}")
                
                try:
                    # Parse header file
                    fs, num_channels, gain, baseline, units, data_format = parse_header_file(files['hea'])
                    print(f"Sampling frequency: {fs} Hz")
                    print(f"Number of channels: {num_channels}")
                    print(f"Data format: {data_format}")
                    print(f"Gain: {gain}")
                    print(f"Baseline: {baseline}")
                    
                    # Read binary data file
                    signal_data = read_dat_file(files['dat'], num_channels, gain, baseline)
                    print(f"Successfully read {len(signal_data)} samples")
                    
                    # Create time array
                    time = np.arange(len(signal_data)) / fs
                    
                    # Use first channel as ECG (or you can modify this)
                    if signal_data.ndim > 1:
                        ecg_signal = signal_data[:, 0]
                    else:
                        ecg_signal = signal_data
                    
                    # Create DataFrame
                    df = pd.DataFrame({
                        'time': time,
                        'ecg': ecg_signal
                    })
                    
                    # Add additional channels if available
                    if signal_data.ndim > 1 and num_channels > 1:
                        for ch in range(1, min(num_channels, signal_data.shape[1])):
                            df[f'channel_{ch+1}'] = signal_data[:, ch]
                    
                    # Convert DataFrame to CSV string
                    csv_data = df.to_csv(index=False)
                    csv_filename = f"{base_name}.csv"
                    
                    print(f"Successfully processed {len(time)} samples at {fs} Hz")
                    return time, ecg_signal, fs, csv_data, csv_filename
                    
                except Exception as e:
                    print(f"Error reading record {base_name}: {str(e)}")
                    print(f"Error type: {type(e).__name__}")
                    # Continue to try other files if available
                    continue
        
        # If no match found or all attempts failed
        available_files = []
        for base_name, files in file_map.items():
            available_files.append(f"{base_name}: {list(files.keys())}")
        
        error_msg = "Could not process WFDB files. Available files:\n"
        error_msg += "\n".join(available_files)
        error_msg += "\n\nPlease ensure:\n"
        error_msg += "1. Both .dat and .hea files are uploaded\n"
        error_msg += "2. Files have matching base names\n"
        error_msg += "3. Files are valid WFDB format"
        
        raise ValueError(error_msg)
        
    except Exception as e:
        print(f"Error processing WFDB file: {str(e)}")
        return None, None, None, None, None

# ============================================================================
# CSV DATA PROCESSING
# ============================================================================

def process_csv_data(csv_file, fs=None, start_time=0, duration=None):
    """
    Process ECG data from CSV file.
    
    Parameters:
    -----------
    csv_file : str or file-like object
        Path to CSV file or file object
    fs : int, optional
        Sampling frequency (if not specified, will try to infer from data)
    start_time : float
        Start time in seconds
    duration : float, optional
        Duration to process in seconds (if None, process entire file)
        
    Returns:
    --------
    tuple : (time, ecg_signal, metadata)
        Time array, ECG signal, and metadata dictionary
    """
    try:
        # Read CSV file
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        else:
            df = pd.read_csv(csv_file)
        
        # Check required columns
        required_cols = ['time', 'ecg']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV file must contain columns: {required_cols}")
        
        # Extract data
        time = df['time'].values
        ecg_signal = df['ecg'].values
        
        # Calculate sampling frequency if not provided
        if fs is None:
            if len(time) > 1:
                fs = 1 / np.mean(np.diff(time))
            else:
                fs = 250  # Default sampling frequency
        
        # Calculate start and end indices
        start_idx = int(start_time * fs)
        if duration is not None:
            end_idx = min(len(time), start_idx + int(duration * fs))
        else:
            end_idx = len(time)
        
        # Extract segment
        time_segment = time[start_idx:end_idx]
        ecg_segment = ecg_signal[start_idx:end_idx]
        
        # Create metadata dictionary
        metadata = {
            'sampling_frequency': fs,
            'signal_length': len(ecg_segment),
            'duration': len(ecg_segment) / fs,
            'start_time': start_time,
            'end_time': start_time + len(ecg_segment) / fs,
            'original_file': str(csv_file),
            'columns': df.columns.tolist()
        }
        
        # Add additional metadata if available
        if 'annotation' in df.columns:
            metadata['has_annotations'] = True
            metadata['annotation_count'] = df['annotation'].sum()
        else:
            metadata['has_annotations'] = False
        
        # Add basic signal statistics
        metadata.update({
            'mean_amplitude': np.mean(ecg_segment),
            'std_amplitude': np.std(ecg_segment),
            'min_amplitude': np.min(ecg_segment),
            'max_amplitude': np.max(ecg_segment)
        })
        
        return time_segment, ecg_segment, metadata
        
    except Exception as e:
        print(f"Error processing CSV data: {e}")
        return None, None, None

# ============================================================================
# ECG SIGNAL PROCESSING
# ============================================================================

def process_ecg(time, ecg_signal, fs, processing_params):
    """
    Process ECG signal with specified parameters.
    Args:
        time (np.ndarray): Time array.
        ecg_signal (np.ndarray): ECG signal (1D or 2D).
        fs (int): Sampling frequency.
        processing_params (dict): Dictionary of processing parameters.
    Returns:
        tuple: (filtered_signal, r_peaks, pqrst_peaks, intervals, heart_rate)
    """
    try:
        from scipy.signal import butter, filtfilt
        
        # Ensure signal is 1D
        if ecg_signal.ndim > 1:
            ecg_signal = ecg_signal.squeeze()
        
        # Bandpass filter
        lowcut = processing_params.get('lowcut', 5.0)
        highcut = processing_params.get('highcut', 30.0)
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Ensure cutoff frequencies are valid
        low = max(0.01, min(low, 0.99))
        high = max(0.01, min(high, 0.99))
        if low >= high:
            low = 0.02
            high = 0.4
        
        b, a = butter(3, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)

        # R-peak detection using simple peak detection if wfdb not available
        try:
            from wfdb.processing import XQRS
            xqrs = XQRS(sig=filtered_signal, fs=fs)
            xqrs.detect()
            r_peaks = xqrs.qrs_inds
        except ImportError:
            # Fallback to scipy peak detection
            from scipy.signal import find_peaks
            # Find peaks with minimum distance between peaks
            min_distance = int(0.6 * fs)  # Minimum 600ms between R peaks
            r_peaks, _ = find_peaks(filtered_signal, 
                                   height=np.percentile(filtered_signal, 75),
                                   distance=min_distance)

        # PQRST detection
        pqrst_peaks = detect_pqrst_peaks(filtered_signal, r_peaks, fs)

        # Calculate intervals
        intervals = calculate_intervals(pqrst_peaks, fs)
        
        # Heart rate
        heart_rate = calculate_heart_rate(r_peaks, fs)

        return filtered_signal, r_peaks, pqrst_peaks, intervals, heart_rate
        
    except Exception as e:
        print(f"Error processing ECG: {e}")
        return None, None, None, None, None

def detect_pqrst_peaks(signal, r_peaks, fs):
    """
    Detect P, Q, S, and T peaks around R peaks.
    Args:
        signal (np.ndarray): 1D ECG signal.
        r_peaks (np.ndarray): Indices of R peaks.
        fs (int): Sampling frequency.
    Returns:
        dict: Dictionary with keys 'p_peaks', 'q_peaks', 'r_peaks', 's_peaks', 't_peaks'
    """
    pqrst_peaks = {
        'p_peaks': [],
        'q_peaks': [],
        'r_peaks': r_peaks,
        's_peaks': [],
        't_peaks': []
    }
    
    for r in r_peaks:
        # Q wave: minimum before R
        q_start = max(0, r - int(0.05 * fs))
        if q_start < r:
            q_idx = np.argmin(signal[q_start:r]) + q_start
            pqrst_peaks['q_peaks'].append(q_idx)
        
        # S wave: minimum after R
        s_end = min(len(signal), r + int(0.08 * fs))
        if r < s_end:
            s_idx = np.argmin(signal[r:s_end]) + r
            pqrst_peaks['s_peaks'].append(s_idx)
        
        # P wave: maximum before Q
        if pqrst_peaks['q_peaks']:
            p_start = max(0, pqrst_peaks['q_peaks'][-1] - int(0.2 * fs))
            if p_start < pqrst_peaks['q_peaks'][-1]:
                p_idx = np.argmax(signal[p_start:pqrst_peaks['q_peaks'][-1]]) + p_start
                pqrst_peaks['p_peaks'].append(p_idx)
        
        # T wave: maximum after S
        if pqrst_peaks['s_peaks']:
            t_start = pqrst_peaks['s_peaks'][-1]
            t_end = min(len(signal), pqrst_peaks['s_peaks'][-1] + int(0.3 * fs))
            if t_start < t_end:
                t_idx = np.argmax(signal[t_start:t_end]) + t_start
                pqrst_peaks['t_peaks'].append(t_idx)
    
    return pqrst_peaks

def calculate_intervals(pqrst_peaks, fs):
    """
    Calculate ECG intervals (PR, QRS, QT, RR) from detected peaks.
    Args:
        pqrst_peaks (dict): Dictionary with keys 'p_peaks', 'q_peaks', 'r_peaks', 's_peaks', 't_peaks'
        fs (int): Sampling frequency.
    Returns:
        dict: Dictionary with mean and std for each interval.
    """
    intervals = {}
    p_peaks = np.array(pqrst_peaks.get('p_peaks', []))
    q_peaks = np.array(pqrst_peaks.get('q_peaks', []))
    r_peaks = np.array(pqrst_peaks.get('r_peaks', []))
    s_peaks = np.array(pqrst_peaks.get('s_peaks', []))
    t_peaks = np.array(pqrst_peaks.get('t_peaks', []))

    # RR intervals
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / fs * 1000  # ms
        intervals['rr_mean'] = np.mean(rr_intervals)
        intervals['rr_std'] = np.std(rr_intervals)

    # PR, QRS, QT intervals
    min_len = min(len(p_peaks), len(q_peaks), len(r_peaks), len(s_peaks), len(t_peaks))
    if min_len > 0:
        pr_intervals = (r_peaks[:min_len] - p_peaks[:min_len]) / fs * 1000
        pr_intervals = pr_intervals[pr_intervals > 0]
        if len(pr_intervals) > 0:
            intervals['pr_mean'] = np.mean(pr_intervals)
            intervals['pr_std'] = np.std(pr_intervals)

        qrs_intervals = (s_peaks[:min_len] - q_peaks[:min_len]) / fs * 1000
        qrs_intervals = qrs_intervals[qrs_intervals > 0]
        if len(qrs_intervals) > 0:
            intervals['qrs_mean'] = np.mean(qrs_intervals)
            intervals['qrs_std'] = np.std(qrs_intervals)

        qt_intervals = (t_peaks[:min_len] - q_peaks[:min_len]) / fs * 1000
        qt_intervals = qt_intervals[qt_intervals > 0]
        if len(qt_intervals) > 0:
            intervals['qt_mean'] = np.mean(qt_intervals)
            intervals['qt_std'] = np.std(qt_intervals)

    return intervals

def calculate_heart_rate(r_peaks, fs):
    """
    Calculate heart rate from R peaks.
    Args:
        r_peaks (np.ndarray): Indices of R peaks.
        fs (int): Sampling frequency.
    Returns:
        float: Heart rate in beats per minute (bpm), or None if not enough peaks.
    """
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / fs  # in seconds
        mean_rr = np.mean(rr_intervals)
        if mean_rr > 0:
            return 60.0 / mean_rr
    return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_ecg_with_peaks(time, signal, peaks, title="ECG Signal with Peaks"):
    """
    Plot ECG signal with detected peaks.
    Args:
        time (np.ndarray): Time array.
        signal (np.ndarray): ECG signal.
        peaks (dict): Dictionary with keys 'p_peaks', 'q_peaks', 'r_peaks', 's_peaks', 't_peaks' (each a list/array of indices).
        title (str): Plot title.
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(time, signal, 'b-', linewidth=1.5, label='ECG Signal')

    colors = {'p_peaks': 'green', 'q_peaks': 'orange', 'r_peaks': 'red', 's_peaks': 'purple', 't_peaks': 'cyan'}
    markers = {'p_peaks': 'o', 'q_peaks': 's', 'r_peaks': '^', 's_peaks': 'v', 't_peaks': 'D'}

    for peak_type, peak_indices in peaks.items():
        if len(peak_indices) > 0:
            color = colors.get(peak_type, 'black')
            marker = markers.get(peak_type, 'o')
            ax.scatter(time[peak_indices], signal[peak_indices], 
                       c=color, marker=marker, s=80, label=f"{peak_type.replace('_peaks', '').upper()} peaks")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_heartbeats(heartbeats, time_windows, title="Individual Heartbeats"):
    """
    Plot individual heartbeats overlaid on each other.
    Args:
        heartbeats (np.ndarray): Array of heartbeats, shape (n_beats, n_samples).
        time_windows (np.ndarray): Array of time windows, shape (n_beats, n_samples).
        title (str): Plot title.
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each heartbeat
    for i, (heartbeat, time_window) in enumerate(zip(heartbeats, time_windows)):
        ax.plot(time_window, heartbeat, alpha=0.3, label=f'Beat {i+1}' if i < 5 else None)

    # Plot mean heartbeat
    mean_heartbeat = np.mean(heartbeats, axis=0)
    ax.plot(time_windows[0], mean_heartbeat, 'r-', linewidth=2, label='Mean Beat')

    # Customize plot
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig

def plot_average_heartbeat(heartbeats, time_windows, title="Average Heartbeat"):
    """
    Plot the average heartbeat with standard deviation.
    Args:
        heartbeats (np.ndarray): Array of heartbeats, shape (n_beats, n_samples).
        time_windows (np.ndarray): Array of time windows, shape (n_beats, n_samples).
        title (str): Plot title.
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    # Calculate mean and standard deviation
    mean_heartbeat = np.mean(heartbeats, axis=0)
    std_heartbeat = np.std(heartbeats, axis=0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean heartbeat
    ax.plot(time_windows[0], mean_heartbeat, 'b-', linewidth=2, label='Mean Beat')

    # Plot standard deviation
    ax.fill_between(time_windows[0], 
                    mean_heartbeat - std_heartbeat,
                    mean_heartbeat + std_heartbeat,
                    alpha=0.2, color='b', label='±1 SD')

    # Customize plot
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig

def plot_feature_distribution(features, feature_name, title=None):
    """
    Plot the distribution of a feature using a histogram and KDE.
    Args:
        features (np.ndarray): Array of feature values.
        feature_name (str): Name of the feature.
        title (str, optional): Plot title. If None, will use feature_name.
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram with KDE
    sns.histplot(features, kde=True, ax=ax)

    # Add vertical line for mean
    mean_value = np.mean(features)
    ax.axvline(mean_value, color='r', linestyle='--', 
               label=f'Mean: {mean_value:.2f}')

    # Add vertical lines for mean ± std
    std_value = np.std(features)
    ax.axvline(mean_value + std_value, color='g', linestyle=':', 
               label=f'Mean ± Std: {mean_value + std_value:.2f}')
    ax.axvline(mean_value - std_value, color='g', linestyle=':')

    # Customize plot
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Count')
    ax.set_title(title or f'Distribution of {feature_name}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add statistics as text
    stats_text = f'Statistics:\nMean: {mean_value:.2f}\nStd: {std_value:.2f}\nMin: {np.min(features):.2f}\nMax: {np.max(features):.2f}'
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return fig

def plot_rr_intervals(r_peaks, fs, title="RR Intervals Analysis"):
    """
    Plot RR intervals over time and their distribution.
    Args:
        r_peaks (np.ndarray): Indices of R peaks.
        fs (int): Sampling frequency.
        title (str): Plot title.
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    if len(r_peaks) < 2:
        print("Not enough R peaks for RR interval analysis")
        return None
    
    # Calculate RR intervals
    rr_intervals = np.diff(r_peaks) / fs * 1000  # in milliseconds
    rr_times = r_peaks[1:] / fs  # time of each RR interval
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot RR intervals over time
    ax1.plot(rr_times, rr_intervals, 'b-', marker='o', markersize=4)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('RR Interval (ms)')
    ax1.set_title('RR Intervals Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Add mean line
    mean_rr = np.mean(rr_intervals)
    ax1.axhline(mean_rr, color='r', linestyle='--', label=f'Mean: {mean_rr:.1f} ms')
    ax1.legend()
    
    # Plot RR interval distribution
    sns.histplot(rr_intervals, kde=True, ax=ax2)
    ax2.axvline(mean_rr, color='r', linestyle='--', label=f'Mean: {mean_rr:.1f} ms')
    ax2.set_xlabel('RR Interval (ms)')
    ax2.set_ylabel('Count')
    ax2.set_title('RR Interval Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add statistics
    std_rr = np.std(rr_intervals)
    heart_rate = 60000 / mean_rr  # BPM
    stats_text = f'Statistics:\nMean: {mean_rr:.1f} ms\nStd: {std_rr:.1f} ms\nHeart Rate: {heart_rate:.1f} BPM'
    ax2.text(0.95, 0.95, stats_text,
            transform=ax2.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_figure_download_link(fig, filename="ecg_plot.png", link_text="Download Plot"):
    """
    Generate a download link for a matplotlib figure.
    Args:
        fig (matplotlib.figure.Figure): The figure to download.
        filename (str): Name of the file to download.
        link_text (str): Text to display for the download link.
    Returns:
        str: HTML link for downloading the figure.
    """
    # Save figure to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    
    # Encode the buffer to base64
    b64 = base64.b64encode(buf.read()).decode()
    
    # Create the download link
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{link_text}</a>'
    
    # Display the link using Streamlit
    st.markdown(href, unsafe_allow_html=True)
    
    return href

def get_download_link(data, filename, link_text="Download File"):
    """
    Generate a download link for data (CSV, JSON, etc.).
    Args:
        data (str or bytes): Data to download.
        filename (str): Name of the file to download.
        link_text (str): Text to display for the download link.
    Returns:
        str: HTML link for downloading the data.
    """
    # Convert data to bytes if it's a string
    if isinstance(data, str):
        data = data.encode()

    # Encode the data to base64
    b64 = base64.b64encode(data).decode()
    
    # Create the download link
    href = f'<a href="data:file/{filename};base64,{b64}" download="{filename}">{link_text}</a>'
    
    # Display the link using Streamlit
    st.markdown(href, unsafe_allow_html=True)
    
    return href

def save_results(results, filename="ecg_results.csv"):
    """
    Save ECG analysis results to a CSV file.
    Args:
        results (dict): Dictionary containing ECG analysis results.
        filename (str): Name of the file to save results to.
    Returns:
        str: Path to the saved file.
    """
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{os.path.splitext(filename)[0]}_{timestamp}.csv"

    # Convert results to DataFrame
    df = pd.DataFrame([results])

    # Save to CSV
    filepath = os.path.join(results_dir, filename)
    df.to_csv(filepath, index=False)

    return filepath

def validate_ecg_data(time, signal, fs):
    """
    Validate ECG data for processing.
    Args:
        time (np.ndarray): Time array.
        signal (np.ndarray): ECG signal.
        fs (int): Sampling frequency.
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check if inputs are numpy arrays
        if not isinstance(time, np.ndarray) or not isinstance(signal, np.ndarray):
            return False, "Time and signal must be numpy arrays"

        # Check if arrays have the same length
        if len(time) != len(signal):
            return False, "Time and signal arrays must have the same length"

        # Check if arrays are not empty
        if len(time) == 0 or len(signal) == 0:
            return False, "Time and signal arrays cannot be empty"

        # Check if sampling frequency is positive
        if fs <= 0:
            return False, "Sampling frequency must be positive"

        # Check if time array is monotonically increasing
        if not np.all(np.diff(time) > 0):
            return False, "Time array must be monotonically increasing"

        # Check if signal contains any NaN or infinite values
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return False, "Signal contains NaN or infinite values"

        # Check if signal is within reasonable range (e.g., -10 to 10 mV)
        if np.max(np.abs(signal)) > 10:
            return False, "Signal amplitude is outside reasonable range (-10 to 10 mV)"

        return True, "Data validation successful"

    except Exception as e:
        return False, f"Error during data validation: {str(e)}"

def load_file(file_path, fs=None):
    """
    Load ECG data from a file (CSV or WFDB format).
    Args:
        file_path (str): Path to the file.
        fs (int, optional): Sampling frequency. If None, will be inferred from data.
    Returns:
        tuple: (time, signal, fs, metadata)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file extension
        _, ext = os.path.splitext(file_path)

        # Load data based on file type
        if ext.lower() == '.csv':
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Check required columns
            if 'time' not in df.columns or 'ecg' not in df.columns:
                raise ValueError("CSV file must contain 'time' and 'ecg' columns")
            
            time = df['time'].values
            signal = df['ecg'].values
            
            # Calculate sampling frequency if not provided
            if fs is None and len(time) > 1:
                fs = 1 / np.mean(np.diff(time))
            elif fs is None:
                fs = 250  # Default sampling frequency

            metadata = {
                'file_type': 'csv',
                'columns': df.columns.tolist(),
                'num_samples': len(time),
                'duration': time[-1] - time[0] if len(time) > 1 else 0
            }

        elif ext.lower() in ['.dat', '.hea']:
            # Load WFDB file
            try:
                import wfdb
                record_name = os.path.splitext(file_path)[0]
                record = wfdb.rdrecord(record_name)
                
                time = np.arange(len(record.p_signal)) / record.fs
                signal = record.p_signal[:, 0]  # Assuming first channel is ECG
                fs = record.fs

                metadata = {
                    'file_type': 'wfdb',
                    'sampling_frequency': record.fs,
                    'num_samples': len(signal),
                    'duration': len(signal) / record.fs,
                    'units': record.units,
                    'comments': record.comments
                }
            except ImportError:
                raise ImportError("wfdb package is required to load WFDB files")

        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return time, signal, fs, metadata

    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None, None, None, None

def extract_heartbeats(signal, r_peaks, fs, window_size=0.4):
    """
    Extract individual heartbeats from ECG signal using R-peak locations.
    Args:
        signal (np.ndarray): ECG signal.
        r_peaks (np.ndarray): Indices of R peaks.
        fs (int): Sampling frequency.
        window_size (float): Window size in seconds (before and after R-peak).
    Returns:
        tuple: (heartbeats, time_windows)
    """
    try:
        # Calculate window size in samples
        window_samples = int(window_size * fs)
        
        # Initialize lists to store heartbeats and time windows
        heartbeats = []
        time_windows = []
        
        # Extract each heartbeat
        for r_peak in r_peaks:
            # Calculate start and end indices
            start_idx = max(0, r_peak - window_samples)
            end_idx = min(len(signal), r_peak + window_samples)
            
            # Extract heartbeat
            heartbeat = signal[start_idx:end_idx]
            
            # Create time window
            time_window = np.arange(-window_samples, len(heartbeat) - window_samples) / fs
            
            # Only keep complete heartbeats
            if len(heartbeat) == 2 * window_samples + 1:
                heartbeats.append(heartbeat)
                time_windows.append(time_window)
        
        # Convert to numpy arrays
        heartbeats = np.array(heartbeats)
        time_windows = np.array(time_windows)
        
        return heartbeats, time_windows
        
    except Exception as e:
        print(f"Error extracting heartbeats: {str(e)}")
        return None, None

def calculate_hrv_features(rr_intervals):
    """
    Calculate Heart Rate Variability (HRV) features.
    Args:
        rr_intervals (np.ndarray): RR intervals in milliseconds.
    Returns:
        dict: Dictionary containing HRV features.
    """
    try:
        if len(rr_intervals) < 2:
            return {}
        
        # Time domain features
        nn_intervals = rr_intervals  # Normal-to-normal intervals (assuming all are normal)
        
        hrv_features = {
            # Basic statistics
            'mean_rr': np.mean(nn_intervals),
            'std_rr': np.std(nn_intervals),
            'min_rr': np.min(nn_intervals),
            'max_rr': np.max(nn_intervals),
            
            # RMSSD: Root mean square of successive differences
            'rmssd': np.sqrt(np.mean(np.diff(nn_intervals)**2)),
            
            # SDNN: Standard deviation of normal-to-normal intervals
            'sdnn': np.std(nn_intervals),
            
            # pNN50: Percentage of successive RR intervals that differ by more than 50ms
            'pnn50': np.sum(np.abs(np.diff(nn_intervals)) > 50) / len(np.diff(nn_intervals)) * 100,
            
            # Triangular index (simplified)
            'triangular_index': len(nn_intervals) / np.max(np.histogram(nn_intervals, bins=50)[0]),
        }
        
        # Additional derived measures
        hrv_features['cv_rr'] = hrv_features['std_rr'] / hrv_features['mean_rr'] * 100  # Coefficient of variation
        hrv_features['range_rr'] = hrv_features['max_rr'] - hrv_features['min_rr']
        
        return hrv_features
        
    except Exception as e:
        print(f"Error calculating HRV features: {str(e)}")
        return {}

def detect_arrhythmias(rr_intervals, heart_rate):
    """
    Simple arrhythmia detection based on RR intervals and heart rate.
    Args:
        rr_intervals (np.ndarray): RR intervals in milliseconds.
        heart_rate (float): Average heart rate in BPM.
    Returns:
        dict: Dictionary containing arrhythmia detection results.
    """
    try:
        results = {
            'bradycardia': heart_rate < 60,
            'tachycardia': heart_rate > 100,
            'irregular_rhythm': False,
            'warnings': []
        }
        
        if len(rr_intervals) > 1:
            # Check for irregular rhythm
            rr_std = np.std(rr_intervals)
            rr_mean = np.mean(rr_intervals)
            cv = rr_std / rr_mean * 100
            
            if cv > 5:  # Coefficient of variation > 5%
                results['irregular_rhythm'] = True
                results['warnings'].append(f"Irregular rhythm detected (CV: {cv:.1f}%)")
        
        if results['bradycardia']:
            results['warnings'].append(f"Bradycardia detected (HR: {heart_rate:.1f} BPM)")
        
        if results['tachycardia']:
            results['warnings'].append(f"Tachycardia detected (HR: {heart_rate:.1f} BPM)")
        
        return results
        
    except Exception as e:
        print(f"Error detecting arrhythmias: {str(e)}")
        return {'error': str(e)}

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_ecg_comprehensive(time, signal, fs, processing_params=None):
    """
    Comprehensive ECG analysis including processing, feature extraction, and visualization.
    Args:
        time (np.ndarray): Time array.
        signal (np.ndarray): ECG signal.
        fs (int): Sampling frequency.
        processing_params (dict, optional): Processing parameters.
    Returns:
        dict: Comprehensive analysis results.
    """
    try:
        # Default processing parameters
        if processing_params is None:
            processing_params = {
                'lowcut': 5.0,
                'highcut': 30.0,
                'window_size': 0.4
            }
        
        # Validate input data
        is_valid, validation_msg = validate_ecg_data(time, signal, fs)
        if not is_valid:
            raise ValueError(f"Data validation failed: {validation_msg}")
        
        # Process ECG signal
        filtered_signal, r_peaks, pqrst_peaks, intervals, heart_rate = process_ecg(
            time, signal, fs, processing_params)
        
        if filtered_signal is None:
            raise ValueError("ECG processing failed")
        
        # Extract heartbeats
        heartbeats, time_windows = extract_heartbeats(
            filtered_signal, r_peaks, fs, processing_params.get('window_size', 0.4))
        
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / fs * 1000 if len(r_peaks) > 1 else np.array([])
        
        # Calculate HRV features
        hrv_features = calculate_hrv_features(rr_intervals)
        
        # Detect arrhythmias
        arrhythmia_results = detect_arrhythmias(rr_intervals, heart_rate)
        
        # Compile comprehensive results
        results = {
            'processing': {
                'filtered_signal': filtered_signal,
                'original_signal': signal,
                'time': time,
                'fs': fs,
                'processing_params': processing_params
            },
            'peaks': pqrst_peaks,
            'intervals': intervals,
            'heart_rate': heart_rate,
            'heartbeats': {
                'individual_beats': heartbeats,
                'time_windows': time_windows,
                'count': len(heartbeats) if heartbeats is not None else 0
            },
            'rr_intervals': rr_intervals,
            'hrv_features': hrv_features,
            'arrhythmia_detection': arrhythmia_results,
            'summary': {
                'duration': time[-1] - time[0] if len(time) > 1 else 0,
                'num_beats': len(r_peaks),
                'avg_heart_rate': heart_rate,
                'signal_quality': 'Good' if heart_rate and 50 <= heart_rate <= 150 else 'Poor'
            }
        }
        
        return results
        
    except Exception as e:
        print(f"Error in comprehensive ECG analysis: {str(e)}")
        return {'error': str(e)}

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def demo_synthetic_ecg():
    """
    Demonstrate the ECG analysis pipeline with synthetic data.
    """
    print("Generating synthetic ECG data...")
    time, signal = generate_synthetic_ecg(duration=10, fs=250, hr=75, noise_level=0.05)
    
    if time is not None and signal is not None:
        print("Running comprehensive analysis...")
        results = analyze_ecg_comprehensive(time, signal, 250)
        
        if 'error' not in results:
            print(f"Analysis completed successfully!")
            print(f"- Duration: {results['summary']['duration']:.1f} seconds")
            print(f"- Number of beats: {results['summary']['num_beats']}")
            print(f"- Average heart rate: {results['summary']['avg_heart_rate']:.1f} BPM")
            print(f"- Signal quality: {results['summary']['signal_quality']}")
        else:
            print(f"Analysis failed: {results['error']}")
    else:
        print("Failed to generate synthetic ECG data")

def extract_ecg_features(signal, r_peaks, fs, window_size=0.4):
    """
    Extract features from ECG signal for each heartbeat.
    
    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal array
    r_peaks : numpy.ndarray
        Array of R-peak locations
    fs : float
        Sampling frequency in Hz
    window_size : float
        Window size in seconds for feature extraction
        
    Returns:
    --------
    dict : Dictionary containing extracted features
    """
    try:
        import numpy as np
        from scipy import signal as sig
        
        features = {
            'heart_rate': [],
            'rr_intervals': [],
            'qrs_duration': [],
            'pr_interval': [],
            'qt_interval': [],
            'p_wave_amplitude': [],
            't_wave_amplitude': [],
            'st_segment': [],
            'heartbeat_morphology': []
        }
        
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / fs * 1000  # Convert to ms
        features['rr_intervals'] = rr_intervals
        
        # Calculate heart rate
        heart_rate = 60 / (np.mean(rr_intervals) / 1000)  # Convert to BPM
        features['heart_rate'] = heart_rate
        
        # Extract features for each heartbeat
        window_samples = int(window_size * fs)
        
        for i, r_peak in enumerate(r_peaks):
            # Extract heartbeat window
            start = max(0, r_peak - window_samples)
            end = min(len(signal), r_peak + window_samples)
            heartbeat = signal[start:end]
            
            # QRS duration (using zero crossing points)
            qrs_window = heartbeat[int(window_samples*0.4):int(window_samples*0.6)]
            zero_crossings = np.where(np.diff(np.signbit(qrs_window)))[0]
            if len(zero_crossings) >= 2:
                qrs_duration = (zero_crossings[-1] - zero_crossings[0]) / fs * 1000  # ms
                features['qrs_duration'].append(qrs_duration)
            
            # P wave amplitude
            p_window = heartbeat[int(window_samples*0.2):int(window_samples*0.4)]
            p_amplitude = np.max(p_window) - np.min(p_window)
            features['p_wave_amplitude'].append(p_amplitude)
            
            # T wave amplitude
            t_window = heartbeat[int(window_samples*0.6):int(window_samples*0.8)]
            t_amplitude = np.max(t_window) - np.min(t_window)
            features['t_wave_amplitude'].append(t_amplitude)
            
            # ST segment (average value in ST segment)
            st_window = heartbeat[int(window_samples*0.5):int(window_samples*0.6)]
            st_segment = np.mean(st_window)
            features['st_segment'].append(st_segment)
            
            # Heartbeat morphology (using wavelet transform)
            coeffs = sig.cwt(heartbeat, sig.ricker, np.arange(1, 31))
            morphology = np.mean(coeffs, axis=1)
            features['heartbeat_morphology'].append(morphology)
        
        # Calculate mean values for scalar features
        for key in ['qrs_duration', 'p_wave_amplitude', 't_wave_amplitude', 'st_segment']:
            if features[key]:
                features[key] = np.mean(features[key])
        
        # Add statistical features
        features['rr_std'] = np.std(rr_intervals)
        features['rr_mean'] = np.mean(rr_intervals)
        features['rr_cv'] = features['rr_std'] / features['rr_mean']  # Coefficient of variation
        
        return features
        
    except Exception as e:
        print(f"Error extracting ECG features: {str(e)}")
        return None

def plot_ecg_features(features, title="ECG Features"):
    """
    Plot extracted ECG features.
    
    Parameters:
    -----------
    features : dict
        Dictionary containing extracted features
    title : str
        Plot title
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title)
        
        # Plot RR intervals
        sns.histplot(features['rr_intervals'], ax=axes[0,0])
        axes[0,0].set_title('RR Intervals Distribution')
        axes[0,0].set_xlabel('RR Interval (ms)')
        axes[0,0].set_ylabel('Count')
        
        # Plot heart rate
        axes[0,1].bar(['Heart Rate'], [features['heart_rate']])
        axes[0,1].set_title('Average Heart Rate')
        axes[0,1].set_ylabel('BPM')
        
        # Plot wave amplitudes
        wave_amplitudes = {
            'P Wave': features['p_wave_amplitude'],
            'T Wave': features['t_wave_amplitude']
        }
        axes[1,0].bar(wave_amplitudes.keys(), wave_amplitudes.values())
        axes[1,0].set_title('Wave Amplitudes')
        axes[1,0].set_ylabel('Amplitude')
        
        # Plot QRS duration
        axes[1,1].bar(['QRS Duration'], [features['qrs_duration']])
        axes[1,1].set_title('Average QRS Duration')
        axes[1,1].set_ylabel('Duration (ms)')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error plotting ECG features: {str(e)}")
        return None

if __name__ == "__main__":
    # Run demo
    demo_synthetic_ecg()