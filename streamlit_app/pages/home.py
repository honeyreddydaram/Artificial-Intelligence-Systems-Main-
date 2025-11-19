"""
Home page for the ECG Analysis Streamlit app.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import helper functions
from streamlit_app.utils.helpers import generate_synthetic_ecg

def show():
    """Show the Home page."""
    st.title("ECG Analysis Tool")
    
    # App description
    st.markdown("""
    ## Welcome to the ECG Analysis Application
    
    This application provides a comprehensive toolkit for analyzing ECG signals, with a focus on:
    
    - **Signal Processing**: Clean and filter ECG signals
    - **PQRST Peak Detection**: Identify important landmarks in ECG waveforms
    - **Feature Extraction**: Extract statistical and morphological features
    - **Visualization**: Interactive plotting of ECG signals and detected peaks
    
    ### Getting Started
    
    1. Navigate to the **ECG Analysis** page to upload or generate ECG data
    2. Process the signal to detect peaks and extract features
    3. Use the **Visualization** page to explore the results in detail
    
    ### Data Sources
    
    This tool works with ECG data from:
    
    - CSV files with ECG signal data
    - MIT-BIH Arrhythmia Database files (.dat, .hea, .atr)
    - Synthetic ECG data generated within the app
    """)
    
    # Demo ECG visualization
    st.header("ECG Signal Example")
    
    # Generate a demo ECG signal
    time, ecg_signal = generate_synthetic_ecg(
        duration=5,
        fs=250,
        hr=70,
        noise_level=0.05
    )
    
    if time is not None and ecg_signal is not None:
        # Plot the demo ECG
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time, ecg_signal)
        
        # Add PQRST labels at specific positions
        # These positions are approximate for visualization
        fs = 250  # sampling frequency
        for i in range(3):
            t_center = i * 0.85 + 0.5  # R-peak positions
            
            # R peak
            ax.plot(t_center, ecg_signal[int(t_center * fs)], 'ro', markersize=8)
            ax.annotate('R', (t_center, ecg_signal[int(t_center * fs)]), 
                       xytext=(0, 10), textcoords='offset points', ha='center')
            
            # P wave (before R)
            p_center = t_center - 0.15
            ax.plot(p_center, ecg_signal[int(p_center * fs)], 'bo', markersize=6)
            ax.annotate('P', (p_center, ecg_signal[int(p_center * fs)]), 
                       xytext=(0, 10), textcoords='offset points', ha='center')
            
            # Q wave (just before R)
            q_center = t_center - 0.04
            ax.plot(q_center, ecg_signal[int(q_center * fs)], 'go', markersize=6)
            ax.annotate('Q', (q_center, ecg_signal[int(q_center * fs)]), 
                       xytext=(0, -15), textcoords='offset points', ha='center')
            
            # S wave (just after R)
            s_center = t_center + 0.04
            ax.plot(s_center, ecg_signal[int(s_center * fs)], 'mo', markersize=6)
            ax.annotate('S', (s_center, ecg_signal[int(s_center * fs)]), 
                       xytext=(0, -15), textcoords='offset points', ha='center')
            
            # T wave (after R)
            t_center = t_center + 0.2
            ax.plot(t_center, ecg_signal[int(t_center * fs)], 'co', markersize=6)
            ax.annotate('T', (t_center, ecg_signal[int(t_center * fs)]), 
                       xytext=(0, 10), textcoords='offset points', ha='center')
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Example ECG Signal with PQRST Waves")
        ax.grid(True)
        
        st.pyplot(fig)
    
    # Features section
    st.header("Key Features")
    
    # Create three columns for features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Signal Processing")
        st.markdown("""
        - Baseline wander removal
        - Powerline interference filtering
        - Bandpass filtering
        - Signal normalization
        """)
    
    with col2:
        st.subheader("Peak Detection")
        st.markdown("""
        - R-peak detection algorithms
        - PQRST wave identification
        - Heart rate calculation
        - Interval measurements (PR, QRS, QT)
        """)
    
    with col3:
        st.subheader("Feature Extraction")
        st.markdown("""
        - Statistical features
        - Morphological features
        - Frequency domain analysis
        - Wavelet-based features
        """)
    
    # Research information
    st.header("About the Research")
    
    st.markdown("""
    This application is part of a research project on ECG signal analysis and arrhythmia detection. 
    The methods implemented here follow established signal processing techniques and novel approaches 
    for feature extraction.
    
    The key objectives of this research include:
    
    1. Developing robust algorithms for ECG peak detection
    2. Extracting clinically relevant features from ECG signals
    3. Building a user-friendly tool for ECG analysis
    
    For more information, please refer to our research paper or contact the research team.
    """)
    
    # Usage instructions
    with st.expander("Usage Instructions", expanded=False):
        st.markdown("""
        ### How to Use This Application
        
        #### Data Input
        - **Upload File**: Navigate to the Analysis page and upload your ECG data file (CSV, TXT, or DAT format)
        - **Generate Synthetic ECG**: If you don't have real ECG data, you can generate synthetic signals for testing
        
        #### Signal Processing
        - Adjust filtering parameters to clean the signal
        - Select R-peak detection method (Pan-Tompkins or XQRS)
        - Process the signal to detect all PQRST peaks
        
        #### Feature Extraction
        - Extract heartbeats from the processed signal
        - Generate statistical and morphological features
        - Analyze feature distributions and correlations
        
        #### Visualization
        - View the raw and processed signals
        - Explore individual heartbeats and average patterns
        - Analyze feature distributions and relationships
        
        #### Results
        - Save analysis results for future reference
        - Download data files and figures for further analysis
        """)