"""
ECG Visualization page for the Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import helper functions
from streamlit_app.utils.helpers import (
    plot_ecg_with_peaks, plot_heartbeats, plot_average_heartbeat, 
    plot_feature_distribution, get_download_link, get_figure_download_link
)

def show():
    """Show the ECG Visualization page."""
    st.title("ECG Visualization")
    
    # Set up tabs for different visualization types
    tabs = st.tabs(["Results Explorer", "Signal Visualization", "Heartbeat Analysis", "Feature Analysis"])
    
    # Tab 1: Results Explorer
    with tabs[0]:
        st.header("Results Explorer")
        
        # Find results directories
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results')
        
        # Create directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            st.info("No analysis results found. Run analyses from the Analysis page first.")
        else:
            # Get list of results directories
            result_dirs = glob.glob(os.path.join(results_dir, "ecg_analysis_*"))
            
            if not result_dirs:
                st.info("No analysis results found. Run analyses from the Analysis page first.")
            else:
                # Sort by modification time (newest first)
                result_dirs.sort(key=os.path.getmtime, reverse=True)
                
                # Format directory names for display
                display_names = [os.path.basename(d) for d in result_dirs]
                
                # Convert timestamp to readable format
                readable_names = []
                for name in display_names:
                    try:
                        # Extract timestamp (assuming format "ecg_analysis_YYYYMMDD_HHMMSS")
                        timestamp_str = name.replace("ecg_analysis_", "")
                        timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        readable_names.append(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    except:
                        readable_names.append(name)
                
                # Create a dictionary mapping readable names to directory paths
                name_to_dir = dict(zip(readable_names, result_dirs))
                
                # Select result to view
                selected_result = st.selectbox(
                    "Select analysis result:",
                    readable_names
                )
                
                if selected_result:
                    selected_dir = name_to_dir[selected_result]
                    
                    # Display metadata if available
                    metadata_file = os.path.join(selected_dir, "metadata.txt")
                    if os.path.exists(metadata_file):
                        st.subheader("Analysis Metadata")
                        
                        # Read metadata
                        with open(metadata_file, "r") as f:
                            metadata_text = f.read()
                        
                        # Display in expandable section
                        with st.expander("View Metadata", expanded=True):
                            st.text(metadata_text)
                    
                    # Available files
                    st.subheader("Available Data Files")
                    
                    # Find CSV files
                    csv_files = glob.glob(os.path.join(selected_dir, "*.csv"))
                    csv_files = [os.path.basename(f) for f in csv_files]
                    
                    if csv_files:
                        # Display as a grid of buttons
                        cols = st.columns(3)
                        for i, file in enumerate(csv_files):
                            col_idx = i % 3
                            with cols[col_idx]:
                                if st.button(file):
                                    # Read and display the file
                                    try:
                                        df = pd.read_csv(os.path.join(selected_dir, file))
                                        st.dataframe(df)
                                        
                                        # Download link
                                        st.markdown(get_download_link(df, file, f"Download {file}"), unsafe_allow_html=True)
                                    except Exception as e:
                                        st.error(f"Error reading file: {e}")
                    else:
                        st.info("No data files found in this result.")
    
    # Tab 2: Signal Visualization
    with tabs[1]:
        st.header("Signal Visualization")
        
        # Check if ECG data is available in session state
        if 'ecg_data' not in st.session_state or st.session_state.ecg_data is None:
            st.info("No ECG data loaded. Go to the Analysis page to load or generate ECG data.")
        else:
            # Get data from session state
            time, ecg_signal, fs = st.session_state.ecg_data
            
            # Signal view options
            st.subheader("View Options")
            
            # Duration selector
            view_duration = st.slider(
                "View Duration (seconds):",
                min_value=1.0,
                max_value=min(30.0, len(time)/fs),
                value=5.0,
                step=1.0,
                key="sig_duration"
            )
            
            # Start time selector
            max_start_time = max(0, len(time)/fs - view_duration)
            start_time = st.slider(
                "Start Time (seconds):",
                min_value=0.0,
                max_value=max_start_time,
                value=0.0,
                step=1.0,
                key="sig_start"
            )
            
            # Channel selector
            channel = st.selectbox(
                "Channel:",
                range(ecg_signal.shape[1]),
                format_func=lambda x: f"Channel {x+1}",
                key="sig_channel"
            )
            
            # Signal view type
            view_type = st.radio(
                "View Type:",
                ["Raw", "Processed", "Raw & Processed", "With Peaks"],
                horizontal=True
            )
            
            # Plot the signal
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Calculate sample range
            start_sample = int(start_time * fs)
            end_sample = min(len(time), start_sample + int(view_duration * fs))
            
            if view_type == "Raw" or view_type == "Raw & Processed":
                ax.plot(time[start_sample:end_sample], 
                        ecg_signal[start_sample:end_sample, channel], 
                        'b-', label="Raw ECG")
            
            if (view_type == "Processed" or view_type == "Raw & Processed") and \
               'processed_signal' in st.session_state and st.session_state.processed_signal is not None:
                ax.plot(time[start_sample:end_sample], 
                        st.session_state.processed_signal[start_sample:end_sample, channel], 
                        'g-', label="Processed ECG")
            
            if view_type == "With Peaks" and \
               'processed_signal' in st.session_state and st.session_state.processed_signal is not None and \
               'pqrst_peaks' in st.session_state and st.session_state.pqrst_peaks is not None:
                # Plot the processed signal
                ax.plot(time[start_sample:end_sample], 
                        st.session_state.processed_signal[start_sample:end_sample, channel], 
                        'g-', label="Processed ECG")
                
                # Colors and markers for each wave
                wave_styles = {
                    'P': {'color': 'blue', 'marker': 'o', 'label': 'P-wave'},
                    'Q': {'color': 'green', 'marker': 's', 'label': 'Q-wave'},
                    'R': {'color': 'red', 'marker': '^', 'label': 'R-peak'},
                    'S': {'color': 'purple', 'marker': 'd', 'label': 'S-wave'},
                    'T': {'color': 'cyan', 'marker': '*', 'label': 'T-wave'}
                }
                
                # Plot peaks
                for wave, indices in st.session_state.pqrst_peaks.items():
                    # Filter peaks within the plotting window
                    indices_in_window = indices[(indices >= start_sample) & (indices < end_sample)]
                    if len(indices_in_window) > 0:
                        ax.plot(time[indices_in_window], 
                                st.session_state.processed_signal[indices_in_window, channel], 
                                marker=wave_styles[wave]['marker'], 
                                color=wave_styles[wave]['color'], 
                                linestyle='none', 
                                markersize=8, 
                                label=wave_styles[wave]['label'])
            
            # Set axis labels and title
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"ECG Signal - Channel {channel+1}")
            ax.grid(True)
            ax.legend()
            
            # Display the plot
            st.pyplot(fig)
            
            # Download link for figure
            st.markdown(get_figure_download_link(fig, "ecg_signal.png", "Download Figure"), unsafe_allow_html=True)
    
    # Tab 3: Heartbeat Analysis
    with tabs[2]:
        st.header("Heartbeat Analysis")
        
        # Check if heartbeats are available in session state
        if 'heartbeats' not in st.session_state or st.session_state.heartbeats is None:
            st.info("No heartbeats extracted. Go to the Analysis page to extract heartbeats.")
        else:
            # Get heartbeats from session state
            heartbeats = st.session_state.heartbeats
            fs = st.session_state.ecg_data[2]  # Get fs from ecg_data tuple
            
            # Heartbeat view options
            st.subheader("View Options")
            
            # View type selector
            view_type = st.radio(
                "View Type:",
                ["Individual Beats", "Average Beat", "All Beats Overlay"],
                horizontal=True,
                key="beat_view_type"
            )
            
            if view_type == "Individual Beats":
                # Number of beats to display
                num_beats = st.slider(
                    "Number of beats to display:",
                    min_value=1,
                    max_value=min(10, len(heartbeats)),
                    value=5,
                    step=1,
                    key="num_beats"
                )
                
                # Start beat selector
                start_beat = st.slider(
                    "Start Beat:",
                    min_value=0,
                    max_value=max(0, len(heartbeats) - num_beats),
                    value=0,
                    step=1,
                    key="start_beat"
                )
                
                # Plot heartbeats
                fig, axes = plt.subplots(num_beats, 1, figsize=(10, 2 * num_beats), sharex=True)
                if num_beats == 1:
                    axes = [axes]
                
                # Create time array for each beat
                beat_samples = heartbeats.shape[1]
                beat_time = np.arange(beat_samples) / fs - 0.25  # Assuming 0.25s before R peak
                
                # Plot each beat
                for i in range(num_beats):
                    beat_idx = start_beat + i
                    if beat_idx < len(heartbeats):
                        axes[i].plot(beat_time, heartbeats[beat_idx])
                        axes[i].axvline(x=0, color='r', linestyle='--', label='R Peak' if i == 0 else '')
                        axes[i].set_title(f"Heartbeat {beat_idx+1}")
                        axes[i].set_ylabel("Amplitude")
                        axes[i].grid(True)
                
                # Set common x-label
                axes[-1].set_xlabel("Time (s)")
                
                # Add legend to first subplot only
                axes[0].legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
            elif view_type == "Average Beat":
                # Plot average heartbeat
                fig = plot_average_heartbeat(heartbeats, fs)
                st.pyplot(fig)
                
            elif view_type == "All Beats Overlay":
                # Plot all beats on a single plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create time array for each beat
                beat_samples = heartbeats.shape[1]
                beat_time = np.arange(beat_samples) / fs - 0.25  # Assuming 0.25s before R peak
                
                # Plot each beat with low alpha
                for i in range(min(100, len(heartbeats))):  # Limit to 100 beats for clarity
                    ax.plot(beat_time, heartbeats[i], 'b-', alpha=0.1)
                
                # Plot mean beat
                mean_beat = np.mean(heartbeats, axis=0)
                ax.plot(beat_time, mean_beat, 'r-', linewidth=2, label='Mean Beat')
                
                # Mark R peak
                ax.axvline(x=0, color='k', linestyle='--', label='R Peak')
                
                # Set labels and title
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_title(f"All Heartbeats Overlay (n={min(100, len(heartbeats))})")
                ax.legend()
                ax.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Download link for figure
            st.markdown(get_figure_download_link(fig, "heartbeat_analysis.png", "Download Figure"), unsafe_allow_html=True)
    
    # Tab 4: Feature Analysis
    with tabs[3]:
        st.header("Feature Analysis")
        
        # Check if features are available in session state
        if 'features_df' not in st.session_state or st.session_state.features_df is None:
            st.info("No features extracted. Go to the Analysis page to extract features.")
        else:
            # Get features from session state
            features_df = st.session_state.features_df
            
            # Feature options
            numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [col for col in numeric_features if col != 'beat_idx']  # Exclude beat index
            
            # Feature statistics
            st.subheader("Feature Statistics")
            
            # Display summary statistics
            with st.expander("View Summary Statistics", expanded=True):
                st.dataframe(features_df[numeric_features].describe())
            
            # Feature visualization options
            st.subheader("Feature Visualization")
            
            # Select visualization type
            viz_type = st.radio(
                "Visualization Type:",
                ["Distribution", "Correlation", "Feature vs. Beat"],
                horizontal=True
            )
            
            if viz_type == "Distribution":
                # Select feature to visualize
                selected_feature = st.selectbox(
                    "Select Feature:",
                    numeric_features
                )
                
                if selected_feature:
                    # Plot feature distribution
                    fig = plot_feature_distribution(features_df, selected_feature)
                    st.pyplot(fig)
                    
                    # Download link for figure
                    st.markdown(get_figure_download_link(fig, "feature_distribution.png", "Download Figure"), unsafe_allow_html=True)
            
            elif viz_type == "Correlation":
                # Number of features to include in correlation matrix
                num_features = st.slider(
                    "Number of features:",
                    min_value=5,
                    max_value=min(30, len(numeric_features)),
                    value=15,
                    step=5
                )
                
                # Select features for correlation matrix
                selected_features = numeric_features[:num_features]
                
                # Calculate correlation matrix
                corr_matrix = features_df[selected_features].corr()
                
                # Plot correlation matrix
                fig, ax = plt.subplots(figsize=(12, 10))
                cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                fig.colorbar(cax)
                
                # Set ticks and labels
                ax.set_xticks(np.arange(len(selected_features)))
                ax.set_yticks(np.arange(len(selected_features)))
                ax.set_xticklabels(selected_features, rotation=90)
                ax.set_yticklabels(selected_features)
                
                # Add correlation values as text
                for i in range(len(selected_features)):
                    for j in range(len(selected_features)):
                        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                                ha="center", va="center", 
                                color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                
                plt.title("Feature Correlation Matrix")
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download link for figure
                st.markdown(get_figure_download_link(fig, "correlation_matrix.png", "Download Figure"), unsafe_allow_html=True)
            
            elif viz_type == "Feature vs. Beat":
                # Select feature to visualize
                selected_feature = st.selectbox(
                    "Select Feature:",
                    numeric_features,
                    key="feature_vs_beat"
                )
                
                if selected_feature:
                    # Plot feature vs. beat number
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Create beat index
                    beat_idx = features_df['beat_idx'] if 'beat_idx' in features_df.columns else np.arange(len(features_df))
                    
                    # Plot feature vs. beat number
                    ax.plot(beat_idx, features_df[selected_feature], 'b-', marker='o', markersize=4)
                    
                    # Add mean line
                    ax.axhline(y=features_df[selected_feature].mean(), color='r', linestyle='--', 
                              label=f'Mean: {features_df[selected_feature].mean():.3f}')
                    
                    # Add standard deviation band
                    ax.fill_between(beat_idx, 
                                   features_df[selected_feature].mean() - features_df[selected_feature].std(),
                                   features_df[selected_feature].mean() + features_df[selected_feature].std(),
                                   color='r', alpha=0.2, label='±1 SD')
                    
                    # Set labels and title
                    ax.set_xlabel("Beat Number")
                    ax.set_ylabel(selected_feature)
                    ax.set_title(f"{selected_feature} vs. Beat Number")
                    ax.legend()
                    ax.grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download link for figure
                    st.markdown(get_figure_download_link(fig, "feature_vs_beat.png", "Download Figure"), unsafe_allow_html=True)
            
            # Feature data download
            st.subheader("Feature Data")
            st.markdown(get_download_link(features_df, "heartbeat_features.csv", "Download All Feature Data"), unsafe_allow_html=True)