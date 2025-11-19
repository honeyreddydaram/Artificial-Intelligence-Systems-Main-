"""
ECG Arrhythmia Classification page for the Streamlit app.
Enhanced with Advanced CNN model for risk assessment.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time as time_module
import joblib
import glob
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    # Import helper functions
    from streamlit_app.utils.helpers import (
        load_file, generate_synthetic_ecg, process_ecg, extract_heartbeats, 
        plot_ecg_with_peaks, get_download_link, get_figure_download_link
    )
    
    # Import processing modules
    from src import data_processing, peak_detection, feature_extraction, arrhythmia_classifier
except ImportError as e:
    st.warning(f"Some helper modules not found: {e}")

@st.cache_resource
def load_advanced_cnn_model():
    """Load the Advanced CNN model for arrhythmia classification."""
    try:
        import tensorflow as tf
    except ImportError:
        st.error("TensorFlow not available. Please install TensorFlow to use classification features.")
        return None, None
    
    # Try different possible paths for your model
    possible_paths = [
        "ecg_model_deployment\Advanced_CNN_production.h5",
        "models/advanced_cnn.h5", 
        "Advanced_CNN.h5",
        "models/Advanced_CNN.h5",
        "./advanced_cnn.h5",
        "../advanced_cnn.h5",
        "../../advanced_cnn.h5"
    ]
    
    model = None
    metadata = None
    
    for model_path in possible_paths:
        try:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                st.success(f"✅ Loaded Advanced CNN model from: {model_path}")
                
                # Try to load metadata
                metadata_paths = [
                    model_path.replace('.h5', '_metadata.json'),
                    model_path.replace('advanced_cnn', 'model_metadata').replace('Advanced_CNN', 'model_metadata'),
                    "ecg_model_deployment\model_metadata.json",
                    "models/model_metadata.json",
                    "../model_metadata.json"
                ]
                
                for metadata_path in metadata_paths:
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            st.info(f"📊 Loaded model metadata: Accuracy = {metadata.get('performance_metrics', {}).get('accuracy', 'N/A')}")
                            break
                        except Exception as e:
                            continue
                
                if metadata is None:
                    # Default metadata if file not found
                    metadata = {
                        "model_name": "Advanced_CNN",
                        "class_names": ["Normal", "Moderate Risk", "High Risk"],
                        "window_size": 180,
                        "sampling_frequency": 360,
                        "performance_metrics": {
                            "accuracy": 0.988
                        }
                    }
                    st.warning("⚠️ Using default metadata (model_metadata.json not found)")
                
                break
        except Exception as e:
            continue
    
    if model is None:
        st.error("❌ Could not load Advanced CNN model. Please ensure 'advanced_cnn.h5' is in the correct directory.")
        st.info("💡 Expected locations: advanced_cnn.h5, models/advanced_cnn.h5, or Advanced_CNN.h5")
        return None, None
    
    return model, metadata

def show():
    """Show the ECG Arrhythmia Classification page."""
    st.title("🫀 ECG Arrhythmia Classification & Risk Assessment")
    
    # Sidebar for options
    with st.sidebar:
        st.header("🔧 Classification Options")
        st.info("This page uses an Advanced CNN model to classify ECG beats and assess arrhythmia risk.")
        
        window = st.slider(
            "Heartbeat Window (samples)",
            min_value=100,
            max_value=300,
            value=180,
            step=10,
            help="Number of samples in each heartbeat window for classification"
        )
        
        st.markdown("---")
        st.subheader("📊 Model Information")
        st.write("**Advanced CNN Model**")
        st.write("- High accuracy arrhythmia detection")
        st.write("- 3-class classification")
        st.write("- Trained on large ECG dataset")
    
    # Load the Advanced CNN model
    model, model_metadata = load_advanced_cnn_model()
    if model is None:
        st.error("Please ensure your Advanced CNN model file is available")
        return
    
    # Display model information if metadata available
    if model_metadata:
        with st.expander("📊 Model Performance Metrics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", model_metadata.get('model_name', 'Advanced_CNN'))
            with col2:
                accuracy = model_metadata.get('performance_metrics', {}).get('accuracy', 0)
                st.metric("Model Accuracy", f"{accuracy:.3f}" if accuracy else "N/A")
            with col3:
                st.metric("Classes", len(model_metadata.get('class_names', ['Normal', 'Moderate Risk', 'High Risk'])))
            
            # Additional metrics if available
            if 'performance_metrics' in model_metadata:
                metrics = model_metadata['performance_metrics']
                st.markdown("**Detailed Performance:**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'f1_macro' in metrics:
                        st.metric("F1 Score (Macro)", f"{metrics['f1_macro']:.4f}")
                    if 'cohen_kappa' in metrics:
                        st.metric("Cohen's Kappa", f"{metrics['cohen_kappa']:.4f}")
                with col2:
                    if 'roc_auc_ovr' in metrics:
                        st.metric("ROC AUC", f"{metrics['roc_auc_ovr']:.4f}")
                    if 'balanced_accuracy' in metrics:
                        st.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}")
                with col3:
                    if 'matthews_corrcoef' in metrics:
                        st.metric("Matthews Correlation", f"{metrics['matthews_corrcoef']:.4f}")
    
    # Main area - Check for ECG data
    if 'ecg_data' not in st.session_state or st.session_state.ecg_data is None:
        st.info("📊 No ECG data loaded. Please go to the Analysis page to load ECG data.")
        return
    
    if 'processed_signal' not in st.session_state or st.session_state.processed_signal is None:
        st.info("🔄 ECG data not processed. Please go to the Analysis page to process the ECG signal.")
        return
    
    if 'r_peaks' not in st.session_state or st.session_state.r_peaks is None:
        st.info("📈 R-peaks not detected. Please run enhanced PQRST detection in the Analysis page.")
        return
    
    # Get data from session state
    time, ecg_signal, fs = st.session_state.ecg_data
    processed_signal = st.session_state.processed_signal
    r_peaks = st.session_state.r_peaks
    
    # Display ECG information
    st.subheader("📋 ECG Data Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sampling Rate", f"{fs} Hz")
    with col2:
        st.metric("Duration", f"{len(time)/fs:.1f} s")
    with col3:
        st.metric("R-peaks Detected", len(r_peaks))
    with col4:
        if len(r_peaks) > 1:
            avg_hr = 60 / (np.mean(np.diff(r_peaks)) / fs)
            st.metric("Average HR", f"{avg_hr:.1f} BPM")
        else:
            st.metric("Average HR", "N/A")
    
    # Extract heartbeats
    st.subheader("🔍 Heartbeat Extraction")
    
    # Channel selection for multi-channel signals
    channel_to_use = 0
    if processed_signal.ndim > 1 and processed_signal.shape[1] > 1:
        channel_to_use = st.selectbox(
            "Select channel for risk classification:",
            range(processed_signal.shape[1]),
            format_func=lambda x: f"Channel {x+1}"
        )
        signal_for_analysis = processed_signal[:, channel_to_use]
        st.info(f"Using channel {channel_to_use + 1} for classification")
    else:
        signal_for_analysis = processed_signal.squeeze() if processed_signal.ndim > 1 else processed_signal
    
    # Extract heartbeats around R-peaks
    beats = []
    for idx in r_peaks:
        start = idx - window//2
        end = idx + window//2
        if start < 0 or end > len(signal_for_analysis):
            continue
        beat = signal_for_analysis[start:end]
        beats.append(beat)
    
    beats = np.array(beats)
    
    if len(beats) == 0:
        st.error("❌ No valid heartbeats could be extracted. Check R-peak detection and window size.")
        return
    
    st.success(f"✅ Extracted {len(beats)} heartbeats from {len(r_peaks)} R-peaks")
    
    # Visualize sample beats
    with st.expander("👁️ Sample Heartbeats Visualization"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.ravel()
        
        sample_indices = np.linspace(0, len(beats)-1, min(6, len(beats)), dtype=int)
        for i, idx in enumerate(sample_indices):
            beat = beats[idx]
            beat_time = np.arange(len(beat)) / fs * 1000  # Convert to ms
            axes[i].plot(beat_time, beat, 'b-', linewidth=1.5)
            axes[i].set_title(f'Beat {idx+1}')
            axes[i].set_xlabel('Time (ms)')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Normalize beats as in the training notebook
    beats_norm = (beats - beats.mean(axis=1, keepdims=True)) / (beats.std(axis=1, keepdims=True) + 1e-8)
    beats_norm = beats_norm[..., np.newaxis]
    
    # Classification
    st.subheader("🧠 Advanced Arrhythmia Classification")
    
    if st.button("🚀 Classify Arrhythmia Risk", key="analyze_risk_button", type="primary"):
        with st.spinner("Analyzing ECG beats for arrhythmia classification..."):
            
            # Get class names from metadata or use defaults
            class_names = model_metadata.get('class_names', ["Normal", "Moderate Risk", "High Risk"]) if model_metadata else ["Normal", "Moderate Risk", "High Risk"]
            
            # Make predictions
            preds = model.predict(beats_norm, verbose=0)
            risk_classes = np.argmax(preds, axis=1)
            risk_labels = np.array(class_names)[risk_classes]
            
            # Calculate overall risk assessment
            high_risk_pct = (risk_classes == 2).sum() / len(risk_classes) * 100
            moderate_risk_pct = (risk_classes == 1).sum() / len(risk_classes) * 100
            normal_pct = (risk_classes == 0).sum() / len(risk_classes) * 100
            
            # Determine overall risk level
            if high_risk_pct > 30:
                overall_risk = "🔴 HIGH RISK"
                risk_color = "red"
            elif high_risk_pct > 10 or moderate_risk_pct > 40:
                overall_risk = "🟡 MODERATE RISK" 
                risk_color = "orange"
            else:
                overall_risk = "🟢 LOW RISK"
                risk_color = "green"
            
            # Display overall assessment
            st.markdown(f"### Overall Assessment: **{overall_risk}**")
            
            # Risk distribution metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Normal Beats", f"{(risk_classes==0).sum()}", f"{normal_pct:.1f}%")
            with col2:
                st.metric("Moderate Risk", f"{(risk_classes==1).sum()}", f"{moderate_risk_pct:.1f}%")
            with col3:
                st.metric("High Risk", f"{(risk_classes==2).sum()}", f"{high_risk_pct:.1f}%")
            with col4:
                avg_confidence = np.max(preds, axis=1).mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            # Risk distribution visualization
            st.subheader("📊 Risk Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                risk_counts = pd.Series(risk_labels).value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#28a745', '#ffc107', '#dc3545']  # Green, Yellow, Red
                bars = ax.bar(risk_counts.index, risk_counts.values, 
                             color=[colors[class_names.index(label)] for label in risk_counts.index])
                ax.set_ylabel('Number of Beats')
                ax.set_title('Arrhythmia Risk Classification Results')
                ax.grid(True, alpha=0.3)
                
                # Add percentage labels
                total_beats = len(risk_labels)
                for bar, count in zip(bars, risk_counts.values):
                    height = bar.get_height()
                    pct = (count / total_beats) * 100
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Detailed statistics
                st.write("**Classification Summary:**")
                for i, class_name in enumerate(class_names):
                    count = (risk_classes == i).sum()
                    percentage = (count / len(risk_classes)) * 100
                    avg_prob = preds[risk_classes == i, i].mean() if count > 0 else 0
                    st.write(f"• **{class_name}**: {count} beats ({percentage:.1f}%) - Avg confidence: {avg_prob:.3f}")
                
                # Risk assessment guidance
                st.markdown("**Risk Level Guidance:**")
                st.write("🟢 **Low Risk**: < 10% high-risk beats")
                st.write("🟡 **Moderate Risk**: 10-30% high-risk beats")
                st.write("🔴 **High Risk**: > 30% high-risk beats")
            
            # ECG visualization with risk annotations
            st.subheader("📈 ECG Signal with Risk Classification")
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot ECG signal
            ax.plot(time, signal_for_analysis, 'b-', alpha=0.7, linewidth=1, label='ECG Signal')
            
            # Color-code R-peaks by risk classification
            color_map = {"Normal": "green", "Moderate Risk": "orange", "High Risk": "red"}
            
            # Plot R-peaks with risk colors
            legend_added = set()
            for i, idx in enumerate(r_peaks[:len(risk_classes)]):
                if idx < len(time):
                    label = risk_labels[i]
                    color = color_map.get(label, "blue")
                    confidence = np.max(preds[i])
                    
                    # Size of marker based on confidence
                    marker_size = 30 + (confidence * 50)
                    
                    # Add to legend only once per class
                    legend_label = label if label not in legend_added else ""
                    if label not in legend_added:
                        legend_added.add(label)
                    
                    ax.scatter(time[idx], signal_for_analysis[idx], 
                              c=color, s=marker_size, alpha=0.8, 
                              edgecolors='black', linewidth=0.5,
                              label=legend_label)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (mV)')
            ax.set_title('ECG Signal with Arrhythmia Risk Classification')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detailed results table
            st.subheader("📋 Detailed Classification Results")
            
            # Create detailed dataframe
            detailed_results = []
            for i, (r_peak_idx, pred_class, pred_probs) in enumerate(zip(r_peaks[:len(risk_classes)], risk_classes, preds)):
                detailed_results.append({
                    'Beat #': i + 1,
                    'R-Peak Index': r_peak_idx,
                    'Time (s)': f"{r_peak_idx / fs:.2f}",
                    'Classification': class_names[pred_class],
                    'Confidence': f"{np.max(pred_probs):.3f}",
                    'Normal Prob': f"{pred_probs[0]:.3f}",
                    'Moderate Prob': f"{pred_probs[1]:.3f}",
                    'High Risk Prob': f"{pred_probs[2]:.3f}"
                })
            
            results_df = pd.DataFrame(detailed_results)
            st.dataframe(results_df, use_container_width=True)
            
            # Download options
            st.subheader("💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Summary CSV
                summary_data = {
                    'Risk_Level': class_names,
                    'Count': [(risk_classes == i).sum() for i in range(len(class_names))],
                    'Percentage': [((risk_classes == i).sum() / len(risk_classes)) * 100 for i in range(len(class_names))],
                    'Avg_Confidence': [preds[risk_classes == i, i].mean() if (risk_classes == i).sum() > 0 else 0 for i in range(len(class_names))]
                }
                summary_df = pd.DataFrame(summary_data)
                
                st.download_button(
                    "📊 Download Risk Summary",
                    data=summary_df.to_csv(index=False),
                    file_name="arrhythmia_risk_summary.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Detailed results CSV
                st.download_button(
                    "📋 Download Detailed Results", 
                    data=results_df.to_csv(index=False),
                    file_name="detailed_arrhythmia_classification.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Beat data with predictions
                beat_data = pd.DataFrame({
                    "R_Peak_Index": r_peaks[:len(risk_classes)],
                    "Risk_Class": risk_labels,
                    "Risk_Prob_Normal": preds[:,0],
                    "Risk_Prob_Moderate": preds[:,1], 
                    "Risk_Prob_High": preds[:,2],
                    "Confidence": np.max(preds, axis=1)
                })
                
                st.download_button(
                    "🫀 Download Beat Classifications",
                    data=beat_data.to_csv(index=False),
                    file_name="beat_risk_classifications.csv",
                    mime="text/csv"
                )
            
            # Clinical insights
            st.subheader("🏥 Clinical Insights & Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Clinical Recommendations:**")
                if high_risk_pct > 30:
                    st.error("⚠️ **HIGH RISK DETECTED**: Immediate medical attention recommended. Consider cardiology consultation.")
                elif high_risk_pct > 10 or moderate_risk_pct > 40:
                    st.warning("⚠️ **MODERATE RISK**: Follow-up with healthcare provider recommended. Monitor symptoms.")
                else:
                    st.success("✅ **LOW RISK**: Routine follow-up sufficient. Continue regular health monitoring.")
                
                st.write("**Key Findings:**")
                st.write(f"• Total beats analyzed: {len(risk_classes)}")
                st.write(f"• High-risk beats: {(risk_classes==2).sum()} ({high_risk_pct:.1f}%)")
                st.write(f"• Average prediction confidence: {avg_confidence:.3f}")
            
            with col2:
                if model_metadata and 'performance_metrics' in model_metadata:
                    st.write("**Model Reliability:**")
                    metrics = model_metadata['performance_metrics']
                    st.write(f"• Model accuracy: {metrics.get('accuracy', 0):.1%}")
                    st.write(f"• F1 score: {metrics.get('f1_macro', 0):.3f}")
                    st.write(f"• Cohen's kappa: {metrics.get('cohen_kappa', 0):.3f}")
                
                st.write("**Important Notes:**")
                st.write("• This is a screening tool, not a diagnostic device")
                st.write("• Consult healthcare providers for medical decisions")
                st.write("• Results should be interpreted by qualified personnel")

if __name__ == "__main__":
    show()