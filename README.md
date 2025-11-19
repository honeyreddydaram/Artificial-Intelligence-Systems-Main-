# AI-Powered ECG Signal Processing and Arrhythmia Detection System

A comprehensive Python-based application for advanced ECG signal analysis, PQRST peak detection, feature extraction, and arrhythmia classification using deep learning models. This system provides both research capabilities and clinical-grade analysis tools through an intuitive Streamlit web interface.

## 🚀 Key Features

### Core Analysis Capabilities
- **Advanced Signal Processing**: Multi-stage filtering, baseline correction, and noise removal
- **PQRST Peak Detection**: Automated detection of all cardiac waveform components using multiple algorithms
- **Feature Extraction**: Comprehensive statistical, morphological, and wavelet-based feature extraction
- **Arrhythmia Classification**: Deep learning-based risk assessment (Normal, Moderate Risk, High Risk)
- **Multi-format Support**: WFDB, CSV, DAT, and other ECG file formats
- **Real-time Processing**: Live ECG analysis and visualization

### Advanced Features
- **Deep Learning Models**: Pre-trained CNN models for arrhythmia classification (98.8% accuracy)
- **Multi-channel Support**: Handles up to 15-lead ECG recordings
- **Signal Quality Assessment**: Automatic quality metrics and artifact detection
- **Interactive Visualization**: Dynamic plots with zoom, pan, and export capabilities
- **Batch Processing**: Process multiple ECG files simultaneously
- **Cloud Deployment**: Azure-ready with scalable architecture

## 📊 Processing Flow

### 1. Data Input & Preprocessing
```
Raw ECG Signal → Format Detection → Signal Validation → Quality Assessment
```

**Supported Formats:**
- WFDB files (.dat, .hea)
- CSV files with time series data
- Binary DAT files
- MIT-BIH Arrhythmia Database format

**Preprocessing Pipeline:**
- DC offset removal
- Bandpass filtering (0.5-40 Hz)
- Baseline drift correction
- Noise artifact removal
- Signal normalization

### 2. Peak Detection & Analysis
```
Preprocessed Signal → R-peak Detection → P,Q,S,T Detection → Interval Analysis
```

**Detection Methods:**
- **XQRS Algorithm**: Primary R-peak detection
- **Pan-Tompkins**: Alternative R-peak detection
- **Enhanced Detection**: Multi-stage validation and correction

**Detected Components:**
- P waves (atrial depolarization)
- QRS complexes (ventricular depolarization)
- T waves (ventricular repolarization)
- Heart rate variability (HRV) metrics

### 3. Feature Extraction
```
Peak Data → Statistical Features → Morphological Features → Wavelet Features → Feature Matrix
```

**Feature Categories:**
- **Statistical**: Mean, std, skewness, kurtosis, RMS
- **Morphological**: RR intervals, PR intervals, QT intervals, ST segments
- **Frequency Domain**: FFT coefficients, power spectral density
- **Wavelet**: Multi-resolution analysis coefficients
- **Time-Frequency**: Short-time Fourier transform features

### 4. Arrhythmia Classification
```
Feature Matrix → CNN Model → Risk Assessment → Classification Report
```

**Model Architecture:**
- **Advanced CNN**: 1D Convolutional Neural Network
- **Input**: 180-sample ECG windows
- **Output**: 3-class risk assessment
- **Performance**: 98.8% accuracy, 0.999 ROC-AUC

**Risk Categories:**
- **Normal**: Healthy cardiac rhythm
- **Moderate Risk**: Minor arrhythmias requiring monitoring
- **High Risk**: Serious arrhythmias requiring immediate attention

## 🏗️ System Architecture

### Core Modules
```
src/
├── data_processing.py      # Signal preprocessing and quality assessment
├── peak_detection.py       # PQRST peak detection algorithms
├── feature_extraction.py   # Comprehensive feature extraction
└── arrhythmia_classifier.py # ML models and classification
```

### Application Interface
```
streamlit_app/
├── app.py                  # Main application entry point
├── pages/
│   ├── home.py            # Welcome and overview
│   ├── analysis.py        # ECG analysis and processing
│   ├── classification.py  # Arrhythmia classification
│   └── visualization.py   # Interactive plotting
└── utils/
    └── helpers.py         # Utility functions
```

### Model Deployment
```
ecg_model_deployment/
├── Advanced_CNN_production.h5  # Pre-trained CNN model
├── model_metadata.json         # Model configuration
├── predictor.py                # Model inference engine
└── test_data.npz              # Test dataset
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM recommended
- GPU support (optional, for faster processing)

### Installation

1. **Clone the repository:**
2. **Create virtual environment:**
   ```bash
   python -m venv ecg_env
   . ecg_env/Scripts/activate 
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Access the web interface:**
   Open your browser to `http://localhost:8501`

### Quick Start Guide

1. **Upload ECG Data**: Use the file uploader to select your ECG file
2. **Configure Analysis**: Set sampling frequency and processing parameters
3. **Run Analysis**: Click "Process ECG" to start the analysis pipeline
4. **View Results**: Explore detected peaks, features, and classification results
5. **Export Data**: Download processed data and analysis reports

## 📈 Usage Examples

### Basic ECG Analysis
```python
from src.data_processing import preprocess_ecg_signal
from src.peak_detection import detect_all_peaks
from src.feature_extraction import extract_comprehensive_features

# Load and preprocess ECG signal
ecg_signal = load_ecg_data('sample_ecg.csv')
processed_signal = preprocess_ecg_signal(ecg_signal, fs=360)

# Detect PQRST peaks
peaks = detect_all_peaks(processed_signal, fs=360)

# Extract features
features = extract_comprehensive_features(processed_signal, peaks)
```

### Arrhythmia Classification
```python
from ecg_model_deployment.predictor import ECGPredictor

# Load pre-trained model
predictor = ECGPredictor('ecg_model_deployment/Advanced_CNN_production.h5')

# Classify ECG signal
risk_assessment = predictor.predict(ecg_signal)
print(f"Risk Level: {risk_assessment['prediction']}")
print(f"Confidence: {risk_assessment['confidence']:.2%}")
```

## 🔧 Configuration

### Signal Processing Parameters
- **Sampling Frequency**: 360 Hz (default), 250 Hz, 500 Hz supported
- **Filter Range**: 0.5-40 Hz (adjustable)
- **Peak Detection**: Sensitivity and threshold parameters
- **Feature Extraction**: Customizable feature sets

### Model Configuration
- **Window Size**: 180 samples (0.5 seconds at 360 Hz)
- **Batch Size**: Configurable for processing speed
- **Confidence Threshold**: Adjustable classification threshold

## 📊 Performance Metrics

### Model Performance
- **Overall Accuracy**: 98.8%
- **Balanced Accuracy**: 98.8%
- **F1-Score (Macro)**: 98.8%
- **ROC-AUC (OVR)**: 99.9%
- **Matthews Correlation**: 98.3%

### Processing Speed
- **Single Lead (10s)**: ~2-3 seconds
- **Multi-lead (10s)**: ~5-8 seconds
- **Batch Processing**: ~50 files/minute


## 📁 Project Structure

```
ECG_Analysis/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── setup.py                       # Package installation
├── README.md                      # This file
│
├── src/                           # Core processing modules
│   ├── data_processing.py         # Signal preprocessing
│   ├── peak_detection.py          # PQRST detection
│   ├── feature_extraction.py      # Feature extraction
│   └── arrhythmia_classifier.py   # ML classification
│
├── streamlit_app/                 # Web application
│   ├── app.py                     # Main app entry
│   ├── pages/                     # Application pages
│   ├── assets/                    # UI assets
│   └── utils/                     # Helper functions
│
├── ecg_model_deployment/          # Pre-trained models
│   ├── Advanced_CNN_production.h5 # CNN model
│   ├── model_metadata.json        # Model info
│   └── predictor.py               # Inference engine
│
├── data_processing/               # Data processing utilities
├── pqrst_detection/               # Peak detection algorithms
├── notebooks/                     # Jupyter notebooks for research
├── results/                       # Analysis results and outputs
└── docs/                          # Documentation and papers
```

## 🔬 Research Applications

### Clinical Research
- Arrhythmia detection and classification
- Heart rate variability analysis
- Cardiac event prediction
- Drug effect assessment

### Academic Research
- Signal processing algorithm development
- Machine learning model training
- Feature engineering research
- Comparative analysis studies

## Acknowledgments

- **MIT-BIH Arrhythmia Database**: PhysioNet for providing the gold standard dataset
- **WFDB Python Package**: For ECG file format support
- **Streamlit**: For the intuitive web interface
- **TensorFlow/Keras**: For deep learning model implementation
- **SciPy/NumPy**: For signal processing algorithms
---

**⚠️ Medical Disclaimer**: This software is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult with qualified medical professionals for clinical applications.