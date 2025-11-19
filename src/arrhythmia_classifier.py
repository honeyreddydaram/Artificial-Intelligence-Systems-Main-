"""
ECG Arrhythmia Classification Module

This module provides functions for classifying arrhythmias in ECG signals
using machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os
import pickle
from collections import Counter

# Try importing wfdb, but handle if it's missing
try:
    import wfdb
    wfdb_available = True
except ImportError:
    wfdb_available = False

# Map of MIT-BIH annotation codes to their descriptions
AAMI_MAPPING = {
    'N': 'Normal',
    'L': 'Normal',
    'R': 'Normal',
    'e': 'Normal',
    'j': 'Normal',
    'A': 'Atrial Premature',
    'a': 'Atrial Premature',
    'J': 'Atrial Premature',
    'S': 'Atrial Premature',
    'V': 'Ventricular Premature',
    'E': 'Ventricular Premature',
    'F': 'Fusion',
    'P': 'Paced',
    'f': 'Fusion',
    '/': 'Paced',
    'Q': 'Unclassifiable',
    '?': 'Unclassifiable'
}

# Simplified mapping for classification
SIMPLIFIED_CLASSES = {
    'Normal': 0,
    'Atrial Premature': 1,
    'Ventricular Premature': 2,
    'Fusion': 3,
    'Paced': 4,
    'Unclassifiable': 5
}

def load_all_records_with_annotations(data_dir):
    """
    Load all records from the MIT-BIH database with their annotations.
    
    Parameters:
        data_dir (str): Path to the MIT-BIH database directory
        
    Returns:
        dict: Dictionary containing record data, signals, and annotations
    """
    record_data = {}
    
    if not wfdb_available:
        print("WFDB library not available. Install with: pip install wfdb")
        return record_data
    
    # Get all record names (files with .hea extension)
    record_names = []
    try:
        for file in os.listdir(data_dir):
            if file.endswith('.hea'):
                record_name = file.replace('.hea', '')
                record_names.append(record_name)
    except Exception as e:
        print(f"Error reading directory {data_dir}: {e}")
        return record_data
    
    for record_name in record_names:
        try:
            # Full path to the record
            record_path = os.path.join(data_dir, record_name)
            
            # Read the record
            record = wfdb.rdrecord(record_path)
            
            # Read the annotation if it exists
            try:
                # First try with 'atr' extension (most common)
                ann = wfdb.rdann(record_path, 'atr')
                
                # Store in dictionary
                record_data[record_name] = {
                    'record': record,
                    'annotation': ann,
                    'signal': record.p_signal,
                    'fs': record.fs,
                    'sig_name': record.sig_name,
                    'n_sig': record.n_sig,
                    'n_samples': record.sig_len,
                    'duration': record.sig_len / record.fs  # in seconds
                }
                
                print(f"Successfully loaded {record_name}")
            except Exception as e:
                print(f"No annotation found for {record_name}: {e}")
        except Exception as e:
            print(f"Error loading {record_name}: {e}")
    
    return record_data

def extract_labeled_heartbeats(record_data, before_r=0.25, after_r=0.45):
    """
    Extract heartbeats with class labels from records.
    
    Parameters:
        record_data (dict): Dictionary of record data from load_all_records_with_annotations
        before_r (float): Time before R-peak in seconds
        after_r (float): Time after R-peak in seconds
        
    Returns:
        tuple: (heartbeats, labels, label_names)
    """
    all_heartbeats = []
    all_labels = []
    
    for record_name, data in record_data.items():
        if 'annotation' not in data or data['annotation'] is None:
            continue
            
        # Get signal and annotation
        signal = data['signal']
        annotation = data['annotation']
        fs = data['fs']
        
        # Convert time to samples
        before_samples = int(before_r * fs)
        after_samples = int(after_r * fs)
        
        # Get beat locations and symbols
        beat_locations = annotation.sample
        beat_symbols = annotation.symbol
        
        for i, (location, symbol) in enumerate(zip(beat_locations, beat_symbols)):
            # Skip if not a beat annotation
            if symbol not in AAMI_MAPPING:
                continue
                
            # Skip if too close to beginning or end
            if location < before_samples or location + after_samples >= len(signal):
                continue
                
            # Extract heartbeat (use first channel if multiple)
            heartbeat = signal[location - before_samples:location + after_samples, 0]
            
            # Get class label
            class_name = AAMI_MAPPING[symbol]
            class_id = SIMPLIFIED_CLASSES[class_name]
            
            # Store heartbeat and label
            all_heartbeats.append(heartbeat)
            all_labels.append(class_id)
    
    # Convert to numpy arrays
    heartbeats = np.array(all_heartbeats)
    labels = np.array(all_labels)
    
    # Get label names for reference
    label_names = {v: k for k, v in SIMPLIFIED_CLASSES.items()}
    
    return heartbeats, labels, label_names

def extract_features_for_classification(heartbeats, feature_module):
    """
    Extract features from heartbeats for classification.
    
    Parameters:
        heartbeats (array): Array of heartbeat segments
        feature_module: Module containing feature extraction functions
        
    Returns:
        DataFrame: Extracted features
    """
    # List to store features
    features_list = []
    
    # Process each heartbeat
    for i, beat in enumerate(heartbeats):
        # Extract features
        features = feature_module.extract_heartbeat_features(beat, include_advanced=False)
        features_list.append(features)
        
        # Print progress
        if (i+1) % 500 == 0:
            print(f"Processed {i+1}/{len(heartbeats)} heartbeats...")
    
    # Create DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Remove NaN values
    features_df = features_df.fillna(0)
    
    return features_df

def train_arrhythmia_classifier(features, labels, model_type='random_forest'):
    """
    Train a classifier for arrhythmia detection.
    
    Parameters:
        features (DataFrame): Features extracted from heartbeats
        labels (array): Class labels
        model_type (str): Type of model to train ('random_forest', 'svm', 'neural_network')
        
    Returns:
        tuple: (model, scaler, accuracy, report)
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    # Create scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model based on type
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == 'svm':
        model = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    elif model_type == 'neural_network':
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return model, scaler, accuracy, report

def create_classification_pipeline(model, scaler):
    """
    Create a pipeline for classification.
    
    Parameters:
        model: Trained classifier
        scaler: Fitted scaler
        
    Returns:
        Pipeline: Scikit-learn pipeline
    """
    return Pipeline([
        ('scaler', scaler),
        ('classifier', model)
    ])

def save_classifier(pipeline, label_names, output_dir):
    """
    Save the classifier pipeline and label mapping.
    
    Parameters:
        pipeline: Classification pipeline
        label_names: Dictionary mapping class IDs to names
        output_dir: Directory to save the model
        
    Returns:
        str: Path to the saved model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save pipeline
    model_path = os.path.join(output_dir, 'arrhythmia_classifier.pkl')
    joblib.dump(pipeline, model_path)
    
    # Save label mapping
    label_path = os.path.join(output_dir, 'label_mapping.pkl')
    joblib.dump(label_names, label_path)
    
    return model_path

def load_classifier(model_dir):
    """
    Load a saved classifier pipeline and label mapping.
    
    Parameters:
        model_dir: Directory containing the saved model
        
    Returns:
        tuple: (pipeline, label_names)
    """
    # Check if files exist
    model_path = os.path.join(model_dir, 'arrhythmia_classifier.pkl')
    label_path = os.path.join(model_dir, 'label_mapping.pkl')
    
    if not os.path.exists(model_path):
        # Try to create a simple demo model
        pipeline, label_names = create_demo_classifier()
        save_classifier(pipeline, label_names, model_dir)
    else:
        # Load pipeline
        pipeline = joblib.load(model_path)
        
        # Load label mapping
        if os.path.exists(label_path):
            label_names = joblib.load(label_path)
        else:
            # Use default mapping if label file is missing
            label_names = {v: k for k, v in SIMPLIFIED_CLASSES.items()}
    
    return pipeline, label_names

def create_demo_classifier():
    """
    Create a simple demo classifier when no model is available.
    
    Returns:
        tuple: (pipeline, label_names)
    """
    # Create a simple Random Forest classifier with dummy data
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, 100)  # 0: Normal, 1: Atrial, 2: Ventricular
    
    # Train on dummy data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', model)
    ])
    
    # Create simplified label mapping
    label_names = {
        0: 'Normal',
        1: 'Atrial Premature',
        2: 'Ventricular Premature'
    }
    
    print("Created demo classifier (not trained on real data)")
    
    return pipeline, label_names

def classify_heartbeat(heartbeat, pipeline, feature_module, label_names):
    """
    Classify a single heartbeat.
    
    Parameters:
        heartbeat (array): Heartbeat signal
        pipeline: Classification pipeline
        feature_module: Module containing feature extraction functions
        label_names: Dictionary mapping class IDs to names
        
    Returns:
        tuple: (class_id, class_name, probabilities)
    """
    # Extract features
    features = feature_module.extract_heartbeat_features(heartbeat, include_advanced=False)
    features_df = pd.DataFrame([features])
    
    # Fill NaN values
    features_df = features_df.fillna(0)
    
    # Predict class
    class_id = pipeline.predict(features_df)[0]
    
    # Handle the case where class_id is not in label_names
    if class_id in label_names:
        class_name = label_names[class_id]
    else:
        class_name = f"Unknown ({class_id})"
    
    # Get probabilities if available
    if hasattr(pipeline, 'predict_proba'):
        probabilities = pipeline.predict_proba(features_df)[0]
    else:
        probabilities = None
    
    return class_id, class_name, probabilities

def classify_multiple_heartbeats(heartbeats, pipeline, feature_module, label_names):
    """
    Classify multiple heartbeats.
    
    Parameters:
        heartbeats (array): Array of heartbeats
        pipeline: Classification pipeline
        feature_module: Module containing feature extraction functions
        label_names: Dictionary mapping class IDs to names
        
    Returns:
        tuple: (class_ids, class_names, all_probabilities)
    """
    # Extract features for all heartbeats
    features_list = []
    for beat in heartbeats:
        features = feature_module.extract_heartbeat_features(beat, include_advanced=False)
        features_list.append(features)
    
    # Create DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Fill NaN values
    features_df = features_df.fillna(0)
    
    # Predict classes
    class_ids = pipeline.predict(features_df)
    class_names = []
    
    # Handle cases where class_id is not in label_names
    for class_id in class_ids:
        if class_id in label_names:
            class_names.append(label_names[class_id])
        else:
            class_names.append(f"Unknown ({class_id})")
    
    # Get probabilities if available
    if hasattr(pipeline, 'predict_proba'):
        all_probabilities = pipeline.predict_proba(features_df)
    else:
        all_probabilities = None
    
    return class_ids, class_names, all_probabilities

def build_advanced_ensemble_classifier(features, labels, output_dir):
    """
    Build an advanced ensemble classifier for arrhythmia detection.
    
    Parameters:
        features (DataFrame): Features extracted from heartbeats
        labels (array): Class labels
        output_dir (str): Directory to save the models
        
    Returns:
        tuple: (ensemble_pipeline, label_names, accuracy, report)
    """
    try:
        from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
        from sklearn.feature_selection import SelectFromModel
    except ImportError:
        print("Required scikit-learn components not available")
        return None, None, 0, "Error: Required libraries not available"
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    # Create scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection using Random Forest
    feature_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42), 
        threshold='median'
    )
    X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = feature_selector.transform(X_test_scaled)
    
    # Create individual classifiers
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight='balanced'
    )
    
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # Create Voting Classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('svm', svm),
            ('mlp', mlp),
            ('gb', gb)
        ],
        voting='soft'
    )
    
    # Train the ensemble
    ensemble.fit(X_train_selected, y_train)
    
    # Evaluate model
    y_pred = ensemble.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Create pipeline with preprocessing steps
    ensemble_pipeline = Pipeline([
        ('scaler', scaler),
        ('feature_selector', feature_selector),
        ('ensemble', ensemble)
    ])
    
    # Get label names for reference
    label_names = {v: k for k, v in SIMPLIFIED_CLASSES.items()}
    
    # Save the pipeline
    save_classifier(ensemble_pipeline, label_names, output_dir)
    
    return ensemble_pipeline, label_names, accuracy, report