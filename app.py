import streamlit as st
import numpy as np
import librosa
import joblib
import json
import pandas as pd
import io
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG
SR = 16000
DURATION = 3
SAMPLES = SR * DURATION
N_MFCC = 20
USE_DELTAS = True
AGG_FUNCS = ("mean", "std", "min", "max")

# Color scheme for emotions
EMOTION_COLORS = {
    "angry": "#FF4444",      # Red
    "disgust": "#8B4513",    # Brown  
    "fear": "#800080",       # Purple
    "happy": "#FF69B4",      # Hot Pink
    "neutral": "#808080",    # Gray
    "sad": "#4169E1"         # Blue
}

def get_emotion_color(emotion):
    """Returns color for given emotion"""
    return EMOTION_COLORS.get(emotion.lower(), "#333333")

# Load evaluation data
@st.cache_data
def load_evaluation_data():
    """Load all evaluation metrics and confusion matrices"""
    evaluations = {}
    
    # Load reports for each model
    eval_files = {
        "XGBoost": "classification_report_xgboost.json",
        "SVM (RBF)": "classification_report_svm_(rbf).json", 
        "Linear SVM": "classification_report_linear_svm.json",
        "Random Forest": "classification_report_randomforest.json"
    }
    
    for model_name, filename in eval_files.items():
        filepath = f"evaluations/{filename}"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                evaluations[model_name] = json.load(f)
    
    # Get confusion matrix data
    cm_files = {
        "XGBoost": "confusion_matrix_xgboost.npy",
        "SVM (RBF)": "confusion_matrix_svm_(rbf).npy",
        "Linear SVM": "confusion_matrix_linear_svm.npy", 
        "Random Forest": "confusion_matrix_randomforest.npy"
    }
    
    confusion_matrices = {}
    for model_name, filename in cm_files.items():
        filepath = f"evaluations/{filename}"
        if os.path.exists(filepath):
            confusion_matrices[model_name] = np.load(filepath)
    
    # Get emotion labels
    with open("evaluations/classes.json", 'r') as f:
        emotion_classes = json.load(f)
    
    return evaluations, confusion_matrices, emotion_classes

# Feature extraction setup (matches training pipeline)
def clean_audio(y):
    y = librosa.util.normalize(y)
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]
    return y

def mfcc_stats_from_audio(y, sr=SR, n_mfcc=N_MFCC, use_deltas=USE_DELTAS):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feats = [mfcc]
    if use_deltas:
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        feats += [d1, d2]
    F = np.concatenate(feats, axis=0)
    stats = []
    if "mean" in AGG_FUNCS: stats.append(F.mean(axis=1))
    if "std"  in AGG_FUNCS: stats.append(F.std(axis=1))
    if "min"  in AGG_FUNCS: stats.append(F.min(axis=1))
    if "max"  in AGG_FUNCS: stats.append(F.max(axis=1))
    return np.concatenate(stats, axis=0).astype(np.float32)



# Initialize app and load everything
@st.cache_resource
def load_models_and_labels():
    # Get all trained models
    xgb_model = joblib.load("trained_models/xgb_pipeline.pkl")
    svm_rbf_model = joblib.load("trained_models/svm_rbf_pipeline.pkl")
    linear_svm_model = joblib.load("trained_models/linear_svm_pipeline.pkl")
    rf_model = joblib.load("trained_models/randomforest_pipeline.pkl")
    
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    
    # Get evaluation metrics from saved files
    metrics = {}
    eval_files = {
        "XGBoost": "evaluations/classification_report_xgboost.json",
        "SVM (RBF)": "evaluations/classification_report_svm_(rbf).json",
        "Linear SVM": "evaluations/classification_report_linear_svm.json",
        "Random Forest": "evaluations/classification_report_randomforest.json"
    }
    
    for model_name, filepath in eval_files.items():
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                report = json.load(f)
                metrics[model_name] = {
                    "accuracy": report["accuracy"],
                    "macro_f1": report["macro avg"]["f1-score"]
                }
    
    models = {
        "XGBoost": xgb_model,
        "SVM (RBF)": svm_rbf_model,
        "Linear SVM": linear_svm_model,
        "Random Forest": rf_model
    }
    
    return models, classes, metrics

models_dict, classes, model_metrics = load_models_and_labels()

# App sidebar
st.sidebar.title("About This App")
st.sidebar.info(
    """
    **Speech Emotion Recognition**
    
    Upload a short speech audio file and get the predicted emotion using your choice of machine learning models trained on RAVDESS and CREMA-D datasets.
    
    - Input: WAV file (3 seconds, mono, 16kHz)
    - Output: Emotion label and confidence
    
    **Available Models:** XGBoost, SVM (RBF), Linear SVM, Random Forest
    """
)
st.sidebar.markdown(f"**XGBoost Validation Accuracy:** <span style='color:#4CAF50;font-size:18px'><b>{model_metrics['XGBoost']['accuracy']*100:.1f}%</b></span>", unsafe_allow_html=True)

# Main interface
page = st.sidebar.radio(
    "Navigation",
    ["Emotion Recognition", "Model Performance Comparison"]
)

if page == "Emotion Recognition":
    st.title("Speech Emotion Recognition")
    st.write("Upload a WAV audio file (speech, 3 seconds recommended) to predict the emotion using your choice of machine learning models.")

    # Choose your model
    model_names = list(models_dict.keys())
    default_idx = model_names.index("XGBoost") if "XGBoost" in model_names else 0
    selected_model_name = st.selectbox("Select Model", model_names, index=default_idx)
    model = models_dict[selected_model_name]
    st.markdown(f"**Validation Accuracy:** {model_metrics[selected_model_name]['accuracy']*100:.1f}%  ", unsafe_allow_html=True)
    st.markdown(f"**Macro F1-score:** {model_metrics[selected_model_name]['macro_f1']:.2f}")

    # Audio input section
    st.subheader("Audio Input")
    
    # Initialize variables
    audio_source = None
    audio_display_name = None
    
    # Use tabs for cleaner UI
    tab1, tab2 = st.tabs(["Test Examples", "Upload File"])
    
    with tab1:
        st.write("Choose from pre-recorded emotion samples")
        test_data_path = "datasets/test_data"
        
        if os.path.exists(test_data_path):
            test_files = glob.glob(os.path.join(test_data_path, "*.wav"))
            
            if test_files:
                # Extract emotion and speaker info for display
                file_options = ["Select an example..."]
                
                for f in test_files:
                    filename = os.path.basename(f)
                    parts = filename.replace('.wav', '').split('_')
                    if len(parts) >= 3:
                        speaker = parts[0]
                        word = parts[1]
                        emotion = parts[2]
                        display_name = f"{emotion.upper()} - {speaker} saying '{word}'"
                    else:
                        display_name = filename
                    file_options.append(display_name)
                
                selected_example = st.selectbox("Choose a test audio file:", file_options, key="test_selector")
                
                if selected_example != "Select an example...":
                    selected_idx = file_options.index(selected_example) - 1
                    selected_file_path = test_files[selected_idx]
                    audio_source = selected_file_path
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.success(f"✓ Selected: `{os.path.basename(selected_file_path)}`")
                    with col2:
                        # Audio player in compact form
                        with open(selected_file_path, 'rb') as f:
                            audio_data = f.read()
                        st.audio(audio_data, format='audio/wav')
            else:
                st.warning("No WAV files found in test dataset")
        else:
            st.warning("Test data folder not found")
    
    with tab2:
        st.write("Upload your own WAV audio file")
        uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"], key="file_uploader")
        
        if uploaded_file is not None:
            audio_source = uploaded_file
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"✓ Uploaded: `{uploaded_file.name}`")
            with col2:
                st.audio(uploaded_file, format='audio/wav')

    # Predict button
    st.write("")  # Add some spacing
    if audio_source is not None:
        predict_button = st.button("Analyze Emotion", type="primary", use_container_width=True)
    else:
        st.info("Please select or upload an audio file first")
        predict_button = False

    if audio_source is not None and predict_button:
        with st.spinner("Analyzing audio and predicting emotion..."):
            # Load audio
            y, sr = librosa.load(audio_source, sr=SR, mono=True)
            y = clean_audio(y)
            # Feature extraction
            x_vec = mfcc_stats_from_audio(y, sr=SR)
            x_vec = x_vec.reshape(1, -1)
            # Prediction
            pred = model.predict(x_vec)[0]
            proba = model.predict_proba(x_vec)[0]
            emotion = classes[pred]
            confidence = float(proba[pred])
            
            # Get color for this emotion
            emotion_color = get_emotion_color(emotion)

            st.markdown(f"<h4 style='color:{emotion_color};'>Predicted Emotion: <b>{emotion.capitalize()}</b></h4>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='color:#555;'>Confidence: <b>{confidence*100:.1f}%</b></h5>", unsafe_allow_html=True)

            # Show probabilities as a sorted table and bar chart
            st.subheader("Prediction Probabilities")
            prob_dict = {classes[i]: float(proba[i]) for i in range(len(classes))}
            prob_df = pd.DataFrame(list(prob_dict.items()), columns=["Emotion", "Probability"])
            prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)
            
            # Create colored bar chart
            chart_data = prob_df.copy()
            chart_data["Color"] = chart_data["Emotion"].apply(lambda x: get_emotion_color(x))
            
            # Display bar chart with custom colors
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(chart_data["Emotion"], chart_data["Probability"], 
                         color=[get_emotion_color(emotion) for emotion in chart_data["Emotion"]], 
                         alpha=0.8)
            
            ax.set_ylabel('Probability')
            ax.set_title('Emotion Prediction Probabilities')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, prob in zip(bars, chart_data["Probability"]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Table with colored emotion names
            def color_emotion_names(row):
                emotion = row['Emotion']
                color = get_emotion_color(emotion)
                return [f'color: {color}; font-weight: bold' if col == 'Emotion' else '' for col in row.index]
            
            styled_df = prob_df.style.format({"Probability": "{:.2%}"}).apply(color_emotion_names, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # Show waveform visualization
            st.subheader("Audio Waveform")
            
            # Create waveform plot
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Time axis
            time = np.arange(0, len(y)) / sr
            
            # Plot waveform
            ax.plot(time, y, color='#2E86AB', linewidth=0.8, alpha=0.8)
            ax.fill_between(time, y, alpha=0.3, color='#2E86AB')
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Audio Waveform Analysis')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, len(y) / sr)
            
            # Add audio stats
            duration = len(y) / sr
            max_amplitude = np.max(np.abs(y))
            rms = np.sqrt(np.mean(y**2))
            
            # Add text box with audio info
            info_text = f'Duration: {duration:.2f}s | Max Amplitude: {max_amplitude:.3f} | RMS: {rms:.3f}'
            ax.text(0.02, 0.95, info_text, transform=ax.transAxes, 
                   bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                   verticalalignment='top', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    elif predict_button and audio_source is None:
        st.warning("Please select an audio file first before predicting!")

elif page == "Model Performance Comparison":
    st.title("Model Performance Comparison")
    
    # Load evaluation data
    evaluations, confusion_matrices, emotion_classes = load_evaluation_data()
    
    if not evaluations:
        st.error("No evaluation data found. Please ensure evaluation files are in the 'evaluations/' folder.")
        st.stop()
    
    # --- Overall Performance Comparison ---
    st.header("Overall Model Performance")
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in evaluations.items():
        comparison_data.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Macro F1": metrics["macro avg"]["f1-score"],
            "Macro Precision": metrics["macro avg"]["precision"],
            "Macro Recall": metrics["macro avg"]["recall"]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("Accuracy", ascending=False)
    
    # Display as styled table
    st.subheader("Model Rankings")
    styled_df = comparison_df.style.format({
        "Accuracy": "{:.1%}",
        "Macro F1": "{:.3f}",
        "Macro Precision": "{:.3f}", 
        "Macro Recall": "{:.3f}"
    }).background_gradient(subset=["Accuracy"], cmap="RdYlGn")
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Performance bar charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(comparison_df["Model"], comparison_df["Accuracy"], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("F1-Score Comparison")
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(comparison_df["Model"], comparison_df["Macro F1"], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_ylabel('Macro F1-Score')
        ax.set_title('Model F1-Score Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    # --- Per-Class Performance Analysis ---
    st.header("Per-Class Performance Analysis")
    
    selected_model = st.selectbox("Select model for detailed analysis:", list(evaluations.keys()))
    
    if selected_model in evaluations:
        metrics = evaluations[selected_model]
        
        # Create per-class performance dataframe
        class_data = []
        for emotion in emotion_classes:
            if emotion in metrics:
                class_data.append({
                    "Emotion": emotion.title(),
                    "Precision": metrics[emotion]["precision"],
                    "Recall": metrics[emotion]["recall"],
                    "F1-Score": metrics[emotion]["f1-score"],
                    "Support": int(metrics[emotion]["support"])
                })
        
        class_df = pd.DataFrame(class_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Horizontal bar chart for per-class metrics
            fig, ax = plt.subplots(figsize=(8, 8))
            
            x = np.arange(len(class_df))
            width = 0.25
            
            bars1 = ax.barh(x - width, class_df["Precision"], width, label='Precision', alpha=0.8)
            bars2 = ax.barh(x, class_df["Recall"], width, label='Recall', alpha=0.8)
            bars3 = ax.barh(x + width, class_df["F1-Score"], width, label='F1-Score', alpha=0.8)
            
            ax.set_xlabel('Score')
            ax.set_title(f'{selected_model} - Per-Class Performance')
            ax.set_yticks(x)
            ax.set_yticklabels(class_df["Emotion"])
            ax.legend()
            ax.set_xlim(0, 1)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Detailed Metrics")
            styled_class_df = class_df.style.format({
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-Score": "{:.3f}"
            }).background_gradient(subset=["Precision", "Recall", "F1-Score"], cmap="RdYlGn")
            
            st.dataframe(styled_class_df, use_container_width=True, hide_index=True)
    
    # --- Confusion Matrices ---
    st.header("Confusion Matrices")
    
    selected_cm_model = st.selectbox("Select model for confusion matrix:", list(confusion_matrices.keys()))
    
    if selected_cm_model in confusion_matrices:
        cm = confusion_matrices[selected_cm_model]
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[e.title() for e in emotion_classes], 
                   yticklabels=[e.title() for e in emotion_classes], 
                   ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'{selected_cm_model} - Confusion Matrix')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Calculate and display per-class accuracy
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        accuracy_df = pd.DataFrame({
            "Emotion": [e.title() for e in emotion_classes],
            "Class Accuracy": class_accuracies
        }).sort_values("Class Accuracy", ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Per-Class Accuracy")
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(accuracy_df["Emotion"], accuracy_df["Class Accuracy"], color='skyblue')
            ax.set_ylabel('Accuracy')
            ax.set_title('Per-Class Accuracy')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Class Accuracy Table")
            styled_acc_df = accuracy_df.style.format({
                "Class Accuracy": "{:.1%}"
            }).background_gradient(subset=["Class Accuracy"], cmap="RdYlGn")
            st.dataframe(styled_acc_df, use_container_width=True, hide_index=True)