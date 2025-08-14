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
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# CONFIG
SR = 16000
DURATION = 3
SAMPLES = SR * DURATION
N_MFCC = 20
USE_DELTAS = True
AGG_FUNCS = ("mean", "std", "min", "max")

# Color scheme for emotions
EMOTION_COLORS = {
    "angry": "#FF0000",      # Red
    "disgust": "#8B4513",    # Brown  
    "fear": "#800080",       # Purple
    "happy": "#00AA00",      # Green
    "neutral": "#808080",    # Gray
    "sad": "#4169E1"         # Blue
}

def get_emotion_color(emotion):
    """Returns color for given emotion"""
    return EMOTION_COLORS.get(emotion.lower(), "#333333")

def safe_predict_proba(model, X):
    """Safely get prediction probabilities, handling models without predict_proba"""
    try:
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[0]
        elif hasattr(model, 'decision_function'):
            # For LinearSVC, use decision function and convert to probabilities
            decision_scores = model.decision_function(X)[0]
            # Convert to probabilities using softmax
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            return exp_scores / np.sum(exp_scores)
        else:
            # Fallback: create uniform probabilities
            n_classes = len(model.classes_) if hasattr(model, 'classes_') else 6
            uniform_proba = np.ones(n_classes) / n_classes
            pred = model.predict(X)[0]
            uniform_proba[pred] = 0.8  # Give higher probability to predicted class
            uniform_proba = uniform_proba / np.sum(uniform_proba)
            return uniform_proba
    except Exception as e:
        print(f"Error getting probabilities: {e}")
        # Return uniform probabilities as fallback
        return np.ones(6) / 6

def create_emotion_radar_chart(probabilities, emotion_labels):
    """Create an interactive radar chart showing emotion probabilities"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=probabilities,
        theta=emotion_labels,
        fill='toself',
        fillcolor='rgba(255, 105, 180, 0.3)',
        line=dict(color='#FF69B4', width=3),
        marker=dict(size=8, color='#FF1493'),
        name='Emotion Confidence'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%',
                gridcolor='rgba(255,255,255,0.3)'
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.3)'
            )
        ),
        showlegend=False,
        title="Emotion Confidence Radar",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_emotion_timeline(predictions_history):
    """Create timeline of emotion predictions"""
    if not predictions_history:
        return None
        
    df = pd.DataFrame(predictions_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = px.line(df, x='timestamp', y='confidence', 
                  color='emotion', title='Emotion Timeline',
                  color_discrete_map=EMOTION_COLORS)
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.2)')
    )
    
    return fig

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

# Initialize session state for amazing features
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False

if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

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
    ["Emotion Recognition", "Model Performance Comparison", "Advanced Features"]
)

if page == "Emotion Recognition":
    st.title("Speech Emotion Recognition")
    st.write("Upload a WAV audio file (speech, 3 seconds recommended) to predict the emotion using your choice of machine learning models.")

    # Choose model with highlighted styling
    st.markdown("""
    <style>
    .stSelectbox > div > div > div {
        background-color: #e3f2fd !important;
        border: 2px solid #2196f3 !important;
        border-radius: 8px !important;
    }
    .stSelectbox label {
        color: #1976d2 !important;
        font-weight: bold !important;
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
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
                        st.success(f"Selected: `{os.path.basename(selected_file_path)}`")
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
                st.success(f"Uploaded: `{uploaded_file.name}`")
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
            proba = safe_predict_proba(model, x_vec)
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
            
            # Save to history
            st.session_state.predictions_history.append({
                'timestamp': datetime.now(),
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': prob_dict
            })
            
            # Create two columns for visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Create radar chart
                radar_fig = create_emotion_radar_chart(
                    list(prob_dict.values()), 
                    list(prob_dict.keys())
                )
                st.plotly_chart(radar_fig, use_container_width=True)
            
            with col2:
                # Create colored bar chart
                chart_data = prob_df.copy()
                chart_data["Color"] = chart_data["Emotion"].apply(lambda x: get_emotion_color(x))
                
                fig = px.bar(chart_data, x="Emotion", y="Probability", 
                           color="Emotion", color_discrete_map=EMOTION_COLORS,
                           title="Emotion Confidence Scores")
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
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

elif page == "Advanced Features":
    st.title("Advanced Features & Analytics")
    st.markdown("Explore powerful features including emotion timeline, batch analysis, and model comparison!")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Analyze Multiple Files", "Model Battle", "Audio Insights", "Emotion Timeline"])
    
    with tab1:
        st.subheader("Analyze Multiple Audio Files")
        st.write("Upload multiple audio files to analyze their emotions all at once!")
        
        # Sub-tabs for test files vs upload
        batch_tab1, batch_tab2 = st.tabs(["Test Examples", "Upload Files"])
        
        uploaded_files = None
        
        with batch_tab1:
            st.write("Select multiple files from our test examples")
            test_data_path = "datasets/test_data"
            
            if os.path.exists(test_data_path):
                test_files = glob.glob(os.path.join(test_data_path, "*.wav"))
                
                if test_files:
                    # Create file selection with emotions
                    file_options = []
                    for f in test_files:
                        filename = os.path.basename(f)
                        parts = filename.replace('.wav', '').split('_')
                        if len(parts) >= 3:
                            speaker = parts[0]
                            word = parts[1]
                            emotion = parts[2]
                            display_name = f"{emotion.title()} - {filename}"
                            file_options.append((display_name, f))
                    
                    # Multi-select for batch
                    selected_files = st.multiselect(
                        "Choose multiple test files:",
                        options=[option[0] for option in file_options],
                        help="Select 2 or more files to see emotion variety"
                    )
                    
                    if selected_files:
                        # Get actual file paths
                        selected_paths = []
                        for display_name in selected_files:
                            for option in file_options:
                                if option[0] == display_name:
                                    selected_paths.append(option[1])
                                    break
                        
                        # For test files, we'll use the file paths directly
                        uploaded_files = selected_paths
                        st.success(f"Selected {len(uploaded_files)} test files")
                else:
                    st.warning("No test files found")
            else:
                st.warning("Test data directory not found")
        
        with batch_tab2:
            st.write("Upload your own multiple WAV audio files")
            uploaded_files_user = st.file_uploader(
                "Choose multiple WAV files", 
                type=["wav"], 
                accept_multiple_files=True,
                key="batch_uploader",
                help="Select 2 or more audio files to see emotion variety in your analysis"
            )
            
            if uploaded_files_user:
                uploaded_files = uploaded_files_user
                st.success(f"Uploaded {len(uploaded_files)} files")
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} files")
            
            # Model selection for batch
            batch_model_name = st.selectbox("Select Model for Analysis", 
                                           list(models_dict.keys()), key="batch_model")
            batch_model = models_dict[batch_model_name]
            
            if st.button("Analyze All Files", type="primary"):
                batch_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Handle both file paths (strings) and uploaded file objects
                    if isinstance(uploaded_file, str):
                        # It's a file path from test examples
                        file_name = os.path.basename(uploaded_file)
                        file_for_processing = uploaded_file
                    else:
                        # It's an uploaded file object
                        file_name = uploaded_file.name
                        file_for_processing = uploaded_file
                    
                    status_text.text(f"Analyzing {file_name}... ({i+1}/{len(uploaded_files)})")
                    
                    try:
                        # Process each file
                        y, sr = librosa.load(file_for_processing, sr=SR, mono=True)
                        y = clean_audio(y)
                        x_vec = mfcc_stats_from_audio(y, sr=SR).reshape(1, -1)
                        
                        pred = batch_model.predict(x_vec)[0]
                        proba = safe_predict_proba(batch_model, x_vec)
                        emotion = classes[pred]
                        confidence = float(proba[pred])
                        
                        batch_results.append({
                            'File': file_name,
                            'Emotion': emotion,
                            'Confidence': f"{confidence:.1%}",
                            'Raw_Confidence': confidence
                        })
                        
                        # Show progress
                        status_text.text(f"✓ {file_name}: {emotion} ({confidence:.1%})")
                        
                    except Exception as e:
                        st.error(f"Error processing {file_name}: {str(e)}")
                        batch_results.append({
                            'File': file_name,
                            'Emotion': 'Error',
                            'Confidence': '0%',
                            'Raw_Confidence': 0.0
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Analysis complete!")
                
                # Display results
                batch_df = pd.DataFrame(batch_results)
                st.subheader("Analysis Results")
                
                # Debug info
                st.write(f"**Total files processed:** {len(batch_results)}")
                st.write(f"**Unique emotions detected:** {batch_df['Emotion'].nunique()}")
                st.write(f"**Emotions found:** {', '.join(batch_df['Emotion'].unique())}")
                
                # Color-coded results
                def color_emotion(val):
                    color = get_emotion_color(val)
                    return f'background-color: {color}; color: white; font-weight: bold'
                
                styled_batch = batch_df.style.applymap(
                    color_emotion, subset=['Emotion']
                ).format({'Raw_Confidence': '{:.1%}'})
                
                st.dataframe(styled_batch, use_container_width=True, hide_index=True)
                
                # Show detailed breakdown
                st.subheader("Detailed Results")
                for i, result in enumerate(batch_results):
                    with st.expander(f"File {i+1}: {result['File']} - {result['Emotion']} ({result['Confidence']})"):
                        st.write(f"**Predicted Emotion:** {result['Emotion']}")
                        st.write(f"**Confidence:** {result['Confidence']}")
                        st.write(f"**Raw Confidence:** {result['Raw_Confidence']:.3f}")
                
                # Analytics charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Emotion distribution pie chart
                    emotion_counts = batch_df['Emotion'].value_counts()
                    pie_fig = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                                   title="Emotion Distribution", color=emotion_counts.index,
                                   color_discrete_map=EMOTION_COLORS)
                    st.plotly_chart(pie_fig, use_container_width=True)
                
                with col2:
                    # Average confidence by emotion
                    avg_confidence = batch_df.groupby('Emotion')['Raw_Confidence'].mean().sort_values(ascending=False)
                    conf_fig = px.bar(x=avg_confidence.index, y=avg_confidence.values,
                                    title="Average Confidence by Emotion",
                                    color=avg_confidence.index, color_discrete_map=EMOTION_COLORS)
                    st.plotly_chart(conf_fig, use_container_width=True)
    
    with tab2:
        st.subheader("Model Battle Arena")
        st.write("Compare predictions from all models on the same audio!")
        
        # Sub-tabs for test files vs upload
        battle_tab1, battle_tab2 = st.tabs(["Test Examples", "Upload File"])
        
        battle_audio = None
        
        with battle_tab1:
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
                            display_name = f"{emotion.title()} - {speaker.upper()} saying '{word}' ({filename})"
                            file_options.append(display_name)
                    
                    selected_example = st.selectbox("Choose a test file:", file_options, key="battle_example")
                    
                    if selected_example != "Select an example...":
                        # Find the corresponding file
                        for f in test_files:
                            filename = os.path.basename(f)
                            if filename in selected_example:
                                battle_audio = f
                                st.success(f"Selected: `{filename}`")
                                st.audio(f, format='audio/wav')
                                break
                else:
                    st.warning("No test files found")
            else:
                st.warning("Test data directory not found")
        
        with battle_tab2:
            st.write("Upload your own WAV audio file")
            battle_audio_user = st.file_uploader("Choose audio for model battle", type=["wav"], key="battle_audio")
            
            if battle_audio_user is not None:
                battle_audio = battle_audio_user
                st.success(f"Uploaded: `{battle_audio_user.name}`")
                st.audio(battle_audio_user, format='audio/wav')
        
        if battle_audio:
            if st.button("Start Battle!", type="primary"):
                with st.spinner("All models are analyzing..."):
                    # Load audio once
                    y, sr = librosa.load(battle_audio, sr=SR, mono=True)
                    y = clean_audio(y)
                    x_vec = mfcc_stats_from_audio(y, sr=SR).reshape(1, -1)
                    
                    battle_results = {}
                    for model_name, model in models_dict.items():
                        pred = model.predict(x_vec)[0]
                        proba = safe_predict_proba(model, x_vec)
                        emotion = classes[pred]
                        confidence = float(proba[pred])
                        
                        battle_results[model_name] = {
                            'Emotion': emotion,
                            'Confidence': confidence,
                            'All_Probabilities': {classes[i]: float(proba[i]) for i in range(len(classes))}
                        }
                    
                    # Display battle results
                    st.subheader("Battle Results")
                    
                    # Create comparison table
                    battle_df = pd.DataFrame({
                        model: {
                            'Predicted Emotion': results['Emotion'],
                            'Confidence': f"{results['Confidence']:.1%}"
                        }
                        for model, results in battle_results.items()
                    }).T
                    
                    # Apply the same color styling as in batch analysis
                    def color_emotion(val):
                        color = get_emotion_color(val)
                        return f'background-color: {color}; color: white; font-weight: bold'
                    
                    styled_battle = battle_df.style.applymap(
                        color_emotion, subset=['Predicted Emotion']
                    )
                    
                    st.dataframe(styled_battle, use_container_width=True)
                    
                    # Create comparison radar charts
                    st.subheader("Model Comparison Radar")
                    cols = st.columns(2)
                    for i, (model_name, results) in enumerate(battle_results.items()):
                        with cols[i % 2]:
                            radar_fig = create_emotion_radar_chart(
                                list(results['All_Probabilities'].values()),
                                list(results['All_Probabilities'].keys())
                            )
                            radar_fig.update_layout(title=f"{model_name} Predictions")
                            st.plotly_chart(radar_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Audio Insights & Analytics")
        st.write("Deep dive into audio characteristics and feature analysis!")
        
        # Sub-tabs for test files vs upload
        insights_tab1, insights_tab2 = st.tabs(["Test Examples", "Upload File"])
        
        insight_audio = None
        
        with insights_tab1:
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
                            display_name = f"{emotion.title()} - {speaker.upper()} saying '{word}' ({filename})"
                            file_options.append(display_name)
                    
                    selected_example = st.selectbox("Choose a test file:", file_options, key="insights_example")
                    
                    if selected_example != "Select an example...":
                        # Find the corresponding file
                        for f in test_files:
                            filename = os.path.basename(f)
                            if filename in selected_example:
                                insight_audio = f
                                st.success(f"Selected: `{filename}`")
                                st.audio(f, format='audio/wav')
                                break
                else:
                    st.warning("No test files found")
            else:
                st.warning("Test data directory not found")
        
        with insights_tab2:
            st.write("Upload your own WAV audio file")
            insight_audio_user = st.file_uploader("Choose audio for deep analysis", type=["wav"], key="insight_audio")
            
            if insight_audio_user is not None:
                insight_audio = insight_audio_user
                st.success(f"Uploaded: `{insight_audio_user.name}`")
                st.audio(insight_audio_user, format='audio/wav')
        
        if insight_audio:
            y, sr = librosa.load(insight_audio, sr=SR, mono=True)
            y = clean_audio(y)
            
            # Audio statistics
            col1, col2, col3, col4 = st.columns(4)
            
            duration = len(y) / sr
            max_amplitude = np.max(np.abs(y))
            rms = np.sqrt(np.mean(y**2))
            zero_crossings = librosa.zero_crossings(y).sum()
            
            with col1:
                st.metric("Duration", f"{duration:.2f}s")
            with col2:
                st.metric("Max Amplitude", f"{max_amplitude:.3f}")
            with col3:
                st.metric("RMS Energy", f"{rms:.3f}")
            with col4:
                st.metric("Zero Crossings", zero_crossings)
            
            # Advanced audio visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Waveform with envelope
                fig, ax = plt.subplots(figsize=(10, 4))
                time = np.arange(0, len(y)) / sr
                ax.plot(time, y, color='#2E86AB', alpha=0.7, linewidth=0.8)
                
                # Add envelope
                envelope = np.abs(y)
                ax.plot(time, envelope, color='red', alpha=0.8, linewidth=1.5, label='Envelope')
                ax.plot(time, -envelope, color='red', alpha=0.8, linewidth=1.5)
                
                ax.set_title('Waveform with Envelope')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Spectral features
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                frames = range(len(spectral_centroids))
                t = librosa.frames_to_time(frames)
                
                ax.plot(t, spectral_centroids, color='green', alpha=0.8, label='Spectral Centroid')
                ax.plot(t, spectral_rolloff, color='orange', alpha=0.8, label='Spectral Rolloff')
                
                ax.set_title('Spectral Features')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Hz')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            # MFCC heatmap
            st.subheader("MFCC Feature Heatmap")
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            img = librosa.display.specshow(mfccs, x_axis='time', ax=ax, cmap='viridis')
            ax.set_title('MFCC Features')
            fig.colorbar(img, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Your Emotion Timeline")
        
        if st.session_state.predictions_history:
            # Create timeline chart
            timeline_fig = create_emotion_timeline(st.session_state.predictions_history)
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Show recent predictions
            st.subheader("Recent Predictions")
            recent_df = pd.DataFrame(st.session_state.predictions_history[-10:])
            if not recent_df.empty:
                recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%H:%M:%S')
                st.dataframe(recent_df[['timestamp', 'emotion', 'confidence']], 
                           use_container_width=True, hide_index=True)
            
            # Clear history button
            if st.button("Clear History"):
                st.session_state.predictions_history = []
                st.rerun()
        else:
            st.info("Make some predictions first to see your emotion timeline!")