# Emotion-Recognition-From-Speech
#  Speech Emotion Recognition (SER) Project

##  Overview
This project focuses on **Speech Emotion Recognition (SER)** using **classical machine learning models** (XGBoost, SVM, Random Forest) and **deep learning architectures** (BiLSTM, 1D CNN, Advanced CRNN).  
The system is designed to classify speech audio into **six emotions**: Angry, Disgust, Fear, Happy, Neutral, and Sad.

A **Streamlit web app** has been built for easy testing and demonstration.  
🔗 **Live Demo:** (https://emotion-recognition-from-speech.streamlit.app/)

---

##  Datasets
We used three widely recognized emotional speech datasets:

### 🎙️ 1. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 24 professional actors (12 male, 12 female)
- 8 emotions: calm, happy, sad, angry, fearful, surprise, disgust, neutral
- Each utterance spoken with two levels of emotional intensity
- Studio-quality recordings, clear emotional expression

###  2. CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)
- 91 actors (48 male, 43 female)
- 6 emotions: happy, sad, anger, fear, disgust, neutral
- Multiple sentences spoken in different emotions and intensities
- More natural variations in tone and accent compared to RAVDESS

### 🎙️ . TESS (Toronto Emotional Speech Set)
- 2 female actors (younger & older voice)
- 7 emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, neutral
- Each emotion recorded for 200+ target words

---

##  Features & Processing
- **Audio Preprocessing**: Silence trimming, normalization
- **Feature Extraction**: MFCCs (Mel-frequency cepstral coefficients)
- **Classical Models**: XGBoost, SVM (RBF & Linear), Random Forest
- **Deep Models**:
  - **BiLSTM** – Temporal sequence learning
  - **1D CNN** – Local feature extraction
  - **Advanced CRNN** – Combined CNN & RNN for spatial + temporal learning


