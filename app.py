import gradio as gr
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load trained model and label encoder
model = joblib.load("genre_classifier.pkl")  # Save your trained model here
label_encoder = joblib.load("label_encoder.pkl")

# Feature extractor (same as training)
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)

    return np.hstack([mfccs_mean, chroma_mean, contrast_mean, tonnetz_mean])


# Prediction function
def predict_genre(audio):
    features = extract_features(audio)
    features = features.reshape(1, -1)
    pred = model.predict(features)[0]
    genre = label_encoder.inverse_transform([pred])[0]
    return f"🎵 Predicted Genre: {genre}"

# Gradio UI
interface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Audio(type="filepath", label="Upload your song (.wav only)"),
    outputs="text",
    title="Music Genre Classifier",
    description="Upload a .wav file and the model will predict its genre."
)

interface.launch()
