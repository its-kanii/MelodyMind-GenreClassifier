import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(spectral_contrast.T, axis=0)
    
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)
    
    return np.hstack([mfccs_mean, chroma_mean, contrast_mean, tonnetz_mean])


def load_dataset(data_path):
    genres = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"Found genres: {genres}")
    
    features_list = []
    labels = []
    
    for genre in genres:
        genre_dir = os.path.join(data_path, genre)
        print(f"Processing genre: {genre}")
        for filename in os.listdir(genre_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(genre_dir, filename)
                try:
                    features = extract_features(file_path)
                    features_list.append(features)
                    labels.append(genre)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return np.array(features_list), np.array(labels)


def plot_genre_distribution(labels):
    plt.figure(figsize=(10,6))
    sns.countplot(labels)
    plt.title("Genre Distribution")
    plt.xlabel("Genre")
    plt.ylabel("Number of Songs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    data_path = r"C:\Users\kanim\Desktop\MusicGenreClassification\data"  # Change this if your dataset folder is elsewhere
    
    print("Loading dataset and extracting features...")
    X, y = load_dataset(data_path)
    print(f"Extracted features from {len(X)} files.")
    
    plot_genre_distribution(y)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, "genre_classifier.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")


    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")
    

if __name__ == "__main__":
    main()
