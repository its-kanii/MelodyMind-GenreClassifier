## 🎵 MelodyMind - Music Genre Classification with Machine Learning 🎶

**MelodyMind** is a Python-based project that classifies songs into different music genres using audio feature extraction and machine learning techniques. Upload any `.wav` file and get instant genre predictions! This project includes a user-friendly Gradio interface for real-time testing.

---

## 🚀 Features

* Extracts meaningful audio features like MFCC, chroma, spectral contrast, and tonnetz using `librosa`
* Trains a robust machine learning model (Random Forest) for genre classification
* Supports 10 popular music genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
* Interactive and easy-to-use Gradio web app for uploading audio and getting predictions
* Visualizes genre distribution and provides playlist recommendations (future work)

---

## 🛠️ Technologies & Tools

* Python 3.x
* Librosa (Audio feature extraction)
* Scikit-learn (Machine learning)
* Joblib (Model serialization)
* Gradio (Web UI)
* Matplotlib & Seaborn (Visualizations)

---

## 📁 Folder Structure

```
MelodyMind-GenreClassifier/
│
├── data/                   # Audio dataset (GTZAN or custom)
│   ├── blues/
│   ├── classical/
│   └── ... (other genres)
│
├── music_genre_classification.py   # Feature extraction + model training script
├── app.py                     # Gradio UI for prediction
├── genre_classifier.pkl       # Saved trained model
├── label_encoder.pkl          # Saved label encoder
├── requirements.txt           # Required packages
└── README.md                  # Project overview and instructions
```

---

## 📥 Dataset

We use the well-known [GTZAN Music Genre Dataset](http://marsyas.info/downloads/datasets.html) consisting of 10 genres and 1000 audio clips (30 seconds each). You can replace it with your own dataset for custom genres or languages.

---

## 💻 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/its-kanii/MelodyMind-GenreClassifier.git
   cd MelodyMind-GenreClassifier
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (optional if you want to retrain):

   ```bash
   python music_genre_classification.py
   ```

4. Run the Gradio app for real-time genre prediction:

   ```bash
   python app.py
   ```

5. Open the provided local link in your browser, upload a `.wav` file, and get your music genre prediction!

---

## 🎯 Future Improvements

* Train a CNN model on spectrograms for better accuracy
* Expand dataset to include Indian/Tamil music genres
* Add playlist recommendation feature based on predicted genres
* Support more audio formats and longer durations
* Add confidence scores and visualization in the UI

---

## 🤝 Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* [GTZAN Dataset](http://marsyas.info/downloads/datasets.html)
* [Librosa](https://librosa.org/)
* [Gradio](https://gradio.app/)
* Inspired by the power of combining AI and music!
