import argparse
import numpy as np
import librosa
from tensorflow.keras.models import load_model

MODEL_PATH = "models/emotion_lstm.h5"
LABEL_PATH = "models/label_classes.npy"

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,
        axis=0
    )
    return mfcc

# -----------------------------
# CLI ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser(description="Speech Emotion Prediction")
parser.add_argument("--audio", required=True, help="Path to wav file")

args = parser.parse_args()

# -----------------------------
# LOAD MODEL + LABELS
# -----------------------------
model = load_model(MODEL_PATH)
labels = np.load(LABEL_PATH, allow_pickle=True)

# -----------------------------
# PREDICT
# -----------------------------
mfcc = extract_mfcc(args.audio)
mfcc = np.expand_dims(mfcc, axis=(0, -1))

pred = model.predict(mfcc)[0]
index = np.argmax(pred)

print("Predicted Emotion:", labels[index])
print("Confidence:", round(float(pred[index]), 4))
