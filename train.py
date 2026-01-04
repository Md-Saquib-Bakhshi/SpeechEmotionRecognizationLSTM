import os
import warnings
import numpy as np
import pandas as pd
import librosa

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

warnings.filterwarnings("ignore")

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "audio")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_lstm.h5")
LABEL_PATH = os.path.join(MODEL_DIR, "label_classes.npy")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# SKIP TRAINING IF MODEL EXISTS
# -----------------------------
if os.path.exists(MODEL_PATH):
    print("[INFO] Model already exists. Skipping training.")
    exit(0)

# -----------------------------
# LOAD DATA (FOLDER-BASED)
# -----------------------------
paths = []
labels = []

for emotion in os.listdir(DATA_DIR):
    emotion_path = os.path.join(DATA_DIR, emotion)

    if not os.path.isdir(emotion_path):
        continue

    for file in os.listdir(emotion_path):
        if file.endswith(".wav"):
            paths.append(os.path.join(emotion_path, file))
            labels.append(emotion.lower())

print(f"[INFO] Dataset loaded: {len(paths)} files")

df = pd.DataFrame({
    "speech": paths,
    "label": labels
})

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

X = np.array([extract_mfcc(x) for x in df["speech"]])
X = np.expand_dims(X, -1)

# -----------------------------
# ENCODE LABELS
# -----------------------------
encoder = OneHotEncoder()
y = encoder.fit_transform(df[["label"]]).toarray()

np.save(LABEL_PATH, encoder.categories_[0])
print("[INFO] Label classes saved")

# -----------------------------
# MODEL
# -----------------------------
model = Sequential([
    LSTM(256, input_shape=(40, 1)),
    Dropout(0.2),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(y.shape[1], activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# TRAIN
# -----------------------------
model.fit(
    X,
    y,
    validation_split=0.2,
    epochs=50,
    batch_size=64
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save(MODEL_PATH)
print(f"[SUCCESS] Model saved at {MODEL_PATH}")
