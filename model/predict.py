# predict.py
#model = load_model("model/artifacts/best_model.h5")

# model/predict.py

import librosa
import numpy as np
import tensorflow as tf

MODEL_PATH = "model/artifacts/best_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

CLASSES = ['Asma', 'Epoc', 'Neumonia', 'Normal']


def preprocess_audio(file_path):

    y, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    mfcc = mfcc.reshape(1, 13, mfcc.shape[1], 1)

    return mfcc


def predict(file_path):

    X = preprocess_audio(file_path)

    pred = model.predict(X)

    class_id = np.argmax(pred)

    return CLASSES[class_id]