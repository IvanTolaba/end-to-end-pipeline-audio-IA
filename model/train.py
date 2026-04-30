#train.py → entrena → guarda en artifacts/
#predict.py → carga artifacts/best_model.h5
#evaluate.py → mide performance
# train.py
#model.save("model/artifacts/best_model.h5")s

# model/train.py

import numpy as np
from pyspark.sql import SparkSession
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_data(parquet_path="data/processed-parquet/"):

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet(parquet_path)

    data = df.select("mfcc", "label").collect()

    X = []
    y = []

    for row in data:
        if len(row["mfcc"]) > 0:
            X.append(row["mfcc"])
            y.append(row["label"])

    X = np.array(X)
    y = np.array(y)

    return X, y


def reshape_data(X):

    # reconstruir MFCC (13 coeficientes)
    X = X.reshape(X.shape[0], 13, -1, 1)
    return X


def build_model(input_shape):

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,1)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,1)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Reshape((-1, 128)),

        tf.keras.layers.LSTM(64, return_sequences=False),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train():

    X, y = load_data()

    X = reshape_data(X)

    y = to_categorical(y, num_classes=4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model(X_train.shape[1:])

    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

    model.save("model/artifacts/best_model.h5")

    print("✅ Modelo entrenado y guardado")


if __name__ == "__main__":
    train()