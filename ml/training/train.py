import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, LSTM, Dense, Bidirectional, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras.utils import to_categorical

from config.settings import DATA_PROCESSED_DIR, N_MFCC, MAX_TIME_STEPS

logger = logging.getLogger(__name__)


def load_parquet_layer(processed_path: str | Path) -> pd.DataFrame:
    """
    Reads the structured data from the Silver Lakehouse folder into a memory-efficient Pandas DataFrame.
    """
    logger.info("Loading tabular feature matrix from Parquet Data Lake storage")
    path_obj = Path(processed_path)
    
    if not path_obj.exists():
        logger.error("Processed Parquet layer path not found: %s", processed_path)
        raise FileNotFoundError(f"Missing processed storage: {processed_path}")
        
    df_pandas = pd.read_parquet(str(path_obj))
    logger.info("Loaded %s records from Parquet store successfully", len(df_pandas))
    return df_pandas


def balance_and_split_pipeline(df: pd.DataFrame, num_classes: int = 4):
    """
    Balances dataset classes and performs a rigorous group split based on original 
    file IDs to completely avoid data leakage across train, val, and test partitions.
    """
    logger.info("Executing class balancing through downsampling strategies")
    
    # Trace target features and grouping columns
    X_raw = df["mfcc"].values
    y_raw = df["label"].values
    group_ids = df["file_id"].values
    
    initial_counts = Counter(y_raw)
    logger.info("Unbalanced dataset distribution: %s", dict(initial_counts))
    
    if not initial_counts:
        raise ValueError("No records found to execute downsampling pipeline")
        
    min_samples_target = min(initial_counts.values())
    
    bal_X, bal_y, bal_groups = [], [], []
    
    # Downsample each category to match minority class length
    for class_id in range(num_classes):
        indices = [idx for idx, y in enumerate(y_raw) if y == class_id]
        
        if len(indices) < min_samples_target:
            logger.warning("Class ID %s exhibits insufficient sample count", class_id)
            continue
            
        sampled_indices = resample(
            indices, 
            n_samples=min_samples_target, 
            random_state=42, 
            replace=False
        )
        
        for i in sampled_indices:
            bal_X.append(X_raw[i])
            bal_y.append(y_raw[i])
            bal_groups.append(group_ids[i])
            
    logger.info("Balanced distribution set at %s segments per class", min_samples_target)

    # --------------------------------------------------
    # Rigorous Partitioning using GroupShuffleSplit
    # --------------------------------------------------
    logger.info("Executing GroupShuffleSplit partition strategy to isolate file domains")
    
    X_arr = np.array(bal_X)
    y_arr = np.array(bal_y)
    groups_arr = np.array(bal_groups)
    
    # First split: 70% Train / 30% Temporal holding
    gss_primary = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, temp_idx = next(gss_primary.split(X_arr, y_arr, groups=groups_arr))
    
    # Split temporal subset into 50% Validation / 50% Test (15% / 15% overall split)
    gss_secondary = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_secondary.split(X_arr[temp_idx], y_arr[temp_idx], groups=groups_arr[temp_idx]))
    
    # Unpack split results
    X_train_raw, y_train = X_arr[train_idx], y_arr[train_idx]
    X_val_raw, y_val = X_arr[temp_idx][val_idx], y_arr[temp_idx][val_idx]
    X_test_raw, y_test = X_arr[temp_idx][test_idx], y_arr[temp_idx][test_idx]
    
    # --------------------------------------------------
    # Signal Feature Normalization & Reshaping
    # --------------------------------------------------
    logger.info("Standardizing feature maps and transforming matrices for deep architectures")
    
    # Reshape 1D flattened rows back into 2D spectrotemporal frames (13 coefficients x 130 timesteps)
    # 13 * 130 = 1690 points per audio window
    num_timesteps = 130 
    
    X_train = np.array([np.array(item).reshape(N_MFCC, num_timesteps) for item in X_train_raw])
    X_val = np.array([np.array(item).reshape(N_MFCC, num_timesteps) for item in X_val_raw])
    X_test = np.array([np.array(item).reshape(N_MFCC, num_timesteps) for item in X_test_raw])
    
    # Apply Standard Scaler row by row to retain independent local distribution curves
    scaler = StandardScaler()
    X_train = np.array([scaler.fit_transform(frame.T).T for frame in X_train]).astype("float32")
    X_val = np.array([scaler.fit_transform(frame.T).T for frame in X_val]).astype("float32")
    X_test = np.array([scaler.fit_transform(frame.T).T for frame in X_test]).astype("float32")
    
    # Fit channel dimension for 2D Convolutions input layer format -> (Batch, Height, Width, Channels)
    X_train = X_train.reshape(-1, N_MFCC, num_timesteps, 1)
    X_val = X_val.reshape(-1, N_MFCC, num_timesteps, 1)
    X_test = X_test.reshape(-1, N_MFCC, num_timesteps, 1)
    
    # Map numerical integers directly to standard categorical cross-entropy One-Hot representation
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    
    logger.info("Partition processing complete. Train shape: %s, Val shape: %s", X_train.shape, X_val.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn_blst_architecture(input_shape=(13, 130, 1), num_classes=4) -> Sequential:
    """
    Instantiates your state-of-the-art hybrid Deep Learning CNN-Bidirectional LSTM 
    network topology tailored for micro-structural acoustic bio-marker mapping.
    """
    logger.info("Building hybrid CNN-Bidirectional LSTM deep neural network structural blocks")
    model = Sequential()

    # 🔹 Convolutional Neural Block 1
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', 
                     kernel_regularizer=l2(0.005), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # 🔹 Convolutional Neural Block 2
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.005)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # 🔹 Convolutional Neural Block 3
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.005)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # 🔹 Re-shaping dimensional arrays to establish standard Recurrent Sequence compatibility: [timesteps, features]
    model.add(Reshape((model.output_shape[1] * model.output_shape[2], model.output_shape[3])))

    # 🔹 Deep Recurrent Bidirectional LSTM Block layers
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)))
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)))

    # 🔹 Global pooling operation over time sequence steps
    model.add(GlobalAveragePooling1D())

    # 🔹 Classification Layer Blocks
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile network using a refined low learning rate schedule
    optimizer = Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    logger.info("Model topology compiled with loss: categorical_crossentropy")
    return model


def execute_model_training():
    """
    Orchestrates the entire end-to-end processing and model training steps.
    """
    # 1. Load Parquet table rows from Data Lake Layer
    df_silver = load_parquet_layer(DATA_PROCESSED_DIR)
    
    # 2. Extract balanced split tensors without file leakages
    X_train, y_train, X_val, y_val, X_test, y_test = balance_and_split_pipeline(df_silver)
    
    # 3. Model assembly
    nn_model = build_cnn_blst_architecture()
    
    # 4. Optimization callback strategies
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
    
    logger.info("Commencing network training fit procedure")
    # 5. Fit neural topology
    nn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=250,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    logger.info("Evaluating optimal weights matrix on unseen Test set arrays")
    loss_metrics, accuracy_metrics = nn_model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test results completed -> Loss: %s | Accuracy: %s", loss_metrics, accuracy_metrics)


if __name__ == "__main__":
    execute_model_training()