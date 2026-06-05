import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from keras.utils import to_categorical

from config.settings import (
    N_MFCC, NUM_CLASSES, RANDOM_STATE, TRAIN_SIZE_RATIO, 
    TEMP_SPLIT_RATIO, NUM_TIMESTEPS, X_TEST_CACHE_PATH, y_TEST_CACHE_PATH
)

logger = logging.getLogger(__name__)

def load_parquet_layer(processed_path: str | Path) -> pd.DataFrame:
    """
    Reads structured feature sequences from the system Data Lake storage.
    """

    logger.info("Loading tabular feature matrix from Parquet Data Lake storage")
    path_obj = Path(processed_path)
    if not path_obj.exists():
        logger.error("Processed Parquet layer path not found: %s", processed_path)
        raise FileNotFoundError(f"Missing processed storage: {processed_path}")
    return pd.read_parquet(str(path_obj))

def balance_and_split_pipeline(df: pd.DataFrame) -> tuple:
    """
    It balances the dataset classes. It performs rigorous grouping 
    based on the original file IDs ("file_id") to completely prevent data leakage 
    between training, validation, and test partitions. 
    Finally, it normalizes the data.
    return:
        X_train: Feature matrix used for training: (N_train_segments, 13, 130, 1)
        y_train: Training label matrix: (N_train_segments, 4)
        X_val: Feature matrix used for validation : (N_val_segments, 13, 130, 1)
        y_val: Validation label matrix: (N_val_segments, 4)
        X_test: Feature matrix used for the final blind testing: (N_test_segments, 13, 130, 1)
        y_test: Test label matrix: (N_test_segments, 4)    
    
    """
    logger.info("Executing class balancing through downsampling strategies")
    
    X_raw = df["mfcc"].values
    y_raw = df["label"].values
    group_ids = df["file_id"].values
    
    initial_counts = Counter(y_raw)
    logger.info("Unbalanced dataset distribution: %s", dict(initial_counts))
    
    if not initial_counts:
        raise ValueError("No records found to execute downsampling pipeline")
        
    min_samples_target = min(initial_counts.values())
    bal_X, bal_y, bal_groups = [], [], []
    
    for class_id in range(NUM_CLASSES):
        indices = [idx for idx, y in enumerate(y_raw) if y == class_id]
        if len(indices) < min_samples_target:
            logger.warning("Class ID %s exhibits insufficient sample count", class_id)
            continue
            
        sampled_indices = resample(
            indices, n_samples=min_samples_target, 
            random_state=RANDOM_STATE, replace=False
        )
        for i in sampled_indices:
            bal_X.append(X_raw[i])
            bal_y.append(y_raw[i])
            bal_groups.append(group_ids[i])
            
    logger.info("Balanced distribution set at %s segments per class", min_samples_target)

    X_arr, y_arr, groups_arr = np.array(bal_X), np.array(bal_y), np.array(bal_groups)

    # Use GroupShuffleSplit to prevent audio segments
    # from the same recording appearing in multiple sets.   
    gss_primary = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE_RATIO, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss_primary.split(X_arr, y_arr, groups=groups_arr))
    
    gss_secondary = GroupShuffleSplit(n_splits=1, train_size=TEMP_SPLIT_RATIO, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss_secondary.split(X_arr[temp_idx], y_arr[temp_idx], groups=groups_arr[temp_idx]))
    
    X_train_raw, y_train = X_arr[train_idx], y_arr[train_idx]
    X_val_raw, y_val = X_arr[temp_idx][val_idx], y_arr[temp_idx][val_idx]
    X_test_raw, y_test = X_arr[temp_idx][test_idx], y_arr[temp_idx][test_idx]
    
    logger.info("Standardizing feature maps and transforming matrices")
    X_train = np.array([np.array(item).reshape(N_MFCC, NUM_TIMESTEPS) for item in X_train_raw])
    X_val = np.array([np.array(item).reshape(N_MFCC, NUM_TIMESTEPS) for item in X_val_raw])
    X_test = np.array([np.array(item).reshape(N_MFCC, NUM_TIMESTEPS) for item in X_test_raw])
    
    scaler = StandardScaler()
    X_train = np.array([scaler.fit_transform(frame.T).T for frame in X_train]).astype("float32")
    X_val = np.array([scaler.fit_transform(frame.T).T for frame in X_val]).astype("float32")
    X_test = np.array([scaler.fit_transform(frame.T).T for frame in X_test]).astype("float32")
    
    X_train = X_train.reshape(-1, N_MFCC, NUM_TIMESTEPS, 1)
    X_val = X_val.reshape(-1, N_MFCC, NUM_TIMESTEPS, 1)
    X_test = X_test.reshape(-1, N_MFCC, NUM_TIMESTEPS, 1)
    
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
    
    # Cache evaluation tensors onto the disk layer for evaluate.py decoupling
    np.save(X_TEST_CACHE_PATH, X_test)
    np.save(y_TEST_CACHE_PATH, y_test)
    logger.info("Evaluation test arrays successfully cached inside artifacts storage folder")
    
    return X_train, y_train, X_val, y_val, X_test, y_test