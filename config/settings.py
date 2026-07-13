# ==============================================================================
# config/settings.py
# ==============================================================================
import os
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. Base Structural Directories Layout
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# We export the root as an absolute string
# for external environments like Airflow
PROJECT_ROOT = str(BASE_DIR)

# Core Pipeline Folders
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Production-Grade MLOps Storage Layout
ML_ROOT_DIR = BASE_DIR / "ml"
ML_ARTIFACTS_DIR = ML_ROOT_DIR / "artifacts"
ML_REPORTS_DIR = ML_ROOT_DIR / "reports"

# 🔹 NEW: Explicit Sub-Artifacts Directory Trees
ML_MODELS_DIR = ML_ARTIFACTS_DIR / "models"
ML_CACHE_DIR = ML_ARTIFACTS_DIR / "cache"
ML_HISTORY_DIR = ML_ARTIFACTS_DIR / "history"#

# Automated directory tree safe-generation
#os.makedirs(ML_ARTIFACTS_DIR, exist_ok=True)
#os.makedirs(ML_REPORTS_DIR, exist_ok=True)
os.makedirs(ML_REPORTS_DIR, exist_ok=True)
os.makedirs(ML_MODELS_DIR, exist_ok=True)
os.makedirs(ML_CACHE_DIR, exist_ok=True)
os.makedirs(ML_HISTORY_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. Audio & Digital Signal Processing (DSP) Configuration
# ------------------------------------------------------------------------------
EXTENSION_AUDIO = ".wav"
SAMPLE_RATE = 22050
N_MFCC = 13
NUM_TIMESTEPS = 130 
MAX_TIME_STEPS = 130   # 🔹 Kept for legacy ETL pipeline compatibility

# Audio Segmentation Parameters (Sliding Window Strategy)
WINDOW_TIME = 3         # Duration of each audio chunk in seconds (SECOND)
OVERLAP = 0.25          # 25% overlap between adjacent window slices

# ------------------------------------------------------------------------------
# 3. Dataset Target Classes (Folder Names Alignment)
# ------------------------------------------------------------------------------
NUM_CLASSES = 4
CLASS_NAMES = ["Asma", "Epoc", "Neumonia", "Normal"]
DISEASE = ["Asma", "Epoc", "Neumonia", "Normal"] # 🔹 Kept for legacy ETL compatibility

# ------------------------------------------------------------------------------
# 4. Multi-Architecture Routing Switch (Experimentation Engine)
# ------------------------------------------------------------------------------
MODEL_ARCHITECTURE_TYPE = 'CNN_BLSTM' 

# ------------------------------------------------------------------------------
# 5. Serialization Artifacts Reference Paths
# ------------------------------------------------------------------------------
# Subfolder: models/
MODEL_SAVE_PATH = ML_MODELS_DIR / "respiratory_cnn_blstm.keras"
MODEL_PATH = MODEL_SAVE_PATH                      

# Subfolder: cache/
X_TEST_CACHE_PATH = ML_CACHE_DIR / "X_test.npy"
y_TEST_CACHE_PATH = ML_CACHE_DIR / "y_test.npy"

# Subfolder: history/
HISTORY_LOSS_PATH = ML_HISTORY_DIR / "history_loss.npy"
HISTORY_VAL_LOSS_PATH = ML_HISTORY_DIR / "history_val_loss.npy"

# ------------------------------------------------------------------------------
# 6. Reports & Verification Outputs Reference Paths
# ------------------------------------------------------------------------------
REPORT_TXT_PATH = ML_REPORTS_DIR / "classification_report.txt"
REPORT_JSON_PATH = ML_REPORTS_DIR / "metrics.json"
PLOT_ROC_PATH = ML_REPORTS_DIR / "roc_curve.png"
PLOT_MATRIX_PATH = ML_REPORTS_DIR / "confusion_matrix.png"
PLOT_HISTORY_PATH = ML_REPORTS_DIR / "training_history.png"

# ------------------------------------------------------------------------------
# 7. Machine Learning Dataset Partitioning Configurations
# ------------------------------------------------------------------------------
RANDOM_STATE = 42
TRAIN_SIZE_RATIO = 0.7
TEMP_SPLIT_RATIO = 0.5  # Splits the remaining 30% into 15% Val / 15% Test

# ------------------------------------------------------------------------------
# 8. Deep Learning Architecture Hyperparameters
# ------------------------------------------------------------------------------
CONV_REGULARIZER_L2 = 0.005
DENSE_REGULARIZER_L2 = 0.0001
CNN_DROPOUT = 0.1
LSTM_DROPOUT = 0.3
FINAL_DROPOUT = 0.5
INITIAL_LEARNING_RATE = 0.00001

# ------------------------------------------------------------------------------
# 9. Neural Network Runtime Optimization Cycles
# ------------------------------------------------------------------------------
#TRAINING_EPOCHS = 250
TRAINING_EPOCHS = 2
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 25
REDUCE_LR_PATIENCE = 10
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

# ------------------------------------------------------------------------------
# 10. Production Fast-API Presentation Layer Meta-Configurations
# ------------------------------------------------------------------------------
API_TITLE = "Detection of lung diseases API"
API_DESCRIPTION = "API for respiratory disease classification using lung sound recordings."










