from pathlib import Path
#import os
# Absolute path to the project's root directory (where api/, model/, etc. are located)
BASE_DIR = Path(__file__).resolve().parent.parent

# Audio
EXTENSION_AUDIO = ".wav"
SAMPLE_RATE = 22050
N_MFCC = 13
MAX_TIME_STEPS = 130  # frames

# Dataset / Output Classes
DISEASE = ["Asma","Epoc", "Neumonia", "Normal"]

# API 
API_TITLE = "Detection of lung diseases API"
API_DESCRIPTION = "API for respiratory disease classification using lung sound recordings."
MODEL_PATH = BASE_DIR / "ml" / "artifacts" / "respiratory_cnn_blstm.keras"