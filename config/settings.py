from pathlib import Path
#import os
# Ruta absoluta al directorio raíz del proyecto (donde está api/, model/, etc.)
BASE_DIR = Path(__file__).resolve().parent.parent

# Audio
EXTENSION_AUDIO = ".wav"
SAMPLE_RATE = 22050
N_MFCC = 13
MAX_TIME_STEPS = 130  # El ancho que espera tu red neuronal

# Dataset / Output Classes
DISEASE = ["Asma","Epoc", "Neumonia", "Normal"]

# API 
API_TITLE = "Detection of lung diseases API"
API_DESCRIPTION = "API para clasificar enfermedades respiratorias a partir de audios respiratorios."
#MODEL_PATH = (BASE_DIR / "model" / "respiratory_cnn_blstm.keras")
MODEL_PATH = BASE_DIR / "ml" / "artifacts" / "respiratory_cnn_blstm.keras"