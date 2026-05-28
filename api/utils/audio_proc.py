# Procesamiento acústico puro (Librosa, MFCC, Normalización, Padding/Truncate).

import io
import logging
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from config.settings import SAMPLE_RATE, N_MFCC, MAX_TIME_STEPS

logger = logging.getLogger(__name__)

def preprocess_audio(file_bytes: bytes) -> np.ndarray:
    """Transforma los bytes de audio en un tensor flotante listo para la CRNN."""
    try:
        audio_buffer = io.BytesIO(file_bytes)
        audio, sr = librosa.load(audio_buffer, sr=SAMPLE_RATE)
        
        #Tranforma a MFCC
        mfcc = librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=N_MFCC).T

        # NORMALIZACIÓN
        scaler = StandardScaler()
        mfcc = scaler.fit_transform(mfcc).T
        
        if mfcc.shape[1] < MAX_TIME_STEPS:
            pad_width = MAX_TIME_STEPS - mfcc.shape[1]
            mfcc_fixed = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc_fixed = mfcc[:, :MAX_TIME_STEPS]
            
        mfcc_fixed = mfcc_fixed.astype(np.float32)
        
        # Dimensiones para Keras: (Batch, Filas, Columnas, Canales)
        mfcc_final = np.expand_dims(mfcc_fixed, axis=-1)
        mfcc_final = np.expand_dims(mfcc_final, axis=0)
        
        return mfcc_final
    except Exception as e:
        logger.exception("Error interno en preprocess_audio")
        raise e