import io
from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
import librosa  

import logging
logger = logging.getLogger(__name__)

from config.settings import (
    API_TITLE,
    API_DESCRIPTION,
    MODEL_PATH,
    EXTENSION_AUDIO,
    SAMPLE_RATE,
    N_MFCC,
    MAX_TIME_STEPS,
    DISEASE
)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION
)
# 1. CARGA DEL MODELO EN MEMORIA
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    #print(f"[SUCCESS] ¡Modelo cargado correctamente desde: {MODEL_PATH}!")
    logger.info("Modelo cargado correctamente desde: %s",MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error crítico al cargar el modelo: {str(e)}")   

#-------------
# 2. FUNCIÓN DE PROCESAMIENTO ACÚSTICO INDIVIDUAL
def preprocess_audio(file_bytes: bytes) -> np.ndarray:
    """
    Versión Senior optimizada para leer archivos de audio grandes en memoria,
    extraer MFCC y adaptarlos exactamente a (1, 13, 130, 1) en float32.
    """
    try:
        # 1. Usar BytesIO para que librosa pueda leer el archivo grande como si estuviera en disco
        audio_buffer = io.BytesIO(file_bytes)
        
        # Cargar el audio (Ajusta el 'sr' al que usaste en tu entrenamiento, ej: 22050 o 16000)
        # Usamos sr=None para mantener el sample rate nativo del archivo
        audio, sr = librosa.load(audio_buffer, sr=SAMPLE_RATE)
        
        # 2. Extraer exactamente 13 coeficientes MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        # 3. Forzar el eje del tiempo a que sea exactamente 130 (Padding / Truncate)
        #max_time_steps = 130
        
        if mfcc.shape[1] < MAX_TIME_STEPS:
            # Si es más corto, rellenamos con ceros
            pad_width = MAX_TIME_STEPS - mfcc.shape[1]
            mfcc_fixed = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            # Si es más largo (como tu audio de 10MB), recortamos los primeros 130 pasos de tiempo
            mfcc_fixed = mfcc[:, :MAX_TIME_STEPS]
            
        # 4. Convertir explícitamente a float32 (crucial para evitar errores en TensorFlow)
        mfcc_fixed = mfcc_fixed.astype(np.float32)
            
        # 5. Agregar dimensiones de Canal (1) y Batch (1) -> de (13, 130) pasa a (1, 13, 130, 1)
        mfcc_final = np.expand_dims(mfcc_fixed, axis=-1)  # (13, 130, 1)
        mfcc_final = np.expand_dims(mfcc_final, axis=0)   # (1, 13, 130, 1)
        
        return mfcc_final

    except Exception as e:
        # Esto imprimirá el error real en tu consola de Uvicorn para que puedas leerlo
        logger.exception("Error interno en preprocess_audio")
        raise 


# 3. ENDPOINT PARA RECIBIR EL AUDIO
@app.post("/predict", tags=["ML Inference"])
async def predict_disease(file: UploadFile = File(...)):
    # Validar defensivamente que sea un archivo de audio    
    #if not file.filename.endswith((EXTENSION_AUDIO)):
    if not file.filename.lower().endswith(EXTENSION_AUDIO):    
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Debe ser WAV.")
    
    try:
        # Leer los bytes del archivo que subió el usuario
        audio_content = await file.read()
        
        # Preprocesar
        features = preprocess_audio(audio_content)
        
        # Ejecutar la inferencia del modelo .h5
        predictions = model.predict(features)
        
        # Obtener el índice de la predicción más alta
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Mapear al nombre de la enfermedad
        result_disease = DISEASE[predicted_class_idx]
        
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": result_disease,
            "confidence": f"{confidence * 100:.2f}%"
        }
        
    except Exception as e:
        # Registrar el error real en tus logs para hacer debug en producción        
        raise HTTPException(status_code=500, detail=f"Error interno procesando el audio: {str(e)}")