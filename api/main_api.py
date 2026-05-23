import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
import librosa  # O la librería que uses para procesar el audio individualmente

app = FastAPI(
    title="Respiratory Disease Detection API",
    description="API para clasificar enfermedades respiratorias a partir de audios de tos/respiración."
)

# 1. CARGA DEL MODELO EN MEMORIA (Se ejecuta UNA sola vez al levantar el servidor)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/respiratory_cnn_blstm.keras")

try:
    # Si usas TensorFlow/Keras completo:
    model = tf.keras.models.load_model(MODEL_PATH)
    # Nota Senior: Mapear las clases exactamente en el mismo orden que tu label encoder
    CLASSES = ["Asma", "Epoc", "Neumonia", "Normal"]
except Exception as e:
    raise RuntimeError(f"Error crítico al cargar el modelo .h5: {str(e)}")
    

#-------------
# 2. FUNCIÓN DE PROCESAMIENTO ACÚSTICO INDIVIDUAL
def preprocess_audio(file_bytes) -> np.ndarray:
    """
    Versión Senior optimizada para leer archivos de audio grandes en memoria,
    extraer MFCC y adaptarlos exactamente a (1, 13, 130, 1) en float32.
    """
    try:
        # 1. Usar BytesIO para que librosa pueda leer el archivo grande como si estuviera en disco
        audio_buffer = io.BytesIO(file_bytes)
        
        # Cargar el audio (Ajusta el 'sr' al que usaste en tu entrenamiento, ej: 22050 o 16000)
        # Usamos sr=None para mantener el sample rate nativo del archivo
        audio, sr = librosa.load(audio_buffer, sr=None)
        
        # 2. Extraer exactamente 13 coeficientes MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # 3. Forzar el eje del tiempo a que sea exactamente 130 (Padding / Truncate)
        max_time_steps = 130
        
        if mfcc.shape[1] < max_time_steps:
            # Si es más corto, rellenamos con ceros
            pad_width = max_time_steps - mfcc.shape[1]
            mfcc_fixed = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            # Si es más largo (como tu audio de 10MB), recortamos los primeros 130 pasos de tiempo
            mfcc_fixed = mfcc[:, :max_time_steps]
            
        # 4. Convertir explícitamente a float32 (crucial para evitar errores en TensorFlow)
        mfcc_fixed = mfcc_fixed.astype(np.float32)
            
        # 5. Agregar dimensiones de Canal (1) y Batch (1) -> de (13, 130) pasa a (1, 13, 130, 1)
        mfcc_final = np.expand_dims(mfcc_fixed, axis=-1)  # (13, 130, 1)
        mfcc_final = np.expand_dims(mfcc_final, axis=0)   # (1, 13, 130, 1)
        
        return mfcc_final

    except Exception as e:
        # Esto imprimirá el error real en tu consola de Uvicorn para que puedas leerlo
        print(f"❌ [ERROR INTERNO PREPROCESAMIENTO]: {str(e)}")
        raise e


# 3. ENDPOINT PARA RECIBIR EL AUDIO
@app.post("/predict", tags=["ML Inference"])
async def predict_disease(file: UploadFile = File(...)):
    # Validar defensivamente que sea un archivo de audio
    if not file.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Debe ser WAV o MP3.")
    
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
        result_disease = CLASSES[predicted_class_idx]
        
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": result_disease,
            "confidence": f"{confidence * 100:.2f}%"
        }
        
    except Exception as e:
        # Registrar el error real en tus logs para hacer debug en producción
        # logger.error("Fallo en inferencia: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error interno procesando el audio: {str(e)}")