# Define los endpoints (el POST /predict), validaciones HTTP.

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from config.settings import EXTENSION_AUDIO
from api.utils.audio_proc import preprocess_audio
from api.services.predictor import predictor_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["IA Inference"])

@router.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(EXTENSION_AUDIO):    
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Debe ser WAV.")
    
    try:
        audio_content = await file.read()

        if not audio_content:
            logger.warning("Archivo vacío recibido: %s", file.filename)
            raise HTTPException(status_code=400, detail="El archivo está vacío.")
        
        # 1. Llamamos a la capa utilitaria de Audio-Se obtiene MFCC
        features = preprocess_audio(audio_content)
        
        # 2. Llamamos a la capa de Servicio de ML, con MFCC predice
        result_disease, confidence = predictor_service.predict(features)

        logger.info(
            "Predicción exitosa | archivo=%s | clase=%s | confianza=%.2f%%",
            file.filename, result_disease, confidence * 100
        )
        
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": result_disease,
            "confidence": f"{confidence * 100:.2f}%"                        
        }
        
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno procesando el audio: {str(e)}")