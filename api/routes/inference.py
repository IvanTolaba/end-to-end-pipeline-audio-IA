# Defines the endpoints (the POST /predict), HTTP validations.

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
        raise HTTPException(status_code=400, detail="Unsupported file format. Must be WAV.")
    
    try:
        audio_content = await file.read()

        if not audio_content:
            logger.warning("Empty file received: %s", file.filename)
            raise HTTPException(status_code=400, detail="The file is empty.")
        
        # 1. We call Utils. We get MFCC
        features = preprocess_audio(audio_content)
        
        # 2. We called Services. With MFCC it predicts.
        result_disease, confidence = predictor_service.predict(features)

        logger.info(
            "Prediction completed | file=%s | class=%s | confidence=%.2f%%",
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
        raise HTTPException(status_code=500, detail=f"Internal error processing audio: {str(e)}")
    

    