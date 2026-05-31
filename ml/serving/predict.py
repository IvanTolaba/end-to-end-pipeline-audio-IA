import logging
import numpy as np
from keras.models import load_model
from config.settings import MODEL_PATH, DISEASE, N_MFCC

logger = logging.getLogger(__name__)

class RespiratoryPredictor:
    """
    Clase Singleton optimizada para la carga en memoria del modelo 
    y ejecución de inferencias en caliente desde la API.
    """
    def __init__(self):
        self.model = None
        self._load_model_artifacts()

    def _load_model_artifacts(self):
        if not MODEL_PATH.exists():
            logger.error(f"Falta el binario del modelo en la ruta estipulada: {MODEL_PATH}")
            raise FileNotFoundError("Artefacto del modelo no disponible para inferencia.")
        
        logger.info("Cargando modelo neuronal en memoria de servicio...")
        self.model = load_model(MODEL_PATH)
        logger.info("Modelo de producción cargado exitosamente.")

    def predict_audio_features(self, mfcc_features: list) -> dict:
        """
        Recibe una lista o array plano de características MFCC extraídas en caliente,
        las moldea al tamaño esperado por la red y computa la predicción.
        """
        # Formatear la lista entrante a un array bidimensional compatible con la red (13, 130, 1)
        # Aseguramos que tenga el shape exacto que requiere el input convolucional
        X_input = np.array(mfcc_features).reshape(1, N_MFCC, 130, 1)
        
        # Ejecutar inferencia síncrona
        preds = self.model.predict(X_input)[0]
        max_idx = np.argmax(preds)
        
        return {
            "prediction": DISEASE[max_idx],
            "confidence": float(preds[max_idx]),
            "probabilities": {DISEASE[i]: float(preds[i]) for i in range(len(DISEASE))}
        }