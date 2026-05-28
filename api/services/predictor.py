# Capa de Machine Learning (Carga el modelo y ejecuta model.predict).

import logging
import tensorflow as tf
import numpy as np
from config.settings import MODEL_PATH, DISEASE

logger = logging.getLogger(__name__)

class RespiratoryPredictor:
    def __init__(self):
        self.model = None

    # Carga el modelo en memoria de forma perezosa/controlada.
    def load_model(self):        
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Modelo cargado correctamente desde: %s", MODEL_PATH)
        except Exception as e:
            logger.critical("Error crítico al cargar el modelo: %s", str(e))
            raise RuntimeError(f"Error crítico al cargar el modelo: {str(e)}")

    # Predice, ejecuta la inferencia en la red neuronal.
    def predict(self, features: np.ndarray) -> tuple[str, float]:        
        if self.model is None:
            raise RuntimeError("El modelo no ha sido cargado en memoria.")
            
        predictions = self.model.predict(features)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        result_disease = DISEASE[predicted_class_idx]
        
        return result_disease, confidence

# Instancia única (Singleton) para usar en las rutas
predictor_service = RespiratoryPredictor()