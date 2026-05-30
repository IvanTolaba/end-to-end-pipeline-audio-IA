# Machine Learning Layer (Load the model and run model.predict).

import logging
import tensorflow as tf
import numpy as np
from config.settings import MODEL_PATH, DISEASE

logger = logging.getLogger(__name__)

class RespiratoryPredictor:
    def __init__(self):
        self.model = None

    # Load the model into memory in a lazy/controlled manner.
    def load_model(self):        
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Model successfully loaded from: %s", MODEL_PATH)
        except Exception as e:
            logger.critical("Critical error loading model: %s", str(e))
            raise RuntimeError(f"Critical error loading model: {str(e)}")

    # Predict, execute inference in the neural network.
    def predict(self, features: np.ndarray) -> tuple[str, float]:        
        if self.model is None:
            raise RuntimeError("The model has not been loaded into memory.")
            
        predictions = self.model.predict(features)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        result_disease = DISEASE[predicted_class_idx]
        
        return result_disease, confidence

# Single instance (Singleton) to use in routes
predictor_service = RespiratoryPredictor()