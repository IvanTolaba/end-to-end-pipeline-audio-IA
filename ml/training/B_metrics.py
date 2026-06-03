

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from config.settings import DISEASE

def generate_classification_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> dict:
    """
    Calcula y muestra por pantalla el reporte estadístico completo y la matriz de confusión.
    """
    # Convertir distribuciones probabilísticas u One-Hot a índices numéricos discretos
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    report = classification_report(y_true_classes, y_pred_classes, target_names=DISEASE)
    
    return {
        "confusion_matrix": cm,
        "classification_report": report
    }

