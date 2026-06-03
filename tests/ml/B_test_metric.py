
## ============================== short test summary info ==============================
#FAILED tests/ml/test_metric.py::test_generate_classification_metrics_math_integrity
#- ValueError: Number of classes, 2, does not match size of target_names, 
# 4. Try sp...

'''
import pytest
import numpy as np
from sklearn.metrics import confusion_matrix
from ml.training.metrics import generate_classification_metrics

def test_generate_classification_metrics_structure():
    """
    Test Unitario: Verifica que la función procese correctamente matrices One-Hot
    y distribuciones de probabilidad Softmax, retornando el diccionario con la estructura esperada.
    """
    # 1. ARRANGE
    # Simulamos valores reales en formato One-Hot (3 muestras, 4 clases)
    # Muestra 0: Clase 0 (Asma) -> [1, 0, 0, 0]
    # Muestra 1: Clase 1 (EPOC)  -> [0, 1, 0, 0]
    # Muestra 2: Clase 3 (Normal)-> [0, 0, 0, 1]
    y_true_mock = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype="float32")

    # Simulamos las predicciones Softmax (probabilidades que suman 1.0 por fila)
    # Muestra 0: Predice Clase 0 con 90% (Acierto)
    # Muestra 1: Predice Clase 2 con 70% (Fallo, era Clase 1)
    # Muestra 2: Predice Clase 3 con 95% (Acierto)
    y_pred_probs_mock = np.array([
        [0.90, 0.05, 0.03, 0.02],
        [0.10, 0.20, 0.70, 0.00],
        [0.01, 0.02, 0.02, 0.95]
    ], dtype="float32")

    # 2. ACT
    metrics_result = generate_classification_metrics(y_true_mock, y_pred_probs_mock)

    # 3. ASSERT
    # Verificamos la existencia de las llaves en el diccionario de salida
    assert isinstance(metrics_result, dict), "El resultado debe ser un diccionario"
    assert "confusion_matrix" in metrics_result, "Falta la llave 'confusion_matrix' en el output"
    assert "classification_report" in metrics_result, "Falta la llave 'classification_report' en el output"

    # Verificar que el reporte de clasificación sea un string formateado listo para imprimir
    assert isinstance(metrics_result["classification_report"], str), "El reporte de clasificación debe ser un String"
    
    # Verificar las dimensiones físicas de la matriz de confusión resultante (debe ser de 4x4 por tus 4 clases)
    cm = metrics_result["confusion_matrix"]
    assert isinstance(cm, np.ndarray), "La matriz de confusión debe ser un array de NumPy"
    assert cm.shape == (4, 4), f"Dimensiones de matriz erróneas. Se esperaba (4,4), se obtuvo {cm.shape}"


def test_generate_classification_metrics_math_integrity():
    """
    Test de Integridad Matemática: Valida que la decodificación interna por argmax
    sea exacta y que la matriz de confusión compute el número correcto de aciertos y fallos.
    """
    # 1. ARRANGE
    # Definimos índices directos para probar la bifurcación 'if len(y_true.shape) > 1' de tu código
    # y_true directo: [Clase 0, Clase 0, Clase 1]
    y_true_flat = np.array([0, 0, 1])
    
    # y_pred_probs: Forzamos a que el modelo elija las clases [0, 1, 1] respectivamente
    y_pred_probs = np.array([
        [0.8, 0.2, 0.0, 0.0],  # argmax -> 0 (Acierto)
        [0.1, 0.9, 0.0, 0.0],  # argmax -> 1 (Fallo, era 0)
        [0.0, 0.6, 0.4, 0.0]   # argmax -> 1 (Acierto)
    ])

    # 2. ACT
    metrics_result = generate_classification_metrics(y_true_flat, y_pred_probs)
    cm = metrics_result["confusion_matrix"]

    # 3. ASSERT
    # Según nuestro diseño:
    # - Para la Clase 0: de 2 muestras, 1 fue clasificada como 0 y la otra como 1.
    # - Para la Clase 1: de 1 muestra, fue clasificada correctamente como 1.
    
    assert cm[0, 0] == 1, "Falló el conteo de Verdaderos Positivos para la clase 0"
    assert cm[0, 1] == 1, "Falló el conteo de Falsos Negativos (Muestra 1 mal clasificada)"
    assert cm[1, 1] == 1, "Falló el conteo de Verdaderos Positivos para la clase 1"
    
    # Comprobar que el reporte de texto contenga los nombres clave de tus patologías
    report_text = metrics_result["classification_report"]
    assert "precision" in report_text.lower(), "El reporte no contiene la métrica fundamental de Precisión"
    assert "f1-score" in report_text.lower(), "El reporte no contiene el indicador F1-score"

'''