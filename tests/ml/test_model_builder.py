import pytest
import numpy as np
from keras.models import Sequential
from ml.training.model_builder import build_cnn_bilstm_model

def test_build_cnn_bilstm_model_structure():
    """
    Test Unitario: Verifica que la arquitectura del modelo se construya,
    tenga las dimensiones correctas en la salida y compile correctamente.
    """
    # 1. ARRANGE
    input_shape = (13, 130, 1)
    num_classes = 4

    # 2. ACT
    model = build_cnn_bilstm_model(input_shape=input_shape, num_classes=num_classes)

    # 3. ASSERT
    # Verificar que herede de la clase Sequential de Keras
    assert isinstance(model, Sequential), "El modelo devuelto debe ser una instancia de Keras Sequential"
    
    # Verificar que esté compilado correctamente buscando el optimizador
    assert model.optimizer is not None, "El modelo debería estar compilado con un optimizador"

    # Verificar que la última capa coincida exactamente con el número de clases (patologías)
    output_shape = model.output_shape  # Retorna algo como (None, 4)
    assert output_shape[-1] == num_classes, f"La capa de salida debería tener tamaño {num_classes}, pero tiene {output_shape[-1]}"


def test_model_inference_shape_with_dummy_data():
    """
    Test de Inferencia Mínima: Simula un lote (batch) de audios dummy procesados
    y verifica que el modelo pueda hacer un forward-pass (predicción) sin romper
    y devolviendo las dimensiones esperadas.
    """
    # 1. ARRANGE
    input_shape = (13, 130, 1)
    num_classes = 4
    batch_size = 2  # Simulamos que ingresan 2 audios en simultáneo al pipeline
    
    model = build_cnn_bilstm_model(input_shape=input_shape, num_classes=num_classes)
    
    # Creamos un tensor de entrada falso usando NumPy con forma (batch, altura, ancho, canales)
    # Ejemplo: (2, 13, 130, 1)
    dummy_input = np.random.rand(batch_size, *input_shape).astype("float32")

    # 2. ACT
    # Evaluamos en caliente
    predictions = model.predict(dummy_input)

    # 3. ASSERT
    # La salida de la predicción debe tener forma (batch_size, num_classes) -> (2, 4)
    assert predictions.shape == (batch_size, num_classes), f"Dimensiones de salida incorrectas: {predictions.shape}"
    
    # Al usar activación 'softmax', la suma de las probabilidades de cada fila debe ser muy cercana a 1.0
    for prob_distribution in predictions:
        assert np.isclose(np.sum(prob_distribution), 1.0, atol=1e-5), "La salida Softmax no suma 1.0"