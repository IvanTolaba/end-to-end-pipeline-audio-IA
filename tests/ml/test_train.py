#ERROR tests/ml/test_train.py - NameError: name 'pd' is not defined
'''

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from ml.training.train import run_pipeline

@patch('ml.training.train.load_dataset')
@patch('ml.training.train.balance_and_split')
@patch('ml.training.train.build_cnn_bilstm_model')
def test_run_pipeline_execution_flow(mock_build_model, mock_balance_split, mock_load_dataset, tmp_path):
    """
    Test de Integración de Flujo: Verifica que run_pipeline orqueste correctamente
    cada paso del pipeline MLOps invocando los componentes en el orden adecuado
    y persistiendo los artefactos finales.
    """
    # 1. ARRANGE (Preparar el entorno y las respuestas simuladas - Mocks)
    # Simulamos que load_dataset devuelve un DataFrame vacío de juguete
    mock_load_dataset.return_compiler = pd.DataFrame()

    # Simulamos que balance_and_split devuelve matrices dummy muy chicas
    X_dummy = np.random.rand(2, 13, 130, 1).astype("float32")
    y_dummy = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype="float32")
    mock_balance_split.return_value = (X_dummy, y_dummy, X_dummy, y_dummy, X_dummy, y_dummy)

    # Creamos un objeto simulador para el modelo de Keras
    mock_model_instance = MagicMock()
    mock_build_model.return_value = mock_model_instance

    # Forzamos rutas temporales inmutables usando fixtures de pytest para evitar sobreescribir datos reales
    dummy_model_path = tmp_path / "artifacts" / "respiratory_cnn_blstm.keras"
    
    # Reemplazamos las configuraciones globales en caliente durante la ejecución del test
    with patch('ml.training.train.MODEL_PATH', dummy_model_path), \
         patch('ml.training.train.Path.__file__', str(tmp_path / "ml" / "training" / "train.py")):
        
        # Simulamos la creación previa del directorio dummy data
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)

        # 2. ACT
        run_pipeline()

        # 3. ASSERT (Validaciones de comportamiento del Orquestador)
        # Verificamos que se haya llamado a la carga de datos del ETL
        mock_load_dataset.assert_called_once()
        
        # Verificamos que se hayan separado los datos protegiendo de data leakage
        mock_balance_split.assert_called_once()
        
        # Verificamos que la red neuronal híbrida haya sido construida
        mock_build_model.assert_called_once()

        # Verificamos que el método .fit() del modelo se haya ejecutado con los parámetros esperados
        mock_model_instance.fit.assert_called_once()
        
        # Verificamos que al finalizar la optimización se invoque la persistencia binaria del modelo
        mock_model_instance.save.assert_called_once_with(dummy_model_path)

        # Verificamos que los conjuntos de prueba se hayan guardado en formato binario .npy
        assert (tmp_path / "data" / "X_test.npy").exists(), "El pipeline no persistió el conjunto X_test"
        assert (tmp_path / "data" / "y_test.npy").exists(), "El pipeline no persistió el conjunto y_test"


@pytest.mark.slow
@patch('ml.training.train.load_dataset')
@patch('ml.training.train.balance_and_split')
def test_run_pipeline_smoke_test(mock_balance_split, mock_load_dataset, tmp_path):
    """
    Smoke Test (Prueba de Humo Real): Ejecuta el pipeline completo usando la arquitectura
    de red neuronal REAL (build_cnn_bilstm_model), pero forzando el entrenamiento a 1 sola época.
    Esto garantiza que las salidas del preprocessing se acoplen perfectamente a las entradas de la CNN-BiLSTM.
    """
    # 1. ARRANGE
    mock_load_dataset.return_value = pd.DataFrame()
    
    # Creamos micro-matrices sintéticas compatibles con la red real (Batch de 2 muestras)
    X_dummy = np.random.rand(2, 13, 130, 1).astype("float32")
    y_dummy = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype="float32")
    mock_balance_split.return_value = (X_dummy, y_dummy, X_dummy, y_dummy, X_dummy, y_dummy)

    dummy_model_path = tmp_path / "artifacts" / "respiratory_model_smoke.keras"

    # Interceptamos la llamada al método .fit() nativo del modelo para cambiar epochs=250 por epochs=1 en caliente
    from keras.models import Sequential
    original_fit = Sequential.fit

    def short_fit_wrapper(self, *args, **kwargs):
        kwargs['epochs'] = 1  # Forzamos 1 sola época de ejecución ultra veloz
        kwargs['verbose'] = 0 # Apagamos barras de progreso molestas en los logs del test
        return original_fit(self, *args, **kwargs)

    # 2. ACT & ASSERT
    with patch('ml.training.train.MODEL_PATH', dummy_model_path), \
         patch('keras.models.Sequential.fit', short_fit_wrapper):
        
        # Configurar estructura de directorios falsa basada en la raíz simulada
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        
        # Ejecutamos el pipeline real interactuando dinámicamente con TensorFlow/Keras
        run_pipeline()
        
        # Validaciones de caja negra: Comprobamos que el archivo .keras real se generó en disco
        assert dummy_model_path.exists(), "El entrenamiento real falló al empaquetar el binario final"
'''