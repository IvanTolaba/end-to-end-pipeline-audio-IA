import logging
import numpy as np
from pathlib import Path
from keras.models import load_model

from config.settings import MODEL_PATH
from ml.training.metrics import generate_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_evaluation():
    logger.info("=== INICIANDO EVALUACIÓN DEL MODELO DE PRODUCCIÓN ===")
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    x_test_path = base_dir / "data" / "X_test.npy"
    y_test_path = base_dir / "data" / "y_test.npy"
    
    if not MODEL_PATH.exists() or not x_test_path.exists():
        logger.error("No se encontraron los artefactos necesarios. Ejecute primero train.py")
        return
        
    # Cargar artefactos limpios de producción
    model = load_model(MODEL_PATH)
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    
    # Realizar inferencias distribuidas de prueba
    y_pred_probs = model.predict(X_test)
    
    # Generar métricas a través del archivo especializado
    metrics = generate_classification_metrics(y_test, y_pred_probs)
    
    print("\n" + "="*60)
    print("📈 REPORTE DE MÉTRICAS CLÍNICAS (TEST SET)")
    print("="*60)
    print("\nMatriz de Confusión:")
    print(metrics["confusion_matrix"])
    print("\nReporte General de Clasificación:")
    print(metrics["classification_report"])
    print("="*60 + "\n")
    
    # Remover los archivos `.npy` temporales de datos para mantener el entorno limpio
    x_test_path.unlink()
    y_test_path.unlink()
    logger.info("Limpieza de almacenamiento local temporario completada.")

if __name__ == "__main__":
    run_evaluation()