import logging
import numpy as np
from pathlib import Path
#from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config.settings import MODEL_PATH, DISEASE, N_MFCC
from ml.training.dataset import load_dataset
from ml.training.preprocessing import balance_and_split
from ml.training.model_builder import build_cnn_bilstm_model

# Configuración global del motor de logs para el proceso de entrenamiento
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("=== INICIANDO PIPELINE DE ENTRENAMIENTO MLOPS ===")
    
    # Inicialización dinámica de rutas locales basadas en la raíz del proyecto
    base_dir = Path(__file__).resolve().parent.parent.parent
    parquet_dir = base_dir / "data" / "processed"
    
    # 1. Carga de datos distribuidos desde almacenamiento columnar
    #transforma parquet en un dataframe
    df = load_dataset(parquet_dir)
    
    # 2. Ingeniería de características y splits robustos en memoria
    #divide los datos en train,test y val,balancea,cada uno 13*130 
    X_train, y_train, X_val, y_val, X_test, y_test = balance_and_split(df)
    
    # 3. Construcción modular de la arquitectura
    # Compila modelo
    model = build_cnn_bilstm_model(input_shape=(N_MFCC, 130, 1), num_classes=len(DISEASE))
    
    # 4. Estrategia de regularización dinámica durante el entrenamiento (Callbacks)
    #da valores a los callback
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    
    logger.info("Comenzando el ajuste de pesos del modelo (Model Fit)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=250,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # 5. Persistencia del artefacto binario para servir en producción
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    logger.info(f"🎉 Éxito: Artefacto guardado correctamente en: {MODEL_PATH}")
    
    # Guardado seguro y aislado de los conjuntos de pruebas para el script evaluador externo
    np.save(base_dir / "data" / "X_test.npy", X_test)
    np.save(base_dir / "data" / "y_test.npy", y_test)
    logger.info("Conjuntos de testeo persistidos temporalmente para verificación.")

if __name__ == "__main__":
    run_pipeline()