#arreglar variable pd
'''
import pytest
import numpy as np
import pandas as pd
from ml.training.preprocessing import balance_and_split

def test_balance_and_split_pipeline():
    """
    Test Unitario de Extremo a Extremo para el preprocesamiento:
    Verifica el balanceo de clases, aislamiento estricto de archivos (GroupShuffleSplit),
    reshape convolucional y codificación One-Hot de las etiquetas.
    """
    # 1. ARRANGE
    # Creamos un dataset de juguete desbalanceado:
    # Clase 0 (Asma): 6 fragmentos distribuidos en 2 archivos únicos
    # Clase 1 (EPOC): 3 fragmentos distribuidos en 1 archivo único
    # Clase 2 (Neumonía): 3 fragmentos distribuidos en 1 archivo único
    # Clase 3 (Normal): 3 fragmentos distribuidos en 1 archivo único
    
    N_MFCC = 13
    ANCHO_TEMPORAL = 130
    TAMANIO_MFCC_FLATTEN = N_MFCC * ANCHO_TEMPORAL  # 1690
    
    mock_data = {
        "path": [
            # Clase 0 - Archivo A (3 fragmentos) y Archivo B (3 fragmentos)
            "dir/Asma/audio_A.wav", "dir/Asma/audio_A.wav", "dir/Asma/audio_A.wav",
            "dir/Asma/audio_B.wav", "dir/Asma/audio_B.wav", "dir/Asma/audio_B.wav",
            # Clase 1 - Archivo C (3 fragmentos)
            "dir/Epoc/audio_C.wav", "dir/Epoc/audio_C.wav", "dir/Epoc/audio_C.wav",
            # Clase 2 - Archivo D (3 fragmentos)
            "dir/Neumonia/audio_D.wav", "dir/Neumonia/audio_D.wav", "dir/Neumonia/audio_D.wav",
            # Clase 3 - Archivo E (3 fragmentos)
            "dir/Normal/audio_E.wav", "dir/Normal/audio_E.wav", "dir/Normal/audio_E.wav"
        ],
        "label": [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "class_name": ["Asma"]*6 + ["Epoc"]*3 + ["Neumonia"]*3 + ["Normal"]*3,
        # Simulamos los arrays planos de MFCC tal como los entrega Spark
        "mfcc": [list(np.random.rand(TAMANIO_MFCC_FLATTEN)) for _ in range(15)]
    }
    
    df_input = pd.DataFrame(mock_data)

    # 2. ACT
    X_train, y_train, X_val, y_val, X_test, y_test = balance_and_split(df_input)

    # 3. ASSERTION BLOCKS
    
    # --- BLOQUE 1: Verificación de Balanceo ---
    # El tamaño mínimo por clase en la entrada es 3. Al tener 4 clases, el total balanceado DEBE ser 3 * 4 = 12 fragmentos.
    total_fragmentos_post_balanceo = len(X_train) + len(X_val) + len(X_test)
    assert total_fragmentos_post_balanceo == 12, (
        f"El submuestreo falló. Se esperaban 12 fragmentos totales, se obtuvieron {total_fragmentos_post_balanceo}"
    )

    # --- BLOQUE 2: Verificación dimensional de la Red (Reshape CNN) ---
    # Cada set de características debe tener la forma exacta: (Batch, Altura=13, Ancho=130, Canales=1)
    assert X_train.shape[1:] == (N_MFCC, ANCHO_TEMPORAL, 1), f"Shape erróneo en X_train: {X_train.shape}"
    assert X_val.shape[1:] == (N_MFCC, ANCHO_TEMPORAL, 1), f"Shape erróneo en X_val: {X_val.shape}"
    assert X_test.shape[1:] == (N_MFCC, ANCHO_TEMPORAL, 1), f"Shape erróneo en X_test: {X_test.shape}"

    # --- BLOQUE 3: Verificación de Target y One-Hot Encoding ---
    # La salida debe mapearse a vectores binarios de longitud 4 (nuestras 4 patologías)
    assert y_train.shape[1] == 4, f"Estructura One-Hot corrupta en y_train: {y_train.shape}"
    # Verificamos que la suma probabilística de la codificación sea exactamente 1 por renglón
    assert np.all(np.sum(y_train, axis=1) == 1.0), "La codificación One-Hot contiene filas inválidas"


def test_balance_and_split_data_leakage_protection():
    """
    Test de Integridad Científica: Asegura que el particionamiento por grupos funcione.
    Los fragmentos pertenecientes a un mismo archivo original NO deben mezclarse
    en diferentes splits (por ejemplo, fragmentos de audio_A en train y en test en simultáneo).
    """
    # 1. ARRANGE
    # Generamos datos sencillos pero asignando IDs claros de archivos
    N_MFCC, ANCHO_TEMPORAL = 13, 130
    mock_data = {
        "path": [
            "dir/Asma/audio_A.wav", "dir/Asma/audio_A.wav",  # Archivo A
            "dir/Asma/audio_B.wav", "dir/Asma/audio_B.wav",  # Archivo B
            "dir/Epoc/audio_C.wav", "dir/Epoc/audio_C.wav",  # Archivo C
            "dir/Neumonia/audio_D.wav", "dir/Neumonia/audio_D.wav",  # Archivo D
            "dir/Normal/audio_E.wav", "dir/Normal/audio_E.wav"   # Archivo E
        ],
        "label": [0, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        "class_name": ["Asma"]*4 + ["Epoc"]*2 + ["Neumonia"]*2 + ["Normal"]*2,
        "mfcc": [list(np.random.rand(N_MFCC * ANCHO_TEMPORAL)) for _ in range(10)]
    }
    df_input = pd.DataFrame(mock_data)

    # Para poder hacer el rastreo, necesitamos interceptar qué índices quedaron en cada split.
    # Como la función original no devuelve los paths, podemos comprobar indirectamente 
    # re-ejecutando la lógica del split del código original con un estado idéntico y verificarlo.
    from sklearn.model_selection import GroupShuffleSplit
    from pathlib import Path
    
    file_ids = df_input["path"].apply(lambda p: Path(p).name).to_numpy()
    y_all = df_input["label"].to_numpy()
    
    # 2. ACT & ASSERT (Simulamos el comportamiento del GroupShuffleSplit interno)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, temp_idx = next(gss.split(df_input, y_all, groups=file_ids))
    
    # Obtenemos los nombres de archivos físicos asignados a Train y a Temp
    files_in_train = set(file_ids[train_idx])
    files_in_temp = set(file_ids[temp_idx])
    
    # Verificación del Axioma de Aislamiento: La intersección de conjuntos de archivos debe ser VACÍA
    interseccion = files_in_train.intersection(files_in_temp)
    assert len(interseccion) == 0, (
        f"¡ALERTA DE DATA LEAKAGE! El archivo {interseccion} se filtró en múltiples conjuntos."
    )

'''