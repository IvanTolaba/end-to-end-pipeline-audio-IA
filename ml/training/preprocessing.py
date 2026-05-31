import logging
import numpy as np
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import resample
from keras.utils import to_categorical
from config.settings import N_MFCC, DISEASE

logger = logging.getLogger(__name__)

def balance_and_split(df: pd.DataFrame):
    """
    Decodifica arrays, balancea clases mediante submuestreo y divide en train/val/test
    aislando los archivos originales mediante GroupShuffleSplit.
    """
    logger.info("Iniciando fase de preprocesamiento, balanceo y segmentación...")
    
    # 1. Decodificar las listas que persistió Spark a matrices de NumPy
    X_all = np.array(df["mfcc"].tolist()).astype("float32")
    y_all = df["label"].to_numpy()
    file_ids = df["path"].apply(lambda p: Path(p).name).to_numpy()

    # 2. Balanceo de Clases (Submuestreo aleatorio controlado)
    conteo_inicial = Counter(y_all)
    min_count = min(conteo_inicial.values())
    logger.info(f"Conteo inicial por clase: {conteo_inicial}. Target por clase: {min_count}")
    
    X_bal, y_bal, file_ids_bal = [], [], []
    for clase_id in range(len(DISEASE)):
        indices = [i for i, y in enumerate(y_all) if y == clase_id]
        if len(indices) < min_count:
            continue
        indices_sample = resample(indices, n_samples=min_count, random_state=42, replace=False)
        for i in indices_sample:
            X_bal.append(X_all[i])
            y_bal.append(y_all[i])
            file_ids_bal.append(file_ids[i])
            
    X_bal = np.array(X_bal)
    y_bal = np.array(y_bal)
    file_ids_bal = np.array(file_ids_bal)

    # 3. División Train (70%) / Temp (30%) cuidando no mezclar pacientes/archivos
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, temp_idx = next(gss.split(X_bal, y_bal, groups=file_ids_bal))
    
    # División Temp -> Val (15%) / Test (15%)
    #para que fragmentos del mismo audio no aparezcan en diferentes grupos
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(X_bal[temp_idx], y_bal[temp_idx], groups=file_ids_bal[temp_idx]))
    
    # 4. Formateo Final y Reshape para capas Convolucionales 2D (Altura, Ancho, Canales)
    # Asumimos que el ancho temporal resultante fijo de tu pipeline es 130 frames
    # covertir a (13,130,1)
    X_train = X_bal[train_idx].reshape(-1, N_MFCC, 130, 1)
    X_val = X_bal[temp_idx][val_idx].reshape(-1, N_MFCC, 130, 1)
    X_test = X_bal[temp_idx][test_idx].reshape(-1, N_MFCC, 130, 1)
    
    # One-Hot Encoding de variables categóricas de salida
    num_classes = len(DISEASE)
    y_train = to_categorical(y_bal[train_idx], num_classes=num_classes)
    y_val = to_categorical(y_bal[temp_idx][val_idx], num_classes=num_classes)
    y_test = to_categorical(y_bal[temp_idx][test_idx], num_classes=num_classes)
    
    logger.info("Preprocesamiento finalizado con éxito.")
    return X_train, y_train, X_val, y_val, X_test, y_test