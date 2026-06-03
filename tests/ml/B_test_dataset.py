# Trabaja con datos reales
import pytest
import pandas as pd
from pathlib import Path
from ml.training.dataset import load_dataset

# 📌 Configuramos la ruta real al Parquet del proyecto de forma dinámica
BASE_DIR = Path(__file__).resolve().parent.parent.parent
REAL_PARQUET_PATH = BASE_DIR / "data" / "processed"

def test_load_dataset_with_real_data():
    """
    Test de Integración: Verifica que la función pueda leer el formato Parquet
    real generado por el pipeline de PySpark en el almacenamiento local.
    """
    # 1. ARRANGE
    # Validamos primero si creaste el parquet real, para dar un mensaje claro si falla
    if not REAL_PARQUET_PATH.exists():
        pytest.skip(
            f"Saltando test: No se encontró el Parquet real en {REAL_PARQUET_PATH}. "
            "Asegurate de correr primero el pipeline de PySpark (save_parquet.py)."
        )

    # 2. ACT
    df_result = load_dataset(REAL_PARQUET_PATH)

    # 3. ASSERT (Validaciones estructurales sobre tus datos reales)
    assert isinstance(df_result, pd.DataFrame), "El resultado debe ser un DataFrame de Pandas."
    assert not df_result.empty, "El Parquet real se cargó pero está completamente vacío."
    
    # Verificamos que las columnas generadas por tu ingest.py y mfcc_pyspark.py estén presentes
    columnas_esperadas = ["path", "label", "class_name", "mfcc"]
    for col in columnas_esperadas:
        assert col in df_result.columns, f"Falta la columna crítica '{col}' en el Parquet real."
    
    # Validación extra de nivel Senior: Verificar que los MFCC no se hayan roto
    first_mfcc = df_result["mfcc"].iloc[0]
    assert len(first_mfcc) > 0, "El vector de características MFCC está vacío o mal decodificado."
    logger_msg = f"✓ Validación exitosa de datos reales. Registros encontrados: {len(df_result)}"
    print(f"\n{logger_msg}")


def test_load_dataset_file_not_found():
    """
    Test Unitario: Verifica que la función responda correctamente con un
    FileNotFoundError si la ruta no existe en el sistema.
    """
    # Usamos una ruta ficticia que sabemos con certeza que no existe
    ruta_falsa = BASE_DIR / "data" / "ruta_inexistente_para_forzar_error"
    
    # Validamos que el bloque levante la excepción esperada
    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset(ruta_falsa)
        
    assert "El directorio Parquet no existe en la ruta" in str(exc_info.value)

'''
# Trabaja con datos ficticios

import pytest
import pandas as pd
from pathlib import Path
from ml.training.dataset import load_dataset

def test_load_dataset_success(tmp_path):
    """
    Testea que la función cargue correctamente un archivo Parquet válido
    y retorne un DataFrame de Pandas con la estructura esperada.
    """
    # 1. ARRANGE (Preparar el escenario temporal)
    # Creamos un DataFrame de juguete con datos simulados de tu ETL
    mock_data = {
        "path": ["/data/raw/Asma/audio1.wav", "/data/raw/Normal/audio2.wav"],
        "label": [0, 3],
        "class_name": ["Asma", "Normal"],
        # Simulamos un vector MFCC plano (flatten) de ejemplo
        "mfcc": [[0.1] * 1690, [0.2] * 1690]  # 13 * 130 = 1690
    }
    df_original = pd.DataFrame(mock_data)
    
    # Definimos una ruta temporal para guardar el Parquet simulado
    parquet_dir = tmp_path / "processed_test.parquet"
    df_original.to_parquet(parquet_dir, index=False)

    # 2. ACT (Ejecutar la función que estamos testeando)
    df_result = load_dataset(parquet_dir)

    # 3. ASSERT (Verificar que todo salió como esperábamos)
    assert isinstance(df_result, pd.DataFrame), "El resultado debe ser un DataFrame de Pandas"
    assert len(df_result) == 2, "El DataFrame debería tener exactamente 2 registros"
    assert list(df_result.columns) == ["path", "label", "class_name", "mfcc"], "Las columnas no coinciden"
    assert df_result["label"].iloc[0] == 0, "Los datos internos sufrieron alteraciones al cargar"


def test_load_dataset_file_not_found():
    """
    Testea que la función lance un FileNotFoundError de forma correcta
    si se le pasa una ruta que no existe en el sistema.
    """
    # Usamos una ruta ficticia que sabemos con certeza que no existe
    ruta_falsa = Path("/ruta/completamente/inexistente/data.parquet")
    
    # Validamos que el bloque levante la excepción esperada
    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset(ruta_falsa)
        
    # Opcional: Verificar que el mensaje de error contenga información útil
    assert "El directorio Parquet no existe en la ruta" in str(exc_info.value)

'''