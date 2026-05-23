from pathlib import Path
from typing import List, Dict
from config.settings import (SAMPLE_RATE,N_MFCC)

import logging
import librosa

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType

logger = logging.getLogger(__name__)

# -----------------------------------
# Extracción MFCC
# -----------------------------------

def extract_mfcc(path: str) -> List[float]:
    """
    Extrae MFCC de un archivo de audio.

    Args:
        path: Ruta del archivo WAV.
    Returns:
        Lista de coeficientes MFCC flatten.
    Raises:
        Exception:
            Si ocurre un error durante
            la extracción de características.
    """

    try:

        logger.info("Procesando audio: %s", path)

        audio_path = Path(path)

        if not audio_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")

        # Cargar audio
        y, sr = librosa.load(path,sr=SAMPLE_RATE)

        # Extraer MFCC
        mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=N_MFCC)
        logger.info("MFCC extraído correctamente: %s",path)

        # Spark no serializa numpy arrays
        return mfcc.flatten().tolist()

    except Exception as error:
        logger.exception("Error procesando audio %s: %s",path,error)
        return []


# -----------------------------------
# Pipeline Spark
# -----------------------------------

def process_mfcc(spark: SparkSession,data: List[Dict]) -> DataFrame:
    """
    Procesa audios usando PySpark y extrae MFCC.

    Args:
        spark:Sesión activa de Spark.

        data: Lista de registros de audio.

    Returns: DataFrame Spark con columna MFCC.
    """

    logger.info("Iniciando pipeline MFCC con PySpark")

    if not data:
        logger.warning("Dataset vacío recibido en process_mfcc")

    # Crear DataFrame Spark
    df = spark.createDataFrame(data)

    logger.info("DataFrame Spark creado correctamente")

    # Registrar UDF
    mfcc_udf = udf(extract_mfcc,ArrayType(FloatType()))
    logger.info("Aplicando extracción MFCC")

    # Aplicar transformación
    df = df.withColumn("mfcc",mfcc_udf(df["path"]))
    logger.info("MFCC generados correctamente")

    return df



