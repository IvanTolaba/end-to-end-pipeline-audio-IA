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
# MFCC Extraction
# -----------------------------------

def extract_mfcc(path: str) -> List[float]:
    """
    Extract MFCC from an audio file.

    Args:
        path: WAV file path.
    Returns:
        List of MFCC flatten coefficients.
    Raises:
        Exception:
            If an error occurs during
            feature extraction.
    """

    try:

        logger.info("Processing audio: %s", path)

        audio_path = Path(path)

        if not audio_path.exists():
            raise FileNotFoundError(f"The file does not exist: {path}")

        # Cargar audio
        y, sr = librosa.load(path,sr=SAMPLE_RATE)

        # Extraer MFCC
        mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=N_MFCC)
        logger.info("MFCC extracted correctly: %s",path)

        # Spark does not serialize numpy arrays
        return mfcc.flatten().tolist()

    except Exception as error:
        logger.exception("Error processing audio %s: %s",path,error)
        return []


# -----------------------------------
# Pipeline Spark
# -----------------------------------

def process_mfcc(spark: SparkSession,data: List[Dict]) -> DataFrame:
    """
    Process audio using PySpark and extract MFCC.

    Args:
        spark: Spark active session.

        data: List of audio recordings.

    Returns: Spark DataFrame with MFCC column.
    """

    logger.info("Starting MFCC pipeline with PySpark")

    if not data:
        logger.warning("Empty dataset received in process_mfcc")

    # Create Spark DataFrame
    df = spark.createDataFrame(data)

    logger.info("Spark DataFrame created successfully")

    # Register UDF
    mfcc_udf = udf(extract_mfcc,ArrayType(FloatType()))
    logger.info("Applying MFCC extraction")

    # Apply transformation
    df = df.withColumn("mfcc",mfcc_udf(df["path"]))
    logger.info("MFCC generated correctly")

    return df



