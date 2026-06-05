import logging
import numpy as np
import librosa
from typing import List, Dict
from config.settings import SAMPLE_RATE, N_MFCC
from etl.segment import segment_audio_file

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, IntegerType

logger = logging.getLogger(__name__)


def process_audio_to_segments_mfcc(path: str, class_name: str, label: int) -> List[Dict]:
    """
    Worker function executed inside the cluster. Slices audio and 
    extracts raw MFCC arrays for each generated segment.
    return:
            {
                "file_id": seg["file_id"],
                "segment_idx": seg["segment_idx"],
                "class_name": class_name,
                "label": label,
                "mfcc": flattened_mfcc
            }
    """
    # 1. Slide window across full signal
    raw_segments = segment_audio_file(audio_path=path, disease_name=class_name)
    
    processed_records = []
    for seg in raw_segments:
        try:
            signal_np = np.array(seg["signal"], dtype=np.float32)
            
            # Extract basic MFCC features (transposed for consistency)
            mfcc_feat = librosa.feature.mfcc(y=signal_np, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
            
            # Flatten to map directly inside Spark Array standard type
            flattened_mfcc = mfcc_feat.flatten().tolist()
            
            processed_records.append({
                "file_id": seg["file_id"],
                "segment_idx": seg["segment_idx"],
                "class_name": class_name,
                "label": label,
                "mfcc": flattened_mfcc
            })
        except Exception as error:
            logger.error("Failed to compute MFCC on segment %s of %s: %s", seg["segment_idx"], path, error)
            continue
            
    return processed_records


def process_mfcc(spark: SparkSession, data: List[Dict]) -> DataFrame:
    """
    Process raw ingested tracks distributedly using Apache Spark RDD transformations.
    
    Args:
        spark: Active context SparkSession.
        data: Initial ingested metadata structure.
                {
                    "path": str(file),
                    "label": label,
                    "class_name": disease
                }
    Returns:
        Spark DataFrame with partitioned acoustic window structures.
    """
    logger.info("Executing parallelized fragmentation and feature extraction with PySpark")

    if not data:
        logger.warning("Empty dataset array received in process_mfcc block")
        return spark.createDataFrame([], StructType([]))

    # Parallelize input metadata into an RDD to distribute processing
    input_rdd = spark.sparkContext.parallelize(data)

    # Map each file tracking information to its corresponding sub-segmented dictionary list
    flat_mapped_rdd = input_rdd.flatMap(
        lambda row: process_audio_to_segments_mfcc(
            path=row["path"], 
            class_name=row["class_name"], 
            label=row["label"]
        )
    )

    # Programmatic schema definition for optimal Spark execution plan
    spark_schema = StructType([
        StructField("file_id", StringType(), False),
        StructField("segment_idx", IntegerType(), False),
        StructField("class_name", StringType(), False),
        StructField("label", IntegerType(), False),
        StructField("mfcc", ArrayType(FloatType()), False)
    ])

    # Instantiate final structural DataFrame
    # "file_id" ! "segment_idx" !"class_name" ! "label" ! "mfcc"
    df_structured = spark.createDataFrame(flat_mapped_rdd, schema=spark_schema)
    logger.info("Distributed features and temporal identifiers generated correctly")

    return df_structured