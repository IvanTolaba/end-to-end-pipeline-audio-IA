
import logging
from pathlib import Path
from pyspark.sql import SparkSession

from config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR
from etl.ingest import ingest_data
from etl.mfcc_pyspark import process_mfcc
from etl.save_parquet import save_data

from ml.training.train import execute_model_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def run_pipeline_etl() -> None:
    """
    Main orchestrator function that sets up environment resources, triggers 
    the step-by-step ETL workflow, and handles lifecycle dependencies.
    return etl: parquet, df
            {
                "file_id": seg["file_id"],
                "segment_idx": seg["segment_idx"],
                "class_name": class_name,
                "label": label,
                "mfcc": flattened_mfcc
            }

    """
    logger.info("==================================================")
    logger.info("STARTING AUDIO DATA ENGINEERING ETL PIPELINE")
    logger.info("==================================================")

    # --------------------------------------------------
    # ingest_data
    # --------------------------------------------------
    try:
        raw_metadata_list = ingest_data(base_path=DATA_RAW_DIR)
    except Exception as error:
        logger.error("Pipeline stopped during Step 1 (Ingestion Failure): %s", error)
        return
    
    # Spark
    logger.info("Initializing centralized Apache Spark Session context")
    spark = (
        SparkSession.builder
        .appName("Respiratory-Audio-BioMarker-ETL")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


    try:
        # --------------------------------------------------
        # mfcc_pyspark -> segment
        # --------------------------------------------------
        df_spark_processed = process_mfcc(spark=spark, data=raw_metadata_list)

        # --------------------------------------------------
        # save_parquet
        # --------------------------------------------------
        # "file_id" ! "segment_idx" !"class_name" ! "label" ! "mfcc"
        save_data(df=df_spark_processed, output_path=DATA_PROCESSED_DIR)

        logger.info("==================================================")
        logger.info("🎉 ETL PIPELINE EXECUTED SUCCESSFULLY AND TERMINATED CLEANLY")
        logger.info("==================================================")

    except Exception as error:
        logger.exception("Pipeline execution failed unexpectedly due to runtime anomalies: %s", error)

    finally:
        logger.info("Shutting down active Spark Session engine to release allocated memory")
        spark.stop()

    
if __name__ == "__main__":
    run_pipeline_etl()

    


