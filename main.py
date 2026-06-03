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

    logger.info("Initializing centralized Apache Spark Session context")
    spark = (
        SparkSession.builder
        .appName("Respiratory-Audio-BioMarker-ETL")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


    try:
        # --------------------------------------------------
        # mfcc_pyspark
        # --------------------------------------------------
        df_spark_processed = process_mfcc(spark=spark, data=raw_metadata_list)

        # --------------------------------------------------
        # save_parquet
        # --------------------------------------------------
        save_data(df=df_spark_processed, output_path=DATA_PROCESSED_DIR)

        logger.info("==================================================")
        logger.info("🎉 ETL PIPELINE EXECUTED SUCCESSFULLY AND TERMINATED CLEANLY")
        logger.info("==================================================")

    except Exception as error:
        logger.exception("Pipeline execution failed unexpectedly due to runtime anomalies: %s", error)

    finally:
        logger.info("Shutting down active Spark Session engine to release allocated memory")
        spark.stop()

    execute_model_training()


if __name__ == "__main__":
    run_pipeline_etl()

'''
#codigo anterior
from etl.ingest import ingest_data
from etl.mfcc_pyspark import process_mfcc
from etl.save_parquet import save_data
import time
from config.settings import DATA_RAW_DIR

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from pyspark.sql import SparkSession

def run_pipeline():
    start = time.time()

    # wav -> List of dictionaries
    data = ingest_data(DATA_RAW_DIR)

    #MFCC procesing
    spark = (
    SparkSession.builder
    .appName("Audio MFCC Pipeline")
    .getOrCreate()
    )
    df = process_mfcc(spark, data)

    #Save parquet
    save_data(df, "data/processed")

    logger.info("Pipeline completed. Records processed: %s",df.count())
    logger.info("Total pipeline execution time: %.2f segundos",time.time() - start)


if __name__ == "__main__":
    run_pipeline()

'''