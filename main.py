from etl.ingest import ingest_data
from etl.mfcc_pyspark import process_mfcc
from etl.save_parquet import save_data
import time

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from pyspark.sql import SparkSession

def run_pipeline():
    start = time.time()
    #Ingest
    data = ingest_data("data/raw")

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

