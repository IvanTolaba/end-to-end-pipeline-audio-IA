# WITH PYTHON OPERATOR

import sys
from datetime import datetime
import logging
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# 1. Define the absolute root dynamically 
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

# 2. Modify the path BEFORE any local import
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 3. Imports of Data (ETL) and Machine Learning (ML) components
from config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR
from etl.ingest import ingest_data
from etl.mfcc_pyspark import process_mfcc
from etl.save_parquet import save_data
from pyspark.sql import SparkSession

from ml.training.train import execute_model_training
from ml.evaluation.evaluate import run_standalone_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# 1. Ingest (Save metadata in XCom)
# -------------------------
def ingest_task(**context):
    logger.info("Starting Ingestion...")
    data = ingest_data(DATA_RAW_DIR) 
    context["ti"].xcom_push(key="raw_data_list", value=data)

# -------------------------
# 2. Spark MFCC + Save
# -------------------------
def spark_processing_task(**context):
    data = context["ti"].xcom_pull(key="raw_data_list", task_ids="ingest_audio")
    
    if not data:
        raise ValueError("No data received from ingest_audio task.")

    logger.info("Initializing Spark Session inside Airflow Worker")
    spark = (
        SparkSession.builder
        .appName("Airflow-Respiratory-Audio-ETL")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    try:
        logger.info("Processing MFCCs with PySpark...")
        df_spark_processed = process_mfcc(spark=spark, data=data)

        logger.info("Saving processed data to Parquet...")
        save_data(df=df_spark_processed, output_path=DATA_PROCESSED_DIR)
        
    finally:
        logger.info("Shutting down Spark Session")
        spark.stop()

# -------------------------
# 3. ML Training Task
# -------------------------
def ml_training_task(**context):
    logger.info("Starting Deep Learning Model Training (CNN_BLSTM)...")
    # Tu script train.py leerá internamente desde DATA_PROCESSED_DIR usando settings.py
    execute_model_training()
    logger.info("Model trained and artifacts saved to ml/artifacts/ successfully.")

# -------------------------
# 4. ML Evaluation Task
# -------------------------
def ml_evaluation_task(**context):
    logger.info("Starting Model Evaluation and Report Generation...")
    # Tu script evaluate.py generará las matrices, curvas ROC y reportes en ml/reports/
    run_standalone_evaluation()
    logger.info("Evaluation finished. Reports updated in ml/reports/.")

# -------------------------
# DAG Definition
# -------------------------
with DAG(
    dag_id="audio_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False
) as dag:

    ingest = PythonOperator(
        task_id="ingest_audio",
        python_callable=ingest_task
    )

    process_and_save = PythonOperator(
        task_id="process_spark_and_save",
        python_callable=spark_processing_task
    )

    # Train the model
    train_ml = PythonOperator(
        task_id="train_ml_model",
        python_callable=ml_training_task
    )

    # Evaluate the trained model
    evaluate_ml = PythonOperator(
        task_id="evaluate_ml_model",
        python_callable=ml_evaluation_task
    )

    # Sequential End-to-End Flow
    ingest >> process_and_save >> train_ml >> evaluate_ml