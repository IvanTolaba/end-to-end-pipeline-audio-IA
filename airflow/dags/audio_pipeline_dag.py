
from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime

from etl.ingest import ingest_data
from etl.mfcc_pyspark import process_mfcc
from etl.save_parquet import save_data

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# -------------------------
# Funciones para Airflow
# -------------------------

def ingest_task(**context):

    data = ingest_data("data/raw")

    context["ti"].xcom_push(key="data", value=data)


def mfcc_task(**context):

    data = context["ti"].xcom_pull(
        key="data",
        task_ids="ingest_audio"
    )

    df = process_mfcc(data)

    context["ti"].xcom_push(key="df", value=df)


def save_task(**context):

    df = context["ti"].xcom_pull(
        key="df",
        task_ids="process_mfcc"
    )

    save_data(df, "data/processed")


# -------------------------
# DAG
# -------------------------

with DAG(

    dag_id="audio_pipeline",

    start_date=datetime(2025, 1, 1),

    schedule="@daily",

    catchup=False

) as dag:

    ingest = PythonOperator(
        task_id="ingest_audio",
        python_callable=ingest_task
    )

    mfcc = PythonOperator(
        task_id="process_mfcc",
        python_callable=mfcc_task
    )

    save = PythonOperator(
        task_id="save_parquet",
        python_callable=save_task
    )

    ingest >> mfcc >> save