# With Docker operator

from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# Inheritance of arguments
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1),
    "force_pull": False,              
}

with DAG(
    dag_id="audio_pipeline_dag",
    default_args=default_args,
    schedule="@daily",
    catchup=False
) as dag:

    # Task 1: Run MFCC Ingestion and Processing with PySpark (ETL Container)
    process_etl = DockerOperator(
        task_id="run_etl_pyspark",
        image="audio-pipeline-etl:latest",
        command="python3 etl/ingest.py", 
        auto_remove="success",
        force_pull=False,             
        timeout=300,
        network_mode="end-to-end-pipeline-audio-ia_default",
        docker_url="unix://var/run/docker.sock",
        mounts=[
            Mount(source="/home/tolaba/end-to-end-pipeline-audio-IA/data", target="/app/data", type="bind"),
            Mount(source="/home/tolaba/end-to-end-pipeline-audio-IA/config", target="/app/config", type="bind"),
            Mount(source="/home/tolaba/end-to-end-pipeline-audio-IA/ml", target="/app/ml", type="bind"),
        ],
        environment={
            "ENV_FOR_DYNACONF": "development",
            "PYTHONPATH": "/app"
        }
    )

    # Task 2: Model Training (TensorFlow Container)
    train_model = DockerOperator(
        task_id="run_ml_training",
        image="audio-pipeline-training:latest",
        command="python3 ml/training/train.py",
        auto_remove="success",
        force_pull=False,             
        timeout=300,
        network_mode="end-to-end-pipeline-audio-ia_default",
        docker_url="unix://var/run/docker.sock",
        mounts=[
            Mount(source="/home/tolaba/end-to-end-pipeline-audio-IA/data", target="/app/data", type="bind"),
            Mount(source="/home/tolaba/end-to-end-pipeline-audio-IA/config", target="/app/config", type="bind"),
            Mount(source="/home/tolaba/end-to-end-pipeline-audio-IA/ml", target="/app/ml", type="bind"),
        ],
        environment={
            "ENV_FOR_DYNACONF": "development",
            "PYTHONPATH": "/app"
        }
    )

    # Task 3: Evaluation and Reports (TensorFlow/Eval Container)
    evaluate_model = DockerOperator(
        task_id="run_ml_evaluation",
        image="audio-pipeline-training:latest",
        command="python3 ml/evaluation/evaluate.py",
        auto_remove="success",
        force_pull=False,             
        timeout=300,
        network_mode="end-to-end-pipeline-audio-ia_default",
        docker_url="unix://var/run/docker.sock",
        mounts=[
            Mount(source="/home/tolaba/end-to-end-pipeline-audio-IA/data", target="/app/data", type="bind"),
            Mount(source="/home/tolaba/end-to-end-pipeline-audio-IA/config", target="/app/config", type="bind"),
            Mount(source="/home/tolaba/end-to-end-pipeline-audio-IA/ml", target="/app/ml", type="bind"),
        ],
        environment={
            "ENV_FOR_DYNACONF": "development",
            "PYTHONPATH": "/app"
        }
    )

    # Sequential execution flow
    process_etl >> train_model >> evaluate_model


