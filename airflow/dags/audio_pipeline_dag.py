from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# Forzamos los argumentos base para que TODO operador Docker los herede sí o sí
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

    # Tarea 1: Ejecutar la Ingesta y Procesamiento MFCC con PySpark (Contenedor ETL)
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

    # Tarea 2: Entrenamiento del Modelo (Contenedor TensorFlow)
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

    # Tarea 3: Evaluación y Reportes (Contenedor TensorFlow/Eval)
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

    # Flujo de ejecución secuencial
    process_etl >> train_model >> evaluate_model


