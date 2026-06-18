# CON DOCKEROPERATOR
from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# Configuración por defecto para los contenedores Docker que lanzará Airflow
docker_common_config = {
    "auto_remove": "success",          # Elimina el contenedor al terminar con éxito para no llenar el disco
    "network_mode": "end-to-end-pipeline-audio-ia_default", # Se une a la red del compose
    "docker_url": "unix://var/run/docker.sock",
    "force_pull": False,
    "mounts": [
        Mount(source="/home/tolaba/end-to-end-pipeline-audio-IA/data", target="/app/data", type="bind"),
        Mount(source="/home/tolaba/end-to-end-pipeline-audio-IA/config", target="/app/config", type="bind"),
        Mount(source="/home/tolaba/end-to-end-pipeline-audio-IA/ml", target="/app/ml", type="bind"),
    ],
    "environment": {
        "ENV_FOR_DYNACONF": "development",
        "PYTHONPATH": "/app"
    }
}

with DAG(
    dag_id="audio_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False
) as dag:

    # Tarea 1: Ejecutar la Ingesta y Procesamiento MFCC con PySpark (Contenedor ETL)
    process_etl = DockerOperator(
        task_id="run_etl_pyspark",
        image="audio-pipeline-etl:latest",
        command="python3 etl/ingest.py", # Aquí puedes apuntar al script orquestador de tu ETL
        **docker_common_config
    )

    # Tarea 2: Entrenamiento del Modelo (Contenedor TensorFlow)
    train_model = DockerOperator(
        task_id="run_ml_training",
        image="audio-pipeline-training:latest",
        command="python3 ml/training/train.py",
        **docker_common_config
    )

    # Tarea 3: Evaluación y Reportes (Contenedor TensorFlow/Eval)
    evaluate_model = DockerOperator(
        task_id="run_ml_evaluation",
        image="audio-pipeline-training:latest",
        command="python3 ml/evaluation/evaluate.py",
        **docker_common_config
    )

    # Flujo de ejecución secuencial
    process_etl >> train_model >> evaluate_model