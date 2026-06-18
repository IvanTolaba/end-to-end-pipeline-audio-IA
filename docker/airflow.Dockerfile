FROM apache/airflow:2.10.5-python3.11

USER root

# Instalar dependencias base y el cliente oficial de Docker
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update && apt-get install -y docker-ce-cli \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

WORKDIR /opt/airflow

ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow"

# Instalar las librerías del gestor (incluyendo apache-airflow-providers-docker)
COPY requirements/airflow_requirements.txt .
RUN pip install --no-cache-dir -r airflow_requirements.txt

# Nota: Tus DAGs se montarán directamente por medio de un volumen en tu docker-compose 
# para que cualquier cambio que hagas en `audio_pipeline_dag.py` se actualice en vivo.