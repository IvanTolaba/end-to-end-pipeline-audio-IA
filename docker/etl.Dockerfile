FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar Java 17 (Esencial para PySpark) y libsndfile1/ffmpeg (Para Librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    libsndfile1 \
    ffmpeg \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configurar variables de entorno de Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark

WORKDIR /app

# Copiar e instalar dependencias específicas de ETL
COPY requirements/etl_requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r etl_requirements.txt

# Copiar el código del pipeline y las configuraciones base
COPY etl/ /app/etl/
COPY config/ /app/config/

# Comando por defecto para correr tu secuencia ETL (Ajustable desde Airflow si es necesario)
CMD ["python3", "etl/ingest.py"]