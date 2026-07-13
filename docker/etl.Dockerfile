FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Java 17 (Essential for PySpark) and libsndfile1/ffmpeg (For Librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    libsndfile1 \
    ffmpeg \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure Spark environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark

WORKDIR /app

# Copy and install specific ETL dependencies
COPY requirements/etl_requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r etl_requirements.txt

# Copy the pipeline code and base configurations
COPY etl/ /app/etl/
COPY config/ /app/config/

# Default command to run the ETL sequence
CMD ["python3", "etl/ingest.py"]