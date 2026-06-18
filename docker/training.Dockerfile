FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias del sistema necesarias para Matplotlib, Seaborn y compilación de extensiones de ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requerimientos e instalar
COPY requirements/training_requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r training_requirements.txt

# Copiar código del módulo de ML, datos estructurados y archivos de configuración
COPY ml/ /app/ml/
COPY config/ /app/config/

CMD ["python3", "ml/training/train.py"]