FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias de producción de la API
COPY requirements/api_requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r api_requirements.txt

# Copiar la estructura de la API, las configuraciones y el artefacto del modelo entrenado
COPY api/ /app/api/
COPY config/ /app/config/


EXPOSE 8000

# Apuntamos exactamente al archivo main_api.py y la instancia de tu app FastAPI
CMD ["uvicorn", "api.main_api:app", "--host", "0.0.0.0", "--port", "8000"]