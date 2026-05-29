
# ==========================================
# 1. IMAGEN BASE OPTIMIZADA
# ==========================================
# Usamos una versión slim de Python para reducir el tamaño de la imagen y el consumo de RAM.
FROM python:3.10-slim

# Evita que Python escriba archivos .pyc en el contenedor y asegura que los logs salgan directo a Render
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==========================================
# 2. DEPENDENCIAS DEL SISTEMA (Para Librosa)
# ==========================================
# Librosa necesita compilar y leer audio mediante bibliotecas nativas de Linux (C++).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Setear el directorio de trabajo donde vivirá la app dentro del contenedor
WORKDIR /app

# ==========================================
# 3. INSTALACIÓN DE DEPENDENCIAS PYTHON
# ==========================================
# Copiamos tu archivo de requerimientos exclusivo de producción con el nuevo nombre
COPY requirements-prod.txt ./requirements.txt

# Instalamos las librerías oficiales de tu API sin guardar la caché (ahorra espacio)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# 4. COPIADO QUIRÚRGICO DE ARCHIVOS
# ==========================================
# En lugar de copiar todo el monorepo (ignoramos Airflow, dags, notebooks y data),
# copiamos únicamente las piezas esenciales para que FastAPI e inferencia funcionen.
COPY api/ ./api
COPY config/ ./config
COPY model/ ./model

# ==========================================
# 5. CONFIGURACIÓN DE EJECUCIÓN
# ==========================================
# Render asigna el puerto dinámicamente mediante la variable de entorno $PORT. 
# Exponemos el puerto estándar por documentación, pero el comando final toma el de Render.
EXPOSE 10000

# Comando de arranque usando tu nueva estructura modular (api/main.py)
# Usamos --workers 1 para controlar estrictamente la RAM en el plan gratuito/básico de Render.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
