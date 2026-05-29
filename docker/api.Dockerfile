# ==========================================
# 1. IMAGEN BASE OPTIMIZADA
# ==========================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==========================================
# 2. DEPENDENCIAS DEL SISTEMA (Para Librosa)
# ==========================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==========================================
# 3. INSTALACIÓN DESDE LA NUEVA CARPETA
# ==========================================
# 🔹 CAMBIO AQUÍ: Vamos a buscar el archivo específico a la carpeta requirements/
# pero lo copiamos dentro del contenedor simplemente como 'requirements.txt'
COPY requirements/api_requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# 4. COPIADO DE MÓDULOS DE LA API
# ==========================================
# Como el contexto de Docker en Render sigue siendo la raíz del monorepo, 
# estas rutas se mantienen igual:
COPY api/ ./api
COPY config/ ./config
#COPY model/ ./model
COPY ml/artifacts/ ./ml/artifacts/

# ==========================================
# 5. CONFIGURACIÓN DE EJECUCIÓN
# ==========================================
EXPOSE 10000

# El comando final mapea dinámicamente el puerto de Render
CMD ["sh", "-c", "uvicorn api.main_api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]