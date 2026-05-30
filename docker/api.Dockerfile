# ==========================================
# 1. OPTIMIZED BASE IMAGE
# ==========================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==========================================
# 2. SYSTEM DEPENDENCIES (For Librosa)
# ==========================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==========================================
# 3. INSTALLATION FROM REQUIREMENTS
# ==========================================

COPY requirements/api_requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# 4. COPYING FROM API MODULES
# ==========================================

COPY api/ ./api
COPY config/ ./config
COPY ml/artifacts/ ./ml/artifacts/

# ==========================================
# 5. EXECUTION CONFIGURATION
# ==========================================
EXPOSE 10000

# dynamically maps the Render port
CMD ["sh", "-c", "uvicorn api.main_api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]