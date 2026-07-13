FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install production API dependencies
COPY requirements/api_requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r api_requirements.txt

# Copy the API structure, configurations, and trained model artifact
COPY api/ /app/api/
COPY config/ /app/config/

# Production model
COPY ml/artifacts/ /app/ml/artifacts/

EXPOSE 8000

# Point to the main_api.py file and the FastAPI instance
CMD ["uvicorn", "api.main_api:app", "--host", "0.0.0.0", "--port", "8000"]