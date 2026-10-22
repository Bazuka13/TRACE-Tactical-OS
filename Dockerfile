# Esports Tactical OS - Production API Container
FROM python:3.11-slim

# libgomp1 is required by LightGBM/CatBoost OpenMP runtimes on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY backend backend
COPY data/maps data/maps
COPY models models

EXPOSE 8000

CMD ["uvicorn", "server:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
