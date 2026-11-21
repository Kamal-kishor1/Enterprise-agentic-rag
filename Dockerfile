# -----------------------------------------------------------------------------
# Stage 1: Base - System & Python Dependencies (Cached across targets)
# -----------------------------------------------------------------------------
FROM python:3.15-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Install essential system dependencies for building packages & health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Backend Target (FastAPI + LangGraph + Guardrails)
# -----------------------------------------------------------------------------
FROM base AS backend

# Copy backend application code and local data/fallback files
COPY app/ ./app/
COPY processed_data/ ./processed_data/

EXPOSE 8003

# Run FastAPI with Uvicorn on port 8003
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]

# -----------------------------------------------------------------------------
# Stage 3: Frontend Target (Streamlit UI)
# -----------------------------------------------------------------------------
FROM base AS frontend

# Copy Streamlit UI code and config
COPY ui/ ./ui/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

# Run Streamlit on port 8501 with CORS disabled for Docker networking
CMD ["streamlit", "run", "ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false", "--browser.gatherUsageStats", "false"]