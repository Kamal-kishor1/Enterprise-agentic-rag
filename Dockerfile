# -----------------------------------------------------------------------------
# Stage 1: Base - System & Python Dependencies (Cached across targets)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

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
RUN pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Backend Target (FastAPI + LangGraph + Guardrails)
# -----------------------------------------------------------------------------
FROM base AS backend

# Copy backend application code and local data/fallback files
COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8003

# Run FastAPI with Uvicorn on port 8003
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]

# -----------------------------------------------------------------------------
# Stage 3: Frontend Target (Streamlit UI)
# -----------------------------------------------------------------------------
FROM base AS frontend

# Copy Streamlit UI code
COPY ui/ ./ui/

EXPOSE 8501

# Run Streamlit on port 8501
CMD ["streamlit", "run", "ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--browser.gatherUsageStats", "false"]