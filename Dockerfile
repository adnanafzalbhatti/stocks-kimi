# ============================================
# stocks-kimi Trading Agent — Multi-stage Build
# ============================================
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY genai_tools.py genai_agents.py stock_agent_v3.py ./
COPY .env.example .env.example

# Health check (Cloud Run liveness probe)
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default entrypoint
CMD ["python", "stock_agent_v3.py", "live"]
