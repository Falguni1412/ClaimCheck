FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p /tmp/huggingface && \
    useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:0 /app /tmp/huggingface && \
    chmod -R g+rwX /app /tmp/huggingface

USER appuser

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
