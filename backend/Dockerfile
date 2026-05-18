# ── Stage 1: Builder ──────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

EXPOSE 5000

ENV FLASK_ENV=production \
    PYTHONUNBUFFERED=1

# Use gunicorn with eventlet for production SocketIO support
CMD ["python", "-m", "gunicorn", "--worker-class", "eventlet", \
     "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "server:app"]
