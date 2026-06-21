# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# - PYTHONUNBUFFERED so logs stream out of the container immediately
# - PYTHONDONTWRITEBYTECODE keeps the image clean
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code + tests + fixtures (tests run inside the container).
COPY src ./src
COPY tests ./tests
COPY fixtures ./fixtures
COPY pytest.ini ./pytest.ini

EXPOSE 8080

# Healthcheck mirrors the eval's readiness probe.
HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "memory_service.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
