# AE3-TL — HTTP API for table extraction from images
# cpu image: onnxruntime + tesseract, weights fetched at first start
FROM python:3.11-slim

LABEL org.opencontainers.image.title="AE3-TL" \
      org.opencontainers.image.description="Table extraction from images: ONNX table-structure model + Tesseract OCR" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/apexe3/ae3-table-extract" \
      org.opencontainers.image.vendor="Team APEX:E3"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AE3_WEIGHTS_DIR=/data/weights \
    AE3_PROVIDER=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src

RUN pip install --no-cache-dir .

# weights live on a volume so they are downloaded once and reused
RUN mkdir -p /data/weights
VOLUME /data/weights

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=4).raise_for_status()"

CMD ["ae3-tl", "serve", "--host", "0.0.0.0", "--port", "8000"]
