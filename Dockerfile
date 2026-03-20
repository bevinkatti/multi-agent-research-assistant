FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

RUN mkdir -p /tmp/faiss_index evaluation/results

RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-MiniLM-L6-v2')" || true

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 777 /tmp
USER appuser

EXPOSE 7860

CMD ["python", "app.py"]