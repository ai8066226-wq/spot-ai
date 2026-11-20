FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m engine && mkdir -p /app/state && chown -R engine:engine /app
USER engine
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
EXPOSE 8080
CMD ["sh","-c","exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
