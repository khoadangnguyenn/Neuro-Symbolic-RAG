FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EXACT_SERVER_BACKEND=flask \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/exact_pipeline

EXPOSE 8000

CMD ["python", "-m", "exact_pipeline.api.cli", "--serve", "--host", "0.0.0.0", "--port", "8000"]
