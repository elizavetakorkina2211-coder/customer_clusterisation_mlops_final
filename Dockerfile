FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md params.yaml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY src/segmentation_mlops/api/templates /usr/local/lib/python3.11/site-packages/segmentation_mlops/api/templates

COPY dvc.yaml ./
COPY models ./models
COPY data ./data
COPY reports ./reports
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["uvicorn", "segmentation_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
