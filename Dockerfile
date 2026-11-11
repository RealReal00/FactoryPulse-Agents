FROM python:3.12.10-slim

WORKDIR /app

COPY requirements-api.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app.py ./
COPY models/random_forest_baseline.joblib models/random_forest_baseline.joblib

RUN useradd --no-create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
