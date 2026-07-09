FROM python:3.11-slim

WORKDIR /app

# System dependencies for Open3D/OpenCV.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY arkit_sensor_fusion ./arkit_sensor_fusion
COPY service ./service
COPY evaluation ./evaluation
COPY profiles ./profiles
COPY tests ./tests
COPY arkit_reconstruct.py .
COPY upload_server.py .
COPY pyproject.toml .
COPY requirements.txt .
COPY README.md .
COPY MYHUT_BACKEND_README.md .

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .[service]

EXPOSE 8000 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3).read()" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "300", "service.wsgi:app"]
