# Split Step Counter API — long-running HTTP server image.
# For AWS App Runner, ECS/Fargate, or any container host. Listens on :8080.
#
#   docker build -t split-step-api .
#   docker run -p 8080:8080 split-step-api

FROM python:3.12-slim

# OpenCV (pulled in by mediapipe) needs these shared libraries at import time
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt api/requirements.txt
# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r api/requirements.txt

COPY split_step split_step
COPY api api
COPY models models

ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
