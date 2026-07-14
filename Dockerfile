# multi-stage: build a venv with the CPU torch wheels, then ship a slim runtime
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch/torchvision come from the CPU index to avoid pulling CUDA
RUN pip install --upgrade pip \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install fastapi "uvicorn[standard]" python-multipart pillow numpy

# ---- runtime ----
FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH" \
    MODEL_KIND=resnet_ft \
    MODEL_DIR=/app/models
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ ./src/
COPY models/ ./models/
ENV PYTHONPATH=/app/src

# drop privileges
RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "galaxy_cnn.serve:app", "--host", "0.0.0.0", "--port", "8000"]
