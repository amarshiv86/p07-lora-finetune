# Local development only — HF Space uses Gradio SDK, not Docker
# Requires NVIDIA GPU + CUDA for training
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

# Training
CMD ["python", "src/train.py"]
