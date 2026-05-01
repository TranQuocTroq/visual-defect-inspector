FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set environment variables
ENV YOLO_CONFIG_DIR="/tmp/Ultralytics"
ENV FORCE_REBUILD="1"

WORKDIR /home/user/app

COPY --chown=user requirements.txt .

# Force install the specific Gradio version to bypass the JSON bool iterable bug
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade gradio==4.44.1 fastapi==0.111.0

# Copy project files
COPY --chown=user . .

# Hugging Face Spaces listens on 7860
EXPOSE 7860

CMD ["python", "app.py"]