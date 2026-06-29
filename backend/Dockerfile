# Use official Python slim image with Debian base (needed for opencv + insightface)
FROM python:3.10-slim

# Install system dependencies required by OpenCV and InsightFace
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the buffalo_l model at build time so it's baked into the image
# This avoids downloading it on every cold start
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l'); \
app.prepare(ctx_id=-1, det_size=(640,640)); \
print('Model downloaded successfully')"

# Copy the rest of the backend code
COPY . .

# Create database directory
RUN mkdir -p database

EXPOSE 5000

CMD ["gunicorn", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]
