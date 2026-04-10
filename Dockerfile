FROM python:3.11-slim

# Install system dependencies
# build-essential is added to fix pip install compilation errors
# libgl1 replaces the deprecated libgl1-mesa-glx
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 7860

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "app:app"]