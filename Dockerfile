# Dockerfile for local development and HuggingFace Spaces deployment
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (7860 is default for HuggingFace Spaces)
EXPOSE 7860

# Run the Panel app
CMD ["panel", "serve", "code/app.py", "--address", "0.0.0.0", "--port", "7860", "--allow-websocket-origin=*"]
