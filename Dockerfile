# Use light Python image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV STREAMLIT_SERVER_PORT 10000
ENV STREAMLIT_SERVER_HEADLESS true
ENV STREAMLIT_SERVER_ADDRESS 0.0.0.0

# Install ONLY minimal system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 10000

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:10000/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py"]
