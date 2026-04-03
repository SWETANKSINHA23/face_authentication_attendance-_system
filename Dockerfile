# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV STREAMLIT_SERVER_PORT 10000
ENV STREAMLIT_SERVER_HEADLESS true
ENV STREAMLIT_SERVER_ADDRESS 0.0.0.0

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    libboost-all-dev \
    libjpeg-dev \
    libpng-dev \
    libgomp1 \
    ffmpeg \
    libfontconfig1 \
    libice6 \
    libxmu6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port Streamlit will run on
EXPOSE 10000

# Command to run the application
CMD ["streamlit", "run", "app.py"]
