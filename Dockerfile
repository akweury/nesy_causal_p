# Base image with PyTorch + CUDA from NVIDIA NGC
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /app

# Set timezone to UTC
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && echo "UTC" > /etc/timezone

# Avoid interactive prompts from apt and pip
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_NO_PROGRESS_BAR=off

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        mesa-utils \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and core Python packaging tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional tools (OpenCV and debugger)
RUN pip install --no-cache-dir \
    opencv-python==4.8.0.74 \
    debugpy \
    pydevd-pycharm~=241.14494.241

# Optional: copy project code (can be mounted instead)
# COPY . .

# Default command (can be overridden by PyCharm run config)
# CMD ["python", "scripts/main.py"]