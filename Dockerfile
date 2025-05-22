# Base image from NVIDIA with PyTorch + CUDA
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /app

# Set timezone to UTC
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && echo "UTC" > /etc/timezone

# Disable APT hook that causes errors in NGC images
RUN rm -f /etc/apt/apt.conf.d/docker-clean || true

# Avoid interactive prompts from apt and pip
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_NO_PROGRESS_BAR=off

# Install system dependencies (safe and minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        mesa-utils \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and core Python packaging tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add optional dev tools (OpenCV, debugger)
RUN pip install --no-cache-dir \
    opencv-python==4.8.0.74 \
    debugpy \
    pydevd-pycharm~=241.14494.241

# Optional: copy source code
# COPY . .

# Optional: run script with debugpy
# CMD ["python3", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "main.py"]