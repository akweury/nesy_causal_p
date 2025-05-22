FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /app

# Set timezone (optional but safe)
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && echo "UTC" > /etc/timezone

# Avoid interactive prompts and reduce pip noise
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_NO_PROGRESS_BAR=off

# Upgrade pip + core tools only
RUN python -m pip install --upgrade pip setuptools wheel

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) If you still need OpenCV
RUN pip install --no-cache-dir opencv-python==4.8.0.74