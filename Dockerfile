FROM nvcr.io/nvidia/pytorch:23.10-py3
WORKDIR /app
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && echo "UTC" > /etc/timezone
# Avoid interactive prompts and disable pip UI that causes thread errors
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_NO_PROGRESS_BAR=off
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir opencv-python==4.8.0.74