# Use the official NVIDIA PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set the working directory inside the container
WORKDIR /app
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
# Ensure SSH key has correct permissions (if using SSH cloning)
#ADD .ssh/ /root/.ssh/

#RUN chmod 600 /root/.ssh/id_ed25519 && ssh-keyscan github.com >> /root/.ssh/known_hosts


#ARG GITHUB_TOKEN
RUN apt update && apt install -y git
RUN #git clone https://$GITHUB_TOKEN@github.com/akweury/nesy_causal_p.git /app

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel
# Install Python dependencies with --no-cache-dir
WORKDIR /app

RUN pip install opencv-python==4.8.0.74
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install debugpy
RUN pip install pydevd-pycharm~=241.14494.241
#CMD ["python3", "-m", "debugpy", "--wait-for-client", "--listen", "0.0.0.0:5678", "play.py"]

# Set the default command for training (adjust as needed)
#CMD ["python", "scripts/main.py"]

