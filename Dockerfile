FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# -------------------------------
# Fix NVIDIA repo key issue (safe)
# -------------------------------
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && rm -f /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 || true \
    && apt-get update && apt-get install -y --no-install-recommends wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb

# -------------------------------
# Install useful system packages
# -------------------------------
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    tmux \
    nano \
    htop \
    wget \
    curl \
    git \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    ffmpeg \
    libxext6 \
    openssh-server \
    cmake \
    libncurses5-dev \
    libncursesw5-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Nice Bash prompt
# -------------------------------
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' ~/.bashrc

# -------------------------------
# CREATE PYTHON VENV (NO CONDA)
# -------------------------------
RUN python3 -m venv /opt/venv

# Make venv Python the default for all shells
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install
COPY requirements.txt /workspace
RUN pip install --upgrade pip && pip install -r /workspace/requirements.txt

# -------------------------------
# Default command
# -------------------------------
CMD ["/bin/bash"]
