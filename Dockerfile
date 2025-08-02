# Multi-stage Dockerfile for spike-snn-event-vision-kit
# Optimized for development and production use

# Base image with CUDA support
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    cmake \
    build-essential \
    pkg-config \
    libopencv-dev \
    libhdf5-dev \
    libeigen3-dev \
    libboost-all-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and use non-root user
RUN useradd -m -s /bin/bash -u 1000 snnuser
USER snnuser
WORKDIR /home/snnuser

# Set up Python virtual environment
RUN python3 -m venv /home/snnuser/venv
ENV PATH="/home/snnuser/venv/bin:$PATH"

# Upgrade pip and install base Python packages
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Development stage
FROM base as development

# Install development dependencies
COPY --chown=snnuser:snnuser requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional dev tools
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    debugpy \
    pre-commit

# Copy source code
COPY --chown=snnuser:snnuser . /home/snnuser/spike-snn-event-vision-kit
WORKDIR /home/snnuser/spike-snn-event-vision-kit

# Install package in development mode
RUN pip install -e ".[dev,cuda,monitoring]"

# Set up pre-commit hooks
RUN pre-commit install || true

# Expose ports for Jupyter and debugging
EXPOSE 8888 5678

# Default command for development
CMD ["bash"]

# Production stage
FROM base as production

# Copy only requirements and install
COPY --chown=snnuser:snnuser requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source code
COPY --chown=snnuser:snnuser . /home/snnuser/spike-snn-event-vision-kit
WORKDIR /home/snnuser/spike-snn-event-vision-kit

# Install package
RUN pip install --no-cache-dir ".[cuda]"

# Clean up
RUN pip cache purge && \
    rm -rf /tmp/* /var/tmp/*

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import spike_snn_event; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "spike_snn_event.cli", "--help"]

# ROS2 stage (extends production)
FROM production as ros2

# Switch back to root for ROS2 installation
USER root

# Install ROS2 Humble
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y \
    ros-humble-desktop-lite \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-cv-bridge \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Switch back to user
USER snnuser

# Install ROS2 Python dependencies
RUN pip install --no-cache-dir ".[ros2]"

# Source ROS2 setup
RUN echo "source /opt/ros/humble/setup.bash" >> /home/snnuser/.bashrc

# Default command for ROS2
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && ros2 run spike_snn_event snn_detection_node"]

# CPU-only stage (for environments without GPU)
FROM ubuntu:24.04 as cpu-only

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (without CUDA)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    cmake \
    build-essential \
    pkg-config \
    libopencv-dev \
    libhdf5-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user and virtual environment
RUN useradd -m -s /bin/bash -u 1000 snnuser
USER snnuser
WORKDIR /home/snnuser

RUN python3 -m venv /home/snnuser/venv
ENV PATH="/home/snnuser/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install CPU-only PyTorch
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements (excluding CUDA)
COPY --chown=snnuser:snnuser requirements.txt /tmp/requirements.txt
RUN grep -v "cupy" /tmp/requirements.txt > /tmp/requirements-cpu.txt && \
    pip install --no-cache-dir -r /tmp/requirements-cpu.txt

# Copy source and install
COPY --chown=snnuser:snnuser . /home/snnuser/spike-snn-event-vision-kit
WORKDIR /home/snnuser/spike-snn-event-vision-kit

RUN pip install --no-cache-dir .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import spike_snn_event; print('OK')" || exit 1

CMD ["python", "-m", "spike_snn_event.cli", "--help"]