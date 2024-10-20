# Start from an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install FFmpeg and other dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    mesa-utils \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy any additional files if needed (e.g., scripts, requirements.txt)
COPY . .

# Install additional Python dependencies if required
RUN pip install -r requirements.txt

# Command to run by default when the container starts
#CMD ["bash"]
ENTRYPOINT [ "python", "demo_video.py"]