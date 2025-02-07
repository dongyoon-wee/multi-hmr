FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install FFmpeg, Mesa, and LLVM
RUN apt-get update && apt-get install -y \
    ffmpeg \
    mesa-utils \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libgl1-mesa-glx \
    libosmesa6 \
    llvm-12 \
    llvm-12-dev \
    clang-12 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable to force software rendering
ENV LIBGL_ALWAYS_SOFTWARE=1

# Set the LLVM paths (replace if you install a different version)
ENV LLVM_DIR=/usr/lib/llvm-12
ENV PATH=$LLVM_DIR/bin:$PATH
ENV LD_LIBRARY_PATH=$LLVM_DIR/lib:$LD_LIBRARY_PATH

# Set the working directory
WORKDIR /workspace

# Copy any additional files if needed (e.g., scripts, requirements.txt)
COPY . .

# Install additional Python dependencies if required
RUN pip install -r requirements.txt

# Command to run by default when the container starts
#CMD ["bash"]
ENTRYPOINT [ "python", "demo_video.py"]