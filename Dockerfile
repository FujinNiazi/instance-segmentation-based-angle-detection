# Use official Python base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Setting time zone (for ffmpeg)
RUN apt-get update && apt-get install -y tzdata
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set work directory
WORKDIR /app

# Install system dependencies 
RUN apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    gcc \
    g++ \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install ninja && \
    pip install -r requirements.txt

# Install detectron2 from GitHub
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copy source code
COPY . .

# Set entrypoint to script
# CMD ["python", "src/main.py"]
