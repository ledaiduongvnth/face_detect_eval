FROM nvidia/cuda:11.1.1-base-ubuntu20.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Install dependency
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
RUN apt-get update && apt-get install -y tzdata
RUN apt-get update && apt-get install -y python3-pip python3-opencv
RUN pip3 install -r requirements.txt

# Set the default command to python3
#CMD ["python3 /app/train.py"]
