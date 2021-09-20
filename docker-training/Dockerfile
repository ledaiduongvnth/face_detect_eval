ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:21.05-py3
FROM $BASE_IMAGE
ARG BASE_IMAGE
RUN mkdir /app
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements_mmdetection.txt /app/requirements_mmdetection.txt
# You may not use this to install mmcv
RUN pip3 install mmcv-full==1.3.3 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
# Replace by your requirements.txt
RUN pip3 install -r requirements_mmdetection.txt