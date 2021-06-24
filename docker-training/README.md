### Install nvidia-container-runtime
https://github.com/NVIDIA/nvidia-container-runtime

### Build docker image
```shell
 docker build -t iva-training .
```
 
### Run docker image with bash
docker run -it --init \
  --gpus=all \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/app" \
  iva-training bash

### Run docker image with default command
mount with current host dir
```shell
docker run -it \
  -d \
  --name=iva-training-container \
  --init \
  --gpus=all \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/app" \
  iva-training python3 train.py
```

### Get log from container 
```shell
docker logs iva-training-container
```