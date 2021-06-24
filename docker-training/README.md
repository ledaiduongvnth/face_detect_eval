### Build docker image
 docker build -t iva-training .
 
### Run docker image with bash
docker run -it --init \
  --gpus=all \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/app" \
  iva-training bash

### Run docker image with default command
docker run -it --init \
  --gpus=all \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/app" \
  iva-training
