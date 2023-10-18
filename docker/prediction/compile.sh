docker image rm prediction
docker build --no-cache -t prediction ./docker/prediction/

# docker run -v ./:/prediction/ prediction
# docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video --group-add render -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY amd-opencl

# docker run --device=/dev/kfd --device=/dev/dri --group-add video -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v ./:/prediction/ prediction