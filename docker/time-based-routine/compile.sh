docker image rm time-based-routine
docker build --no-cache -t time-based-routine ./docker/time-based-routine/

# docker run -v ./:/prediction/ time-based-routine