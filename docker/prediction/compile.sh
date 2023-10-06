docker image rm prediction
docker build --no-cache -t prediction ./docker/prediction/

# docker run -v ./:/prediction/ prediction