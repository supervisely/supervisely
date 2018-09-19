#!/bin/bash

show_usage() {
	echo -e "Usage: $0 NETWORK ENTRYPOINT"
	echo ""
}

if [ $# -lt 2 ]
then
	show_usage
	exit 1
fi

NN=$1
ENTRYPOINT=$2

./build-docker-image.sh $NN dev

nvidia-docker run \
	--rm \
	-ti \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority \
	-v "$PWD/${NN}/src:/workdir/src" \
	-v "$PWD/${NN}/data:/sly_task_data" \
	--entrypoint=$ENTRYPOINT \
	"${@:3}" \
	"${NN}-dev"