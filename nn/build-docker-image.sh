#!/bin/bash

show_usage() {
	echo -e "Usage: $0 NETWORK MODE"
	echo ""
}

if [ $# -lt 2 ]
then
	show_usage
	exit 1
fi

NN=$1
MODE=$2

docker build \
	--build-arg SOURCE_PATH="nn/${NN}/src" \
	--build-arg RUN_SCRIPT="${MODE}.py" \
	-f "${NN}/Dockerfile" \
	-t "${NN}-${MODE}" \
	..
