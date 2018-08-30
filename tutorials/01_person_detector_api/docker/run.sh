docker build -t supervisely_detector_api . && \
docker run --rm -it \
			-p 8888:8888 \
			-v `pwd`/../src:/src \
			supervisely_detector_api bash
