docker build -t 02_unet_inference . && \
docker run --rm -it --init \
	--runtime=nvidia \
	-v `pwd`/../../../supervisely_lib:/src/supervisely_lib \
	-v `pwd`/../src:/src \
	-v `pwd`/../../../nn/unet_v2/src:/src/unet_src \
	-v `pwd`/../data:/sly_task_data \
	-p 8888:8888 02_unet_inference:latest bash
