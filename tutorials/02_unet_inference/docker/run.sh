#docker build -t supervisely_detector_api . && \
#docker run --rm -it \
#			-p 8888:8888 \
#			-v `pwd`/../src:/src \
#			supervisely_detector_api bash
#docker run --rm -it -v `pwd`/supervisely_lib:/src/supervisely_lib -v `pwd`/nn:/src/nn -v `pwd`/tutorials/01_person_detector_api/src:/src -p 8888:8888 01_person_detector_api:latest bash 
docker build -t 02_unet_inference .
docker run --rm -it --init --runtime=nvidia -v `pwd`/supervisely_lib:/src/supervisely_lib -v `pwd`/nn/unet_v2/src:/src/unet_src -v `pwd`/tutorials/02_unet_inference/src:/src -v `pwd`/task_data:/sly_task_data  -p 8888:8888 02_unet_inference:latest  bash
