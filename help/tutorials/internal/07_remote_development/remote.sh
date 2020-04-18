docker build \
    --build-arg SSH_PASSWORD=17711771 \
    --build-arg DOCKER_IMAGE=docker.deepsystems.io/supervisely/five/import-video-sly:master \
    -f Dockerfile_remote \
    .
    
#docker build -f Dockerfile_remote -t remote_dev .

# nvidia-docker run \
#     --name test_sshd \
#     --rm \
#     -ti \
#     -v ${PWD}/src:/workdir/src \
#     -v ${PWD}/../../../supervisely_lib:/workdir/supervisely_lib \
#     -v '/home/ds/work/sly_private:/sly_private' \
#     -v '/home/ds/work/task_data_videos_sly_import example:/sly_task_data' \
#     -p 49154:22 \
#     remote_dev \
#     bash

    #/usr/sbin/sshd -D

# ssh root@192.168.1.210 -p 49154