#!/bin/bash

MY_PROJ_NAME="deeplab_v3plus__ag"

PYCHARM_ROOT_LOCAL="/pycharm/installation/path"
PYCHARM_ROOT_CONT="/pycharm"
PYCHARM_SETTINGS_LOCAL="/home/user/pycharm-settings/import_export_nn__${MY_PROJ_NAME}"
PYCHARM_SETTINGS_CONT="/root/.PyCharm2018.1"
PYCHARM_SETTINGS_LOCAL2="/home/user/pycharm-settings/import_export_nn__${MY_PROJ_NAME}__idea"
PYCHARM_SETTINGS_CONT2="/workdir/.idea"

PYCHARM_VOLUMES="-v ${PYCHARM_ROOT_LOCAL}:${PYCHARM_ROOT_CONT} \
                 -v ${PYCHARM_SETTINGS_LOCAL}:${PYCHARM_SETTINGS_CONT} \
                 -v ${PYCHARM_SETTINGS_LOCAL2}:${PYCHARM_SETTINGS_CONT2}"


TOP_DATA_DIR="/my/own/data/directory"

# TASK_SUBPATH="task_train/task000"
# TASK_SUBPATH="task_inf/task000"

##############################################################################

LOCAL_ENVS="-e DEBUG_LOG_TO_FILE=1 -e DEBUG_PATCHES_PROB=0.1 -e DEBUG_COPY_IMAGES=1"

LOCAL_VOLUMES="${PYCHARM_VOLUMES} \
               -v ${TOP_DATA_DIR}:/data"

# SLY_TASK_DATA="${TOP_DATA_DIR}/deeplab_v3plus-trains/${TASK_SUBPATH}"
