# coding: utf-8

import docker


def remove_containers(label_filter):
    dc = docker.from_env()
    stop_list = dc.containers.list(all=True, filters=label_filter)
    for cont in stop_list:
        cont.remove(force=True)
    return stop_list
