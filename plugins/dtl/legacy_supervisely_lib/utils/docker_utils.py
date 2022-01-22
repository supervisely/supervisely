# coding: utf-8

def remove_containers(label_filter):
    import docker
    dc = docker.from_env()
    stop_list = dc.containers.list(all=True, filters=label_filter)
    for cont in stop_list:
        cont.remove(force=True)
    return stop_list
