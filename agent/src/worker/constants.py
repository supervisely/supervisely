# coding: utf-8

import os
from urllib.parse import urlparse
import supervisely_lib as sly


def HOST_DIR():
    return os.environ['AGENT_HOST_DIR']


def SERVER_ADDRESS():
    str_url = os.environ['SERVER_ADDRESS']
    if ('http://' not in str_url) and ('https://' not in str_url):
        str_url = os.path.join('http://', str_url) #@TODO: raise with error
    parsed_uri = urlparse(str_url)
    server_address = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
    return server_address


def TOKEN():
    return os.environ['ACCESS_TOKEN']


def TASKS_DOCKER_LABEL():
    return 'supervisely_{}'.format(TOKEN())


def DOCKER_LOGIN():
    return os.environ['DOCKER_LOGIN']


def DOCKER_PASSWORD():
    return os.environ['DOCKER_PASSWORD']


def DOCKER_REGISTRY():
    return os.environ['DOCKER_REGISTRY']


def AGENT_TASKS_DIR_HOST():
    return os.path.join(HOST_DIR(), 'tasks')


def DELETE_TASK_DIR_ON_FINISH():
    return sly.env.flag_from_env(os.getenv('DELETE_TASK_DIR_ON_FINISH', 'true'))


def DELETE_TASK_DIR_ON_FAILURE():
    return sly.env.flag_from_env(os.getenv('DELETE_TASK_DIR_ON_FAILURE', 'false'))


def AGENT_ROOT_DIR():
    return '/sly_agent'


def AGENT_LOG_DIR():
    return os.path.join(AGENT_ROOT_DIR(), 'logs')


def AGENT_TASKS_DIR():
    return os.path.join(AGENT_ROOT_DIR(), 'tasks')


def AGENT_TMP_DIR():
    return os.path.join(AGENT_ROOT_DIR(), 'tmp')


def AGENT_IMPORT_DIR():
    return os.path.join(AGENT_ROOT_DIR(), 'import')


def AGENT_STORAGE_DIR():
    return os.path.join(AGENT_ROOT_DIR(), 'storage')


def WITH_LOCAL_STORAGE():
    return sly.env.flag_from_env(os.getenv('WITH_LOCAL_STORAGE', 'true'))


def UPLOAD_RESULT_IMAGES():
    return sly.env.flag_from_env(os.getenv('UPLOAD_RESULT_IMAGES', 'true'))


def PULL_ALWAYS():
    return sly.env.flag_from_env(os.getenv('PULL_ALWAYS', 'true'))


def TIMEOUT_CONFIG_PATH():
    use_default_timeouts = sly.env.flag_from_env(os.getenv('DEFAULT_TIMEOUTS', 'true'))
    return None if use_default_timeouts else '/workdir/src/configs/timeouts_for_stateless.json'


def NETW_CHUNK_SIZE():
    return 1048576


def BATCH_SIZE_GET_IMAGES_INFO():
    return 100


def BATCH_SIZE_DOWNLOAD_IMAGES():
    return 20


def BATCH_SIZE_DOWNLOAD_ANNOTATIONS():
    return 1000


def BATCH_SIZE_UPLOAD_IMAGES():
    return 1000


def BATCH_SIZE_UPLOAD_ANNOTATIONS():
    return 1000


def BATCH_SIZE_ADD_IMAGES():
    return 1000


def BATCH_SIZE_LOG():
    return 100


def init_constants():
    sly.fs.mkdir(AGENT_LOG_DIR())
    sly.fs.mkdir(AGENT_TASKS_DIR())
    sly.fs.mkdir(AGENT_STORAGE_DIR())
    sly.fs.mkdir(AGENT_TMP_DIR())
    sly.fs.mkdir(AGENT_IMPORT_DIR())
    os.chmod(AGENT_IMPORT_DIR(), 0o777)  # octal