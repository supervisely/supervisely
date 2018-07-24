# coding: utf-8

import os

import supervisely_lib as sly
from supervisely_lib import logger

from worker.agent import Agent
from worker import constants


def parse_envs():
    args_req = {x: sly.required_env(x) for x in [
        'AGENT_HOST_DIR',
        'SERVER_ADDRESS',
        'ACCESS_TOKEN',
        'DOCKER_LOGIN',
        'DOCKER_PASSWORD',
        'DOCKER_REGISTRY',
    ]}
    args_opt = {x: os.getenv(x, def_val) for x, def_val in [
        ('WITH_LOCAL_STORAGE', 'true'),
        ('UPLOAD_RESULT_IMAGES', 'false'),
        ('PULL_ALWAYS', 'true'),
        ('DELETE_TASK_DIR_ON_FINISH', 'true'),
    ]}
    args = {**args_opt, **args_req}
    return args


def flag_from_env(s):
    res = s.upper() in ['TRUE', 'YES', '1']
    return res


def main(args):
    logger.info('ENVS', extra={**args, 'DOCKER_PASSWORD': 'hidden'})

    constants.HOST_DIR = args['AGENT_HOST_DIR']
    constants.AGENT_TASKS_DIR_HOST = os.path.join(constants.HOST_DIR, 'tasks')

    constants.SERVER_ADDRESS = args['SERVER_ADDRESS']
    constants.TOKEN = args['ACCESS_TOKEN']
    constants.TASKS_DOCKER_LABEL = 'supervisely_{}'.format(constants.TOKEN)

    constants.DOCKER_LOGIN = args['DOCKER_LOGIN']
    constants.DOCKER_PASSWORD = args['DOCKER_PASSWORD']
    constants.DOCKER_REGISTRY = args['DOCKER_REGISTRY']

    constants.WITH_LOCAL_STORAGE = flag_from_env(args['WITH_LOCAL_STORAGE'])
    constants.UPLOAD_RESULT_IMAGES = flag_from_env(args['UPLOAD_RESULT_IMAGES'])
    constants.PULL_ALWAYS = flag_from_env(args['PULL_ALWAYS'])

    constants.DELETE_TASK_DIR_ON_FINISH = flag_from_env(args['DELETE_TASK_DIR_ON_FINISH'])

    agent = Agent()
    agent.inf_loop()
    agent.wait_all()


if __name__ == '__main__':
    sly.add_default_logging_into_file(logger, constants.AGENT_LOG_DIR)
    sly.main_wrapper('agent', main, parse_envs())
