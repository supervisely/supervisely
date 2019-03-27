# coding: utf-8

import os
import supervisely_lib as sly

from worker import constants
constants.init_constants()
from worker.agent import Agent


def parse_envs():
    args_req = {x: os.environ[x] for x in [
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
        ('DEFAULT_TIMEOUTS', 'true'),
        ('DELETE_TASK_DIR_ON_FINISH', 'true'),
        ('DELETE_TASK_DIR_ON_FAILURE', 'false'),
    ]}
    args = {**args_opt, **args_req}
    return args


def main(args):
    sly.logger.info('ENVS', extra={**args, 'DOCKER_PASSWORD': 'hidden'})
    agent = Agent()
    agent.inf_loop()
    agent.wait_all()


if __name__ == '__main__':
    sly.add_default_logging_into_file(sly.logger, constants.AGENT_LOG_DIR())
    sly.main_wrapper('agent', main, parse_envs())
