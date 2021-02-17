# coding: utf-8

import os

import supervisely_lib as sly

from worker import constants
from worker.agent import Agent


def parse_envs():
    args_req = {x: os.environ[x] for x in constants.get_required_settings()}
    args_opt = {x: constants.read_optional_setting(x) for x in constants.get_optional_defaults().keys()}
    args = {**args_opt, **args_req}
    return args


def main(args):
    sly.logger.info('ENVS', extra={**args, constants._DOCKER_PASSWORD: 'hidden'})
    agent = Agent()
    agent.inf_loop()
    agent.wait_all()


if __name__ == '__main__':
    constants.init_constants()  # Set up the directories.
    sly.add_default_logging_into_file(sly.logger, constants.AGENT_LOG_DIR())
    sly.main_wrapper('agent', main, parse_envs())
