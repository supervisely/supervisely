import os
from pathlib import Path
import shlex
import subprocess
from supervisely.io.fs import mkdir
from supervisely.api.api import Api
from supervisely.sly_logger import logger

VPN_CONFIGURATION_DIR = "~/supervisely-network"


def connect_to_supervisely_vpn_network():
    api = Api()
    current_dir = Path(__file__).parent.absolute()
    script_path = os.path.join(current_dir, "sly-net.sh")
    network_dir = os.path.expanduser(VPN_CONFIGURATION_DIR)
    mkdir(network_dir)

    process = subprocess.run(
        shlex.split(f"{script_path} up {api.token} {api.server_address} {network_dir}"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    try:
        process.check_returncode()
        logger.info("You have been successfully connected to Supervisely VPN Network")
    except subprocess.CalledProcessError as e:
        e.cmd[2] = "***-api-token-***"
        if "wg0' already exists" in e.stderr:
            logger.info("You already connected to Supervisely VPN Network")
            pass
        else:
            raise e


def create_debug_task(team_id):
    api = Api()
    me = api.user.get_my_info()
    session_name = me.login + "-development"
    module_id = api.app.get_ecosystem_module_id(
        "supervisely-ecosystem/while-true-script-v2"
    )
    sessions = api.app.get_sessions(
        team_id, module_id, only_running=True, session_name=session_name
    )
    if len(sessions) == 0:
        # run task
        api.task.start(
            agent_id=None,
        )
        pass
    else:
        task = sessions[0]
    x = 10
