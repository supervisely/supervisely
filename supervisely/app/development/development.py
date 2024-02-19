import os
import shutil
import stat
import subprocess
from typing import Optional

import requests

from supervisely.api.api import Api
from supervisely.io.fs import mkdir
from supervisely.sly_logger import logger

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

VPN_CONFIGURATION_DIR = "~/supervisely-network"


def supervisely_vpn_network(
    action: Optional[Literal["up", "down"]] = "up",
    raise_on_error: Optional[bool] = True,
) -> None:
    """Connects to Supervisely network using WireGuard VPN.
    Reads Supervisely API settings from the environment variables.

    :param action: The action to perform, either "up" or "down".
    :type action: Optional[Literal["up", "down"]]
    :param raise_on_error: If True, an exception will be raised if an error occurs.
    :type raise_on_error: Optional[bool]
    :raises RuntimeError: If wg-quick is not available in the system and raise_on_error is True.
    :raises subprocess.CalledProcessError: If an error occurs while connecting and raise_on_error is True.
    :raises RuntimeError: If an error occurs while connecting and raise_on_error is True.
    """
    if shutil.which("wg-quick") is None:
        if raise_on_error:
            raise RuntimeError(
                "wg-quick is not available in the system. "
                "Please refer to this documentation to install required packages: "
                "https://developer.supervisely.com/app-development/advanced/advanced-debugging#prepare-environment"
            )
        else:
            logger.error(
                "wg-quick is not available in the system. "
                "Please refer to this documentation to install required packages: "
                "https://developer.supervisely.com/app-development/advanced/advanced-debugging#prepare-environment"
            )
            return
    else:
        logger.info("wg-quick is available in the system, will try to connect to VPN.")

    logger.info("wg-quick reqires root privileges, you may be asked to enter your password.")
    network_dir = os.path.expanduser(VPN_CONFIGURATION_DIR)
    mkdir(network_dir)
    os.chdir(network_dir)

    config_file = os.path.join(network_dir, "wg0.conf")
    public_key = os.path.join(network_dir, "public.key")
    private_key = os.path.join(network_dir, "private.key")

    logger.info(f"VPN configuration directory: {network_dir}")
    api = Api()

    # Try to bring down the connection if it's active.
    logger.info("Trying to bring down the connection...")
    try:
        subprocess.run(
            ["wg-quick", "down", config_file],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info("Connection has been brought down successfully.")
    except subprocess.CalledProcessError as e:
        logger.info(f"The connection was not active: {e.stderr.decode()}.")

    # Generate the private and public keys if they don't exist.
    if not os.path.exists(public_key):
        logger.info(f"Public key not found in {public_key}, generating a new one...")
        with open(private_key, "w") as priv_key_file:
            subprocess.run(["wg", "genkey"], stdout=priv_key_file)
        os.chmod(private_key, stat.S_IRUSR | stat.S_IWUSR)  # Change file permissions to 600
        with open(private_key) as priv_key_file, open(public_key, "w") as pub_key_file:
            subprocess.run(["wg", "pubkey"], stdin=priv_key_file, stdout=pub_key_file)
    else:
        logger.info(f"Public key found in {public_key}, using it...")

    # Register the connection with the server.
    logger.info(f"Registering the connection with the server: {api.server_address}...")
    response = requests.post(
        f"{api.server_address}/net/register/{api.token}/{open(public_key).read().strip()}",
        timeout=5,
    )
    response.raise_for_status()
    logger.info(f"Connection registered with the server, status: {response.status_code}.")
    ip, server_public_key, server_endpoint, subnet_base, gateway = response.text.split(";")

    # Update the configuration file with the server's parameters.
    logger.info(
        f"Updating wg0.conf with the following parameters: IP: {ip} "
        f"Server Public Key: {server_public_key}, Server endpoint: {server_endpoint} "
        f"Subnet base: {subnet_base}, Gateway: {gateway}"
    )
    with open(config_file, "w") as f:
        f.write(
            f"""
            [Interface]
            PrivateKey = {open(private_key).read().strip()}
            Address = {ip}/16
            [Peer]
            PublicKey = {server_public_key}
            AllowedIPs = {subnet_base}
            Endpoint = {server_endpoint}
            PersistentKeepalive = 25
            """
        )
    os.chmod(config_file, 0o600)

    logger.info("Configuration saved, bringing up the connection...")
    try:
        subprocess.run(
            ["wg-quick", action, config_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(
            "WireGuard interface has been successfully brought up, checking the connection..."
        )
    except subprocess.CalledProcessError as e:
        if raise_on_error:
            raise
        else:
            logger.warning(f"Error while connecting to VPN, try again. Error: {e.stderr.decode()}")
            return

    # Check the connection.
    test_response = requests.get(f"http://{gateway}", timeout=5)
    test_response.raise_for_status()

    if not test_response.ok:
        logger.warning(f"Error while connecting to VPN, try again. Error: {test_response.text}")
        if raise_on_error:
            raise RuntimeError(f"Error while connecting to VPN: {test_response.text}")
    else:
        logger.info(f"VPN connection has been successfully established to {gateway}")


def create_debug_task(team_id, port="8000"):
    api = Api()
    me = api.user.get_my_info()
    session_name = me.login + "-development"
    module_id = api.app.get_ecosystem_module_id("supervisely-ecosystem/while-true-script-v2")
    sessions = api.app.get_sessions(team_id, module_id, session_name=session_name)
    redirect_requests = {"token": api.token, "port": port}
    task = None
    for session in sessions:
        if (session.details["meta"].get("redirectRequests") == redirect_requests) and (
            session.details["status"] == str(api.app.Status.QUEUED)
        ):
            task = session.details
            if "id" not in task:
                task["id"] = task["taskId"]
            logger.info(f"Debug task already exists: {task['id']}")
            break
    workspaces = api.workspace.get_list(team_id)
    if task is None:
        task = api.task.start(
            agent_id=None,
            module_id=module_id,
            workspace_id=workspaces[0].id,
            task_name=session_name,
            redirect_requests=redirect_requests,
            proxy_keep_url=False,  # to ignore /net/<token>/endpoint
        )
        if type(task) is list:
            task = task[0]
        logger.info(f"Debug task has been successfully created: {task['taskId']}")
    return task
