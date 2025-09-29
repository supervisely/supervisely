from typing import Optional, Tuple

from supervisely.api.api import Api
from supervisely.api.entities_collection_api import EntitiesCollectionInfo


def get_interval_period(sec: int) -> Tuple[str, int]:
    """
    Returns the period and interval based on the given seconds.
    :param sec: Number of seconds
    :return: Tuple of (period, interval)
    """
    if sec is None:
        return None, None
    if sec < 60:
        period = "s"
        interval = sec
    elif sec // 60 < 60:
        period = "min"
        interval = sec // 60
    elif sec // 3600 < 24:
        period = "h"
        interval = sec // 3600
    else:
        period = "d"
        interval = sec // 86400
    return period, interval


def get_seconds_from_period_and_interval(period: str, interval: int) -> int:
    """
    Returns the number of seconds based on the given period and interval.
    :param period: Period of time ('min', 'h', 'd')
    :param interval: Interval value
    :return: Number of seconds
    """
    if period == "min":
        return interval * 60
    elif period == "h":
        return interval * 3600
    elif period == "d":
        return interval * 86400
    else:
        raise ValueError(f"Unknown period: {period}")


def find_agents(api: Api, team_id: int) -> int:
    agents = api.agent.get_list_available(team_id, show_public=True, has_gpu=True)
    if len(agents) == 0:
        raise ValueError("No available agents found.")
    agent_id_memory_map = {}
    kubernetes_agents = []
    for agent in agents:
        if "4090" not in agent.name:  # ! TODO: remove after testing
            continue
        if agent.type == "sly_agent":
            # No multi-gpu support, always take the first one
            agent_id_memory_map[agent.id] = agent.gpu_info["device_memory"][0]["available"]
        elif agent.type == "kubernetes":
            kubernetes_agents.append(agent.id)
    if len(agent_id_memory_map) > 0:
        return sorted(agent_id_memory_map, key=agent_id_memory_map.get, reverse=True)
    if len(kubernetes_agents) > 0:
        return kubernetes_agents


def find_agent(api: Api, team_id: int) -> int:
    agents = find_agents(api, team_id)
    if agents is not None:
        if len(agents) > 0:
            return agents[0]
    raise ValueError("No available agents found.")


def get_last_split_collection(
    api: Api, project_id: int, prefix: str
) -> Tuple[Optional[EntitiesCollectionInfo], int]:
    last_collection_idx = 0
    last_collection = None
    for collection_info in api.entities_collection.get_list(project_id):
        if collection_info.name.startswith(prefix):
            try:
                suffix = collection_info.name.split("_")[-1]
                if suffix == "latest":
                    continue
                elif suffix.isdigit():
                    collection_idx = int(suffix)
                    if collection_idx > last_collection_idx:
                        last_collection_idx = collection_idx
                        last_collection = collection_info
            except Exception:
                continue
    return last_collection, last_collection_idx


def get_last_val_collection(
    api: Api, project_id: int
) -> Tuple[Optional[EntitiesCollectionInfo], int]:
    val_collection, val_idx = get_last_split_collection(api, project_id, "val_")
    return val_collection, val_idx
